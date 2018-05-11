import os, sys, time, argparse, random, glob
import numpy as np
import tensorflow as tf
os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['PYTHONHASHSEED'] = '54321'
np.random.seed(54321)
random.seed(54321)
tf.set_random_seed(54321)
from keras import backend as K
if os.environ['KERAS_BACKEND'] == 'tensorflow':
    session_conf = tf.ConfigProto(intra_op_parallelism_threads=8, inter_op_parallelism_threads=8)
    sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
    K.set_session(sess)
import cPickle as pickle
from sklearn.svm import SVC
from sklearn.utils import shuffle
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from scipy import signal

import keras
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Dropout, Concatenate, Lambda, Add, LeakyReLU
from keras.optimizers import Adam
from keras.layers.noise import GaussianNoise

import util

def prepareData(dataRaw, dataset, wavelengths, materials, objects_train, objects_test, Xtrain=None, ytrain=None, Xtest=None, ytest=None, filterLuminiData=False):
    if dataset == 'scio':
        if Xtrain is None:
            Xtrain, ytrain, _ = util.processScioDataset(dataRaw, materials, objects_train, sampleCount=100, spectrumRaw='spectrum')
            Xtest, ytest, _ = util.processScioDataset(dataRaw, materials, objects_test, sampleCount=100, spectrumRaw='spectrum')
    elif dataset == 'lumini':
        if Xtrain is None:
            Xtrain, ytrain, _ = util.processLuminiDataset(dataRaw, materials, objects_train, sampleCount=100, exposure=500, correctedValues=True)
            Xtest, ytest, _ = util.processLuminiDataset(dataRaw, materials, objects_test, sampleCount=100, exposure=500, correctedValues=True)
        if filterLuminiData:
            Xtrain = lowpassfilter(Xtrain, order=5, freq=0.2)
            Xtest = lowpassfilter(Xtest, order=5, freq=0.2)
    # Numerical differentiation
    Xtrain = util.firstDeriv(Xtrain, wavelengths)
    Xtest = util.firstDeriv(Xtest, wavelengths)

    # Zero mean unit variance
    scaler = preprocessing.StandardScaler()
    Xtrain = scaler.fit_transform(Xtrain)
    Xtest = scaler.transform(Xtest)
    return Xtrain, ytrain, Xtest, ytest

def lowpassfilter(data, order=5, freq=0.5, realtime=False):
    b, a = signal.butter(order, freq, analog=False)
    if realtime:
        # Real time filtering
        zi = signal.lfilter_zi(b, a)
        filtered = []
        for d in data:
            s_filtered, zi = signal.lfilter(b, a, [d], zi=zi)
            filtered.append(s_filtered)
    else:
        # Forward backward filtering (no time delay, but cannot be used for real time filtering)
        filtered = signal.filtfilt(b, a, data)
    return np.array(filtered)

def learnNNSVM(Xtrain, ytrain, Xtest, ytest, numLabeled=None, epochs=100, batchSize=64, materialCount=5, verbose=False, algorithm='svm', objectsTrain=None):
    np.random.seed(54321)
    random.seed(54321)
    tf.set_random_seed(54321)

    # Select labeled data
    if numLabeled is None:
        x_labeled = Xtrain
        y_labeled = ytrain
    else:
        x_labeled = np.concatenate([Xtrain[objectsTrain==objectName][:numLabeled] for objectName in set(objectsTrain)], axis=0)
        y_labeled = np.concatenate([ytrain[objectsTrain==objectName][:numLabeled] for objectName in set(objectsTrain)], axis=0)
    if verbose:
        print numLabeled, 'x_labeled:', np.shape(x_labeled), 'y_labeled:', np.shape(y_labeled), 'Xtrain:', np.shape(Xtrain), 'ytrain:', np.shape(ytrain)
    x_labeled, y_labeled = shuffle(x_labeled, y_labeled)

    if algorithm == 'nn' or algorithm == 'residualnn':
        y_labeled = keras.utils.to_categorical(y_labeled, num_classes=materialCount)
        ytest2 = keras.utils.to_categorical(ytest, num_classes=materialCount)

        if algorithm == 'nn':
            d = [32, 32, 32, 32, 64, 64, 64, 64, 128, 128, 128, 128, 256, 256, 256, 256]
            model = Sequential()
            model.add(Dense(d[0], activation='linear', input_dim=np.shape(Xtrain)[-1]))
            model.add(Dropout(0.25))
            model.add(LeakyReLU())
            for dd in d[1:]:
                model.add(Dense(dd, activation='linear'))
                model.add(Dropout(0.25))
                model.add(LeakyReLU())
            model.add(Dense(materialCount, activation='softmax'))
            model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.0005), metrics=['accuracy'])

        elif algorithm == 'residualnn':
            d = [32, 64, 128, 256]
            cardinality = 4
            dc = [dd/cardinality for dd in d]
            disc_input = Input(shape=(Xtrain.shape[1],))
            for j in xrange(len(d)):
                x = Dense(d[j], activation='linear')(disc_input if j == 0 else x)
                x = Dropout(0.25)(x)
                xx = LeakyReLU()(x)
                groups = []
                for i in xrange(cardinality):
                    group = Lambda(lambda z: z[:, i*dc[j]:(i+1)*dc[j]])(xx)
                    x = Dense(d[j], activation='linear')(group)
                    x = Dropout(0.25)(x)
                    x = LeakyReLU()(x)
                    x = Dense(dc[j], activation='linear')(x)
                    x = Dropout(0.25)(x)
                    x = LeakyReLU()(x)
                    groups.append(Dense(d[j], activation='linear')(x))
                x = Add()(groups)
                x = Dropout(0.25)(x)
                x = LeakyReLU()(x)
                x = Add()([xx, x])
            disc_output = Dense(materialCount, activation='softmax')(x)
            model = Model(inputs=disc_input, outputs=disc_output)
            model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.0005), metrics=['accuracy'])

        model.fit(x_labeled, y_labeled, epochs=epochs, batch_size=batchSize, validation_split=0.0, verbose=(1 if verbose else 0), validation_data=(Xtest, ytest2))
        cm = confusion_matrix(ytest, model.predict(Xtest, verbose=0).argmax(axis=-1), labels=range(materialCount))
        # Return accuracy
        return model.evaluate(Xtest, ytest2, batch_size=batchSize, verbose=0)[-1], cm

    elif algorithm == 'svm':
        svm = SVC(kernel='linear')
        svm.fit(x_labeled, y_labeled)

        cm = confusion_matrix(ytest, svm.predict(Xtest), labels=range(materialCount))
        # Return accuracy
        return svm.score(Xtest, ytest), cm


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Classification of Household Materials via Spectroscopy')
    parser.add_argument('-t', '--test', nargs='+', help='Which test? (0) K-fold CV, (1) Leave-one-object-out, (2) Leave-one-object-out with varying number of training objects', required=True)
    parser.add_argument('-a', '--algorithm', nargs='+', help='svm, nn, residualnn', required=True)
    parser.add_argument('-v', '--verbose', help='Verbose', action='store_true')
    args = parser.parse_args()
    algorithm = args.algorithm[0]

    materials = ['plastic', 'fabric', 'paper', 'wood', 'metal']
    plastics = ['HDPE', 'PET', 'polyethyleneBlue', 'polyethyleneGreen', 'polyethyleneRed', 'polyethyleneYellow', 'PP', 'PVC', 'thermoPolypropylene', 'thermoTeflon']
    fabrics = ['cottonCanvas', 'cottonSweater', 'cottonTowel', 'denim', 'felt', 'flannel', 'gauze', 'linen', 'satin', 'wool']
    papers = ['cardboard', 'constructionPaperGreen', 'constructionPaperOrange', 'constructionPaperRed', 'magazinePaper', 'newspaper', 'notebookPaper', 'printerPaper', 'receiptPaper', 'textbookPaper']
    woods = ['ash', 'cherry', 'curlyMaple', 'hardMaple', 'hickory', 'redCedar', 'redElm', 'redOak', 'walnut', 'whiteOak']
    metals = ['aluminum', 'aluminumFoil', 'brass', 'copper', 'iron', 'lead', 'magnesium', 'steel', 'titanium', 'zinc']
    objects = [plastics, fabrics, papers, woods, metals]

    epochs = 150
    batchSize = 64

    # Check that the datasets have been downloaded
    luminiFilenames = glob.glob(os.path.join('data', 'lumini*'))
    scioFilenames = glob.glob(os.path.join('data', 'scio*'))
    if not luminiFilenames and not scioFilenames:
        raise Exception('The SMM50 dataset must be downloaded before running this script.\nThis can be done using the command: \'wget -O smm50.tar.gz https://goo.gl/2X276V\'\nSee the following webpage for more details on downloading the SMM50 dataset: https://github.com/Healthcare-Robotics/smm50')

    t = time.time()
    if '0' in args.test:
        # K-fold cross validation
        # NOTE: Tables I and II in paper

        for dataset in ['lumini', 'scio']:
            print '\n', '-'*30
            print 'Using %s measurements' % dataset
            print '-'*30, '\n'
            if dataset == 'lumini':
                dataRaw, wavelengths = util.loadLuminiDataset()
                X, y, scioluminiObjects = util.processLuminiDataset(dataRaw, materials, objects, sampleCount=100, exposure=500, correctedValues=True)
            elif dataset == 'scio':
                dataRaw, wavelengths = util.loadScioDataset()
                X, y, scioluminiObjects = util.processScioDataset(dataRaw, materials, objects, sampleCount=100, spectrumRaw='spectrum')
            X = np.array(X)
            y = np.array(y)
            # Create a new y list for which each object has its own label, rather than all objects in the same material class having the same y label.
            # This is used for stratified k-fold CV
            objectSet = list(set(scioluminiObjects))
            yObjects = [objectSet.index(objectName) for objectName in scioluminiObjects]

            for numLabeledPerObject in [1, 10, 40, 80]:
                print 'Num training samples per object:', numLabeledPerObject
                accuracies = []
                confusionMatrix = None
                skf = StratifiedKFold(n_splits=5, shuffle=True)
                for trainIdx, testIdx in list(skf.split(X, yObjects)):
                    Xtrain, ytrain, Xtest, ytest = prepareData(dataRaw, dataset, wavelengths, materials, None, None, X[trainIdx], y[trainIdx], X[testIdx], y[testIdx], filterLuminiData=True)
                    accuracy, cm = learnNNSVM(Xtrain, ytrain, Xtest, ytest, numLabeled=numLabeledPerObject, epochs=epochs, batchSize=batchSize, materialCount=len(materials), verbose=args.verbose, algorithm=algorithm, objectsTrain=np.array(scioluminiObjects)[trainIdx])
                    accuracies.append(accuracy)
                    if confusionMatrix is None:
                        confusionMatrix = cm
                    else:
                        confusionMatrix += cm
                    print 'Test accuracy:', accuracies[-1]
                    sys.stdout.flush()
                print 'Average accuracy:', np.mean(accuracies)
                print 'Confusion matrix:'
                print np.array2string(confusionMatrix, separator=', ')
                print 'Confusion matrix normalized:'
                print np.array2string(confusionMatrix.astype('float') / confusionMatrix.sum(axis=1)[:, np.newaxis], separator=', ')
                sys.stdout.flush()

    elif '1' in args.test:
        # NOTE: Table III, confusion matrix, and full classification figure (Fig. 13) in paper

        for dataset in ['lumini', 'scio']:
            print '\n', '-'*30
            print 'Using %s measurements' % dataset
            print '-'*30, '\n'
            dataRaw, wavelengths = util.loadScioDataset() if dataset == 'scio' else util.loadLuminiDataset()

            genCVAccuracies = []
            confusionMatrix = None
            objSet = []
            objectConfusionMatrix = []
            for i, objectSet in enumerate(objects):
                for objName in objectSet:
                    newSet = [x for x in objectSet if x != objName]
                    objects_train = [x if i != j else newSet for j, x in enumerate(objects)]
                    objects_test = [[]]*i + [[objName]] + [[]]*(len(materials) - 1 - i)
                    Xtrain, ytrain, Xtest, ytest = prepareData(dataRaw, dataset, wavelengths, materials, objects_train, objects_test, filterLuminiData=True)
                    accuracy, cm = learnNNSVM(Xtrain, ytrain, Xtest, ytest, numLabeled=None, epochs=epochs, batchSize=batchSize, materialCount=len(materials), verbose=args.verbose, algorithm=algorithm)
                    genCVAccuracies.append(accuracy)
                    if confusionMatrix is None:
                        confusionMatrix = cm
                    else:
                        confusionMatrix += cm
                    objSet.append(objName)
                    objectConfusionMatrix.append(np.copy(cm[i]))
                    print objects_test, accuracy
                    sys.stdout.flush()
            print 'Average accuracy:', np.mean(genCVAccuracies)
            print 'Confusion matrix:'
            print np.array2string(confusionMatrix, separator=', ')
            print 'Confusion matrix normalized:'
            print np.array2string(confusionMatrix.astype('float') / confusionMatrix.sum(axis=1)[:, np.newaxis], separator=', ')
            print 'Object label set for confusion matrix:'
            print objSet
            print 'Object confusion matrix:'
            print np.array2string(np.array(objectConfusionMatrix), separator=', ')
            sys.stdout.flush()

    elif '2' in args.test:
        # NOTE: LOOO performance as we increase the number of training objects. Fig. 15 in paper

        numTrainObjects = range(1, 11)
        for dataset in ['lumini', 'scio']:
            print '\n', '-'*30
            print 'Using %s measurements' % dataset
            print '-'*30, '\n'
            dataRaw, wavelengths = util.loadScioDataset() if dataset == 'scio' else util.loadLuminiDataset()

            for nto in numTrainObjects:
                print 'Number of training objects:', nto
                genCVAccuracies = []
                confusionMatrix = None
                for i, objectSet in enumerate(objects):
                    for objName in objectSet:
                        newSet = [x for x in objectSet if x != objName]
                        objects_train = [x[:nto] if i != j else newSet[:nto] for j, x in enumerate(objects)]
                        objects_test = [[]]*i + [[objName]] + [[]]*(len(materials) - 1 - i)
                        Xtrain, ytrain, Xtest, ytest = prepareData(dataRaw, dataset, wavelengths, materials, objects_train, objects_test, filterLuminiData=True)
                        accuracy, cm = learnNNSVM(Xtrain, ytrain, Xtest, ytest, numLabeled=None, epochs=epochs, batchSize=batchSize, materialCount=len(materials), verbose=args.verbose, algorithm=algorithm)
                        genCVAccuracies.append(accuracy)
                        if confusionMatrix is None:
                            confusionMatrix = cm
                        else:
                            confusionMatrix += cm
                        print objects_test, accuracy
                        sys.stdout.flush()
                avgAccuracy = np.mean(genCVAccuracies)
                print 'Average accuracy:', avgAccuracy
                print 'Confusion matrix:'
                print np.array2string(confusionMatrix, separator=', ')
                print 'Confusion matrix normalized:'
                print np.array2string(confusionMatrix.astype('float') / confusionMatrix.sum(axis=1)[:, np.newaxis], separator=', ')
                sys.stdout.flush()

    print 'Run time:', time.time() - t, 'seconds'
    sys.stdout.flush()

