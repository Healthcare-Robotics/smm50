import os, sys, time, argparse, glob
import numpy as np
import cPickle as pickle
from sklearn.utils import shuffle
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from scipy import signal

import keras
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Dropout, LeakyReLU
from keras.optimizers import Adam

def firstDeriv(x, wavelengths):
    # First derivative of measurements with respect to wavelength
    x = np.copy(x)
    for i, xx in enumerate(x):
        dx = np.zeros(xx.shape, np.float)
        dx[0:-1] = np.diff(xx)/np.diff(wavelengths)
        dx[-1] = (xx[-1] - xx[-2])/(wavelengths[-1] - wavelengths[-2])
        x[i] = dx
    return x

def prepareData(Xtrain, Xtest, wavelengths, filterData=False, deriv=True):
    if filterData:
        Xtrain = lowpassfilter(Xtrain, order=5, freq=0.2)
        Xtest = lowpassfilter(Xtest, order=5, freq=0.2)

    if deriv:
        # Finite difference (Numerical differentiation)
        Xtrain = firstDeriv(Xtrain, wavelengths)
        Xtest = firstDeriv(Xtest, wavelengths)

    # Zero mean unit variance
    scaler = preprocessing.StandardScaler()
    Xtrain = scaler.fit_transform(Xtrain)
    Xtest = scaler.transform(Xtest)
    return Xtrain, Xtest

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

def learn(Xtrain, ytrain, Xtest, ytest, numLabeled=None, epochs=100, batchSize=64, materialCount=5, verbose=False, objectsTrain=None):
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

    y_labeled = keras.utils.to_categorical(y_labeled, num_classes=materialCount)
    ytest2 = keras.utils.to_categorical(ytest, num_classes=materialCount)

    d = [64]*2 + [32]*2
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

    model.fit(x_labeled, y_labeled, epochs=epochs, batch_size=batchSize, validation_split=0.0, verbose=(1 if verbose else 0), validation_data=(Xtest, ytest2))
    cm = confusion_matrix(ytest, model.predict(Xtest, verbose=0).argmax(axis=-1), labels=range(materialCount))
    # Return accuracy
    return model.evaluate(Xtest, ytest2, batch_size=batchSize, verbose=0)[-1], cm


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Classification of Household Materials via Spectroscopy')
    parser.add_argument('-t', '--test', nargs='+', help='Which test? (0) K-fold CV, (1) Leave-one-object-out, (2) Leave-one-object-out with varying number of training objects', required=True)
    parser.add_argument('-v', '--verbose', help='Verbose', action='store_true')
    args = parser.parse_args()

    materials = ['plastic', 'fabric', 'paper', 'wood', 'metal']
    plastics = np.array(['HDPE', 'PET', 'polyethyleneBlue', 'polyethyleneGreen', 'polyethyleneRed', 'polyethyleneYellow', 'PP', 'PVC', 'thermoPolypropylene', 'thermoTeflon'])
    fabrics = np.array(['cottonCanvas', 'cottonSweater', 'cottonTowel', 'denim', 'felt', 'flannel', 'gauze', 'linen', 'satin', 'wool'])
    papers = np.array(['cardboard', 'constructionPaperGreen', 'constructionPaperOrange', 'constructionPaperRed', 'magazinePaper', 'newspaper', 'notebookPaper', 'printerPaper', 'receiptPaper', 'textbookPaper'])
    woods = np.array(['ash', 'cherry', 'curlyMaple', 'hardMaple', 'hickory', 'redCedar', 'redElm', 'redOak', 'walnut', 'whiteOak'])
    metals = np.array(['aluminum', 'aluminumFoil', 'brass', 'copper', 'iron', 'lead', 'magnesium', 'steel', 'titanium', 'zinc'])
    objectNames = np.array([plastics, fabrics, papers, woods, metals])

    epochs = 300
    batchSize = 32

    # Check that the datasets have been downloaded
    luminiScioFilenames = glob.glob(os.path.join('data', 'smm50_*.pkl'))
    pr2Filenames = glob.glob(os.path.join('data', 'pr2_*.pkl'))
    if not luminiScioFilenames or not pr2Filenames:
        raise Exception('The SMM50 dataset must be downloaded and extracted before running this script.\nThis can be done using the command: \'wget -O smm50.tar.gz https://goo.gl/2X276V\'\nSee the following webpage for more details on downloading the SMM50 dataset: https://github.com/Healthcare-Robotics/smm50')

    t = time.time()

    for dataset in ['lumini', 'scio']:
        print '\n', '-'*30
        print 'Using %s measurements' % dataset
        print '-'*30, '\n'
        saveFilename = os.path.join('data', 'smm50_%s.pkl' % dataset)
        if os.path.isfile(saveFilename):
            with open(saveFilename, 'rb') as f:
                X, y, scioluminiObjects, wavelengths = pickle.load(f)
        X = np.array(X)
        y = np.array(y)
        scioluminiObjects = np.array(scioluminiObjects)
        wavelengths = np.array(wavelengths)

        if '0' in args.test:
            # NOTE: K-fold cross validation. Table I in paper

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
                    Xtrain, Xtest = prepareData(X[trainIdx], X[testIdx], wavelengths, filterData=(dataset=='lumini'), deriv=True)
                    ytrain = y[trainIdx]
                    ytest = y[testIdx]
                    accuracy, cm = learn(Xtrain, ytrain, Xtest, ytest, numLabeled=numLabeledPerObject, epochs=epochs, batchSize=batchSize, materialCount=len(materials), verbose=args.verbose, objectsTrain=np.array(scioluminiObjects)[trainIdx])
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
            # NOTE: Leave-one-object-out cross-validation. Figure 11 and 12 in paper

            genCVAccuracies = []
            confusionMatrix = None
            objSet = []
            objectConfusionMatrix = []
            for i, objectSet in enumerate(objectNames):
                for objName in objectSet:
                    # Set up leave-one-object-out training and test sets
                    Xtrain = X[scioluminiObjects != objName]
                    ytrain = y[scioluminiObjects != objName]
                    Xtest = X[scioluminiObjects == objName]
                    ytest = y[scioluminiObjects == objName]

                    Xtrain, Xtest = prepareData(Xtrain, Xtest, wavelengths, filterData=(dataset=='lumini'), deriv=True)
                    accuracy, cm = learn(Xtrain, ytrain, Xtest, ytest, numLabeled=None, epochs=epochs, batchSize=batchSize, materialCount=len(materials), verbose=args.verbose)
                    genCVAccuracies.append(accuracy)
                    if confusionMatrix is None:
                        confusionMatrix = cm
                    else:
                        confusionMatrix += cm
                    objSet.append(objName)
                    objectConfusionMatrix.append(np.copy(cm[i]))
                    print objName, accuracy
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
            # NOTE: LOOO performance as we increase the number of training objects. Fig. 14 in paper

            numTrainObjects = range(1, 11)
            for nto in numTrainObjects:
                print 'Number of training objects:', nto
                genCVAccuracies = []
                confusionMatrix = None
                for i, objectSet in enumerate(objectNames):
                    for objName in objectSet:
                        trainObjects = [(x[:nto] if objName not in x else x[x != objName][:nto]) for x in objectNames]
                        trainObjects = [xx for x in trainObjects for xx in x] # flatten
                        indices = [x in trainObjects for x in scioluminiObjects]
                        Xtrain = X[indices]
                        ytrain = y[indices]
                        Xtest = X[scioluminiObjects == objName]
                        ytest = y[scioluminiObjects == objName]

                        Xtrain, Xtest = prepareData(Xtrain, Xtest, wavelengths, filterData=(dataset=='lumini'), deriv=True)
                        accuracy, cm = learn(Xtrain, ytrain, Xtest, ytest, numLabeled=None, epochs=epochs, batchSize=batchSize, materialCount=len(materials), verbose=args.verbose)
                        genCVAccuracies.append(accuracy)
                        if confusionMatrix is None:
                            confusionMatrix = cm
                        else:
                            confusionMatrix += cm
                        print objName, accuracy
                        sys.stdout.flush()
                avgAccuracy = np.mean(genCVAccuracies)
                print 'Average accuracy:', avgAccuracy
                print 'Confusion matrix:'
                print np.array2string(confusionMatrix, separator=', ')
                print 'Confusion matrix normalized:'
                print np.array2string(confusionMatrix.astype('float') / confusionMatrix.sum(axis=1)[:, np.newaxis], separator=', ')
                sys.stdout.flush()

        elif '3' in args.test:
            # NOTE: Evaluation on data collected from PR2 with household objects

            plastics_test = ['waterbottle', 'vaselinebottle', 'coffeecontainer', 'pillbottle', 'bag']
            fabrics_test = ['cardigan', 'tshirt', 'khakishorts', 'gown', 'sweatpants']
            papers_test = ['book', 'cardboard', 'cup', 'plate', 'napkins']
            woods_test = ['soapdispenser', 'bowl', 'spoon', 'largebowl', 'potter']
            metals_test = ['bottle', 'bowl', 'can', 'aluminumpan', 'steelpan']
            objects_test = [plastics_test, fabrics_test, papers_test, woods_test, metals_test]

            saveFilename = os.path.join('data', 'pr2_%s.pkl' % dataset)
            if os.path.isfile(saveFilename):
                with open(saveFilename, 'rb') as f:
                    Xtest, ytest, scioluminiObjectsTest, wavelengthsTest = pickle.load(f)
            Xtest = np.array(Xtest)
            ytest = np.array(ytest)
            scioluminiObjectsTest = np.array(scioluminiObjectsTest)
            wavelengthsTest = np.array(wavelengthsTest)
            Xtrain = X
            ytrain = y

            Xtrain, Xtest = prepareData(Xtrain, Xtest, wavelengths, filterData=(dataset=='lumini'), deriv=True)
            accuracy, cm = learn(Xtrain, ytrain, Xtest, ytest, numLabeled=None, epochs=epochs, batchSize=batchSize, materialCount=len(materials), verbose=args.verbose)

            print 'Accuracy:', accuracy
            print 'Confusion matrix:'
            print np.array2string(cm, separator=', ')

    print 'Run time:', time.time() - t, 'seconds'
    sys.stdout.flush()

