import plotly
import plotly.graph_objs as go
import numpy as np
from sklearn import preprocessing, decomposition, manifold
from scipy import signal
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

import util

def lowpassfilter(data, order=1, freq=0.01, realtime=False):
    b, a = signal.butter(3, 0.05, analog=False) # responsive, less smooth lines
    b, a = signal.butter(order, freq, analog=False) # smooth lines
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

''' Plot one sample from each object. Color code lines by material, or plot one sample from each material category. '''
def plot(dataSL, objects, materials, scio=True, spectrumExposure='spectrum', numSamples=1, materialIndex=None, filterData=False, xtitle=None, ytitle=None, wavelengths=None, filename='materialComparison', width=1000, height=750, title='Material Comparison', normalizeData=False, showlegend=True, legend=None, linedict=dict(), alpha=1.0):
    if normalizeData:
        mins = []
        maxs = []
        for i, objectSet in enumerate(objects):
            m = i if materialIndex is None else materialIndex
            for j, objName in enumerate(objectSet):
                if type(objName) == list:
                    objName, fancyName = objName
                else:
                    fancyName = objName
                if scio:
                    X, _, _ = util.processScioDataset(dataSL, materials[m:m+1], [[objName]], sampleCount=numSamples, spectrumRaw=spectrumExposure)
                else:
                    X, _, _ = util.processLuminiDataset(dataSL, materials[m:m+1], [[objName]], sampleCount=numSamples, exposure=spectrumExposure, correctedValues=True)
                if filterData:
                    X = lowpassfilter(X, order=5, freq=0.2)
                mins.append(np.min(X))
                maxs.append(np.max(X))
        minimum = np.min(mins)
        maximum = np.max(maxs)
        print 'Min:', minimum, 'Max:', maximum

    colors = ['rgb(57, 118, 175)', 'rgb(240, 133, 54)', 'rgb(80, 157, 62)', 'rgb(198, 58, 50)', 'rgb(142, 107, 184)']
    colorsalpha = [c.replace('rgb', 'rgba').replace(')', ', %f)'%alpha) for c in colors]
    data = []
    for i, objectSet in enumerate(objects):
        m = i if materialIndex is None else materialIndex
        for j, objName in enumerate(objectSet):
            if type(objName) == list:
                objName, fancyName = objName
            else:
                fancyName = objName
            if scio:
                X, _, _ = util.processScioDataset(dataSL, materials[m:m+1], [[objName]], sampleCount=numSamples, spectrumRaw=spectrumExposure)
            else:
                X, _, _ = util.processLuminiDataset(dataSL, materials[m:m+1], [[objName]], sampleCount=numSamples, exposure=spectrumExposure, correctedValues=True)
            if filterData:
                X = lowpassfilter(X, order=5, freq=0.2)
            if normalizeData:
                X = (X - minimum) / (maximum - minimum)
            line = dict(color=colors[i])
            line.update(linedict)
            linealpha = dict(color=colorsalpha[i])
            linealpha.update(linedict)
            for ii, x in enumerate(X):
                data.append(go.Scatter(x=wavelengths, y=x, name=fancyName, line=line if ii==0 else linealpha, showlegend=ii==0))

    layout = dict(title=title,
                  xaxis=dict(title=xtitle, titlefont=dict(size=18), tickfont=dict(size=18)),
                  yaxis=dict(title=ytitle, titlefont=dict(size=18), tickfont=dict(size=18), dtick=0.25),
                  width=width,
                  height=height,
                  legend=dict(font=dict(size=14)) if legend is None else legend,
                  showlegend=showlegend)

    plotly.offline.plot({'data': data, 'layout': layout}, filename='plots/%s_%s.html' % (filename, 'scio' if scio else 'lumini'))

if __name__ == '__main__':
    materials = ['plastic', 'fabric', 'paper', 'wood', 'metal']
    plastics = ['HDPE', 'PET', 'polyethyleneBlue', 'polyethyleneGreen', 'polyethyleneRed', 'polyethyleneYellow', 'PP', 'PVC', 'thermoPolypropylene', 'thermoTeflon']
    fabrics = ['cottonCanvas', 'cottonSweater', 'cottonTowel', 'denim', 'felt', 'flannel', 'gauze', 'linen', 'satin', 'wool']
    papers = ['cardboard', 'constructionPaperGreen', 'constructionPaperOrange', 'constructionPaperRed', 'magazinePaper', 'newspaper', 'notebookPaper', 'printerPaper', 'receiptPaper', 'textbookPaper']
    woods = ['ash', 'cherry', 'curlyMaple', 'hardMaple', 'hickory', 'redCedar', 'redElm', 'redOak', 'walnut', 'whiteOak']
    metals = ['aluminum', 'aluminumFoil', 'brass', 'copper', 'iron', 'lead', 'magnesium', 'steel', 'titanium', 'zinc']
    objects = [plastics, fabrics, papers, woods, metals]
    scioData, wavelengthsScio = util.loadScioDataset()
    luminiData, wavelengthsLumini = util.loadLuminiDataset()

    # Plot samples from each of the material categories
    sampleObjects = [['HDPE'], [['wool', 'Wool']], [['cardboard', 'Cardboard']], [['whiteOak', 'White Oak']], [['aluminum', 'Aluminum']]]
    plot(luminiData, sampleObjects, materials, scio=False, spectrumExposure=500, numSamples=1, filterData=True, xtitle='Wavelength (nm)', ytitle='Normalized Intensity', wavelengths=wavelengthsLumini, filename='sampleSignals_filtered_norm', width=750, height=350, title='Lumini Spectral Samples', normalizeData=True, legend=dict(font=dict(size=14), x=0.75, y=0.95))
    plot(scioData, sampleObjects, materials, scio=True, spectrumExposure='spectrum', numSamples=1, xtitle='Wavelength (nm)', ytitle='Normalized Intensity', wavelengths=wavelengthsScio, filename='sampleSignals_norm', width=750, height=350, title='SCiO Spectral Samples', normalizeData=True, showlegend=True, legend=dict(font=dict(size=14)))

    # Plot all samples from three objects to show variance in data
    sampleObjects = [[['PVC', 'PVC']], [['denim', 'Denim']], [['cardboard', 'Cardboard']]]
    mats = ['plastic', 'fabric', 'paper']
    plot(luminiData, sampleObjects, mats, scio=False, spectrumExposure=500, numSamples=100, filterData=True, xtitle='Wavelength (nm)', ytitle='Normalized Intensity', wavelengths=wavelengthsLumini, filename='lowvariance', width=750, height=300, title='Lumini Spectral Samples', normalizeData=True, legend=dict(font=dict(size=14), x=0.75, y=0.95), linedict=dict(width=1), alpha=0.25)
    plot(scioData, sampleObjects, mats, scio=True, spectrumExposure='spectrum', numSamples=100, xtitle='Wavelength (nm)', ytitle='Normalized Intensity', wavelengths=wavelengthsScio, filename='lowvariance', width=750, height=300, title='SCiO Spectral Samples', normalizeData=True, showlegend=True, legend=dict(font=dict(size=14)), linedict=dict(width=1), alpha=0.25)

    # Plot samples from different color variations of the same object to see variance across color
    sampleObjects = [[['polyethyleneBlue', 'Blue Polyethylene']], [['polyethyleneYellow', 'Yellow Polyethylene']], [['polyethyleneGreen', 'Green Polyethylene']], [['polyethyleneRed', 'Red Polyethylene']]]
    plot(scioData, sampleObjects, materials, scio=True, spectrumExposure='spectrum', numSamples=1, materialIndex=0, xtitle='Wavelength (nm)', ytitle='Normalized Intensity', wavelengths=wavelengthsScio, filename='colorComparisonPlastic', width=750, height=350, title='SCiO Color Comparison', normalizeData=True)

    # Plot how performance increases as we increase the number of training objects
    data = []
    data.append(go.Scatter(x=range(1, 11), y=np.array([0.3322, 0.435, 0.569, 0.6648, 0.723, 0.7662, 0.7518, 0.818, 0.7756, 0.814])*100.0, name='SCiO', mode='lines'))
    data.append(go.Scatter(x=range(1, 11), y=np.array([0.262, 0.406, 0.4756, 0.491, 0.5346, 0.573, 0.6084, 0.6282, 0.6404, 0.654])*100.0, name='Lumini', mode='lines'))
    layout = dict(title='Leave-One-Object-Out Cross-Validation Performance with ResNeXt',
                  xaxis=dict(title='Number of training objects per material', titlefont=dict(size=18), tickfont=dict(size=18), dtick=1.0),
                  yaxis=dict(title='Average accuracy (%)', titlefont=dict(size=18), tickfont=dict(size=18), range=[0.0, 100.0]),
                  width=750,
                  height=375,
                  legend=dict(font=dict(size=14), x=0.8, y=0.1),
                  showlegend=True)
    plotly.offline.plot({'data': data, 'layout': layout}, filename='plots/looo_luminiscio.html')


    # Plot confusion matrices

    objects = ['HDPE', 'PET', 'Blue Polyethylene', 'Green Polyethylene', 'Red Polyethylene', 'Yellow Polyethylene', 'PVC', 'Thermo Polypropylene', 'PP', 'Thermo Teflon', 'Cotton Canvas', 'Cotton Sweater', 'Cotton Towel', 'Denim', 'Felt', 'Satin', 'Flannel', 'Gauze', 'Linen', 'Wool', 'Green Constr. Paper', 'Orange Constr. Paper', 'Red Constr. Paper', 'Magazine Paper', 'Newspaper', 'Notebook Paper', 'Printer Paper', 'Receipt Paper', 'Textbook Paper', 'Cardboard', 'Ash', 'Cherry', 'Curly Maple', 'Hard Maple', 'Hickory', 'Red Cedar', 'Red Elm', 'Red Oak', 'White Oak', 'Walnut', 'Aluminum', 'Brass', 'Iron', 'Steel', 'Titanium', 'Copper', 'Aluminum Foil', 'Zinc', 'Lead', 'Magnesium']

    # SCiO Reorganized
    results = [[100, 0, 0, 0, 0], # HDPE
    [100, 0, 0, 0, 0], # PET
    [100, 0, 0, 0, 0], # Blue Poly
    [100, 0, 0, 0, 0], # Green Poly
    [100, 0, 0, 0, 0], # Red Poly
    [100, 0, 0, 0, 0], # Yellow Poly
    [100, 0, 0, 0, 0], # PVC
    [100, 0, 0, 0, 0], # Thermo Poly
    [ 98, 0, 0, 0, 2], # PP
    [ 0, 88, 12, 0, 0], # Thermo Teflon
    [ 0, 100, 0, 0, 0], # Cotton Canvas
    [ 0, 100, 0, 0, 0], # Cotton Sweater
    [ 0, 100, 0, 0, 0], # Cotton Towel
    [ 0, 100, 0, 0, 0], # Denim
    [ 0, 100, 0, 0, 0], # Felt
    [ 1, 59, 40, 0, 0], # Satin
    [ 0, 22, 78, 0, 0], # Flannel
    [ 97, 0, 3, 0, 0], # Gauze
    [ 0, 0, 75, 0, 25], # Linen
    [ 0, 0, 0, 0, 100], # Wool
    [ 0, 0, 100, 0, 0], # Green Constr.
    [ 0, 0, 100, 0, 0], # Orange Constr.
    [ 0, 0, 100, 0, 0], # Red Constr.
    [ 0, 0, 100, 0, 0], # Magazine
    [ 0, 0, 100, 0, 0], # Newspaper
    [ 0, 0, 100, 0, 0], # Notebook
    [ 0, 0, 100, 0, 0], # Printer
    [ 0, 0, 100, 0, 0], # Receipt
    [ 0, 0, 100, 0, 0], # Textbook
    [ 0, 90, 0, 0, 10], # Cardboard
    [ 0, 0, 0, 100, 0], # Ash
    [ 0, 0, 0, 100, 0], # Cherry
    [ 0, 0, 0, 100, 0], # Curly Maple
    [ 0, 0, 0, 100, 0], # Hard Maple
    [ 0, 0, 0, 100, 0], # Hickory
    [ 0, 0, 0, 100, 0], # Red Cedar
    [ 0, 0, 0, 100, 0], # Red Elm
    [ 0, 0, 0, 100, 0], # Red Oak
    [ 0, 0, 0, 100, 0], # White Oak
    [ 0, 0, 0, 0, 100], # Walnut
    [ 0, 0, 0, 0, 100], # Aluminum
    [ 0, 0, 0, 0, 100], # Brass
    [ 0, 0, 0, 0, 100], # Iron
    [ 0, 0, 0, 0, 100], # Steel
    [ 0, 0, 0, 0, 100], # Titanium
    [ 0, 0, 5, 0, 95], # Copper
    [ 4, 0, 3, 0, 93], # Aluminum Foil
    [ 0, 0, 0, 49, 51], # Zinc
    [ 0, 0, 0, 50, 50], # Lead
    [ 0, 100, 0, 0, 0]] # Magnesium

    results = np.array(results) / 100.0

    materials = ['Plastic', 'Fabric', 'Paper', 'Wood', 'Metal']
    df_cm = pd.DataFrame(results, index=objects, columns=materials)
    plt.figure(figsize=(6, 10))
    cmap = sn.cubehelix_palette(as_cmap=True)
    ax = sn.heatmap(df_cm, annot=True, cmap=cmap, vmin=0.0, vmax=1.0, cbar=False)
    plt.yticks(rotation=30)
    plt.xticks(rotation=0)
    plt.ylabel('Object', fontsize=12)
    ax.set_xlabel('Predicted label', fontsize=12, labelpad=10)
    plt.tight_layout()
    # plt.savefig('plots/scio_object_confusionmatrix.pdf', format='pdf')
    # plt.show()

    # SCiO
    array = [[0.898, 0.088, 0.012, 0., 0.002],
    [0.098, 0.581, 0.196, 0., 0.125],
    [0., 0.09, 0.9, 0., 0.01],
    [0., 0., 0., 0.9, 0.1],
    [0.004, 0.1, 0.008, 0.099, 0.789]]

    sn.set(font_scale=1.7)
    materials = ['Plastic', 'Fabric', 'Paper', 'Wood', 'Metal']
    df_cm = pd.DataFrame(array, index=materials, columns=materials)
    plt.figure(figsize=(10, 6))
    cmap = sn.cubehelix_palette(as_cmap=True)
    ax = sn.heatmap(df_cm, annot=True, cmap=cmap, vmin=0.0, vmax=1.0)
    plt.yticks(rotation=45)
    plt.xticks(rotation=0)
    plt.ylabel('True label')
    ax.set_xlabel('Predicted label', labelpad=15)
    plt.tight_layout()
    # plt.savefig('plots/scio_material_confusionmatrix.pdf', format='pdf')
    plt.show()


