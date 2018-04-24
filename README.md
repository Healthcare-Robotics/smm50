# Classification of Household Materials via Spectroscopy

Z. Erickson, N. Luskey, S. Chernova, and C. C. Kemp, "Classification of Household Materials via Spectroscopy", arXiv, 2018.

Project webpage: http://healthcare-robotics.com/smm50

## Download the SMM50 dataset
SMM50 dataset (170 MB): https://goo.gl/2X276V  
Raw data collected from the robot and spectrometers (260 MB): https://goo.gl/n6biJE  
Dataset details can be found on the [project webpage](http://healthcare-robotics.com/smm50).

Use the following commands to download and extract the SMM50 dataset.
```bash
cd data
wget -O smm50.tar.gz https://goo.gl/2X276V
tar -xvzf smm50.tar.gz
rm smm50.tar.gz
```

## Running the code
Our residual and vanilla neural networks are implemented in Keras with the Tensorflow backend.  
Results presented in tables I and II from the paper can be computed using the following.
```bash
python learn.py -t 0 -a svm
python learn.py -t 0 -a nn
python learn.py -t 0 -a residualnn
```
Generalization with leave-one-object-out validation results from table III can be computed using the commands below. These results are also used for Fig. 12 and 13 in the paper.
```bash
python learn.py -t 1 -a svm
python learn.py -t 1 -a nn
python learn.py -t 1 -a residualnn
```
The generalization results with increasing numbers of obects can be recomputed using the commands below. This corresponds to Fig. 15 in the paper.
```bash
python learn.py -t 2 -a residualnn
```
All of the plots from the paper can be regenerated using `plot.py`. This requires [plotly](https://plot.ly/python/).
```bash
python plot.py
```

### Dependencies
Python 2.7  
Keras 2.2.1  
Tensorflow 1.7.0  
Scikit-learn 0.18.1  
Numpy 1.14.2  
Scipy 1.0.1  
Plotly 2.5.1
