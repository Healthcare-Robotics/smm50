# Classification of Household Materials via Spectroscopy

Z. Erickson, N. Luskey, S. Chernova, and C. C. Kemp, ["Classification of Household Materials via Spectroscopy"](https://arxiv.org/abs/1805.04051), IEEE Robotics and Automation Letters (RA-L), 2019.

Project webpage: [https://pwp.gatech.edu/hrl/smm50/](https://pwp.gatech.edu/hrl/smm50/)

## Download the SMM50 dataset
SMM50 dataset (16 MB): [https://goo.gl/Xjh6x4](https://goo.gl/Xjh6x4)  
Dataset details can be found on the [project webpage](https://pwp.gatech.edu/hrl/smm50/).

Use the following commands to download and extract the SMM50 dataset.
```bash
cd data
wget -O smm50.tar.gz https://goo.gl/2X276V
tar -xvzf smm50.tar.gz
rm smm50.tar.gz
```

## Running the code
Our models are implemented in Keras with the Tensorflow backend.  
Results presented in table I from the paper can be computed using the following.
```bash
python learn.py -t 0
```
Generalization with leave-one-object-out validation results from figures 11 and 12 can be computed using the command below.
```bash
python learn.py -t 1
```
The generalization results with increasing numbers of objects can be recomputed using the command below. This corresponds to figure 14 in the paper.
```bash
python learn.py -t 2
```
Generalization results with everyday objects and a PR2 (figure 15 in the paper) can be computed using the below command.
```bash
python learn.py -t 3
```

### Dependencies
Python 2.7  
Keras 2.2.1  
Tensorflow 1.7.0  
Scikit-learn 0.18.1  
Numpy 1.14.2  
Scipy 1.0.1  
