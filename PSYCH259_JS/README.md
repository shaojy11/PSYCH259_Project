# PSYCH259: Voice Gender Recognition

### Prerequisites


- Tensorflow 1.0


### Source Code

#### python code
`/src` folder contains all my source codes. *.py files are used to implement RNN models and preprocess audio raw data; rnn_main.py file implements the whole training and testing pipeline. 
Execute it in source code directory `~/yourpath/src` by command: 
```
python rnn_main.py -g device -m mode
```

Parameters:
-g: device number
-m: 'train' or 'test'


#### R
R scripts are used to do experiments with SVM, Logistic Regression, Bayesian, Random Forest and Xgboost models;
#### MATLAB
MATLAB script is used for plotting training loss and errors.


### Gender-recognition-voice
Project paper
