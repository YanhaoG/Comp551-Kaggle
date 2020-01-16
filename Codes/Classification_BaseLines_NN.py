# -*- coding: utf-8 -*-
"""
Team: MTL551
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
#from matplotlib.pyplot import savefig

def to_cat(t_train):
    t_train = t_train.astype(int)
    nClass = np.max(t_train)+1
    y_train = np.zeros((t_train.shape[0], nClass))
    for i in range(t_train.shape[0]):
        y_train[i,t_train[i,0]] = 1
    return y_train

def to_label(y):
    return np.argmax(y, axis = 1)

def sigmoid_func(x):
    return 1/(1+np.exp(-x))

def deriv_sigmoid(z):
    return z*(1-z)

def feedforward(a0, w0, w1):
    a1 = sigmoid_func(np.dot(a0, w0))
    a2 = sigmoid_func(np.dot(a1, w1))
    yp = np.argmax(a2,axis=1)
    return yp



Classification_Method = 'NN' 
##Classification_Method = 'KNN' 
##Classification_Method = 'SVM'
##Classification_Method = 'any baseline'
##Classification_Method = 'xgboost'


#processingMethod = 1
processingMethod = 2
normalizingMethod = 'Quantizing'
#normalizingMethod = 'Binarizing'



print('Loading preprocessed data...')
print('Preprocessing Method: {}'.format(processingMethod))

if processingMethod == 1:
    """ Old PreProcessing """ 
    [processedImgTrain, trainLabel, processedImgTest, filteredImgSize, uniqueLabelWords, idxTrainValid, nTrain] = np.load('PreProcessing_Method1.npy')
    #[processedImgTrain, trainLabel, processedImgTest, filteredImgSize, uniqueLabelWords, idxTrainValid, nTrain] = np.load('./PreProcessing_Method1.npy')
elif processingMethod == 2:
    """ New PreProcessing """
    [processedImgTrain, trainLabel, processedImgTest, filteredImgSize, uniqueLabelWords, idxTrainValid, nTrain] = np.load('PreProcessing_Method2.npy')
    #[processedImgTrain, trainLabel, processedImgTest, filteredImgSize, uniqueLabelWords, idxTrainValid, nTrain] = np.load('./PreProcessing_Method2.npy')



if normalizingMethod == 'Quantizing':
    print('Quantizing...')
    NumLevel = 30
    featMatrixTrain = np.floor(NumLevel * processedImgTrain) / NumLevel
    featMatrixTest  = np.floor(NumLevel * processedImgTest ) / NumLevel
    featMatrixTrain = featMatrixTrain.astype('float32')
    featMatrixTest  = featMatrixTest.astype('float32')
elif normalizingMethod == 'Binarizing':
    print('Binarizing...')
    binarizingTH = 0.1 # range (0,1)
    featMatrixTrain = 1 * (processedImgTrain>binarizingTH)
    featMatrixTest  = 1 * (processedImgTest>binarizingTH)
    featMatrixTrain = featMatrixTrain.astype('float32')
    featMatrixTest  = featMatrixTest.astype('float32')

"""  Extracting Train & Test Sets Sizes and Original Feature & Image Sizes  """
trainSize = processedImgTrain.shape[0]               #Number of Training Examples
testSize  = processedImgTest.shape[0]                #Number of Testing Examples
featureSize = processedImgTrain.shape[1]             #Number of Features


nTrain = int(0.9*trainSize)
idxTrainValid = np.random.choice(trainSize, [trainSize,1],replace = False)

t_train = trainLabel[idxTrainValid[:nTrain],0]
t_valid = trainLabel[idxTrainValid[nTrain:],0]
X_train = featMatrixTrain[idxTrainValid[:nTrain,0],:]
X_valid = featMatrixTrain[idxTrainValid[nTrain:,0],:]
X_test = featMatrixTest

y_train = to_cat(t_train)
y_valid = to_cat(t_valid)



alpha = 0.001  #learning rate
nEpoch = 5000 
nN = 50 # number of neurons
nF = X_train.shape[1] #number of features
nC = y_train.shape[1] #number of classes

w0 = 2*np.random.random((nF, nN)) - 1 
w1 = 2*np.random.random((nN, nC)) - 1 

CCR_train = np.zeros((nEpoch,1))
CCR_valid = np.zeros((nEpoch,1))


for i in range(nEpoch):
    #FّّّF
    a0 = X_train
    a1 = sigmoid_func(np.dot(a0, w0))
    a2 = sigmoid_func(np.dot(a1, w1))

    #BP
    e2 = a2 - y_train
    delt2 = e2 * deriv_sigmoid(a2)
    
    e1 = delt2.dot(w1.T)
    delt1 = e1 * deriv_sigmoid(a1)
        
    w1 -= a1.T.dot(delt2) * alpha
    w0 -= a0.T.dot(delt1) * alpha
    
    #FF
    tp_train = feedforward(X_train, w0, w1)
    tp_valid = feedforward(X_valid, w0, w1)

    #Evaluation
    CCR_train[i] = np.sum(tp_train.ravel() == t_train.ravel())/t_train.shape[0]
    CCR_valid[i] = np.sum(tp_valid.ravel() == t_valid.ravel())/t_valid.shape[0]
    if i%20==0:   
        print('Epoch: {} and Train Acc= {}'.format(i, CCR_train[i]))
        print('Epoch: {} and Valid Acc= {}'.format(i, CCR_valid[i]))


plt.plot(CCR_train, 'b')
plt.plot(CCR_valid, 'k')
plt.xlabel('Epoch')
plt.ylabel('CCR')
plt.show()

#FF
t_train_predicted = feedforward(X_train, w0, w1)
t_valid_predicted = feedforward(X_valid, w0, w1)
t_test_predicted = feedforward(X_test, w0, w1)

#Evaluation
accuracy_train = np.sum(t_train_predicted.ravel() == t_train.ravel())/t_train.shape[0]
accuracy_valid = np.sum(t_valid_predicted.ravel() == t_valid.ravel())/t_valid.shape[0]
f1_score_train = f1_score(t_train_predicted.ravel(), t_train.ravel(), average='weighted')
f1_score_valid = f1_score(t_valid_predicted.ravel(), t_valid.ravel(), average='weighted')


np.set_printoptions(precision=3)
print('Results for Classification Method:' + Classification_Method)
print('PreProcessing Method :{}'.format(processingMethod) + ', and Normalization Method: '+ normalizingMethod)
print('\n')
print('F1-Measure Train:\n', f1_score_train)
print('\n')
print('F1-Measure Valid:\n', f1_score_valid)
print('\n')
print('Accuracy Train:\n', accuracy_train)
print('\n')
print('Accuracy Valid:\n', accuracy_valid)
print('\n')

print('Results have been saved')
results_data = [f1_score_train, f1_score_valid, accuracy_train, accuracy_valid]#, t_train_predicted, t_train, t_valid_predicted, t_valid, t_test_predicted]#, processingMethod, normalizingMethod]
np.save('Results/'+Classification_Method + "_Results_" +str(normalizingMethod) + "_" +str(processingMethod), results_data)