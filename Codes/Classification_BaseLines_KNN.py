# -*- coding: utf-8 -*-
"""
Team: MTL551
"""

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
#from matplotlib.pyplot import savefig

Classification_Method = 'KNN' 
##Classification_Method = 'SVM'
##Classification_Method = 'any baseline'
##Classification_Method = 'xgboost'


#processingMethod = 1
processingMethod = 2
#normalizingMethod = 'Quantizing'
normalizingMethod = 'Binarizing'



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
    NumLevel = 20
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


""" Cross Validation """
kFold = 5
print('Generating {}-Fold Cross-Validation Samples...'.format(kFold) )
nTest = testSize
nValid = int(trainSize / kFold)
nTrain = trainSize - nValid
idxTrain_CV = np.zeros((kFold , nTrain))
idxValid_CV = np.zeros((kFold , nValid))

idxShuffled = np.random.choice(trainSize, [trainSize,1],replace = False)
for k in range(kFold):
    idxValid_CV[k,:] = np.copy(idxShuffled[k*nValid:(k+1)*nValid].ravel())
    idxTrain_CV[k,:] = np.copy(np.delete(idxShuffled, np.arange(k*nValid,(k+1)*nValid)))

idxTrain_CV = idxTrain_CV.astype(int)
idxValid_CV = idxValid_CV.astype(int)

""" Classification Step """
print('Initializing ' + Classification_Method +' ...')
from sklearn.neighbors import KNeighborsClassifier

typeHyperParameter = 'K'
hyperParameter = [1, 2, 5, 10, 15, 20, 100]  #k in KNN

print('Type of Hyperparameter:', typeHyperParameter)
print('Range of Hyperparameter:', hyperParameter)

f1_score_train = np.zeros((kFold, len(hyperParameter)))
f1_score_valid = np.zeros((kFold, len(hyperParameter)))
accuracy_train = np.zeros((kFold, len(hyperParameter))) 
accuracy_valid = np.zeros((kFold, len(hyperParameter))) 

t_train_predicted = np.zeros((nTrain, kFold))
t_valid_predicted = np.zeros((nValid, kFold))
t_test_predicted  = np.zeros((nTest, kFold))
for k in range(kFold):
    print('Fold = {}'.format(k+1))
    t_train = trainLabel[idxTrain_CV[k,:],0]
    t_valid = trainLabel[idxValid_CV[k,:],0]
    X_train = featMatrixTrain[idxTrain_CV[k,:]]
    X_valid = featMatrixTrain[idxValid_CV[k,:]]
    X_test  = featMatrixTest

    X_train = csr_matrix(X_train)
    X_valid = csr_matrix(X_valid)
    X_test = csr_matrix(X_test)
    cntr = 0
    for item  in hyperParameter:
        print('Hyperparameter = {}'.format(item))
        neigh = KNeighborsClassifier(n_neighbors=item)
        neigh.fit(X_train, t_train.ravel())         
        t_train_predicted[:,k] = neigh.predict(X_train.toarray())
        t_valid_predicted[:,k] = neigh.predict(X_valid.toarray())
        t_test_predicted[:,k] = neigh.predict(X_test.toarray())
        
                
        f1_score_train[k, cntr] = f1_score(t_train,t_train_predicted[:,k], average='weighted')
        f1_score_valid[k, cntr]  = f1_score(t_valid,t_valid_predicted[:,k], average='weighted')
        
        accuracy_train[k, cntr] = np.sum(t_train.ravel() == t_train_predicted[:,k].ravel())/t_train.shape[0]
        accuracy_valid[k, cntr] = np.sum(t_valid.ravel() == t_valid_predicted[:,k].ravel())/t_valid.shape[0]
        cntr += 1


fig, ax = plt.subplots()
ax.plot(hyperParameter, np.mean(f1_score_train, axis = 0), c='k')
ax.plot(hyperParameter, np.mean(f1_score_valid, axis= 0), c='b')
ax.set(xlabel='HyperParameter', ylabel='F1-Score', title=Classification_Method+' Average F1-Score')
ax.legend(['Train' , 'Valid'])
ax.grid()
fig.savefig('Figures/'+Classification_Method +"_F1_Score_" +str(normalizingMethod) + "_" +str(processingMethod) + "_" + typeHyperParameter+".png", dpi=600)

fig, ax = plt.subplots()
ax.plot(hyperParameter, np.mean(accuracy_train, axis = 0), c='k')
ax.plot(hyperParameter, np.mean(accuracy_valid, axis = 0), c='b')
ax.set(xlabel='HyperParameter', ylabel='Accuracy', title=Classification_Method + ' Average Accuracy')
ax.legend(['Train' , 'Valid'])
ax.grid()
fig.savefig('Figures/'+Classification_Method + "_Accuracy_" +str(normalizingMethod) + "_" +str(processingMethod) + "_" + typeHyperParameter+".png", dpi=600)
print('Figures have been saved')


np.set_printoptions(precision=3)
print('Results for Classification Method:' + Classification_Method)
print('PreProcessing Method :{}'.format(processingMethod) + ', and Normalization Method: '+ normalizingMethod)
print('\n')
print('Hyperparameter:\n', hyperParameter)
print('F1-Measure Train (Average):\n', np.mean(f1_score_train, axis = 0))
print('F1-Measure Train (Std):\n', np.std(f1_score_train, axis = 0))
print('\n')
print('Hyperparameter:\n', hyperParameter)
print('F1-Measure Valid (Average):\n', np.mean(f1_score_valid, axis = 0))
print('F1-Measure Valid (Std):\n', np.std(f1_score_valid, axis = 0))
print('\n')
print('Hyperparameter:\n', hyperParameter)
print('Accuracy Train (Average):\n', np.mean(accuracy_train, axis = 0))
print('Accuracy Train (Std):\n', np.std(accuracy_train, axis = 0))
print('\n')
print('Hyperparameter:\n', hyperParameter)
print('Accuracy Valid (Average):\n', np.mean(accuracy_valid, axis = 0))
print('Accuracy Valid (Std):\n', np.std(accuracy_valid, axis = 0))
print('\n')

print('Results have been saved')
np.save('Results/'+Classification_Method + "_Results_" +str(normalizingMethod) + "_" +str(processingMethod)+ "_" + typeHyperParameter, [f1_score_train, f1_score_valid, accuracy_train, accuracy_valid, t_train_predicted, t_train, t_valid_predicted, t_valid, t_test_predicted, idxTrain_CV, idxValid_CV, typeHyperParameter, hyperParameter, processingMethod, normalizingMethod])