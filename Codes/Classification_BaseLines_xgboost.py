# -*- coding: utf-8 -*-
"""
Team: MTL551
"""

import numpy as np
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import xgboost as xgb
#from matplotlib.pyplot import savefig
import warnings
warnings.filterwarnings('ignore')

def main():
    #Classification_Method = 'KNN'
    #Classification_Method = 'SVM'
    #Classification_Method = 'any baseline'
    Classification_Method = 'xgboost'


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


    ##################################### HyperParameters range ###################################################
    # typeHyperParameter = 'max_depth'
    # hyperParameter = [4, 6, 8, 10, 12]  # max_depth in KNN

    # typeHyperParameter = 'eta'
    # hyperParameter = [0.1, 0.2, 0.3, 0.4, 0.5]  # max_depth in KNN


    typeHyperParameter = 'gamma'
    hyperParameter = [0.1, 0.2, 0.3, 0.4, 0.5]


    print('Type of Hyperparameter:', typeHyperParameter)
    print('Range of Hyperparameter:', hyperParameter)
    ###############################################################################################################
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

        xg_train = xgb.DMatrix(X_train, label=t_train)
        xg_valid = xgb.DMatrix(X_valid, label=t_valid)
        xg_test = xgb.DMatrix(X_test)

        num_round = 5
        cntr = 0
        for item  in hyperParameter:
            ################### Construct xgb model ##############################
            param = {}
            param['objective'] = 'multi:softmax'
            # parameters can be tuned
            # param['eta'] = item
            # param['max_depth'] = 8

            param['eta'] = 0.1
            param['max_depth'] = 8
            param['gamma'] = item
            ##No need to be tuned
            param['silent'] = 1
            param['nthread'] = 4
            param['num_class'] = 31
            print('Hyperparameter = {}'.format(item))
            model = xgb.train(param, xg_train, num_round)

            ######################################################################
            t_train_predicted[:,k] = model.predict(xg_train)
            t_valid_predicted[:,k] = model.predict(xg_valid)
            t_test_predicted[:,k] = model.predict(xg_test)


            f1_score_train[k, cntr] = f1_score(t_train,t_train_predicted[:,k], average='weighted')
            f1_score_valid[k, cntr]  = f1_score(t_valid,t_valid_predicted[:,k], average='weighted')

            accuracy_train[k, cntr] = np.sum(t_train.ravel() == t_train_predicted[:,k].ravel())/t_train.shape[0]
            accuracy_valid[k, cntr] = np.sum(t_valid.ravel() == t_valid_predicted[:,k].ravel())/t_valid.shape[0]
            cntr += 1
    ###############################################################################################################

    fig, ax = plt.subplots()
    ax.plot(hyperParameter, np.mean(f1_score_train, axis = 0), c='k')
    ax.plot(hyperParameter, np.mean(f1_score_valid, axis= 0), c='b')
    ax.set(xlabel='HyperParameter', ylabel='F1-Score', title=Classification_Method+' Average F1-Score')
    ax.legend(['Train' , 'Valid'])
    ax.grid()
    fig.savefig(Classification_Method +"_F1_Score_" +str(normalizingMethod) + "_" +str(processingMethod) + "_" + typeHyperParameter+".png", dpi=600)

    fig, ax = plt.subplots()
    ax.plot(hyperParameter, np.mean(accuracy_train, axis = 0), c='k')
    ax.plot(hyperParameter, np.mean(accuracy_valid, axis = 0), c='b')
    ax.set(xlabel='HyperParameter', ylabel='Accuracy', title=Classification_Method + ' Average Accuracy')
    ax.legend(['Train' , 'Valid'])
    ax.grid()
    fig.savefig(Classification_Method + "_Accuracy_" +str(normalizingMethod) + "_" +str(processingMethod) + "_" + typeHyperParameter+".png", dpi=600)

    print('Figures have been saved')



    np.set_printoptions(precision=3)
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



if __name__ == '__main__':
    main()
