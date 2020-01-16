# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 12:52:11 2018

@author: enateg
"""
import cv2
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from skimage import measure, feature, transform, morphology
from sklearn.metrics import classification_report, confusion_matrix, f1_score

import keras
from keras import backend as K
from keras.utils import np_utils
from keras.models import Sequential
from keras.utils import to_categorical
from keras.layers import Dense, Conv2D, Flatten, Dropout, BatchNormalization, Lambda
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

import os



def CNN_model(filteredImgSize):
    model = Sequential([
        # First two convolutional layers
        Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same', input_shape=(1, filteredImgSize, filteredImgSize)),
        Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'),
        # normalization layer
        BatchNormalization(),
        # pooling layer
        MaxPooling2D(pool_size=(2, 2)),
        # add regularization
        Dropout(0.25),
        # Second two convolutional layers
        Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'),
        Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'),
        # normalization layer
        BatchNormalization(),
        # pooling layer
        MaxPooling2D(pool_size=(2, 2)),
        # add regularization
        Dropout(0.25),

        Flatten(),

        # FC layer
        Dense(1024, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(1024, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(1024, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(31, activation='softmax')
    ])
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(lr=0.001),
                  metrics=['accuracy'])
    return model


def main():
    ##Classification_Method = 'KNN'
    ##Classification_Method = 'SVM'
    ##Classification_Method = 'any baseline'
    ##Classification_Method = 'xgboost'
    Classification_Method = 'CNN'

    processingMethod = 1
    #processingMethod = 2
    #normalizingMethod = 'Quantizing'
    normalizingMethod = 'Binarizing'


    print('Loading preprocessed data...')
    print('Preprocessing Method: {}'.format(processingMethod))

    if processingMethod == 1:
        """ Old PreProcessing """
        [processedImgTrain, trainLabel, processedImgTest, filteredImgSize, uniqueLabelWords, idxTrainValid, nTrain] = np.load('./results/PreProcessing_Method1.npy')
        #[processedImgTrain, trainLabel, processedImgTest, filteredImgSize, uniqueLabelWords, idxTrainValid, nTrain] = np.load('./PreProcessing_Method1.npy')
    elif processingMethod == 2:
        """ New PreProcessing """
        [processedImgTrain, trainLabel, processedImgTest, filteredImgSize, uniqueLabelWords, idxTrainValid, nTrain] = np.load('./results/PreProcessing_Method2.npy')
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
        binarizingTH = 0.01 # range (0,1)
        featMatrixTrain = 1 * (processedImgTrain>binarizingTH)
        featMatrixTest  = 1 * (processedImgTest>binarizingTH)
        featMatrixTrain = featMatrixTrain.astype('float32')
        featMatrixTest  = featMatrixTest.astype('float32')


    """  Extracting Train & Test Sets Sizes and Original Feature & Image Sizes  """
    trainSize = featMatrixTrain.shape[0]               #Number of Training Examples
    testSize  = featMatrixTest.shape[0]                #Number of Testing Examples
    featureSize = featMatrixTrain.shape[1]             #Number of Features


    nTrain = int(0.9*trainSize)
    idxTrainValid = np.random.choice(trainSize, [trainSize,1],replace = False)

    t_train = trainLabel[idxTrainValid[:nTrain],0]
    t_valid = trainLabel[idxTrainValid[nTrain:],0]
    X_train = featMatrixTrain[idxTrainValid[:nTrain,0],:]
    X_valid = featMatrixTrain[idxTrainValid[nTrain:,0],:]
    X_test = featMatrixTest


    X_train = X_train.reshape(X_train.shape[0], 1, filteredImgSize, filteredImgSize)
    X_valid = X_valid.reshape(X_valid.shape[0], 1, filteredImgSize, filteredImgSize)
    X_test = X_test.reshape(X_test.shape[0], 1, filteredImgSize, filteredImgSize)

    #one-hot encode target column
    y_train = to_categorical(t_train)
    y_valid = to_categorical(t_valid)


    K.set_image_dim_ordering('th')

    # fix random seed for reproducibility
    seed = 7
    np.random.seed(seed)
    num_classes = y_valid.shape[1]


    model = CNN_model(filteredImgSize)
    # apply image augmentation
    # define hyper-parameters
    batch_size=55
    epoch_aug1=40
    epoch_aug2=400

    earlyStopping = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.001, patience=30, verbose=0, mode='min')

    gen = ImageDataGenerator(rotation_range=15, width_shift_range=0.1, shear_range=0.3, 
                             height_shift_range=0.1, zoom_range=0.1)

    batches = gen.flow(X_train, y_train, batch_size=batch_size)
    val_batches = gen.flow(X_valid, y_valid, batch_size=batch_size)
     

    model.fit_generator(batches, steps_per_epoch=X_train.shape[0] // batch_size, epochs=epoch_aug1, 
                        validation_data=val_batches, validation_steps=X_valid.shape[0] // batch_size,
                        use_multiprocessing=False, callbacks=[earlyStopping])

    # change the learning rate 
    model.optimizer.lr=0.0001

    model.fit_generator(batches, steps_per_epoch=X_train.shape[0] // batch_size, epochs=epoch_aug2,
                        validation_data=val_batches, validation_steps=X_valid.shape[0] // batch_size,
                        use_multiprocessing=False, callbacks=[earlyStopping])


    # Final evaluation of the model
    scores = model.evaluate(X_train, y_train, verbose=1)
    print("Large CNN Train Error: %.2f%%" % (100-scores[1]*100))
    scores = model.evaluate(X_valid, y_valid, verbose=1)
    print("Large CNN Valid Error: %.2f%%" % (100-scores[1]*100))


    #predict images in the test set
    y_train_CNN = model.predict(X_train)
    y_valid_CNN = model.predict(X_valid)
    y_test_CNN = model.predict(X_test)
    t_train_CNN = np.argmax(y_train_CNN,axis=1)
    t_valid_CNN = np.argmax(y_valid_CNN,axis=1)
    t_test_CNN = np.argmax(y_test_CNN,axis=1)

    testLabelCNN = np.zeros((testSize,1)).astype('str')

    # map the predict result to classes name
    for i in range(NumberLabel):
        testLabelCNN[t_test_CNN == i,0] = uniqueLabelWords[i]


    accuracy_train = np.sum(t_train.ravel()==t_train_CNN.ravel())/t_train.shape[0]
    accuracy_valid = np.sum(t_valid.ravel()==t_valid_CNN.ravel())/t_valid.shape[0]
    f1_score_train = f1_score(t_train.ravel(), t_train_CNN.ravel(), average='weighted')
    f1_score_valid = f1_score(t_valid.ravel(), t_valid_CNN.ravel(), average='weighted')
    print('train accuracy: ', accuracy_train)
    print('valid accuracy: ', accuracy_valid)
    print('train f1-score: ', f1_score_train )
    print('valid f1-score: ',  f1_score_valid)

    #######################Save the result to a text file##########################################
    with open('./results/tar84.txt', 'w') as f:
        for item in testLabelCNN:
            f.write("%s\n" % item)


if __name__ == '__main__':
    main()



