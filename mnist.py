# -*- coding: utf-8 -*-
"""
Digit classification on the MNIST data-set.

Inspiration:
https://www.kaggle.com/yassineghouzam/introduction-to-cnn-keras-0-997-top-6

"""

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools

from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau

class Mnist(object):
    """ Class for training MNIST classifier. """
    
    def __init__(self):
        self.train = None
        self.test = None
        
        self.xtrain = None
        self.ytrain = None
        
        self.xval = None
        self.yval = None
        
        self.ptrain = None
        
        self.model = None
        
        self.history = None
        
    def load_data(self, path1, path2):
        """Load training and test data from csv."""
        self.train = pd.read_csv(path1)
        self.test = pd.read_csv(path2)
        
    def process_train(self):
        """Process and normalize the training and test data."""
        self.ytrain = self.train["label"]
        self.ytrain = to_categorical(self.ytrain, num_classes=10)
        
        self.xtrain = self.train.drop(labels=["label"], axis=1)
        self.xtrain = self.xtrain / 255.0
        self.xtrain = self.xtrain.values.reshape(-1,28,28,1)

        self.test = self.test / 255.0
        self.test = self.test.values.reshape(-1,28,28,1)
        
    def split(self):
        """Process and normalize the training and test data."""
        RANDOM_SEED = 2
        self.xtrain, self.xval, self.ytrain, self.yval = train_test_split(
                self.xtrain, self.ytrain, test_size=0.1,
                random_state = RANDOM_SEED)

    def create_model(self):
        """Create a convolution neural network model in Keras."""
        print("Creating model.")
        self.model = Sequential()
        self.model.add(Conv2D(filters=32, kernel_size=(5,5), padding='Same',
                              activation='relu', input_shape=(28,28,1)))
        self.model.add(Conv2D(filters=32, kernel_size=(5,5), padding='Same',
                              activation='relu'))
        self.model.add(MaxPool2D(pool_size=(2,2)))
        self.model.add(Dropout(0.25))
        
        self.model.add(Conv2D(filters=64, kernel_size=(3,3), padding='Same',
                              activation='relu'))
        self.model.add(Conv2D(filters=64, kernel_size=(3,3), padding='Same',
                              activation='relu'))
        self.model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
        self.model.add(Dropout(0.25))

        self.model.add(Flatten())
        self.model.add(Dense(256,activation="relu"))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(10,activation="softmax"))
        
        optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
        self.model.compile(optimizer=optimizer, loss = "categorical_crossentropy",
                      metrics=["accuracy"])


    def train_model(self):
       """"Specify parameters and train model."""
       print("Training model.")
       
       learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)
       
       epochs = 75 # Turn epochs to 30 to get 0.9967 accuracy
       batch_size = 86
       
       datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image 
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images

       datagen.fit(self.xtrain)

       self.history = self.model.fit_generator(datagen.flow(self.xtrain, self.ytrain, batch_size=batch_size), 
                                      epochs = epochs, validation_data = (self.xval,self.yval), 
                                      verbose = 2, steps_per_epoch=self.xtrain.shape[0] // batch_size,
                                      callbacks=[learning_rate_reduction])
       
    def predict(self):
        """Predict the digits in the test set."""
        results = self.model.predict(self.test)
        results = np.argmax(results, axis=1)
        results = pd.Series(results,name="Label")
        submission = pd.concat([pd.Series(range(1,28001),
                                          name = "ImageId"),results],axis = 1)
        submission.to_csv("cnn_mnist_datagen.csv",index=False)

    def run_all(self):
        
        train_path = "./train.csv"
        #train_path = "C:/Users/Daniel/.kaggle/competitions/digit-recognizer/train.csv"
        test_path = "./test.csv"
        #test_path = "C:/Users/Daniel/.kaggle/competitions/digit-recognizer/test.csv"

        self.load_data(train_path, test_path)

        #Process training data
        self.process_train()

        #Slit the training data.
        self.split()

        #Create the model
        self.create_model()

        #Fit the model
        self.train_model()
        
        #Make predictions
        self.predict()
        
        print("Done!")
        
#Perform training and prediction.
m = Mnist()
m.run_all()