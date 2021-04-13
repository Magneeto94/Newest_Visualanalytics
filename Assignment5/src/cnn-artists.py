# data tools
'''
---------------------Importing packages-----------------------------------
'''
import os
import cv2
from pathlib import Path
import re
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pydot

# sklearn tools
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report

# tf tools
#from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv2D, 
                                     MaxPooling2D, 
                                     Activation, 
                                     Flatten, 
                                     Dense)
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import backend as K

import argparse

def plot_history(H, epochs):
    # visualize performance
    plt.style.use("fivethirtyeight")
    fig = plt.figure()
    plt.plot(np.arange(0, epochs), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, epochs), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, epochs), H.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, epochs), H.history["val_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.show()
    fig.savefig("../output_lenet/lenet_graph.png")

#Defining main function
def main():
    
    
    ap = argparse.ArgumentParser(description = "[INFO] creating LeNet CNN")
    
    
    ap.add_argument("-k", #flag
                    "--kernal", 
                    required=False, # You do not need to give any arguments, but you can.
                    default=5, # If you don't the default is 0.8 or 80%
                    type = int,
                    help = "The size of your kernal x*x. The size should be 3,5 or 7")
    
    ap.add_argument("-ps", #flag
                    "--picture_size", 
                    required=False, # You do not need to give any arguments, but you can.
                    default=120, # If you don't the default is 0.8 or 80%
                    type = int,
                    help = "The height and width that the paintings are resized to")
    
    args = vars(ap.parse_args())
    
    kernal_size = args["kernal"]
    picture_size = args["picture_size"]
    
    
    '''
    --------------------------Finding training data and creating label-names:----------
    '''
    #Path to training folder with painters

    training_dir = os.path.join("../data/small_training")

    #Names as a string
    label_names = []
    #Training labels
    y_train = []

    #First we find the painters 
    i = 0
    
    for folder in Path(training_dir).glob("*"):
        #Finding the painters name and taking it from folder names
        #findall returns a list.
        #Using regex to find the right painter with matching pictures.
        painter = re.findall(r"(?!.*/).+", str(folder))
    
        # We want the first element of the list returned every time, therefore index 0
        label_names.append(painter[0])
    
        #This forlook runs through every painter folder and appends them to y_train
        for img in folder.glob("*"):
            y_train.append(i)
        
        i +=1
    
    
    '''
    -----------------------------------Finding test data:---------------------------------
    '''
    #Labels for validation
    #Path to training folder with painters

    validation_dir = os.path.join("../data/small_validation")

    #Names as a string

    #Test labels
    y_test = []

    #
    i = 0
    
    for folder in Path(validation_dir).glob("*"):
        for img in folder.glob("*"):
            y_test.append(i)
        i +=1
        
    
    
    '''
    -----------------------------Making data binary:-----------------
    '''
    # integers to one-hot vectors
    lb = LabelBinarizer()

    #Transforming the labels into binary names
    trainY = lb.fit_transform(y_train)
    testY = lb.fit_transform(y_test)
    
    
    '''
    ----------------------------Resizing training data:-----------------
    '''
    
    filepath = os.path.join("../data/small_training")

    X_train=[]

    #Running through the small_training data
    for folder in Path(filepath).glob("*"):
        #Running throught the pictures in the small_training data
        for file in Path(folder).glob("*"):
            #reading the image
            image_open = cv2.imread(str(file))
            #redefining dimentions for the pictures to height and width to be "picture_size", that way the pictures have the same size.
            dim = (picture_size, picture_size)
            #Saving rezised image
            resize_image = cv2.resize(image_open, dim, interpolation = cv2.INTER_AREA)
            #Pushing the data to X_train
            X_train.append(resize_image.astype("float") / 255.)
    
    
    
    '''
    ----------------------------Resizing test data:-----------------
    '''
   
    #Same procidure as before for the test data.

    # The path were the images are located
    
    filepath = os.path.join("../data/small_validation")

    X_test=[]

    for folder in Path(filepath).glob("*"):
        for file in Path(folder).glob("*"):
            image_open = cv2.imread(str(file))
            dim = (picture_size, picture_size)
            resize_image = cv2.resize(image_open, dim, interpolation = cv2.INTER_AREA)
            X_test.append(resize_image.astype("float") / 255.)

            
    '''
    -------------------------convert data to numpy arrays:-----------------------
    '''
    trainX = np.array(X_train)
    testX = np.array(X_test)
    
    
    '''
    -------------------------Create model:------------------------------------------
    '''
    # define model
    model = Sequential()

    # first set of CONV => RELU => POOL
    model.add(Conv2D(32, (kernal_size, kernal_size), 
                     padding="same", #Det samme for hver kolonne
                     input_shape=(picture_size, picture_size, 3)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), 
                           strides=(2, 2)))

    # second set of CONV => RELU => POOL
    model.add(Conv2D(50, (3, 3), 
                     padding="same"))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), 
                           strides=(2, 2)))

    # FC => RELU
    model.add(Flatten())
    model.add(Dense(500))
    model.add(Activation("relu"))

    # softmax classifier
    model.add(Dense(10))
    model.add(Activation("softmax"))
    
    opt = SGD(lr=0.01)
    model.compile(loss="categorical_crossentropy",
                  optimizer=opt,
                  metrics=["accuracy"])
    
    print(model.summary())
    
    
    
    # train model
    H = model.fit(trainX, trainY, 
                  validation_data=(testX, testY), 
                  batch_size=32,
                  epochs=20,
                  verbose=1)
    
    predictions = model.predict(testX, batch_size=32)
    print(classification_report(testY.argmax(axis=1),
                                predictions.argmax(axis=1),
                                target_names=label_names))
    
    '''
    -------------------------plotting models:------------------------------------
    '''
    plot_model(model, to_file = "../output_lenet/lenet_model.png", show_shapes=True, show_layer_names=True)
    plot_history(H, 20)
    
    
if __name__ == '__main__':
    main()