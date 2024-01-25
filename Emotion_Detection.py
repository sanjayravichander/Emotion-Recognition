# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 22:46:54 2024

@author: DELL
"""
## Emotion Recognition ##

from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Dense,Dropout,Flatten
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
import scipy.stats
import scipy
import numpy as np
import cv2


## ImageDataGenerator is used to preprocess the Images ##
## We have downloaded the training images and testing images from kaggle FER-2013 ##
## ImageDataGenerator is used to preprocess the training images and testing images ##
train_data_gen=ImageDataGenerator(rescale=1./255)
validation_data_gen=ImageDataGenerator(rescale=1./255)

## Preprocess all test images ##
train_generator= train_data_gen.flow_from_directory(
    'C:\\Users\\DELL\\Downloads\\Emotion Recognition\\train', ## Giving the file location of training dataset
    target_size=(48,48), ## giving the size of the image
    batch_size=64, ## takes 64 images at one rate
    color_mode='grayscale', ## giving color to grayscale, imp step in preprocessing
    class_mode='categorical') ## the training data set has different category such as several emotions. This code will seperate them in individual category

## Preprocess all train images ##
validation_generator=validation_data_gen.flow_from_directory(
    'C:\\Users\\DELL\\Downloads\\Emotion Recognition\\test',
    target_size=(48,48),
    batch_size=64,
    color_mode='grayscale',
    class_mode='categorical')

## Creating CNN Model ##
emotion_model=Sequential()
emotion_model.add(Conv2D(32,kernel_size=(3,3),activation='relu',input_shape=(48,48,1)))
#Convolutional 2 Dimension is added to read grid like data like images,number of filters is given as 32
#this will learns the image data,kernal size is used to specify the height and width of convolutional 2D layer means filter is given at size of 3x3
#activation is output function.Many types are there like sigmoid,tanh etc but relu is best one
#input shape is given only in the first line and its about resizing the image and 1 specify the grayscale image
emotion_model.add(Conv2D(64, kernel_size=(3,3), activation='relu')) ## Adding this line again because we are adding more filters to learn the image data
emotion_model.add(MaxPooling2D(pool_size=(2,2)))
#MaxPooling2D is a pooling layer in convolutional neural networks (CNNs)
#--that is often used to downsample the spatial dimensions of the input data.
#Pooling layers help reduce the computational load and number of parameters in the network 
#--while retaining important features
#The "pool size" parameter in MaxPooling2D determines the size of the small rectangular regions(photo grid), 
#--called pooling windows, that slide over the input data during the pooling operation.Smaller the pool size more the
#--spatial information we learn
emotion_model.add(Dropout(0.25))
#Dropout is a regularization technique commonly used in neural networks,including convolutional neural networks (CNNs) and deep learning models.
#It helps prevent overfitting by randomly setting a fraction of input units to zero during training, effectively "dropping out" those units.
#This helps the model generalize better to unseen data.

emotion_model.add(Conv2D(128, kernel_size=(3,3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2,2)))
emotion_model.add(Conv2D(128, kernel_size=(3,3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2,2)))
emotion_model.add(Dropout(0.25))

emotion_model.add(Flatten())
#The Flatten layer in neural networks is used to flatten the input data,converting it from 
#--a multi-dimensional tensor into a one-dimensional tensor. 
#--It's typically used when transitioning from convolutional layers (2D) to dense layers (1D) in a model architecture.
emotion_model.add(Dense(1024, activation='relu'))
emotion_model.add(Dropout(0.5))
emotion_model.add(Dense(7, activation='softmax'))
emotion_model.compile(loss='categorical_crossentropy',optimizer=Adam(learning_rate=0.0001,epsilon=1e-6),metrics='accuracy')

## To Train the Model
emotion_model_info=emotion_model.fit_generator(train_generator,
                                     steps_per_epoch=28709//64,
                                     epochs=50,
                                     validation_data=validation_generator,
                                     validation_steps=7178//64)


## Creating key value pairs for all the emotions that we have downloaded 
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# start the webcam feed
#video_capture = cv2.VideoCapture(0)

video_capture = cv2.VideoCapture("C:\\Users\\DELL\\Downloads\\Sofya_ed.mp4")

while True:
    ## Reading the Video frame
    ret,frame=video_capture.read()
    frame = cv2.resize(frame, (400,680))
    if not ret: ##If there is no video detected break
        break
    ## Importing Face Classifier ##
    face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_frontalface_default.xml")
    gray_img = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    ## Detecting Multi Faces
    faces = face_classifier.detectMultiScale(gray_img,scaleFactor=1.3,minNeighbors=5)
    ## Drawing Rectangle in face
    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y-50), (x+w,y+h+10),(0,255,0),2)#subtracting 50 from y moves the rectangle upwards.Adding 10 to h extends the rectangle downwards.
        # This denotes the range of rows (vertical pixels) to be included in the ROI. 
        #--The starting row is y, and it extends to y+h.
        #--This is determined by the y coordinate and the height h of the detected object.
        #--This denotes the range of columns (horizontal pixels) to be included in the ROI. 
        #The starting column is x, and it extends to x+w.
        #--This is determined by the x coordinate and the width w of the detected object.
        #--This is often done to perform further processing or analysis on the specific region where the object is located in the image.
        roi_gray_frame=gray_img[y:y+h,x:x+w]
        #This adds a new axis at the end of the array, effectively converting a 2D grayscale image into a 3D image by adding a channel dimension. 
        #--This is necessary because many deep learning models expect input images to have three dimensions: height, width, and channels.
        #--This adds another axis at the beginning of the array, creating a batch dimension. 
        #--In deep learning, input data is often organized into batches, and this additional dimension represents the batch size.
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48,48)),-1),0)
        ## Predict the Emotion using trained model we built
        emotion_prediction=emotion_model.predict(cropped_img)
        #This function returns the index of the maximum value in the array emotion_prediction.
        #--It finds the position of the highest value in the array.
        #-- This is often used in scenarios where you have a set of predictions 
        #--(e.g., probabilities for different emotions) and you want to identify the index corresponding 
        #--to the predicted emotion with the highest confidence or probability.
        max_index=int(np.argmax(emotion_prediction))
        #cv2.putText(frame, text=emotion_dict[max_index], org=(x+5,y-20), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255,0,0),thickness=2,cv2.LINE_AA)
        cv2.putText(frame, emotion_dict[max_index], (x+5, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.imshow('Emotion Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()
    