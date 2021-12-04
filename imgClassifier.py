import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import os 
import keras

from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras import optimizers
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense, Activation
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras import callbacks
from keras.applications.vgg16 import VGG16

import sklearn.metrics as metrics


train_data_path = 'data/newtrain'
validation_data_path = 'data/newtest'



################################## TRAINING #########################################
epochs = 20


#Parameters
img_width, img_height = 150, 150
batch_size = 32
samples_per_epoch = 100
validation_steps = 300
nb_filters1 = 32
nb_filters2 = 64
conv1_size = 3
conv2_size = 2
pool_size = 2
classes_num = 7
lr = 0.0004


model = Sequential()

model.add(Convolution2D(nb_filters1, conv1_size, conv1_size, padding ="same", input_shape=(img_width, img_height, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))

model.add(Convolution2D(nb_filters2, conv2_size, conv2_size, padding ="same"))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(pool_size, pool_size), data_format="channels_last"))

model.add(Flatten())
model.add(Dense(256))
model.add(Activation("relu"))
model.add(Dropout(0.5))
model.add(Dense(classes_num, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.RMSprop(learning_rate=lr),
              metrics=['accuracy'])


train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_path,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    validation_data_path,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')


model.fit(
    train_generator,
    #steps_per_epoch=10,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=validation_steps)


predictions = model.predict(validation_generator)
# Get most likely class
predicted_classes = np.argmax(predictions, axis=1)
print(len(predicted_classes))
print(predicted_classes)

true_classes = validation_generator.classes
class_labels = list(validation_generator.class_indices.keys())

#print(true_classes)

#print(testdata)

report = metrics.classification_report(true_classes, predicted_classes, target_names=class_labels)
print(report)    

confusion_matrix = metrics.confusion_matrix(y_true=true_classes, y_pred=predicted_classes)
print(confusion_matrix) 
print("Accuracy: ", metrics.accuracy_score(true_classes, predicted_classes))
print("Precision: ", metrics.precision_score(true_classes, predicted_classes, average='micro'))



################################## TESTING #########################################
test_path = 'data/alien_test'

img_width, img_height = 150, 150


#Prediction Function
def predict(file):
  x = load_img(file, target_size=(img_width,img_height))
  x = img_to_array(x)
  x = np.expand_dims(x, axis=0)
  array = model.predict(x)
  result = array[0]
  #print(result)
  answer = np.argmax(result)
  if answer == 1:
    print("Predicted: Predator")
  elif answer == 0:
    print("Predicted: Alien")


  return answer

'''
for i, ret in enumerate(os.walk(test_path)):
  for i, filename in enumerate(ret[2]):
    if filename.startswith("."):
      continue
    
    print(ret[0] + '/' + filename)
    result = predict(ret[0] + '/' + filename)
    print(" ")'''