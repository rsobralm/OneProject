from tensorflow.keras.applications.inception_v3 import InceptionV3
import os
import zipfile
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers
from tensorflow.keras import Model
import numpy as np
import sklearn.metrics as metrics


train_data_path = 'data/newtrain'
validation_data_path = 'data/newtest'

img_width, img_height = 150, 150
batch_size = 10

# Import the inception model

pre_trained_model = InceptionV3(input_shape=(150, 150, 3),  # Shape of our images
                                include_top=False,  # Leave out the last fully connected layer
                                weights='imagenet')


for layer in pre_trained_model.layers:
    layer.trainable = False


class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('accuracy') > 0.959):
            print("\nReached 99.9% accuracy so cancelling training!")
            self.model.stop_training = True

from tensorflow.keras.optimizers import RMSprop

# Flatten the output layer to 1 dimension
x = layers.Flatten()(pre_trained_model.output)
# Add a fully connected layer with 1,024 hidden units and ReLU activation
x = layers.Dense(1024, activation='relu')(x)
# Add a dropout rate of 0.2
x = layers.Dropout(0.2)(x)                  
# Add a final sigmoid layer for classification
x = layers.Dense  (5, activation='sigmoid')(x)           

model = Model( pre_trained_model.input, x) 

model.compile(optimizer = RMSprop(learning_rate=0.0001), 
              loss = 'categorical_crossentropy', 
              metrics = ['accuracy', 'FalseNegatives', 'FalsePositives', 'TrueNegatives', 'TruePositives'])

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

test_generator = test_datagen.flow_from_directory(
    validation_data_path,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    shuffle=False,
    class_mode='categorical')


callbacks = myCallback()
history = model.fit(
            train_generator,
            validation_data = validation_generator,
            steps_per_epoch = 60,
            epochs = 10,
            validation_steps = 50,
            verbose = 2,
            callbacks=[callbacks])


predictions = model.predict(test_generator)
# Get most likely class
predicted_classes = np.argmax(predictions, axis=1)
#print(len(predicted_classes))
print(predicted_classes)
true_classes = test_generator.classes
class_labels = list(test_generator.class_indices.keys())
print(true_classes)
#print(testdata)
report = metrics.classification_report(true_classes, predicted_classes, target_names=class_labels)
print(report)    
confusion_matrix = metrics.confusion_matrix(y_true=true_classes, y_pred=predicted_classes)
print(confusion_matrix) 
print("Accuracy: ", metrics.accuracy_score(true_classes, predicted_classes))
print("Precision: ", metrics.precision_score(true_classes, predicted_classes, average='micro'))