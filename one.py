import os
import keras
from keras.backend import print_tensor
import numpy as np
import sklearn.metrics as metrics
from keras.models import Model
from keras.layers import Dense
from tensorflow.keras import optimizers
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.preprocessing import image



trdata = ImageDataGenerator()
traindata = trdata.flow_from_directory(directory="data/newtrain",target_size=(224,224))
tsdata = ImageDataGenerator()
testdata = tsdata.flow_from_directory(directory="data/newtest", target_size=(224,224))


from keras.applications.vgg16 import VGG16
vggmodel = VGG16(weights='imagenet', include_top=True)

#vggmodel.summary()

for layers in (vggmodel.layers)[:19]:
    print(layers)
    layers.trainable = False

X= vggmodel.layers[-2].output
predictions = Dense(7, activation="softmax")(X)
model_final = Model(vggmodel.input, predictions)

model_final.compile(loss = "categorical_crossentropy", optimizer = optimizers.SGD(learning_rate=0.0001, momentum=0.9), metrics=['accuracy', 'AUC', 'Recall', 'FalseNegatives', 'FalsePositives', 'TrueNegatives', 'TruePositives'])


from keras.callbacks import ModelCheckpoint, EarlyStopping
checkpoint = ModelCheckpoint("vgg16_1.h5", monitor='val_accuracy', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
early = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=40, verbose=1, mode='auto')
hist = model_final.fit(traindata, steps_per_epoch= 10, epochs= 20, validation_data= testdata, validation_steps=1, callbacks=[checkpoint,early])
'''

predictions = model_final.predict(testdata)
# Get most likely class
predicted_classes = numpy.argmax(predictions, axis=1)
print(len(predicted_classes))
print(predicted_classes)

true_classes = testdata.classes
class_labels = list(testdata.class_indices.keys())

print(true_classes)

print(testdata)

report = metrics.classification_report(true_classes, predicted_classes, target_names=class_labels)
print(report)    

confusion_matrix = metrics.confusion_matrix(y_true=true_classes, y_pred=predicted_classes)
print(confusion_matrix) 
print("Accuracy: ", metrics.accuracy_score(true_classes, predicted_classes))
print("Precision: ", metrics.precision_score(true_classes, predicted_classes, average='micro'))
'''

test_path = 'data/newtest'

img_width, img_height = 224, 224


#Prediction Function
def predict(file):
  x = load_img(file, target_size=(img_width,img_height))
  x = img_to_array(x)
  x = np.expand_dims(x, axis=0)
  array = model_final.predict(x)
  result = array[0]
  #print(result)
  answer = np.argmax(result)

  return answer


predictedList = []
real = []
cat_index = -1
for i, ret in enumerate(os.walk(test_path)):
    for i, filename in enumerate(ret[2]):
        if filename.startswith("."):
            continue
        real.append(cat_index)
        predictedList.append(predict(ret[0] + '/' + filename))
        #print(ret[0] + '/' + filename)
    cat_index+=1
        
    #  result = predict(ret[0] + '/' + filename)
    #print(" ")

true_classes = testdata.classes
class_labels = list(testdata.class_indices.keys())
print(true_classes)
print("\n")
print(class_labels)

print("\n")
confusion_matrix = metrics.confusion_matrix(y_true=real, y_pred=predictedList)
print(confusion_matrix) 
print("Accuracy: ", metrics.accuracy_score(true_classes, predictedList))
print("Precision: ", metrics.precision_score(true_classes, predictedList, average='micro'))

'''
import matplotlib.pyplot as plt
plt.plot(hist.history["accuracy"])
plt.plot(hist.history['val_accuracy'])
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title("model accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend(["Accuracy","Validation Accuracy","loss","Validation Loss"])
plt.show()'''