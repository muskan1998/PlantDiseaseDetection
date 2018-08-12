import keras
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD, Adam, Adamax
from keras.preprocessing.image import ImageDataGenerator
import cv2
import numpy as np
import h5py

#'vgg16_weights.h5''vgg16_weights.h5'

model=keras.applications.vgg16.VGG16(include_top=True, weights=None, input_tensor=None, input_shape=None, pooling=None, classes=1000)

top_model = Sequential()
top_model.add(Dropout(0.5))
top_model.add(Dense(38, activation='softmax'))



# CREATE AN "REAL" MODEL FROM VGG16
# BY COPYING ALL THE LAYERS OF VGG16
new_model = Sequential()
for l in model.layers:
    new_model.add(l)


# CONCATENATE THE TWO MODELS

new_model.add(top_model)


# LOCK THE TOP CONV LAYERS

for layer in model.layers:
    layer.trainable = False





new_model.summary()
new_model.load_weights('my_model_weights2.h5')

"""

train_datagen = ImageDataGenerator(
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
"""



train_datagen = ImageDataGenerator()

test_datagen = ImageDataGenerator()
                                  
nb_train_samples = 1202
nb_validation_samples = 771
epochs = 3
batch_size = 16

train_generator = train_datagen.flow_from_directory(
        '../Data2/train',  
        batch_size=batch_size,
        shuffle=True,
        target_size=(224,224),
        class_mode='categorical')  

validation_generator = test_datagen.flow_from_directory(
        '../Data2/val',  
        batch_size=batch_size,
        target_size=(224, 224),
        shuffle=True,
        class_mode='categorical')

new_model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=None, decay=1e-8, amsgrad=False),
              metrics=['accuracy'])



new_model.fit_generator(
    train_generator,
    epochs=epochs,
    validation_data=validation_generator,
    )

new_model.save_weights('my_model_weights2.h5')


