# Building the CNN
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

'''
This import code has been added to get rid of the error while compiling
ValueError: The shape of the input to "Flatten" is not fully defined
This is related to the Convolution2D input_shape dimension ordering
'''
import keras.backend as K
K.set_image_dim_ordering('th')

# Initializing the CNN
classifier = Sequential()

# step1 - Convolution
classifier.add(Convolution2D(32, 3, 3, input_shape=(3, 64, 64), activation= 'relu'))

# step2 - pooling
classifier.add(MaxPooling2D(pool_size= (2, 2)))

# Adding second Convolution layer to improve accuracy
classifier.add(Convolution2D(32, 3, 3, activation= 'relu'))
classifier.add(MaxPooling2D(pool_size= (2, 2)))

#step3 - flattening
classifier.add(Flatten())

# step4 - Full connection
# hidden layer
classifier.add(Dense(output_dim= 128, activation= 'relu'))
# output layer
classifier.add(Dense(output_dim = 1, activation = 'sigmoid'))

# compiling the CNN
classifier.compile(optimizer= 'adam', loss= 'binary_crossentropy', metrics= ['accuracy'])

# Fitting the CNN to the images
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        'dataset/training_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

test_set = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

classifier.fit_generator(
        training_set,
        steps_per_epoch=8000,
        epochs=10,
        validation_data=test_set,
        validation_steps=2000)

