#This is the main script for training and saving a model given data.
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
import PIL
import numpy as np



def image_reshape(image):
	if image.ndim > 2:
		#print(image)
		print(image.ndim)
		print(image.shape)
		image = np.dot(image[...,:3], [0.299, 0.587, 0.114])


''' Target dimensions of our images -- Keras' ImageDataGenerator class will automatically reform an image of any size to a target
	This is set smaller than the actual images entirely because of memory constraints. 1024x1024 images require more RAM than I have
	available to run the system, and it segfaults, coredumps, or otherwise fails when set too high for the machine it is running on.
	As more memory is available on a larger instance or new system, increase this as possible and decrease the batch_size to work within
	the new memory constraints. This will allow for better resolution of the original images and less data loss.
'''
img_width, img_height = 256, 256

#number of classes -- this has been constrained from the original 15 to test the original set up. see create_data_structure.py for classes used
num_classes = 5

#set up variables
train_data_dir = 'data/train'
validation_data_dir = 'data/validation'
nb_train_samples = 1449
nb_validation_samples = 168
epochs = 1
batch_size = 70

#make sure the numpy arrays are formatted correctly based on the system settings
if K.image_data_format() == 'channels_first':
    input_shape = (1, img_width, img_height)
else:
    input_shape = (img_width, img_height, 1)

#instantiate an empty sequential model
model = Sequential()

#create first convolutional + pooling layer
model.add(Conv2D(256, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(4, 4)))

#create second convolutional + pooling layer
model.add(Conv2D(512, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(4, 4)))

#create a fully connected layer followed by a filter layer of size 1k and then the output layer
model.add(Flatten())
model.add(Dense(1000, activation='relu'))
model.add(Dense(num_classes, activation='relu'))

#compile the model with the assigned loss and optimizer -- metrics here are only for training info.
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True)#,
    #preprocessing_function=image_reshape)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1. / 255)

#generate train image data using Keras' built in image loader -- set to grayscale
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    color_mode="grayscale")

#generate validation image data using Keras' built in image loader -- set to grayscale
validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    color_mode="grayscale")

#calculate epoch steps and run training
model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size)

#save the model weights for use later
model.save_weights('training_weights_1.h5')