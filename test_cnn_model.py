#This is the main script for training and saving a model given data.
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
import PIL
import numpy as np
import os
import tensorflow as tf
from PIL import Image
from fnmatch import fnmatch
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
import csv

#for vector creation purposes -- not used
pathogens = ['Cardiomegaly','Pneumothorax','Mass','Effusion','No Finding']

'''
Generates and returns the lookup dictionary with keys as file names and values as arrays of length 5
indicating the presence (or lack of) the above pathogens in the order of the pathogens array
'''
def create_lookup_dict():
	lookup_dict = {}
	with open("Data_Entry_2017.csv") as a:
		reader = csv.reader(a)
		next(reader,None)
		for row in reader:
			a = [0,0,0,0,0]
			if "Cardiomegaly" in row[1]:
				a[0]=1
			if "Effusion" in row[1]:
				a[1]=1
			if "Mass" in row[1]:
				a[2]=1
			if "No Finding" in row[1]:
				a[3]=1
			if "Pneumothorax" in row[1]:
				a[4]=1
			lookup_dict[row[0]] = a
	return lookup_dict


LD = create_lookup_dict()

'''
Takes 2 inputs, the true value vector and the prediction vector for a single file. It calculates the confusion
matrix for THAT file. We can then sum the confusion matrices to get the result for the whole test set
'''
def create_individual_confusion_matrix(true,pred):
	conf_mat = np.zeros((5,5))
	for i,j in enumerate(true):
		if j == 1:
			if pred[i] == 1:
				#if they match, put one on the diagonal
				conf_mat[i,i] = 1
			else:
				#if they don't, put one at the first index one is found
				conf_mat[i,pred.index(1)] = 1
	return conf_mat


'''
Takes the prediction vector (which is a vector of sigmoided values, not 0 or 1 activations) and turns it into a one-hot
vector corresponding to the classes above. The thresholds were calculated after a run of returning the collection of sigmoid values
(e.g. .059234, .124325, .052342, .995324, .091234) and calculating appropriate top percentiles to determine actual activation.
This idea of constraining the output was taken from adapting the idea presented here https://arxiv.org/abs/1707.09457
0.124887514	0.20296256	0.037007332	0.997478092	0.181568922

'''
def activate_pred(pred):
	new_pred = [0,0,0,0,0]
	if pred[0] > 0.124887514:
		new_pred[0] = 1
	else:
		new_pred[0] = 0
	new_pred
	if pred[1] > 0.20296256:
		new_pred[1] = 1
	else:
		new_pred[1] = 0
	new_pred
	if pred[2] > 0.037007332:
		new_pred[2] = 1
	else:
		new_pred[2] = 0
	new_pred
	if pred[3] > 0.997478092:
		new_pred[3] = 1
	else:
		new_pred[3] = 0
	new_pred
	if pred[4] > 0.181568922:
		new_pred[4] = 1
	else:
		new_pred[4] = 0
	if np.sum(new_pred) < 1:
		new_pred[3] = 1
	return new_pred


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
test_data_dir = 'data/test'
nb_test_samples = 5
batch_size = 5

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
model.add(Dense(num_classes, activation='sigmoid'))



model.load_weights("training_weights_1.h5")

print("Prediction Time!")


total_conf_matrix = np.zeros((5,5))
for path, subdirs, files in os.walk("data/test"):
    for name in files:
        if fnmatch(name, "*.png"):
            #print(os.path.join(path, name))
            img = Image.open(os.path.join(path, name))
            img = img.resize((img_width,img_height), Image.ANTIALIAS)
            np_array = np.asarray(img)
            np_array = np_array / 255.
            np_array = np.reshape(np_array,[1,img_width,img_height,1])
            #tensor = tf.convert_to_tensor(np_array,dtype=tf.float32)
            prediction = model.predict(np_array)
            new_pred = activate_pred(prediction[0])
            true_val = LD[name]
            conf_mat = create_individual_confusion_matrix(true_val,new_pred)
            total_conf_matrix += conf_mat



print(total_conf_matrix)
