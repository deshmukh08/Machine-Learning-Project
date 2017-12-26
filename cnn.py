
# -*- coding: utf-8 -*-

"""
Based on the tflearn example located here:
https://github.com/tflearn/tflearn/blob/master/examples/images/convnet_cifar10.py
"""
from __future__ import division, print_function, absolute_import
import numpy as np
# Import tflearn and some helpers
import tflearn
from tflearn.data_utils import shuffle
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
import pickle
import csv

import segment as sg
import csv
import os
import sys
import numpy as np
import tensorflow as tf
# Load the data set
#X, Y, X_test, Y_test = pickle.load(open("full_dataset.pkl", "rb"))
testList = []
angle = int(sys.argv[1])
zone = int(sys.argv[2])
with open('csvFiles/Zone'+str(zone)+'Test.csv') as csvfile:
	readCSV = csv.reader(csvfile, delimiter=',')
	for row in readCSV:
		testList.append(row[0])

trainList = []
with open('csvFiles/Zone'+str(zone)+'Train.csv') as csvfile:
	readCSV = csv.reader(csvfile, delimiter=',')
	for row in readCSV:
		trainList.append(row[0])
models = []
#change the 0 to 1, 2 and so on and so forth
for i in range(0,17):
	models.append(i)

for i in range(angle,angle+1):

	print("Begin - "+str(i))
	testZones = sg.get_cropped_zones('../aps/', filelist = testList, file_extension='aps', angle=i)
	trainZones = sg.get_cropped_zones('../aps/', filelist = trainList, file_extension='aps', angle=i)
	if(len(testZones[zone-1][0])==0):
		continue

	testZones = testZones[zone-1]
	trainZones = trainZones[zone-1]
	X = trainZones[0]
	Y = trainZones[1]
	X_test = testZones[0]
	Y_test = testZones[1]
	if(zone == 14 or zone == 13):
		len_files_train = 917
		len_files_test = 230
		dimX = 256
		dimY = 75
	elif(zone == 11 or zone == 12):
		len_files_train = 916
		len_files_test = 231
		dimX = 256
		dimY = 75
	elif(zone == 10):
		len_files_train = 917
		len_files_test = 230
		dimX = 237
		dimY = 80
	elif(zone == 9):
		len_files_train = 917
		len_files_test = 230
		dimX = 50
		dimY = 80
	elif(zone == 8):
		len_files_train = 917
		len_files_test = 230
		dimX = 225
		dimY = 80
		if(angle == 9):
			dimX = 237
			dimY = 80
	elif(zone == 7):
		len_files_train = 917
		len_files_test = 230
		dimX = 256
		dimY = 60
	elif(zone == 6):
		len_files_train = 916
		len_files_test = 231
		dimX = 256
		dimY = 60
	elif(zone == 5):
		len_files_train = 916
		len_files_test = 231
		dimX = 512
		dimY = 80
	else:
		len_files_train = 0
		len_files_test = 0
		dimX = 0
		dimY = 0

	X = np.asfarray(X).reshape(len_files_train, dimY,dimX,1)
	Y = np.asarray(Y)
	Y = Y.reshape(len_files_train)
	temp = np.zeros((len_files_train, 2), dtype=np.uint8)
	temp[np.arange(len_files_train), Y] = 1
	Y = temp

	X_test = np.asfarray(X_test).reshape(len_files_test, dimY,dimX,1)
	Y_test = np.asarray(Y_test)

	Y_test = Y_test.reshape(len_files_test)
	temp = np.zeros((len_files_test, 2), dtype=np.uint8)
	temp[np.arange(len_files_test), Y_test] = 1
	Y_test = temp

	print(X.shape)
	print(X_test.shape)
	# Shuffle the data
	X, Y = shuffle(X, Y)

	# Normalzation of data

	img_prep = ImagePreprocessing()
	img_prep.add_featurewise_zero_center()
	img_prep.add_featurewise_stdnorm()

	# Image Augmentation

	img_aug = ImageAugmentation()
	img_aug.add_random_flip_leftright()
	img_aug.add_random_rotation(max_angle=25.)
	img_aug.add_random_blur(sigma_max=3.)

	# CNN  architecture:

	# Input image tensor

	network = input_data(shape=[None, dimY, dimX, 1],
	                     data_preprocessing=img_prep,
	                     data_augmentation=img_aug)

	# Convolution

	network = conv_2d(network, 32, 3, activation='relu')

	# Max pooling

	network = max_pool_2d(network, 2)

	# Second layer of  Convolution 

	network = conv_2d(network, 64, 3, activation='relu')

	# Third Layer of Convolution

	network = conv_2d(network, 64, 3, activation='relu')

	# Second Layer Max pool

	network = max_pool_2d(network, 2)

	# Fully-connected 128 node neural network

	network = fully_connected(network, 128, activation='relu')

	# dropout
	network = dropout(network, 0.5)

	# Fully-connected  neural network with two outputs (0=safe,1=threat)
	network = fully_connected(network, 2, activation='softmax')

	# Train the network

	network = regression(network, optimizer='adam',
	                     loss='categorical_crossentropy',
	                     learning_rate=0.001)

	# Make the model object 

	models[i] = tflearn.DNN(network, tensorboard_verbose=0, checkpoint_path='TSA-Zone'+str(zone)+'-Angle-'+str(i)+'.tfl.ckpt')

	# Training the network

	models[i].fit(X, Y, n_epoch=100, shuffle=True, validation_set=(X_test, Y_test),
	          show_metric=True, batch_size=50,
	          snapshot_epoch=True,
	          run_id='TSA-Zone'+str(zone)+'-Angle-'+str(i))

	# Save model when training is complete to a file
	models[i].save("models/zone"+str(zone)+"/"+str(i)+"/TSA-Zone"+str(zone)+"-Angle-"+str(i)+".tfl")
	print("Network trained and saved as TSA-Zone"+str(zone)+"-Angle-"+str(i)+".tfl!")

	print("End - "+str(i))

