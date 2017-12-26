# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
import csv
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
import scipy
import numpy as np
import argparse

import segment as sg

import os
import os.path
from os import listdir
from os.path import isfile, join
import pandas as pd

def evaluate(index, zone):

	if(not os.path.isfile('models/zone'+str(zone)+'/'+str(index)+'/checkpoint')): 
		return [],[],[]

	# Same network definition as before
	img_prep = ImagePreprocessing()
	img_prep.add_featurewise_zero_center()
	img_prep.add_featurewise_stdnorm()
	img_aug = ImageAugmentation()
	img_aug.add_random_flip_leftright()
	img_aug.add_random_rotation(max_angle=25.)
	img_aug.add_random_blur(sigma_max=3.)
#adding here
	if(zone == 13 or zone==14):
		dimX = 256
		dimY = 75
	elif(zone == 11 or zone==12):
		dimX = 256
		dimY = 75
	elif(zone == 10):
		dimX = 237
		dimY = 80
	elif(zone == 9):
		dimX = 50
		dimY = 80
	elif(zone == 8):
		if(index == 1 or index == 13):
			dimX = 225
			dimY = 80
		elif(index == 9):
			dimX = 237
			dimY = 80
	elif(zone == 6 or zone == 7):
		dimX = 256
		dimY = 60
	elif(zone == 5):
		dimX = 512
		dimy = 80
	else:
		dimX = 0
		dimY = 0
	network = input_data(shape=[None, dimY, dimX, 1], #zone14,13
	#network = input_data(shape=[None, 80, 512, 1], #zone5
	                     data_preprocessing=img_prep,
	                     data_augmentation=img_aug)
	network = conv_2d(network, 32, 3, activation='relu')
	network = max_pool_2d(network, 2)
	network = conv_2d(network, 64, 3, activation='relu')
	network = conv_2d(network, 64, 3, activation='relu')
	network = max_pool_2d(network, 2)
	network = fully_connected(network, 128, activation='relu')
	network = dropout(network, 0.5)
	network = fully_connected(network, 2, activation='softmax')
	network = regression(network, optimizer='adam',
	                     loss='categorical_crossentropy',
	                     learning_rate=0.001)

	model = tflearn.DNN(network, tensorboard_verbose=0, checkpoint_path='models/zone'+str(zone)+'/'+str(index)+'/TSA-Zone'+str(zone)+'-Angle-'+str(index)+'.tfl') #zone14
	model.load("models/zone"+str(zone)+"/"+str(index)+"/TSA-Zone"+str(zone)+"-Angle-"+str(index)+".tfl") #zone14




	apsid = []
	predArray = []
	bnotb = []

	mypath = "../aps/test_data/"
	onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
	tp = 0
	tn = 0
	fp = 0
	fn = 0
	for fname in onlyfiles:
		if fname.endswith(".aps"):
	# Load the image file
	#img = scipy.ndimage.imread(args.image, mode="RGB")
			apsid.append(fname.split('.')[0])
			single_image = sg.get_single_image("../aps/test_data/"+fname, index)
			img = sg.convert_to_grayscale(single_image)
			crop_dim =sg.get_crop_dimensions(index, zone)
			img = img[crop_dim[0]:crop_dim[1],crop_dim[2]:crop_dim[3]]
			img = np.asfarray(img).reshape(dimY,dimX,1) #zone14,13

			# Predict
			prediction = model.predict([img])
			#print(prediction)
			predArray.append(prediction)
			# Check the result.
			is_threat = np.argmax(prediction[0]) == 1

			bnotb.append(is_threat)
			final_result = []
			filename = 'test_labels.csv'
			with open(filename) as csvfile:
				readCSV = csv.reader(csvfile, delimiter=',')
				for row in readCSV:
					final_result.append(row)
			if is_threat:
				print("That's detetced: "+str(fname))
				flag = True
				for value in final_result:
					if value[0]+".aps" == fname:
						label = int(value[1])
						if zone == label:
							tp = tp+1
							flag = False
							break
				if flag:
					fp = fp+1                
			else:
				flag = True
				for value in final_result:
					if value[0]+".aps" == fname:
						label = int(value[1])
						if zone == label:
							fn = fn+1
							flag = False
							break
				if flag:
					tn = tn+1
	print('True positives',tp)
	print('False positives',fp)
	print('True negatives',tn)
	print('False negatives',fn)
	return apsid,predArray,bnotb
