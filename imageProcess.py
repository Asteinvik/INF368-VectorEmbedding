#import keras
#from keras.models import Sequential, Model, model_from_json
#from keras.layers import Dense, Input, Conv2D, AveragePooling2D, Flatten
import numpy as np
#import matplotlib
#matplotlib.use('tkagg')
#from matplotlib import pyplot as plt
#from sklearn.model_selection import train_test_split
#import pandas as pd
from PIL import Image, ImageOps
import glob
import os
import random as rnd

minw = 10000000
minh = 10000000
minr = 10000000
maxw = 0
maxh = 0
maxr = 0
nTRAIN = 2
nTEST = 0
nVALID = 0

def preprocess():
	global minw, minh, minr, maxw, maxh, maxr
	
	src_dir = "imgs/"
	dst_dir = "data/train/"
	test_dir = "data/test/"
	valid_dir = "data/validate/"
	classes = readClassnames()
	
	for dirs, i in zip(classes,range(len(classes))):
		
		train_directory=dst_dir+dirs+"/"
		test_directory=test_dir+dirs+"/"
		valid_directory=valid_dir+dirs+"/"
		createDir(train_directory)
		createDir(test_directory)
		createDir(valid_directory)
		n_images=len([name for name in os.listdir(src_dir+dirs) if os.path.isfile(os.path.join(src_dir+dirs, name))])
		print(dirs,n_images)

		size_diff = nTRAIN-n_images
		test_diff = nTEST
		valid_diff = nVALID
		in_trainset = 0
		for mediafile in glob.iglob(src_dir+dirs+"/*.jpg"):
			if in_trainset>=nTRAIN:
				break
			im=Image.open(mediafile)
			mediafile=mediafile.split("/")
			image=im.copy()
			#print(image.size,mediafile[-1])

			ratio=image.size[0]/image.size[1]
			if(ratio<1):
				image = image.rotate(90,expand=True)
				
			maxw, maxh, maxr = sizeStats(max,image.size)
			minw, minh, minr = sizeStats(min,image.size)
			

			if(max(image.size)<100):
				factor=4
				size=(image.size[0]*factor,image.size[1]*factor)
				image=image.resize(size)
			elif(max(image.size)<200):
				factor=2
				size=(image.size[0]*factor,image.size[1]*factor)
				image=image.resize(size)

			maxsize = (299, 299)
			image.thumbnail(maxsize, Image.LANCZOS)
			image = pad(image)
			image = image.convert(mode="RGB")

			if(test_diff>0 and rnd.random()<0.15):
				test_diff-=1
				image.save(test_directory+mediafile[-1],"JPEG")
			elif(valid_diff>0 and rnd.random()<0.30):
				valid_diff-=1
				image.save(valid_directory+mediafile[-1],"JPEG")
			else:
				if(size_diff<0 and in_trainset<nTRAIN): #if set is larger than wanted some will be dropped ranomly
					in_trainset+=1
					image.save(train_directory+mediafile[-1],"JPEG")

				elif(size_diff>0 and in_trainset<nTRAIN):#if set is smaller than wanted some will be augmented
					in_trainset+=1
					image.save(train_directory+mediafile[-1],"JPEG")

					if(rnd.random()<size_diff/n_images):
						mirror = ImageOps.mirror(image)
						in_trainset+=1
						mirror.save(train_directory+"flip"+mediafile[-1],"JPEG")
						if(size_diff/n_images>3):
							mirror =im.rotate(90)
							in_trainset+=1
							mirror.save(train_directory+"rotate"+mediafile[-1],"JPEG")
					if(size_diff/n_images>2):
						image =im.rotate(90)
						in_trainset+=1
						image.save(train_directory+"rotate"+mediafile[-1],"JPEG")
		

	print("maximum: ","width: ",maxw," height: ",maxh," ratio: ",maxr)
	print("minimum: ","width: ",minw," height: ",minh," ratio: ",minr)


def pad(image):
	goal=299

	padding_width = (goal-image.size[0])//2
	padding_height = (goal-image.size[1])//2

	padding = (padding_width, padding_height, goal-image.size[0]-padding_width, goal-image.size[1]-padding_height)
	
	image = ImageOps.expand(image, padding, fill=255)
	return image


def readClassnames(filename="Ex1_selected_categories"):
	classes = []
	with open(filename,"r")as f:
	    for line in f:
	        classes.append(line.strip())
	return classes

def sizeStats(func,im_size):
	 return (func((maxw,im_size[0])), 
	 		func((maxh,im_size[1])), 
	 		func((maxr,(im_size[0]/im_size[1]))))

def createDir(directory):
		if not os.path.exists(directory):
			os.makedirs(directory)

preprocess()


