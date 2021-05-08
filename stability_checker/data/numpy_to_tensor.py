import os
import glob
import cv2
import numpy as np
import torch
from sklearn.model_selection import train_test_split
COLOR=  False
IMAGE_SIZE = 65

folders = ['stable','unstable']   ########### Fix here
print (folders)

num_classes = len(folders)
data_dictionary= {}

for i in range(num_classes): ###fix here
	data_dictionary[folders[i]] = sorted(glob.glob('/home/saeid/stack/data/'+folders[i]+'/*.jpg')) 

#print (data_dictionary)
# Creating a numpy array to get 3 images, each has a 4096 * 3072* 3 dimensions
#all_data = np.zeros((len(data_dictionary['stable'])+len(data_dictionary['unstable']),512,512,3)) # got memory error
#all_data = np.zeros((len(data_dictionary['stable'])+len(data_dictionary['unstable']),25,25,3)) # got memory error
if COLOR:
	all_data = np.zeros((len(data_dictionary['stable'])+len(data_dictionary['unstable']),IMAGE_SIZE,IMAGE_SIZE,3))
else:
	all_data = np.zeros((len(data_dictionary['stable'])+len(data_dictionary['unstable']),IMAGE_SIZE,IMAGE_SIZE))
all_labels = np.zeros((len(data_dictionary['stable'])+len(data_dictionary['unstable'])))   #dog = [1,0], #cat = [0,1]
counter=0




for i, folder_name in enumerate(folders):
	for j in range(len(data_dictionary[folder_name])):

		if COLOR:
			img = cv2.imread(data_dictionary[folder_name][j])
			img =cv2.cvtColor(img, cv2.COLOR_BGR2RGB)			
		else:

			img = cv2.imread(data_dictionary[folder_name][j],0)
		#img = cv2.imread('cat.jpg',0)   #This converts the image to grayscale
		#print (folder_name)
		#print ('\nimg name is ',data_dictionary[folder_name][j])

		img = cv2.resize(img,(IMAGE_SIZE,IMAGE_SIZE))
		#print ('\nimg shape is:')
		#print (img.shape)

		# Converting BGR to RGB because cv2 works in BGR
		# While matplotlib works in RGB

		#
		#print (i*len(data_dictionary[folder_name])+j)
		#all_data[counter,:,:,:] = img
		all_data[counter,:,:] = img

		if folder_name =='stable':
			all_labels[counter] =np.array([1])
		else:
			all_labels[counter] =np.array([0])

		counter+=1
		if counter%1000 ==0:
			print (counter)

#print (all_labels[:,:])

X_train,X_test, y_train, y_test = train_test_split(all_data,all_labels,test_size=0.33, random_state=42)
del all_data
del all_labels

print (X_train.shape)
print (y_train.shape)
print (X_test.shape)
print (y_test.shape)
#input()

tensor_X = torch.from_numpy(X_train)
tensor_y = torch.from_numpy(y_train)
with open('/home/saeid/stack/data/training.pt', 'wb') as f:
            torch.save((tensor_X, tensor_y), f)

tensor_X = torch.from_numpy(X_test)
tensor_y = torch.from_numpy(y_test)
with open('/home/saeid/stack/data/test.pt', 'wb') as f:
            torch.save((tensor_X, tensor_y), f)

