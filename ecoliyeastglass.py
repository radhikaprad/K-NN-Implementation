from __future__ import division
import os
import pdb
import csv
import math as mt
import collections
import pandas as pd
import numpy as np
from operator import itemgetter
from math import exp
from sklearn.cross_validation import KFold

# knn class with distance measure using euclidean distance
class knn(object):

	# Iterating and storing each distance calculated using euclidean between a test instance and train instance
	def distancemeasure(self,x_train,y_train,x_test,kcluster,distantmetric):
		testdistance = []
		rbfdistancecalcs = []
		sorteddistances = []
		neartestdistance = []
		polynormial_kernel_array = [] 
		
		for index in range(x_train.shape[0]):	
			if int(distantmetric) == 1:
				dist = self.euclideandistancecalc(x_train[index],x_test,calc_sqrt=True)
				testdistance.append((x_train[index],y_train[index],dist))
			elif int(distantmetric) == 3:
				dist = self.rbf(x_train[index],x_test,x_train.shape[0])
				testdistance.append((x_train[index],y_train[index],dist))
			elif int(distantmetric) == 2:
				dist = self.polynormial_kernel(x_train[index],x_test)
				testdistance.append((x_train[index],y_train[index],dist))
			# rbfdistancecalcs.append((x_train[index],y_train[index],rbfdist))
			# polynormial_kernel_array.append((x_train[index],y_train[index],polynormial_kernel))
		sorteddistances=sorted(testdistance, key=itemgetter(2))
		for index in range(kcluster):
			neartestdistance.append((sorteddistances[index][0],sorteddistances[index][1]))
		return neartestdistance

	# measures distance between a test sample and all train samples
	def euclideandistancecalc(self,x1,x2,calc_sqrt=True):
		dist = np.sum(pow(np.subtract(x1,x2),2))
		if calc_sqrt:
			dist = mt.sqrt(dist)	
		return round(dist,3)

	# getting votes based on the nearby distances
	def rbf(self,xtrain,xtest,nsamples):
		#pdb.set_trace()
		gamma=1.0/nsamples
		# k = dist
		k = 1
		k_exp =2 - 2 * np.exp(-k*np.linalg.norm(np.subtract(xtrain,xtest))/np.square(xtest.std()))
		# sigma=1
		# k_exp = np.exp(-k*np.linalg.norm(np.subtract(xtrain,xtest))/np.square(sigma))
		return k_exp

	def polynormial_kernel(self,xtrain,xtest,gamma=50,delta=1,degree=5):
		return  pow(gamma*np.dot(xtrain.T,xtrain) + delta,degree) + pow(gamma*np.dot(xtest.T,xtest) + delta,degree) - 2 * pow(gamma*np.dot(xtrain.T,xtest) + delta,degree) 

	def votes(self,neartestdistance,sortorder):
		#pdb.set_trace()
		neartestdistance = np.array(neartestdistance)
		neighbourvotes = {}
		for index,x in enumerate(neartestdistance[:,1]):
			if x[0] not in neighbourvotes:
				neighbourvotes[x[0]] = 1
			else:
				neighbourvotes[x[0]] += 1
		neighboursorted = sorted(neighbourvotes.iteritems(), key=itemgetter(1), reverse=sortorder)
		return neighboursorted[0][0]

	#calculating the accuracy between the actual test class and predicted test class
	def getAccuracy(self,retrievedvotes,k):
		count = 0
		totalpercentage = 0
		for i,x in enumerate(retrievedvotes):
			if x[0] in x[1]:
				count += 1
		return round((float(count)/len(retrievedvotes) * 100),3)

# Reads csv file and converts them to matrix. Gets X and Y dataset for classification
def readcsv(filename):
	if "ecoli" in str(filename):
		dataframes = pd.read_csv(filename,skiprows=range(0,0),usecols=range(8))
		sample_x = dataframes.ix[: , :6]
		sample_y = dataframes.ix[: , 7:] 
		return sample_x.as_matrix(), sample_y.as_matrix()
	elif "yeast" in str(filename):
		dataframes = pd.read_csv(filename,usecols=range(1,10),header=None)
		sample_x = dataframes.ix[: , :8]
		sample_y = dataframes.ix[: , 9:]
		return sample_x.as_matrix(), sample_y.as_matrix()
	elif "glass" in str(filename):
		dataframes = pd.read_csv(filename,usecols=range(1,11),header=None)
		sample_x = dataframes.ix[: , :9]
		sample_y = dataframes.ix[: , 10:]
		return sample_x.as_matrix(), sample_y.as_matrix()

#normalising dataset using the formulae (value - minValue) / (maxValue - minValue)
def normalizedata(normalizex):
	minx = normalizex.min(axis = 0) # Get minimum value in each column
	maxx = normalizex.max(axis = 0) # Get maximum value in each column
	#print ("min :", minx)
	#print ("max :", maxx)
	normalizex = normalizex - normalizex.min(axis = 0)
	for rows in range(0, len(normalizex)):
		for column in range (0, len(minx) - 1):
			normalizex[rows][column] = round((normalizex[rows][column]/(maxx[column] - minx[column])), 2)
	return normalizex

def kfold(x_train,y_train,k,kcluster,distantmetric):
	i = 0
	retrievedvotes = []
	totalofallpercentage = []
	totalpercentage = 0
	knnclassobject = knn()
	kf = KFold(n=len(y_train), n_folds=k, shuffle=True, random_state=None)
	for train_index, test_index in kf:
		if i != len(kf):
			#x training and y training datasets
			X_train, X_test = x_train[train_index], x_train[test_index]
			Y_train, Y_test = y_train[train_index], y_train[test_index]
			i += 1
			for ind in range(len(X_test)):
				distancenear = knnclassobject.distancemeasure(X_train, Y_train, X_test[ind], kcluster, distantmetric)
				if distantmetric<2:
					thisvote = knnclassobject.votes(distancenear,True)
				else:
					thisvote = knnclassobject.votes(distancenear,True)

				retrievedvotes.append((thisvote,Y_test[ind]))
			totalofallpercentage.append(knnclassobject.getAccuracy(retrievedvotes,k))
			totalpercentage = sum(totalofallpercentage)
			retrievedvotes = []
	print (totalofallpercentage)
	print (totalpercentage/k)

if __name__ == "__main__":

	filenames={}
	filenames[1] = "ecoli.csv"
	filenames[2] = "glass.csv"
	filenames[3] = "yeast.csv"
	validindices = [1,2,3]
	kfoldnum = 10

	flag = False
	while(flag == False):
		dataset = input("Enter the dataset index for prediction (1.Ecoli 2.Glass 3.Yeast) : ")
		if int(dataset) in validindices:
			flag = True
		else:
			print ("Enter correct index!!!")

	flag1 = False
	while(flag1 == False):
		distancemetric = input("Enter the distant metric index (1. Euclidean 2. Polynomial 3. RBF) : ")
		if int(distancemetric) in validindices:
			flag1 = True
		else:
			print ("Enter correct index!!!")

	data_x, train_y = readcsv(filenames[dataset]) # Read the input file and get X. Y matrices
	#print ("X : ", data_x)
	#print ("Y : ", train_y)
	train_x = normalizedata(data_x) # Normalize the X matrix 
	#print ("Normalized X : ", train_x)
	k_values=[1,2,3,5,10,15]
	for kcluster in k_values:
		print ("K value : ", kcluster)
		kfold(train_x, train_y, kfoldnum, kcluster, distancemetric)
