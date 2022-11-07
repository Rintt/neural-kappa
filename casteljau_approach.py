#! /usr/bin/python3
import copy
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import torch
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import PolynomialFeatures
import re
import torch
from sklearn import svm
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.neural_network import MLPRegressor
from neural_curves import casteljau, casteljau_diff
from neural_curves import FC
from neural_curves import HaltonSampler
import pickle
#Open Training 
f = open('training.txt', 'r')
training = True
training_size = 59
testing_size = 41
control_size = 3
xy = np.ones((training_size, control_size,2))
output = np.ones((training_size, control_size * 4))
xy_testing = np.ones((testing_size, control_size,2))
output_testing = np.ones((testing_size, control_size * 4))

count, count1, count2 = -1,0,0
def findall(val):
  return float(re.findall(r"[-+]?(?:\d*\.\d+|\d+)", line)[val])

#read training line by line into input and output arrays
for line in f:
    if(line[0] == "S"):
            count = count + 1
            count1 = 0
            count2 = 0
            count3 = 0
    elif(training_size == count):
      count = -1
      training = False
      break
    elif(training and line != "/n" and line[0] == "x"):
        xy[count][count1][0] = float(re.findall(r"[-+]?(?:\d*\.\d+|\d+)", line)[0])
        xy[count][count1][1] = float(re.findall(r"[-+]?(?:\d*\.\d+|\d+)", line)[1])
        count1 = count1 + 1
    elif(training and line != "" and line != "\n" and line[0] != "S"):
        if(count2 % (control_size*2) < 4):
          print(count2 % (control_size*2))
          print(count2)
          output[count][count3] = float(re.findall(r"[-+]?(?:\d*\.\d+|\d+)", line)[0])
          count3 = count3 + 1
        count2 = count2 + 1
f.close()
d = open('testing.txt', 'r')
for line in d:
  if(line[0] == "S"):
            count = count + 1
            count1 = 0
            count2 = 0
            count3 = 0            
  elif(not training and line != "/n" and line[0] == "x"):
        xy_testing[count][count1][0] = float(re.findall(r"[-+]?(?:\d*\.\d+|\d+)", line)[0])
        xy_testing[count][count1][1] = float(re.findall(r"[-+]?(?:\d*\.\d+|\d+)", line)[1])
        count1 = count1 + 1
  elif(not training and line != "" and line != "\n" and line[0] != "S"):
        if(count2 % (control_size*2) < 4):
          output_testing[count][count3] = float(re.findall(r"[-+]?(?:\d*\.\d+|\d+)", line)[0])
          count3 = count3 + 1
        count2 = count2 + 1

xy.resize(training_size, control_size*2)
xy_testing.resize(testing_size, control_size*2)
poly = PolynomialFeatures(2)
# xy = poly.fit_transform(xy)
print("xy:")
print(xy)
print("output:")
print(output)
print("xy_testing:")
print(xy_testing)
print("output_testing:")
print(output_testing)
# regr = svm.SVR()
regr_3 = MLPRegressor(random_state=1, max_iter=1000000,activation='identity', alpha=0.0001, learning_rate='invscaling', learning_rate_init=0.0005, batch_size=10, solver='sgd', hidden_layer_sizes= (120,), verbose=True, tol=1e-11, n_iter_no_change=100)# regr_3 = SGDRegressor(verbose=True)

regr_3.fit(xy, output)
filename = 'finalized_model.sav'
pickle.dump(regr_3, open(filename, 'wb'))
print("Predict:")
sum = 0
for test in range(xy_testing.shape[0]):
  pred = regr_3.predict([xy_testing[test]])
  sum = sum + mean_absolute_error([output_testing[test]], pred)

print("sum:")
#0.07196238730632361 best value
#regr_3 = MLPRegressor(random_state=1, max_iter=1000000,activation='identity', alpha=0.0001, learning_rate='constant', solver='adam', hidden_layer_sizes= (100), verbose=True, tol=1e-11, n_iter_no_change=100)
print(sum/xy_testing.shape[0])

p = regr_3.predict([xy_testing[0]])
print(p)
print(output_testing[0])
# y_pred =  regr_3.predict([[0., 0., 0.5, 0.2, -0.8, -0.7, 0.1, -0.1, -0.3, 0.9]])
# y_pred1 = regr_3.predict([[-0.5, 0.8, 0.6, 0.7, 0., -1., 0.6, 0.2, -1., 0.8]])
# y_pred2 = regr_3.predict([[0.4, 0.4, -1., -0.6, -0.7, -0.5, -0.8, 0, 0.9, -0.7]])
# y_pred3 = regr_3.predict([[-0.2, 0.2, 0.8, -0.5, 0.7, 0.4, -1, 0.1, -0.2, -0.4]])
# y_pred4 = regr_3.predict([[0.6, -0.8, 0.1, 0.6, -0.2, -0.4, -0.3, -1., 0.1, 0.4]])
# y_pred5 = regr_3.predict([[0.4, 0.5, -0.4, -0.6, 0.6, -0.3, -1., -0.4, 0.9, 0.]])

# y_pred5.resize(1,30)
# print(y_pred5)
# y_true=[[-0.312871, 0.607454, -0.16799, -0.339212, 0.455625, 0.15854,  
#           0.455625, 0.15854,   0.6336,   0.300594, 0.100044,-0.0387515,
#           0.100044,-0.0387515,-1.23193, -0.885896,-0.595625,-0.661066, 
#          -0.595625,-0.661066,  0.540018,-0.2598,  -0.108063, 0.710518, 
#          -0.108063,  0.710518,-0.394213,  1.13895,-0.312871, 0.607454]]
# y_true1=[[-0.987639, 0.798992, -0.603813, 0.770605,-0.227201, 0.833524,
#           -0.227201, 0.833524,  1.36148,  1.09894,  0.140957,-0.751153,
#            0.140957,-0.751153, -0.141045,-1.17862,  0.135635,-0.873607,
#            0.135635,-0.873607,  1.51554,  0.64758, -0.977923, 0.798787,
#           -0.977923, 0.798787, -1.01652,  0.801127,-0.987639, 0.798992
# ]]
# y_true2=[[0.883955,
# -0.206073,
# 0.450919,
#  1.04151,
# -0.731962,
# -0.196357,
# -0.731962,
# -0.196357,
#  -1.19411,
# -0.679983,
# -0.879299,
# -0.610147,
# -0.879299,
# -0.610147,
# -0.554321,
# -0.538055,
# -0.755986,
# -0.176663,
# -0.755986,
# -0.176663,
# -0.996866,
#  0.255002,
# -0.169345,
# -0.294216,
# -0.169345,
# -0.294216,
#  1.23898,
# -1.22891,
#  0.883955,
# -0.206073
# ]]
# y_true3=[[
#   -0.19037,
# -0.0360034,
# -0.335887,
#   0.47225,
#   0.179126,
# -0.0901725,
#   0.179126,
# -0.0901725,
#   0.91851,
# -0.897619,
# 0.901781,
# -0.15932,
# 0.901781,
# -0.15932,
# 0.879473,
# 0.825254,
# -0.394984,
#  0.474897,
# -0.394984,
#  0.474897,
# -1.46894,
# 0.179659,
# -0.638817,
# -0.295642,
# -0.638817,
# -0.295642,
# -0.0135285,
#  -0.653661,
#   -0.19037,
# -0.0360034,
# ]]
# y_true4=[[
#  0.579605,
# -0.718509,
#  0.620425,
# -0.891164,
#  0.579796,
# -0.698018,
#  0.579796,
# -0.698018,
# 0.0727692,
#   1.71232,
# -0.193809,
# -0.352865,
# -0.193809,
# -0.352865,
# -0.227415,
# -0.613211,
# -0.273825,
# -0.872014,
# -0.273825,
# -0.872014,
# -0.322174,
#  -1.14162,
#  -0.28287,
# -0.843293,
#  -0.28287,
# -0.843293,
# 0.036313,
#  1.57943,
#  0.579605,
# -0.718509,
# ]]
# y_true5=[[
#  0.762336, 0.307695,  0.266974, 0.828612,-0.257765, -0.0609567,
# -0.257765,-0.0609567,-0.780032,-0.946335, 0.275961, -0.504046,
#  0.275961,-0.504046,  1.21042, -0.11266, -0.545264, -0.325985,
# -0.545264,-0.325985, -1.3782,  -0.427191,-0.685476, -0.405557, 
# -0.685476,-0.405557,  1.37928, -0.341076, 0.762336,  0.307695
# ]]

# print(mean_absolute_error(y_true, y_pred))
# print(mean_absolute_error(y_true1, y_pred1))
# print(mean_absolute_error(y_true2, y_pred2))
# print(mean_absolute_error(y_true3, y_pred3))
# print(mean_absolute_error(y_true4, y_pred4))
# print(mean_absolute_error(y_true5, y_pred5))
# sum = mean_absolute_error(y_true, y_pred) + mean_absolute_error(y_true1, y_pred1) + mean_absolute_error(y_true2, y_pred2) + mean_absolute_error(y_true3, y_pred3) +mean_absolute_error(y_true4, y_pred4) + mean_absolute_error(y_true5, y_pred5)
# print("average mean absolute error: ")
# print(sum/6)