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
import torch.nn as nn
import torch.nn.functional as Func
import torch.utils.data as utils_data
from torch.autograd import Variable

#Open Training 
f = open('training.txt', 'r')
training = True
training_size = 10000
testing_size = 460
control_size = 3
xy = np.ones((training_size, control_size, 2))
output = np.ones((training_size, control_size * 4))
xy_testing = np.ones((testing_size, control_size, 2))
output_testing = np.ones((testing_size, control_size * 4))

count, count1, count2 = -1,0,0
def findall(val):
  return float(re.findall(r"[-+]?(?:\d*\.\d+|\d+)", line)[val])

#read training line by line into input and output arrays
for line in f:
    if(line[0] == "S" or line[0] == "s"):
            count = count + 1
            count1 = 0
            count2 = 0
            count3 = 0
            # print(count)
    elif(training_size == count + 1):
      print("training over time for testing")
      count = -1
      training = False
      break
    elif(training and line != "/n" and line[0] == "x"):
        # print("xy:")
        # print(line)
        xy[count][count1][0] = float(re.findall(r"[-+]?(?:\d*\.\d+|\d+)", line)[0])
        xy[count][count1][1] = float(re.findall(r"[-+]?(?:\d*\.\d+|\d+)", line)[1])
        count1 = count1 + 1
    elif(training and line != "" and line != "\n" and line[0] != "S"):
        # print("output:")
        # print(line)
        if(count2 % (control_size*2) < 4):
          output[count][count3] = float(re.findall(r"[-+]?(?:\d*\.\d+|\d+)", line)[0])
          count3 = count3 + 1
        count2 = count2 + 1
f.close()
d = open('testing.txt', 'r')
for line in d:
  if(line[0] == "S" or line[0] == "s"):
            count = count + 1
            count1 = 0
            count2 = 0
            count3 = 0  
            # print("training count:")          
            # print(count)
            # print(training)
  elif(not training and line != "/n" and line[0] == "x"):
        # print("testing_xy:")
        # print(line)
        xy_testing[count][count1][0] = float(re.findall(r"[-+]?(?:\d*\.\d+|\d+)", line)[0])
        xy_testing[count][count1][1] = float(re.findall(r"[-+]?(?:\d*\.\d+|\d+)", line)[1])
        count1 = count1 + 1
  elif(not training and line != "" and line != "\n" and line[0] != "S"):
        # print("testing_output:")
        # print(line)
        if(count2 % (control_size*2) < 4):
          output_testing[count][count3] = float(re.findall(r"[-+]?(?:\d*\.\d+|\d+)", line)[0])
          count3 = count3 + 1
        count2 = count2 + 1

xy.resize(training_size, control_size*2)
xy_testing.resize(testing_size, control_size*2)
poly = PolynomialFeatures(2)
# xy = poly.fit_transform(xy)
# print("xy:")
# print(xy)
# print("output:")
# print(output)
# print("xy_testing:")
# print(xy_testing)
# print("output_testing:")
# print(output_testing)
# # regr = svm.SVR()
# regr_3 = MLPRegressor(random_state=1, max_iter=1000000,activation='identity', alpha=0.0001, learning_rate='constant', learning_rate_init=0.0001, batch_size=10, solver='sgd', hidden_layer_sizes= (120,), verbose=True, tol=1e-11, n_iter_no_change=100)# regr_3 = SGDRegressor(verbose=True)

# regr_3.fit(xy, output)
# filename = 'finalized_model.sav'
# pickle.dump(regr_3, open(filename, 'wb'))
# print("Predict:")
# sum = 0
# for test in range(xy_testing.shape[0]):
#   pred = regr_3.predict([xy_testing[test]])
#   sum = sum + mean_absolute_error([output_testing[test]], pred)

# print("sum:")
# #0.07196238730632361 best value
# #regr_3 = MLPRegressor(random_state=1, max_iter=1000000,activation='identity', alpha=0.0001, learning_rate='constant', solver='adam', hidden_layer_sizes= (100), verbose=True, tol=1e-11, n_iter_no_change=100)
# print(sum/xy_testing.shape[0])

# p = regr_3.predict([xy_testing[0]])
# print(p)
# print(output_testing[0])



########################################################################################################################

# class MLPRegressorTorch(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size):
#         super(MLPRegressorTorch, self).__init__()
        
#         self.flatten = nn.Flatten()
#         self.linear_relu_stack = nn.Sequential(
#             nn.Linear(input_size, output_size),
          
#             # nn.Linear(512, 512),
#             # nn.ReLU(),
#             # nn.Linear(512, 10),
#         )
#     def forward(self, x):
#         x = self.flatten(x)
#         logits = self.linear_relu_stack(x)
#         return logits

class MLPRegressorTorch(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLPRegressorTorch, self).__init__()
        
        self.flatten = nn.Flatten()
        self.linear_relu_stack1 = nn.Sequential(
          nn.Linear(input_size, hidden_size),
          nn.Identity(),
          nn.Linear(hidden_size, hidden_size),
          nn.SELU(),
          nn.Linear(hidden_size, hidden_size),
          nn.Identity(),
          nn.Linear(hidden_size, output_size),
        )
          
    def forward(self, x):
        x = self.flatten(x)
        f = self.linear_relu_stack1(x)
        return f
    #     self.fc1 = nn.Linear(input_size, hidden_size)
    #     self.fc2 = nn.Linear(hidden_size, hidden_size)
    #     self.fc3 = nn.Linear(hidden_size, output_size)
    #     self.relu = nn.ReLU()

    # def forward(self, x):
    #     x = self.fc1(x)
    #     x = self.relu(x)
    #     x = self.fc2(x)
    #     x = self.relu(x)
    #     x = self.fc3(x)
    #     return x


X = Variable(torch.tensor(xy, dtype=torch.float))
Y = Variable(torch.tensor(output, dtype=torch.float))

X_size = X.size()[1]
Y_size = Y.size()[1]
X_Testing = Variable(torch.tensor(xy_testing, dtype=torch.float))
Y_Testing = Variable(torch.tensor(output_testing, dtype=torch.float))


hidden_size = 128;#int((X_size)*2/3 + Y_size)
max_iter=300
learning_rate_init=0.0001

snapshot = MLPRegressorTorch(input_size = X_size, hidden_size = hidden_size, output_size=Y_size)

print(snapshot)
optimizer = torch.optim.Adam(snapshot.parameters(), lr = learning_rate_init, weight_decay=0.0001)
loss_fct = nn.L1Loss()

training_samples = utils_data.TensorDataset(X,Y)
data_loader = utils_data.DataLoader(training_samples, shuffle=True)

for iteration in range(max_iter):
        loss_total = 0
        for batch, (data, target) in enumerate(data_loader):
            training_x, training_y = data.float(), target.float()
            loss = loss_fct(snapshot(training_x), training_y.unsqueeze(1)) 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_total += loss.item()
            
        if iteration % 10 == 0:
            print ('Iteration %d/%d, Loss: %.4f' %(iteration, max_iter, loss_total))


# final_prediction = snapshot(X)
# final_pred_np = final_prediction.clone().detach().numpy()

# print(np.corrcoef(final_pred_np.squeeze(), output)[0,1])
# for test in range(xy_testing.shape[0]):
#   # pred = regr_3.predict([xy_testing[test]])
#   final_prediction = snapshot([xy_testing[test]])
#   final_pred_np = final_prediction.clone().detach().numpy()
#   sum = sum + mean_absolute_error([output_testing[test]], final_pred_np)
# data = [[1, 2], [3, 4]]
final_prediction = snapshot(X_Testing)
final_pred_np = final_prediction.clone().detach().numpy()

# print("training input:")
# print(X)
# print("training output:")
# print(Y)
# print("mean absolute error:")
print("Mean Absolute Error:")
print(mean_absolute_error(Y_Testing, final_pred_np))
sample=1
print("random sample example")
print(xy[sample])
print("predicted:")
print(snapshot(X_Testing)[sample])
print("actual")
print(Y_Testing[sample])
