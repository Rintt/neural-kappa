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
import copy
import matplotlib.pyplot as plt
import numpy as np
from neural_curves import casteljau 
import joblib

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
          nn.SELU(),
          nn.Linear(hidden_size, hidden_size),
          nn.SELU(),
          nn.Linear(hidden_size, hidden_size),
          nn.SELU(),
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
    
def findall(val, line):
  return float(re.findall(r"[-+]?(?:\d*\.\d+|\d+)", line)[val])
def reader(control_size, training_size, testing_size):
  f = open('training.txt', 'r')
  training = True
  training_size = 10000
  count, count1, count2 = -1,0,0
  xy = np.ones((training_size, control_size, 2))
  output = np.ones((training_size , control_size * 4))
  xy_testing = np.ones((testing_size, control_size, 2))
  output_testing = np.ones((testing_size, control_size * 4))

  f = open('training.txt', 'r')
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
          if(count2 % (3*2) < 4):
            # print(float(re.findall(r"[-+]?(?:\d*\.\d+|\d+)", line)[0]))
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
          count2 = 0

    elif(not training and line != "" and line != "\n" and line[0] != "S"):
          # print("testing_output:")
          # print(line)
          if(count2 % (3*2) < 4):
            output_testing[count][count3] = float(re.findall(r"[-+]?(?:\d*\.\d+|\d+)", line)[0])
            count3 = count3 + 1
          count2 = count2 + 1
  return xy, output, output_testing, xy_testing
def main():
  training_size = 10000
  testing_size = 460
  control_size = 10
 
  device = torch.device('cuda:0')
  xy, output, output_testing, xy_testing = reader(f, xy, output, training_size, xy_testing, output_testing)
  
#read training line by line into input and output arrays
  reader(f, xy, output, training_size, xy_testing, output_testing)
  xy.resize(training_size, control_size*2)
  xy_testing.resize(testing_size, control_size*2)

  X = Variable(torch.tensor(xy, dtype=torch.float))
  Y = Variable(torch.tensor(output, dtype=torch.float))

  X_size = X.size()[1]
  Y_size = Y.size()[1]
  X_Testing = Variable(torch.tensor(xy_testing, dtype=torch.float))
  Y_Testing = Variable(torch.tensor(output_testing, dtype=torch.float))


  hidden_size = 500#int(((X_size) + Y_size)*2 + 50)
  print("Hidden Size:")
  print(hidden_size)
  max_iter=10
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

  with open('model_pickle10', 'wb') as f:
      pickle.dump(snapshot, f)



  # final_prediction = snapshot(X)
  # final_pred_np = final_prediction.clone().detach().numpy()

  # print(np.corrcoef(final_pred_np.squeeze(), output)[0,1])
  # for test in range(xy_testing.shape[0]):
  #   # pred = regr_3.predict([xy_testing[test]])
  #   final_prediction = snapshot([xy_testing[test]])
  #   final_pred_np = final_prediction.clone().detach().numpy(torch)
  #   sum = sum + mean_absolute_error([output_testing[test]], final_pred_np)
  # data = [[1, 2], [3, 4]]
  final_prediction = snapshot(X_Testing)
  final_pred_np = final_prediction.clone().detach().numpy()

  joblib.dump(snapshot, 'model_joblib10.pkl')

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

if __name__=='__main__':
    main()