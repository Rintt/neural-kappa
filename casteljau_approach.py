#! /usr/bin/python3
import numpy as np
import torch
from sklearn.metrics import mean_absolute_error
import re
import torch
import pickle
import torch.nn as nn
import torch.utils.data as utils_data
from torch.autograd import Variable
import numpy as np
import joblib
import math 
import time 
import sys

#class for model architecture
class MLPRegressorTorch(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLPRegressorTorch, self).__init__()
        self.dropout = nn.Dropout(p=0.8),
        self.flatten = nn.Flatten()
        self.linear_relu_stack1 = nn.Sequential(
          nn.Linear(input_size, int(hidden_size*4/3)),
          nn.Identity(),
          # nn.Dropout(p=0.5),
          nn.Linear(int(hidden_size*4/3), int(hidden_size*2)),
          nn.SELU(),
          nn.Linear(int(hidden_size*2), hidden_size*2),
          nn.SELU(),
          nn.Linear(hidden_size*2, int(hidden_size*4/3)),
          nn.SELU(),
          #nn.Dropout(p=0.2),
          # nn.Linear(hidden_size, hidden_size),
          # nn.Hardswish(),
          # nn.Linear(hidden_size, hidden_size),
          # nn.Identity(),
          nn.Linear(int(hidden_size*4/3), output_size),
        )
          
    def forward(self, x):
        x = self.flatten(x)
        #x = self.dropout(x)
        f = self.linear_relu_stack1(x)
        return f
class CNNRegressorTorch(nn.Module):
    def __init__(self, input_channels, hidden_size, output_size):
        super(CNNRegressorTorch, self).__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv1d(input_channels, hidden_size, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
        )

        self.linear_layers = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)  # Swap axis to convert from (batch_size, input_size) to (batch_size, input_channels, input_size)
        f = self.conv_layers(x)
        f = torch.flatten(f, 1)
        f = self.linear_layers(f)
        return f
    
#helper function for reader
def findall(val, line):
  return float(re.findall(r"[-+]?(?:\d*\.\d+|\d+)", line)[val])

#read training line by line into input and output arrays
def reader(control_size, training_size, testing_size):
  f = open('training.txt', 'r')
  training = True
  count, count1, count2 = -1,0,0
  xy = np.ones((training_size, control_size, 2))
  output = np.ones((training_size , control_size * 4))
  xy_testing = np.ones((testing_size, control_size, 2))
  output_testing = np.ones((testing_size, control_size * 4))
  for line in f:
      if(line[0] == "S" or line[0] == "s"):
              count = count + 1
              count1 = 0
              count2 = 0
              count3 = 0

      elif(training_size == count + 1):
        print("training over time for testing")
        count = -1
        training = False
        break

      elif(training and line != "/n" and line[0] == "x"):
        xy[count][count1][0] = float(re.findall(r"[-+]?(?:\d*\.\d+|\d+)", line)[0])
        xy[count][count1][1] = float(re.findall(r"[-+]?(?:\d*\.\d+|\d+)", line)[1])
        count1 = count1 + 1

      elif(training and line != "" and line != "\n" and line[0] != "S"):
        if(count2 % (3*2) < 4):
          output[count][count3] = float(re.findall(r"[-+]?(?:\d*\.\d+|\d+)", line)[0])
          count3 = count3 + 1
        count2 = count2 + 1
  count = -1
  training = False
  f.close()
  d = open('testing3.txt', 'r')

  for line in d:
    if(line[0] == "S" or line[0] == "s"):
        count = count + 1
        count1 = 0
        count2 = 0
        count3 = 0  

    elif(not training and line != "/n" and line[0] == "x"):
        xy_testing[count][count1][0] = float(re.findall(r"[-+]?(?:\d*\.\d+|\d+)", line)[0])
        xy_testing[count][count1][1] = float(re.findall(r"[-+]?(?:\d*\.\d+|\d+)", line)[1])
        count1 = count1 + 1

    elif(not training and line != "" and line != "\n" and line[0] != "S"):
        if(count2 % (3*2) < 4):
          output_testing[count][count3] = float(re.findall(r"[-+]?(?:\d*\.\d+|\d+)", line)[0])
          count3 = count3 + 1
        count2 = count2 + 1

  return xy, output, output_testing, xy_testing

def main():
  start = time.time() #timer to see how long  things are taking
  training_size = 25000
  testing_size = 460
  control_size = 3
  device = torch.device('cuda')

  xy, output, output_testing, xy_testing = reader(control_size, training_size, testing_size)
  
  xy.resize(training_size, control_size*2)
  xy_testing.resize(testing_size, control_size*2)

  X = Variable(torch.tensor(xy, dtype=torch.float, device=device))
  Y = Variable(torch.tensor(output, dtype=torch.float, device=device))

  X_size = X.size()[1]
  Y_size = Y.size()[1]
  X_Testing = Variable(torch.tensor(xy_testing, dtype=torch.float, device=device))
  Y_Testing = Variable(torch.tensor(output_testing, dtype=torch.float, device=device))


  hidden_size = 96 #int(((X_size) * (Y_size)) * 2/3)
  print(hidden_size)
  max_iter=5000 #training_size * control_size /150
  learning_rate_init=0.0001

  # net = CNNRegressorTorch(2, 64, control_size * 4)
  # net = net.to(device)

  snapshot = MLPRegressorTorch(input_size = X_size, hidden_size = hidden_size, output_size=Y_size)
  snapshot = snapshot.to(device)
  print(snapshot)
  # l1_lambda = torch.norm(snapshot.parameters(), p=1)
  optimizer = torch.optim.Adam(snapshot.parameters(), lr = learning_rate_init, # weight_decay=0.0001;before
                               #*l1_lambda
                               )
  # optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate_init, weight_decay=0.0001)
  loss_fct = nn.MSELoss()

  training_samples = utils_data.TensorDataset(X,Y)
  data_loader = utils_data.DataLoader(training_samples, shuffle=True)
  loss_prev = float('inf')
  fool_me_once = 0
  l2_alpha = 0.0005
  big_lossmark = sys.maxsize

 
  for iteration in range(max_iter):
          loss_total = 0
          for batch, (data, target) in enumerate(data_loader):
              training_x, training_y = data.float(), target.float()
              loss = loss_fct(snapshot(training_x), training_y.unsqueeze(1))
              # l2_reg = torch.tensor(0., device=device)
              # for param in snapshot.to(device).parameters():
              #   l2_reg += torch.norm(param, 2)
              # loss += l2_alpha * l2_reg  
              optimizer.zero_grad()
              loss.backward()
              optimizer.step()
              loss_total += loss.item()
              
          if iteration % 5 == 0:
              
          #     if(loss_prev - loss_total < 0.1):
          #       if(fool_me_once == 1 ):
          #          break
          #       print("fool me once")
          #       fool_me_once = 1
          #     elif(fool_me_once == 1):
          #        fool_me_once = 0
              loss_prev = loss_total
              print ('Iteration %d/%d, Loss: %.4f' %(iteration, max_iter, loss_total))
          if iteration % 25 == 0:
            print('big lossmark ' + str(big_lossmark) + ' previous trunc = ' + str(math.trunc(big_lossmark)))
            print('loss total ' + str(loss_total) + ' current trunc = ' + str(math.trunc(loss_total)))
            if(math.trunc(big_lossmark) <= math.trunc(loss_total)):
              print("big loss")
              print(big_lossmark)
              break
            big_lossmark = loss_total


  with open('model_pickle3.2', 'wb') as f:
    pickle.dump(snapshot, f)


  #snapshot = snapshot.to('cpu')

  final_prediction = snapshot(X_Testing)
  final_pred_np = final_prediction.clone().detach().cpu().numpy()

  #joblib.dump(snapshot, 'model_joblib10.pkl')


  print("Mean Absolute Error:")
  print(mean_absolute_error(Y_Testing.cpu().numpy(), final_pred_np))
  sample=1
  print("random sample example")
  print(xy_testing[sample])
  print("predicted:")
  print(snapshot(X_Testing)[sample])
  print("actual")
  print(Y_Testing[sample])
  end = time.time()
  print('total time of operation:')
  print(str((end - start)) + ' seconds')

if __name__=='__main__':
    main()