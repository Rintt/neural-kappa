import copy
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import torch
import pickle
from sklearn.metrics import mean_absolute_error
import torch.nn as nn
import re

from neural_curves import casteljau, casteljau_diff
from neural_curves import FC
from neural_curves import HaltonSampler
from torch.autograd import Variable
from casteljau_approach import MLPRegressorTorch, reader
import joblib


def findall(val):
  return float(re.findall(r"[-+]?(?:\d*\.\d+|\d+)", line)[val])

def add_connections(output):
    count = -4 
    count2 = 0
    s1 = 0
    s2 = 0
    r1 = 0
    r2 = 0
    o1 = 0
    o2 = 0
    new = []
    for s in output:
        #new.cat()
        if count2 == 0:
          o1 = s.item()
        if count2 == 1:
          o2 = s.item()
        if (count2 % 2) == 0:
          s1 = s.item()
        if (count2 % 2) == 1:
          s2 = s.item()
          new.append([s1, s2])
        if (count >= 0) and ((count % 4) == 0):
          r1 = s.item()
        if (count >= 0) and ((count % 4) == 1):
          r2 = s.item()
          new.append([r1,r2])
        count += 1
        count2 += 1
    new.append([o1,o2])
    count = 0                   
    return [torch.Tensor(new[i:i+3]) for i in range(0, len(new), 3)]
    


def plot_curve(ax, control_points):
    ts = torch.linspace(0, 1, 100)
    xys = torch.stack([casteljau(control_points, t)[0] for t in ts])
    #ax.scatter(control_points[:, 0].cpu().numpy(), control_points[:, 1].cpu().numpy(), c='red')
    ax.plot(xys[:, 0].cpu().numpy(), xys[:, 1].cpu().numpy(), c="blue")
def plot_curve_red(ax, control_points):
    ts = torch.linspace(0, 1, 100)
    xys = torch.stack([casteljau(control_points, t)[0] for t in ts])
    #ax.scatter(control_points[:, 0].cpu().numpy(), control_points[:, 1].cpu().numpy(), c='red')
    ax.plot(xys[:, 0].cpu().numpy(), xys[:, 1].cpu().numpy(), c="red")

def compute_closest_point_on_bezier_curve(control_points, x):
    #x = control_points[1]# torch.tensor([2.5, 0.5], dtype=torch.float32)
    t = torch.tensor([0.75], dtype=torch.float32, requires_grad=True)
    optimizer = torch.optim.Adam([t], lr=0.01)
    ts = []
    fs = []
    for i in range(200):
        ts += [float(t.cpu().detach())]
        #p, pp = casteljau_diff(control_points, t)
        # Setting the derivative of the least squares functional to zero yields this:
        #loss = (p - x).dot(pp - x) 
        p = casteljau_diff(control_points, t)[0]
        f = p - x
        loss = torch.linalg.norm(f)
        #loss = f.dot(f)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return casteljau(control_points, t.detach())[0]
if __name__=='__main__':
  training = True
  training_size = 10000
  testing_size = 460
  control_size = 10
  xy, output, output_testing, xy_testing = reader(control_size, training_size, testing_size)
  X = Variable(torch.tensor(xy, dtype=torch.float))
  Y = Variable(torch.tensor(output, dtype=torch.float))
  X_size = X.size()[1]
  Y_size = Y.size()[1]
  X_Testing = Variable(torch.tensor(xy_testing, dtype=torch.float))
  Y_Testing = Variable(torch.tensor(output_testing, dtype=torch.float))
  count, count1, count2 = -1,0,0
  device = torch.device('cuda:0')
  sample=1


  with open('model_pickle10', 'rb') as f:
      snapshot = pickle.load(f)
      
  # print("random sample example")
  # print(xy[sample])
  # print("predicted:")
  # print(snapshot(X_Testing)[sample])
  # print("actual")
  # print(Y_Testing[sample])



  #print("actual")
  #print(type(Y_Testing[sample]))
  ##print(Y_Testing[sample])
  #print(add_connections(Y_Testing[sample]))

  # control_points = torch.tensor([[0.9, 0.2], [-0.7, 0.6], [-0.2, -0.7], [0.3, -0.3], [-0.5, -0.7]], device=device, dtype=torch.float32)
  # control_points = torch.tensor([[[xy[sample][0]][0], [xy[sample][1]][0]], [[xy[sample][2]][0], [xy[sample][3]][0]], [[xy[sample][4]][0], [xy[sample][5]][0]]], device=device, dtype=torch.float32)
  #control_points = torch.tensor([[2.1, 4.5], [1.9, 2.5], [2.3, 1.8], [2.7, 3.7], [2.9, 4.5], [3.3, 6.0], [3.9, 7.0], [2.5, 7.8], [1.2, 7.1], [1.6, 5.8]], dtype=torch.float32)
  fig = plt.figure()
  ax = fig.add_subplot()

  count = 0
  for c in add_connections(Y_Testing[sample]):
      plot_curve(ax, c)
  #print(snapshot(X_Testing)[sample].cpu().detach())
  final_pred = add_connections(snapshot(X_Testing)[sample].cpu().detach())
  print(final_pred)
  #print("here")
  for c in final_pred:
    print(c)
    plot_curve_red(ax, c)
  X = X_Testing[sample]
  ax.scatter([(X[x][0]) for x in range(X_size)], [(X[x][1]) for x in range(X_size)], c="black")

  #pred = snapshot(X_Testing)[sample]
  print("xtesting")
  print((X_Testing)[sample])

  print("snaphot")
  print(snapshot(X_Testing)[sample])
  real = Y_Testing[sample]
  # control_points = torch.tensor([[pred[0], pred[1]], [pred[2], pred[3]], [pred[4], pred[5]], [pred[6], pred[7]], [pred[8], pred[9]],[pred[10], pred[11]]], device=device, dtype=torch.float32)
  # true_points = torch.tensor([[real[0], real[1]], [real[2], real[3]], [real[4], real[5]], [real[4], real[5]], [real[6], real[7]], [real[8], real[9]], [real[8], real[9]], [real[10], real[11]], [real[0], real[1]]], device=device, dtype=torch.float32)
  print("Pred")
  true_points = torch.tensor([[real[0], real[1]], [real[2], real[3]], [real[4], real[5]], [real[4], real[5]], [real[6], real[7]], [real[8], real[9]] , [real[8], real[9]], [real[10], real[11]], [real[12], real[13]] , [real[12], real[13]], [real[14], real[15]], [real[16], real[17]] , [real[16], real[17]], [real[18], real[19]], [real[0], real[1]]], device=device, dtype=torch.float32)
  #true_points = torch.tensor([[real[x], real[x+1]] for x in range(0, len(real), 2)], device=device, dtype=torch.float32)

  # print("real[0], real[1]")
  # print([real[0], real[1]])
  # print("real[4], real[5]")
  # print([real[4], real[5]])
  # fig = plt.figure()
  # points = fig.add_subplot()
  #plot_curve(points, true_points)

  print(X_Testing[sample])
  # points.scatter([X[0][0], X[1][0], X[2][0]], [X[0][1], X[1][1], X[2][1]], c="black")
  #points.scatter([(X[x][0]) for x in range(X_size)], [(X[x][1]) for x in range(X_size)], c="black")

  # points.scatter([-1., 0.1, -0.8], [0.8, -0.1, -0.3], c="black")

  plt.show()

