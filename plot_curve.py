import matplotlib.pyplot as plt
import torch
import pickle
from neural_curves import casteljau, casteljau_diff
from torch.autograd import Variable
from casteljau_approach import MLPRegressorTorch, reader


def add_connections(output):
    count = -4 
    count2 = 0
    s1, s2, r1, r2, o1, o2, new = 0, 0, 0, 0, 0, 0, []
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

def plot_curve(ax, control_points, color):
    ts = torch.linspace(0, 1, 100)
    xys = torch.stack([casteljau(control_points, t)[0] for t in ts])
    ax.plot(xys[:, 0].cpu().numpy(), xys[:, 1].cpu().numpy(), c=color)
    
if __name__=='__main__':
  training = True
  training_size = 10000
  testing_size = 460
  control_size = 3
  xy, output, output_testing, xy_testing = reader(control_size, training_size, testing_size)
  X = Variable(torch.tensor(xy, dtype=torch.float))
  Y = Variable(torch.tensor(output, dtype=torch.float))
  X_size = X.size()[1]
  Y_size = Y.size()[1]
  X_Testing = Variable(torch.tensor(xy_testing, dtype=torch.float))
  Y_Testing = Variable(torch.tensor(output_testing, dtype=torch.float))
  count, count1, count2 = -1,0,0
  device = torch.device('cuda:0')
  sample=120


  with open('model_pickle', 'rb') as f:
      snapshot = pickle.load(f)

  
  for i in range(sample, sample+10):
    fig = plt.figure()
    ax = fig.add_subplot()
    for c in add_connections(Y_Testing[i]):
      plot_curve(ax, c, "red")
    final_pred = add_connections(snapshot(X_Testing)[i].cpu().detach())
    print(final_pred)
    for c in final_pred:
      print(c)
      plot_curve(ax, c, "blue")
    X = X_Testing[i]
    ax.scatter([(X[x][0]) for x in range(X_size)], [(X[x][1]) for x in range(X_size)], c="black")

  plt.show()

