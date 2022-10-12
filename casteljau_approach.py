#! /usr/bin/python3
import copy
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import torch
import re

from neural_curves import casteljau, casteljau_diff
from neural_curves import FC
from neural_curves import HaltonSampler

# device = torch.device('cuda:0')
f = open('training.txt', 'r')
training_size = 10
xy = np.ones((training_size,5,2))
output = np.ones((training_size,30))
count, count1, count2 = -1,0,0
for line in f:
    if(line[0] == "S"):
            count = count + 1
            count1 = 0
            count2 = 0
    elif(line != "/n" and line[0] == "x"):
        xy[count][count1][0] = float(re.findall(r"[-+]?(?:\d*\.\d+|\d+)", line)[0])
        xy[count][count1][1] = float(re.findall(r"[-+]?(?:\d*\.\d+|\d+)", line)[1])
        count1 = count1 + 1
    elif(line != "" and line != "\n" and line[0] != "S"):
        output[count][count2] = float(re.findall(r"[-+]?(?:\d*\.\d+|\d+)", line)[0])
        count2 = count2 + 1     

print("xy:")
print(xy[2])
print("output:")
print(output[2])