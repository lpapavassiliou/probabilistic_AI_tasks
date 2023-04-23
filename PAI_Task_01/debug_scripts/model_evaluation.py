from random import random
from re import T
import numpy as np
import math
import matplotlib.pyplot as plt
from random import randrange

def gaussian_sampler(x, mean, covariance):
    K = 1/(2*math.pi*math.sqrt(np.linalg.det(covariance)))    
    return  K* math.exp(-0.5 * (np.transpose(x-mean) @np.linalg.inv(covariance) @ (x-mean)))

n_points = 100

train_data  = [[i/float(n_points),j/float(n_points)] for i in range(0,n_points) for j in range(0,n_points) ]

x1 = []
x2 = []

for i in range(0,n_points):
    for j in range(0,n_points):
        x1.append(i/float(n_points))
        x2.append(j/float(n_points))

x1 = np.array(x1)
x2 = np.array(x2)

mean = np.transpose([0, 0])
covariance = np.eye(2,2)

train_y = [gaussian_sampler(np.transpose([x1[i], x2[i]]), mean, covariance)+(randrange(10)/500.0) for i in range(0,n_points*n_points)]


ax = plt.axes(projection='3d')
ax.scatter3D(x1, x2, train_y, cmap='Greens');
plt.show()

points = np.vstack((x1, x2)).transpose()
np.savetxt("gauss_features.csv", points, delimiter=",")
np.savetxt("gauss_GT.csv", np.array(train_y).transpose())
"""
model.fitting_model(train_y, train_data)
predictions, gp_mean, gp_std = model.make_predictions(np.array([[0.32,0.32],[0.41,0.41]]))
print(predictions, gp_mean, gp_std)
print(gaussian_sampler(np.transpose([0.32, 0.32]), mean, covariance), gaussian_sampler(np.transpose([0.41, 0.41]), mean, covariance))
#print(cost_function(gaussian_sampler(np.transpose(train_data), mean, covariance), predictions))
"""
"""
train_data = np.array(train_data)
train_y = np.array(train_y)

model.fitting_model(train_y, train_data)
predictions, gp_mean, gp_std = model.make_predictions(np.array([[1.5,1.5],[2,2]]))
print(predictions, gp_mean, gp_std)

ground_truth = np.array([53550, 53550])

print(cost_function(ground_truth, predictions))
"""