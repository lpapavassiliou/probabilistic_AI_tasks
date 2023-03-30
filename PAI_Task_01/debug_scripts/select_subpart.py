from ast import Gt
import numpy as np
import matplotlib.pyplot as plt 

def save_xyz(train_features, train_GT, name, norm=False):
    x1 = train_features[:, 0]
    if norm:
        min_x1 = np.min(x1)
        max_x1 = np.max(x1)
        x1 = (x1-min_x1)/(max_x1-min_x1)

    x2 = train_features[:, 1]
    if norm:
        min_x2 = np.min(x2)
        max_x2 = np.max(x2)
        x2 = (x2-min_x2)/(max_x2-min_x2)

    y = np.array(train_GT)
    if norm:
        min_y = np.min(y)
        max_y = np.max(y)
        y = (y-min_y)/(max_y-min_y)

    points = np.vstack((x1, x2, y)).transpose()
    np.savetxt(name + ".xyz", points, delimiter=",")
    print("Saved file "+ name +".xyz")

train_features = np.loadtxt('train_x.csv', delimiter=',', skiprows=1)
train_GT = np.loadtxt('train_y.csv', delimiter=',', skiprows=1)

feature_to_save = []
GT_to_save = []

for i in range(0,train_features.shape[0]):
    if train_features[i][0] < 0.15 and train_features[i][1] < 0.15:
        feature_to_save.append(train_features[i])
        GT_to_save.append(train_GT[i])

feature_to_save = np.array(feature_to_save)
GT_to_save = np.array(GT_to_save)
save_xyz(feature_to_save, GT_to_save, "partial")

plt.plot(GT_to_save)
plt.show()
print(np.min(GT_to_save), np.max(GT_to_save))