import numpy as np

x1 = []
x2 = []
y = []
for i in np.arange(1.0,4.1,0.1):
    for j in np.arange(1.0,4.1,0.1):
        x1.append(i)
        x2.append(j)
        y.append(0)

points = np.vstack((x1, x2)).transpose()

np.savetxt("4x4_1_features.csv", points)
np.savetxt("4x4_1_GT.csv", np.array(y).transpose())