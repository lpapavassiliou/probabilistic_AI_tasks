import numpy as np
import math
from scipy.stats import norm
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d

def cost(f, f_star):
    weight = 0

    if f <= f_star:
        weight = 25
    elif f >= 1.2*f_star:
        weight = 10   
    else:
        weight = 1 

    return weight * abs(f-f_star)**2

mu = 0
N = 10000

sigmaSet = np.arange(0.0, 10.0, 0.05)

results_x = []
results_y = []

for sigma in sigmaSet:
    print(sigma)
    deltaF = np.arange(0.0, 1.0, 0.001)* float(sigma)

    points_to_plot_x = []
    points_to_plot_y = []

    for delta in deltaF:
        f_star = norm.rvs(loc=mu, scale=math.sqrt(sigma), size=N)

        f = [mu+delta]*N

        cost_vector = [cost(f[i], f_star[i]) for i in range(0,N)]

        cost_sum = np.sum(np.array(cost_vector))

        points_to_plot_x.append(delta)
        points_to_plot_y.append(cost_sum)


    points_to_plot_y = uniform_filter1d(points_to_plot_y, size=50)
    min_y = np.min(points_to_plot_y)

    points_to_plot_x = np.array(points_to_plot_x)

    min_delta = points_to_plot_x[np.where(points_to_plot_y == min_y)]

    results_x.append(sigma)
    results_y.append(min_delta[0])

points = np.vstack((results_x, results_y)).transpose()
np.savetxt("fit.csv", points, delimiter=",")
plt.plot(results_x, results_y)
plt.show()

