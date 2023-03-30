from math import inf
from math import floor
import os
import typing
from sklearn.gaussian_process.kernels import *
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.path import Path
import time
from random import randrange

# Set `EXTENDED_EVALUATION` to `True` in order to visualize your predictions.
EXTENDED_EVALUATION = False
EVALUATION_GRID_POINTS = 300  # Number of grid points used in extended evaluation
EVALUATION_GRID_POINTS_3D = 50  # Number of points displayed in 3D during evaluation


# Cost function constants
COST_W_UNDERPREDICT = 25.0
COST_W_NORMAL = 1.0
COST_W_OVERPREDICT = 10.0

def save_xyz(train_features, train_GT, name):
    x1 = train_features[:, 0]
    min_x1 = np.min(x1)
    max_x1 = np.max(x1)
    x1 = (x1-min_x1)/(max_x1-min_x1)

    x2 = train_features[:, 1]
    min_x2 = np.min(x2)
    max_x2 = np.max(x2)
    x2 = (x2-min_x2)/(max_x2-min_x2)

    y = np.array(train_GT)
    min_y = np.min(y)
    max_y = np.max(y)
    y = (y-min_y)/(max_y-min_y)

    points = np.vstack((x1, x2, y)).transpose()
    np.savetxt(name + ".xyz", points, delimiter=",")
    print("Saved file "+ name +".xyz")

class Model(object):
    """
    Model for this task.
    You need to implement the fit_model and predict methods
    without changing their signatures, but are allowed to create additional methods.
    """
    sqrtN = 2
    kernel = []
    gp_model = []

    minX1 = inf
    minX2 = inf
    maxX1 = -inf
    maxX2 = -inf

    gridX1Side = 0
    gridX2Side = 0

    def __init__(self):
        """
        Initialize your model here.
        We already provide a random number generator for reproducibility.
        """
        self.rng = np.random.default_rng(seed=0)

        for i in range(0, self.sqrtN*self.sqrtN):
            self.kernel.append(RBF(0.01))
            self.gp_model.append(GaussianProcessRegressor(kernel=self.kernel[i], random_state=0, alpha=0.0475))
        
    def _getPointsInArea(self, i, train_features, train_GT):
        epsilon = 0.00001

        gridX1 = i % self.sqrtN
        gridX2 = floor(i/float(self.sqrtN))

        x1 = train_features[:, 0]
        x2 = train_features[:, 1]

        points = np.vstack((x1,x2)).transpose()
        squarePoly = Path([(gridX1 * self.gridX1Side + self.minX1 - epsilon,  gridX2*self.gridX2Side + self.minX2 - epsilon), (gridX1 * self.gridX1Side + self.minX1 - epsilon,  (gridX2+1)*self.gridX2Side + self.minX2 + epsilon), ((gridX1+1)*self.gridX1Side + self.minX1 + epsilon, (gridX2+1)*self.gridX2Side  + self.minX2 + epsilon),  ((gridX1+1) * self.gridX1Side  + self.minX1+ epsilon,  gridX2*self.gridX2Side  + self.minX2 - epsilon)])

        polyMask = squarePoly.contains_points(points)

        return points[polyMask], train_GT[polyMask]

    def _getAreaFromPoint(self, point):
        epsilon = 0.00001

        point[0] = max(point[0], self.minX1 + epsilon)
        point[1] = max(point[1], self.minX2 + epsilon)
        point[0] = min(point[0], self.maxX1 - epsilon)
        point[1] = min(point[1], self.maxX2 - epsilon)

        gridX1 = floor((point[0]-self.minX1) / self.gridX1Side)
        gridX2 = floor((point[1]-self.minX2) / self.gridX2Side)

        return gridX1 + gridX2*self.sqrtN

    def _calculateMuOffsetFromSigma2(self, sigma2):
        return  0.365* math.sqrt(sigma2) + 1.0
        
    def make_predictions(self, test_features: np.ndarray) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict the pollution concentration for a given set of locations.
        :param test_features: Locations as a 2d NumPy float array of shape (NUM_SAMPLES, 2)
        :return:
            Tuple of three 1d NumPy float arrays, each of shape (NUM_SAMPLES,),
            containing your predictions, the GP posterior mean, and the GP posterior stddev (in that order)
        """
        all_predictions = []
        all_gp_mean = []
        all_gp_std = []

        for i in test_features:
            modelIndex = self._getAreaFromPoint(i)
         
            gp_mean, gp_std = self.gp_model[modelIndex].predict(np.array([i]), return_std=True, return_cov=False)

            #predictions = self.gp_model[modelIndex].sample_y(np.array([i]), random_state=0)
            predictions = gp_mean[0] + self._calculateMuOffsetFromSigma2(gp_std[0]**2)

            all_predictions.append(predictions)#[0])#[0])
            all_gp_mean.append(gp_mean[0])
            all_gp_std.append(gp_std[0])
        
        predictions = np.array(all_predictions)
        gp_mean = np.array(all_gp_mean)
        gp_std = np.array(all_gp_std)
       
        return predictions, gp_mean, gp_std

    def fitting_model(self, train_GT: np.ndarray, train_features: np.ndarray):
        """
        Fit your model on the given training data.
        :param train_features: Training features as a 2d NumPy float array of shape (NUM_SAMPLES, 2)
        :param train_GT: Training pollution concentrations as a 1d NumPy float array of shape (NUM_SAMPLES,)
        """

        self.minX1 = np.min(train_features[:, 0])
        self.minX2 = np.min(train_features[:, 1])
        self.maxX1 = np.max(train_features[:, 0])
        self.maxX2 = np.max(train_features[:, 1])

        self.gridX1Side = (self.maxX1 - self.minX1)/self.sqrtN
        self.gridX2Side = (self.maxX2 - self.minX2)/self.sqrtN
     
        for i in range(0, self.sqrtN*self.sqrtN):
            tempFeature, tempGT = self._getPointsInArea(i, train_features, train_GT)

            self.gp_model[i].fit(np.array(tempFeature), np.array(tempGT))        

            print(self.gp_model[i].kernel_)

def cost_function(ground_truth: np.ndarray, predictions: np.ndarray) -> float:
    """
    Calculates the cost of a set of predictions.

    :param ground_truth: Ground truth pollution levels as a 1d NumPy float array
    :param predictions: Predicted pollution levels as a 1d NumPy float array
    :return: Total cost of all predictions as a single float
    """
    assert ground_truth.ndim == 1 and predictions.ndim == 1 and ground_truth.shape == predictions.shape

    # Unweighted cost
    cost = (ground_truth - predictions) ** 2
    weights = np.ones_like(cost) * COST_W_NORMAL

    # Case i): underprediction
    mask_1 = predictions < ground_truth
    weights[mask_1] = COST_W_UNDERPREDICT

    # Case ii): significant overprediction
    mask_2 = (predictions >= 1.2*ground_truth)
    weights[mask_2] = COST_W_OVERPREDICT

    # Weigh the cost and return the average
    return np.mean(cost * weights)

def perform_extended_evaluation(model: Model, output_dir: str = '/results'):
    """
    Visualizes the predictions of a fitted model.
    :param model: Fitted model to be visualized
    :param output_dir: Directory in which the visualizations will be stored
    """
    print('Performing extended evaluation')
    fig = plt.figure(figsize=(30, 10))
    fig.suptitle('Extended visualization of task 1')

    # Visualize on a uniform grid over the entire coordinate system
    grid_lat, grid_lon = np.meshgrid(
        np.linspace(0, EVALUATION_GRID_POINTS - 1,
                    num=EVALUATION_GRID_POINTS) / EVALUATION_GRID_POINTS,
        np.linspace(0, EVALUATION_GRID_POINTS - 1,
                    num=EVALUATION_GRID_POINTS) / EVALUATION_GRID_POINTS,
    )
    visualization_xs = np.stack(
        (grid_lon.flatten(), grid_lat.flatten()), axis=1)

    # Obtain predictions, means, and stddevs over the entire map
    predictions, gp_mean, gp_stddev = model.make_predictions(visualization_xs)

    predictions = np.reshape(
        predictions, (EVALUATION_GRID_POINTS, EVALUATION_GRID_POINTS))
    gp_mean = np.reshape(
        gp_mean, (EVALUATION_GRID_POINTS, EVALUATION_GRID_POINTS))
    gp_stddev = np.reshape(
        gp_stddev, (EVALUATION_GRID_POINTS, EVALUATION_GRID_POINTS))

    vmin, vmax = 0.0, 65.0
    vmax_stddev = 35.5

    # Plot the actual predictions
    ax_predictions = fig.add_subplot(1, 3, 1)
    predictions_plot = ax_predictions.imshow(predictions, vmin=vmin, vmax=vmax)
    ax_predictions.set_title('Predictions')
    fig.colorbar(predictions_plot)

    # Plot the raw GP predictions with their stddeviations
    ax_gp = fig.add_subplot(1, 3, 2, projection='3d')
    ax_gp.plot_surface(
        X=grid_lon,
        Y=grid_lat,
        Z=gp_mean,
        facecolors=cm.get_cmap()(gp_stddev / vmax_stddev),
        rcount=EVALUATION_GRID_POINTS_3D,
        ccount=EVALUATION_GRID_POINTS_3D,
        linewidth=0,
        antialiased=False
    )
    ax_gp.set_zlim(vmin, vmax)
    ax_gp.set_title('GP means, colors are GP stddev')

    # Plot the standard deviations
    ax_stddev = fig.add_subplot(1, 3, 3)
    stddev_plot = ax_stddev.imshow(gp_stddev, vmin=vmin, vmax=vmax_stddev)
    ax_stddev.set_title('GP estimated stddev')
    fig.colorbar(stddev_plot)

    # Save figure to pdf
    figure_path = os.path.join(output_dir, 'extended_evaluation.pdf')
    fig.savefig(figure_path)
    print(f'Saved extended evaluation to {figure_path}')

    plt.show()


def main():
    # Load the training dateset and test features
    train_features = np.loadtxt('train_x.csv', delimiter=',', skiprows=1)
    train_GT = np.loadtxt('train_y.csv', delimiter=',', skiprows=1)
    #test_features = np.loadtxt('test_x.csv', delimiter=',', skiprows=1)

    validation_GT = []
    validation_features = []

    val_set_percentage = 0.1

    for _ in range(0,int(train_features.shape[0]*val_set_percentage)):
        index = randrange(train_GT.shape[0])

        validation_GT.append(train_GT[index])
        validation_features.append(train_features[index])

        train_features = np.delete(train_features, index, axis=0)
        train_GT = np.delete(train_GT, index, axis=0)

    validation_GT = np.array(validation_GT)
    validation_features = np.array(validation_features)
    time.sleep(2)

    print("Validation perc:", validation_features.shape[0]/train_features.shape[0])
    # Fit the model
    print('Fitting model')
    model = Model()
    model.fitting_model(train_GT, train_features)

    print('Predicting on validation features')
    predictions, _, _ = model.make_predictions(validation_features)
    print("Total reward: ", cost_function(validation_GT, predictions))
    time.sleep(2)

    if EXTENDED_EVALUATION:
        perform_extended_evaluation(model, output_dir='.')

if __name__ == "__main__":
    main()
