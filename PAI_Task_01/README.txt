0. TASK DESCRIPTION
Given a dataset of pollution indices measured in different locations of a 2D map, fit a gaussian process to predict the pollution values in other locations of the map.

1.	DIVISION IN GRID
Since the dataset was large we decided to divide the domain of x in N rectangles. For every rectangle there will be a different model, trained only with data that belongs to that region.
We calculate the side of a grill on x1 axis by doing (max {x1} – min{x1})/sqrtN and same for x2 axis. Every rectangle is being identified with a progressive numeration.
We then write two python functions that will be helpful later The first one, _getAreaFromPoint(), takes the number of a rectangle as the input and returns all the data points that belong to that rectangle. The second one takes a point as input and returns the number of the rectangle to which it belongs.

2.	MODEL TRAINING
We tried to train the whole dataset at the same time, but the computation time was too high and we opted for dividing the dataset spatially into an N by N grid. In particular, dividing the training set into 4 cells proved to be the optimal division, since it allowed us to run the training in a reasonable time without losing too much local information. Each cell was individually trained with a separate gaussian process and also the evaluation was performed the same way, by obtaining first the cell to which the point belonged to and then using the model trained on that specific cell to obtain an estimation.
As a prior, we decided to go for a gaussian kernel with a small lenght scale h=0.01., since the plotted data did not show high frequency components. Also we tried some Matern kernels and exponential ones, and the RBF was the one that gave us better scores.
The models have all the same prior.

3. PREDICTION
Given a test point of which we want to predict the label, we first need to understand which rectangular it belongs to, in order to use the right model for the prediction:
And then we get the mean function and cov function with
            gp_mean, gp_std = self.gp_model[modelIndex].predict(np.array([i]), return_std=True, return_cov=False)
To make a prediction we could have simply used the gp_mean function. This would have been sufficient to go under the cost  bottom-line. However, we went for a more accurate prediction.
This approach takes for true the posterior distribution of the prediction. That means that the prediction f* is distributed as a Gaussian with mean gp_mean(x*):=mu and variance k(x*,x*) := sigma^2. Now since the cost function penalizes underestimations more, we want our prediction to be a bit greater than mu. The best delta to add to mu we need to pick the delta which minimizes the expectation
	E[costFunction( mu+delta , f*)]
Which we can approximate with averaging.
It is pretty obvious that this delta does not depend on mu, but only on sigma. So we extracted many values from a gaussian distribution variance sigma and we picked the f which minimized the average, for different values of sigma.
We fitted the function best_delta,sigma and we discovered by fitting that the relation between best_delta and sigma is a sqrt:
	best_delta = 0.365*math.sqrt(sigma**2)
We than discovered that
	best_delta = 0.365*math.sqrt(sigma**2)+1
Performes better. This only means that our model to predict f* tends to underestimate the pollution by 1.