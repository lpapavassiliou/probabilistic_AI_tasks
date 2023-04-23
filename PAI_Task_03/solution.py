import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from sklearn.gaussian_process.kernels import *
from sklearn.gaussian_process import GaussianProcessRegressor
from scipy.stats import norm
import matplotlib.pyplot as plt 
import os

from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)
simplefilter("ignore", category=np.VisibleDeprecationWarning)

domain = np.array([[0, 5]])
SAFETY_THRESHOLD = 1.2
SEED = 0

""" Solution """


class BO_algo():
    gp_f = None
    gp_v = None
    x_data = np.empty((1,1), dtype=float)
    f_data = np.empty((1,1), dtype=float)
    v_data = np.empty((1,1), dtype=float)

    n_uniform_sample = 5
    current_sample = 0

    def __init__(self):
        """Initializes the algorithm with a parameter configuration. """
        self.gp_f = GaussianProcessRegressor(kernel=0.5*Matern(length_scale=0.5, length_scale_bounds="fixed", nu=2.5), random_state=0, alpha=0.15**2)
        self.gp_v = GaussianProcessRegressor(kernel=(1.5 + np.sqrt(2)*Matern(length_scale=0.5, length_scale_bounds="fixed", nu=2.5)), random_state=0, alpha=0.0001**2)

    def next_recommendation(self):
        """
        Recommend the next input to sample.

        Returns
        -------
        recommendation: np.ndarray
            1 x domain.shape[0] array containing the next point to evaluate
        """

        # TODO: enter your code here
        # In implementing this function, you may use optimize_acquisition_function() defined below.
        res = 0
        if self.current_sample < self.n_uniform_sample:
            res = (self.current_sample)*(domain[0][1])/(self.n_uniform_sample-1)
        else:
            res = self.optimize_acquisition_function()

        self.current_sample +=1
        return res
        #raise NotImplementedError


    def optimize_acquisition_function(self):
        """
        Optimizes the acquisition function.

        Returns
        -------
        x_opt: np.ndarray
            1 x domain.shape[0] array containing the point that maximize the acquisition function.
        """

        def objective(x):
            return -self.acquisition_function(x)


        f_values = []
        x_values = []

        # Restarts the optimization 20 times and pick best solution
        for _ in range(20):
            x0 = domain[:, 0] + (domain[:, 1] - domain[:, 0]) * \
                 np.random.rand(domain.shape[0])
            result = fmin_l_bfgs_b(objective, x0=x0, bounds=domain,
                                   approx_grad=True)
            x_values.append(np.clip(result[0], *domain[0]))
            f_values.append(-result[1])

        ind = np.argmax(f_values)
        return np.atleast_2d(x_values[ind])

    def expected_improvement(self, x):
        # mu(f(x)), sigma(f(x))
        mu, sigma = self.gp_f.predict([x], return_std=True)
        mu_sample = self.gp_f.predict(self.x_data)

        sigma = sigma.reshape(-1, 1)
        
        # f(x*) <- best so far
        mu_sample_opt = np.max(mu_sample)

        with np.errstate(divide='warn'):
            imp = mu - mu_sample_opt
            Z = imp / sigma
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)

            if sigma == 0.0:
                ei = 0.0

        return ei

    def probability_safe(self, x):
        mu, sigma = self.gp_v.predict([x], return_std=True)

        return norm.cdf(-(SAFETY_THRESHOLD-mu)/sigma)

    def acquisition_function(self, x):
        """
        Compute the acquisition function.

        Parameters
        ----------
        x: np.ndarray
            x in domain of f

        Returns
        ------
        af_value: float
            Value of the acquisition function at x
        """

        #raise NotImplementedError
        #mu_f, std_f = self.gp_f.predict([x], return_std=True)
        #return (mu_f + 100*std_f)[0]
        return ((self.expected_improvement(x))*(self.probability_safe(x)**2))[0]

    def add_data_point(self, x, f, v):
        """
        Add data points to the model.

        Parameters
        ----------
        x: np.ndarray
            Hyperparameters
        f: np.ndarray
            Model accuracy
        v: np.ndarray
            Model training speed
        """
        if x.shape == (1,):
            x = np.array([x])
        if v.shape == (1,):
            v = np.array([v])
        if f.shape == (1,):
            f = np.array([f])

        self.x_data = np.append(self.x_data, x, axis=0)
        self.f_data = np.append(self.f_data, f, axis=0)
        self.v_data = np.append(self.v_data, v, axis=0)

        self.gp_f.fit(self.x_data, self.f_data)
        self.gp_v.fit(self.x_data, self.v_data)
        #raise NotImplementedError

    def get_solution(self):
        """
        Return x_opt that is believed to be the maximizer of f.

        Returns
        -------
        solution: np.ndarray
            1 x domain.shape[0] array containing the optimal solution of the problem
        """

        # TODO: enter your code here
        #return self.optimize_acquisition_function()
        values = []
        
        mu_f_values = []
        sigma_f_values = []
        mu_v_values = []
        sigma_v_values = []
        prob_values = []

        resolution = 0.001
        for i in np.arange(0,5,resolution):
            mu_f, sigma_f = self.gp_f.predict([[i]], return_std=True)
            mu_v, sigma_v = self.gp_v.predict([[i]], return_std=True)

            mu_f_values.append(mu_f[0][0])

            sigma_f_values.append(sigma_f[0])
            mu_v_values.append(mu_v[0][0])
            sigma_v_values.append(sigma_v[0])

            prob_values.append(self.probability_safe([i])[0][0])
            if self.probability_safe([i])[0][0] > 0.99:
                values.append(mu_f)
            else:
                values.append(-np.Inf)
        
        value = np.argmax(np.array(values)) * resolution

        plot = False
        if plot:
            fig = plt.figure()
            ax = fig.gca()

            ax.errorbar(np.arange(0,5,resolution), mu_f_values, yerr=3.0*np.array(sigma_f_values), fmt="or")
            ax.errorbar(np.arange(0,5,resolution), mu_v_values, yerr=3.0*np.array(sigma_v_values), fmt="og")
            ax.scatter(self.x_data, self.x_data*0.0 - 0.6)
            ax.plot(np.arange(0,5,resolution), prob_values)

            ax.vlines(value, -0.6,2.0, "b")
            ax.hlines(SAFETY_THRESHOLD, 0,5.0, "orange")
            
            a = 0
            while os.path.isfile("/root/" + str(a) + ".png"):
                a += 1
            fig.savefig("/root/" +str(a) + ".png", dpi=600)

        return value


""" Toy problem to check code works as expected """

def check_in_domain(x):
    """Validate input"""
    x = np.atleast_2d(x)
    return np.all(x >= domain[None, :, 0]) and np.all(x <= domain[None, :, 1])


def f(x):
    """Dummy objective"""
    mid_point = domain[:, 0] + 0.5 * (domain[:, 1] - domain[:, 0])
    return - np.linalg.norm(x - mid_point, 2)  # -(x - 2.5)^2


def v(x):
    """Dummy speed"""
    return 2.0

def get_initial_safe_point():
    """Return initial safe point"""
    x_domain = np.linspace(*domain[0], 4000)[:, None]
    c_val = np.vectorize(v)(x_domain)
    x_valid = x_domain[c_val > SAFETY_THRESHOLD]
    np.random.seed(SEED)
    np.random.shuffle(x_valid)
    x_init = x_valid[0]
    return x_init



def main():
    # Init problem
    agent = BO_algo()

    # Add initial safe point
    x_init = get_initial_safe_point()
    obj_val = f(x_init)
    cost_val = v(x_init)
    agent.add_data_point(x_init, obj_val, cost_val)


    # Loop until budget is exhausted
    for j in range(20):
        # Get next recommendation
        x = agent.next_recommendation()

        # Check for valid shape
        assert x.shape == (1, domain.shape[0]), \
            f"The function next recommendation must return a numpy array of " \
            f"shape (1, {domain.shape[0]})"

        # Obtain objective and constraint observation
        obj_val = f(x)
        cost_val = v(x)
        agent.add_data_point(x, obj_val, cost_val)

    # Validate solution
    solution = np.atleast_2d(agent.get_solution())
    assert solution.shape == (1, domain.shape[0]), \
        f"The function get solution must return a numpy array of shape (" \
        f"1, {domain.shape[0]})"
    assert check_in_domain(solution), \
        f'The function get solution must return a point within the ' \
        f'domain, {solution} returned instead'

    # Compute regret
    if v(solution) < 1.2:
        regret = 1
    else:
        regret = (0 - f(solution))

    print(f'Optimal value: 0\nProposed solution {solution}\nSolution value '
          f'{f(solution)}\nRegret{regret}')


if __name__ == "__main__":
    main()