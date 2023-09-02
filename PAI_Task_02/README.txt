Task description is available in the file "Task2.pdf"

SOLUTION:

To resolve the problem we decided to use the Ensemble Learning method. 
We went for a tipycal NN structure of 1 input layer, 3 hidden layers and one output layer with respectively 28*28, 200 and 10 neurons each. Each neuron had ReLu as the activation function with batch normalization. As for the standard method for classification we trasformed the output of the NN in probabilities through the softmax function.

We trained 5 different NN of this structure, with 5 different optimizers, with learning rate 1e-5, through 100 batches of size 128 each extracted randomly from the dataset. For the training we decided to choose as the loss function the negative log of the softmax of the results of the forward propagation of the NNs. 

To get the prediction we just calculated the mean of the results of the 5 NNs. Being a classification problem, this mean is a discrete distribution with a mean and a variance. This last number can be interpreted as the level of uncertainty of our results.

To finish our implementation we did some correction on the outputs of the NNS. We found out that our model was  overconfident on its predictions. To improve the calibration of our results we decided to decrease the value of the maximum output of each net and increase all the others to obtain distributions with higher tails.