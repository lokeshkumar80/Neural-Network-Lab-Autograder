import numpy as np

class FullyConnectedLayer:
	def __init__(self, in_nodes, out_nodes):
		# Method to initialize a Fully Connected Layer
		# Parameters
		# in_nodes - number of input nodes of this layer
		# out_nodes - number of output nodes of this layer
		self.in_nodes = in_nodes
		self.out_nodes = out_nodes
		# Stores the outgoing summation of weights * feautres 
		self.data = None

		# Initializes the Weights and Biases using a Normal Distribution with Mean 0 and Standard Deviation 0.1
		self.weights = np.random.normal(0,0.1,(in_nodes, out_nodes))	
		self.biases = np.random.normal(0,0.1, (1, out_nodes))
		###############################################
		# NOTE: You must NOT change the above code but you can add extra variables if necessary 

	def forwardpass(self, X):
		# print('Forward FC ',self.weights.shape)
		# Input
		# activations : Activations from previous layer/input
		# Output
		# activations : Activations after one forward pass through this layer
		
		n = X.shape[0]  # batch size
		# INPUT activation matrix  		:[n X self.in_nodes]
		# OUTPUT activation matrix		:[n X self.out_nodes]

		###############################################
		# TASK 1 - YOUR CODE HERE
		self.input = X
		z=np.dot(X, self.weights) + self.biases
		
		a=sigmoid(z)
		self.data = a
		self.z=z
		return a

		# raise NotImplementedError
		###############################################
		
	def backwardpass(self, lr, activation_prev, delta):
		# Input
		# lr : learning rate of the neural network
		# activation_prev : Activations from previous layer
		# delta : del_Error/ del_activation_curr
		# Output
		# new_delta : del_Error/ del_activation_prev
		
		# Update self.weights and self.biases for this layer by backpropagation
		n = activation_prev.shape[0] # batch size

		###############################################
		# TASK 2 - YOUR CODE HERE
		n=activation_prev.shape[0]
		delta=delta*derivative_sigmoid(self.z)

		dw=np.dot(activation_prev.T, delta)/n
		db=np.sum(delta, axis=0, keepdims=True)/n

		self.weights -= lr * dw
		self.biases -= lr * db

		new_delta=np.dot(delta, self.weights.T)
		return new_delta

		raise NotImplementedError
		###############################################

class ConvolutionLayer:
	def __init__(self, in_channels, filter_size, numfilters, stride):
		# Method to initialize a Convolution Layer
		# Parameters
		# in_channels - list of 3 elements denoting size of input for convolution layer
		# filter_size - list of 2 elements denoting size of kernel weights for convolution layer
		# numfilters  - number of feature maps (denoting output depth)
		# stride	  - stride to used during convolution forward pass
		self.in_depth, self.in_row, self.in_col = in_channels
		self.filter_row, self.filter_col = filter_size
		self.stride = stride

		self.out_depth = numfilters
		self.out_row = int((self.in_row - self.filter_row)/self.stride + 1)
		self.out_col = int((self.in_col - self.filter_col)/self.stride + 1)

		# Stores the outgoing summation of weights * feautres 
		self.data = None
		
		# Initializes the Weights and Biases using a Normal Distribution with Mean 0 and Standard Deviation 0.1
		self.weights = np.random.normal(0,0.1, (self.out_depth, self.in_depth, self.filter_row, self.filter_col))	
		self.biases = np.random.normal(0,0.1,self.out_depth)
		

	def forwardpass(self, X):
		# print('Forward CN ',self.weights.shape)
		# Input
		# X : Activations from previous layer/input
		# Output
		# activations : Activations after one forward pass through this layer
		n = X.shape[0]  # batch size
		# INPUT activation matrix  		:[n X self.in_depth X self.in_row X self.in_col]
		# OUTPUT activation matrix		:[n X self.out_depth X self.out_row X self.out_col]

		###############################################
		# TASK 1 - YOUR CODE HERE
		self.input = X

		out = np.zeros((n, self.out_depth, self.out_row, self.out_col))

		for i in range(n):  # iterate over batch size
			for f in range(self.out_depth):  # iterate over number of filters
				for r in range(self.out_row):
					for c in range(self.out_col):
						r_start = r * self.stride
						r_end = r_start + self.filter_row
						c_start = c * self.stride
						c_end = c_start + self.filter_col

						patch = X[i, :, r_start:r_end, c_start:c_end]
						out[i, f, r, c] = np.sum(patch * self.weights[f]) + self.biases[f]

		self.data = sigmoid(out)
		self.z=out
		return self.data

		# raise NotImplementedError
		###############################################

	def backwardpass(self, lr, activation_prev, delta):
		# Input
		# lr : learning rate of the neural network
		# activation_prev : Activations from previous layer
		# delta : del_Error/ del_activation_curr
		# Output
		# new_delta : del_Error/ del_activation_prev
		
		# Update self.weights and self.biases for this layer by backpropagation
		n = activation_prev.shape[0] # batch size

		###############################################
		# TASK 2 - YOUR CODE HERE
		n=activation_prev.shape[0]
		delta=delta*derivative_sigmoid(self.z)
		dw=np.zeros_like(self.weights)
		db=np.sum(delta, axis=(0, 2, 3),keepdims=True)/n
		new_delta=np.zeros_like(activation_prev)

		for i in range(n):  # iterate over batch size
			for f in range(self.out_depth):  # iterate over number of filters
				for r in range(self.out_row):
					for c in range(self.out_col):
						r_start = r * self.stride
						r_end = r_start + self.filter_row
						c_start = c * self.stride
						c_end = c_start + self.filter_col

						patch = activation_prev[i, :, r_start:r_end, c_start:c_end]

						dw[f] += patch * delta[i, f, r, c]
						new_delta[i, :, r_start:r_end, c_start:c_end] += self.weights[f] * delta[i, f, r, c]
		dw /= n
		self.weights -= lr * dw
		self.biases -= lr * db.flatten()
		return new_delta

		# raise NotImplementedError
		###############################################
	
class AvgPoolingLayer:
	def __init__(self, in_channels, filter_size, stride):
		# Method to initialize a Convolution Layer
		# Parameters
		# in_channels - list of 3 elements denoting size of input for max_pooling layer
		# filter_size - list of 2 elements denoting size of kernel weights for convolution layer

		# NOTE: Here we assume filter_size = stride
		# And we will ensure self.filter_size[0] = self.filter_size[1]
		self.in_depth, self.in_row, self.in_col = in_channels
		self.filter_row, self.filter_col = filter_size
		self.stride = stride

		self.out_depth = self.in_depth
		self.out_row = int((self.in_row - self.filter_row)/self.stride + 1)
		self.out_col = int((self.in_col - self.filter_col)/self.stride + 1)

	def forwardpass(self, X):
		# print('Forward MP ')
		# Input
		# X : Activations from previous layer/input
		# Output
		# activations : Activations after one forward pass through this layer
		
		n = X.shape[0]  # batch size
		# INPUT activation matrix  		:[n X self.in_depth X self.in_row X self.in_col]
		# OUTPUT activation matrix		:[n X self.out_depth X self.out_row X self.out_col]

		###############################################
		# TASK 1 - YOUR CODE HERE
		self.input = X
		out = np.zeros((n, self.out_depth, self.out_row, self.out_col))

		for i in range(n):  # iterate over batch size
			for d in range(self.out_depth):  # iterate over depth
				for r in range(self.out_row):
					for c in range(self.out_col):
						r_start = r * self.stride
						r_end = r_start + self.filter_row
						c_start = c * self.stride
						c_end = c_start + self.filter_col

						patch = X[i, d, r_start:r_end, c_start:c_end]
						out[i, d, r, c] = np.mean(patch)
		self.data = out
		return out
		# raise NotImplementedError
		###############################################


	def backwardpass(self, alpha, activation_prev, delta):
		# Input
		# lr : learning rate of the neural network
		# activation_prev : Activations from previous layer
		# activations_curr : Activations of current layer
		# delta : del_Error/ del_activation_curr
		# Output
		# new_delta : del_Error/ del_activation_prev
		
		n = activation_prev.shape[0] # batch size

		###############################################
		# TASK 2 - YOUR CODE HERE
		f=self.filter_row
		s=self.stride

		new_delta=np.zeros(activation_prev.shape)
		for i in range(n):  # iterate over batch size
			for d in range(self.out_depth):  # iterate over depth
				for r in range(self.out_row):
					for c in range(self.out_col):
						r_start = r * s
						r_end = r_start + f
						c_start = c * s
						c_end = c_start + f

						avg_grad = delta[i, d, r, c] / (f * f)
						new_delta[i, d, r_start:r_end, c_start:c_end] += np.ones((f, f)) * avg_grad
		return new_delta

		# raise NotImplementedError
		###############################################


# Helper layer to insert between convolution and fully connected layers
class FlattenLayer:
    def __init__(self):
        pass
    
    def forwardpass(self, X):
        self.in_batch, self.r, self.c, self.k = X.shape
        return X.reshape(self.in_batch, self.r * self.c * self.k)

    def backwardpass(self, lr, activation_prev, delta):
        return delta.reshape(self.in_batch, self.r, self.c, self.k)


# Helper Function for the activation and its derivative
def sigmoid(x):
	return 1 / (1 + np.exp(-x))

def derivative_sigmoid(x):
	return sigmoid(x) * (1 - sigmoid(x))
