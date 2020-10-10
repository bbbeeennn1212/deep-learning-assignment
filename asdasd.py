# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 21:21:12 2020

@author: UseR
"""






import numpy as np

def sigmoid(x):
  # Sigmoid activation function: f(x) = 1 / (1 + e^(-x))
  return 1 / (1 + np.exp(-x))

def deriv_sigmoid(x):
  # Derivative of sigmoid: f'(x) = f(x) * (1 - f(x))
  fx = sigmoid(x)
  return fx * (1 - fx)

def mse_loss(y_true, y_pred):
  # y_true and y_pred are numpy arrays of the same length.
  return ((y_true - y_pred) ** 2).mean()

class OurNeuralNetwork:
  '''
  A neural network with:
    - 2 inputs
    - a hidden layer with 2 neurons (h1, h2)
    - an output layer with 1 neuron (o1)

  *** DISCLAIMER ***:
  The code below is intended to be simple and educational, NOT optimal.
  Real neural net code looks nothing like this. DO NOT use this code.
  Instead, read/run it to understand how this specific network works.
  '''
  def __init__(self):
    # Weights
    self.w=np.array([[0.5,0.5],[0.5,0.5]])
    self.wO=np.array([[0.5],[0.5]])
    # Biases
    self.b1 = np.random.normal()
    self.b2 = np.random.normal()
    self.b3 = np.random.normal()

  def feedforward(self, x):
    # x is a numpy array with 2 elements.
    h1 = sigmoid(self.w[0][0] * x[0] + self.w[0][1] * x[1] + self.b1)
    h2 = sigmoid(self.w[1][0] * x[0] + self.w[1][1] * x[1] + self.b2)
    o1 = sigmoid(self.wO[0][0] * h1 + self.wO[1][0] * h2 + self.b3)
    return o1

  def train(self, data, all_y_trues):
    '''
    - data is a (n x 2) numpy array, n = # of samples in the dataset.
    - all_y_trues is a numpy array with n elements.
      Elements in all_y_trues correspond to those in data.
    '''
    learn_rate = 0.1
    epochs = 1000 # number of times to loop through the entire dataset
    
    for epoch in range(epochs):
      for x, y_true in zip(data, all_y_trues):
        R=np.array([[x[0]],[x[1]]])
        # --- Do a feedforward (we'll need these values later)
        sum1=np.dot(self.w,R)
        h1=sigmoid(sum1[0][0]+self.b1)
        h2=sigmoid(sum1[1][0]+self.b2)
        sum2=np.array([[h1,h2]])
        sum_o1=np.dot(self.wO,sum2)
        o1=sigmoid(sum_o1[0][0]+self.b3)
        y_pred = o1

        # --- Calculate partial derivatives.
        # --- Naming: d_L_d_w1 represents "partial L / partial w1"
        d_L_d_ypred = -2 * (y_true - y_pred)
        # Neuron o1
        matrix=np.array([[h1,h2,1]])
        matrix_mul_deriv=matrix*deriv_sigmoid(sum_o1[0][0])
        
        d_ypred_d_h1 = self.wO[0][0] * deriv_sigmoid(sum_o1[0][0]) ###
        d_ypred_d_h2 = self.wO[1][0] * deriv_sigmoid(sum_o1[0][0]) ###

        # Neuron h1
        N=np.array([[x[0],x[1],1]])
        matrix2_mul_deriv=N*deriv_sigmoid(sum1+self.b1)
        
        # Neuron h2
        N2=N*deriv_sigmoid(sum1[1][0]+self.b1)
        

        # --- Update weights and biases
        # Neuron h1
        self.w[0][0] -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * matrix2_mul_deriv[0][0]
        self.w[0][1] -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * matrix2_mul_deriv[0][1]
        self.b1 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * matrix2_mul_deriv[0][2]

        # Neuron h2
        self.w[1][0] -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * N2[0][0]
        self.w[1][1] -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * N2[0][1]
        self.b2 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * N2[0][2]

        # Neuron o1
        self.wO[0][0] -= learn_rate * d_L_d_ypred * matrix_mul_deriv[0][0]
        self.wO[1][0] -= learn_rate * d_L_d_ypred * matrix_mul_deriv[0][1]
        self.b3 -= learn_rate * d_L_d_ypred * matrix_mul_deriv[0][2]

      # --- Calculate total loss at the end of each epoch
      if epoch % 10 == 0:
        y_preds = np.apply_along_axis(self.feedforward, 1, data)
        loss = mse_loss(all_y_trues, y_preds)
        print("Epoch %d loss: %.3f" % (epoch, loss))

# Define dataset
data = np.array([
  [-2, -1],  # Alice
  [25, 6],   # Bob
  [17, 4],   # Charlie
  [-15, -6], # Diana
])
all_y_trues = np.array([
  1, # Alice
  0, # Bob
  0, # Charlie
  1, # Diana
])

# Train our neural network!
network = OurNeuralNetwork()
network.train(data, all_y_trues)
# Make some predictions
emily = np.array([-7, -3]) # 128 pounds, 63 inches
frank = np.array([20, 2])  # 155 pounds, 68 inches
print("Emily: %.3f" % network.feedforward(emily)) # 0.951 - F
print("Frank: %.3f" % network.feedforward(frank)) # 0.039 - M



