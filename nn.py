import math
import numpy as np


from numpy import genfromtxt

import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.python.framework import ops


my_data = genfromtxt('train_data.csv', delimiter=',')

my_data_app = genfromtxt('test_data.csv', delimiter=',')

my_data[0,0] = 12
my_data_app[0,0] = 12

M = np.shape(my_data)[0]
N = np.shape(my_data)[1]

L_Z = N
Y_I = np.shape(my_data_app)[1] - 1
M_app = np.shape(my_data)[0]

t_m = int(M_app*0.5)

train_set = my_data[:,:]

test_set = my_data_app[0:t_m,:]

X_train = train_set[:,0:Y_I].T

Y_train = train_set[:,Y_I].T.reshape(1,-1)

utrain = np.mean(X_train,axis=1).reshape(-1,1)
stdtrain = np.std(X_train,axis=1).reshape(-1,1)


X_train_norm = (X_train - utrain) / stdtrain

X_test = test_set[:,0:Y_I].T

X_test_norm = (X_test - utrain) / stdtrain

Y_test = test_set[:,Y_I].T.reshape(1,-1)

def create_placeholders(n_x, n_y):
    """
    Creates the placeholders for the tensorflow session.
    
    Arguments:
    n_x -- scalar, size of an image vector (num_px * num_px = 64 * 64 * 3 = 12288)
    n_y -- scalar, number of classes (from 0 to 5, so -> 6)
    
    Returns:
    X -- placeholder for the data input, of shape [n_x, None] and dtype "float"
    Y -- placeholder for the input labels, of shape [n_y, None] and dtype "float"
    
    Tips:
    - You will use None because it let's us be flexible on the number of examples you will for the placeholders.
      In fact, the number of examples during test/train is different.
    """

    ### START CODE HERE ### (approx. 2 lines)
    X = tf.placeholder(tf.float32, shape = [n_x,None])
    Y = tf.placeholder(tf.float32, shape = [n_y,None])
    ### END CODE HERE ###
    
    return X, Y

def initialize_parameters(n_x,l_z,utrain,stdtrain):
    print("IP")
    
    l_z2 = l_z + l_z/2
    ### START CODE HERE ### (approx. 6 lines of code)
    W1 = tf.get_variable("W1", [l_z,n_x], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    b1 = tf.get_variable("b1", [l_z,1], initializer = tf.zeros_initializer())
    W2 = tf.get_variable("W2", [l_z2,l_z], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    b2 = tf.get_variable("b2", [l_z2,1], initializer = tf.zeros_initializer())
    W3 = tf.get_variable("W3", [1,l_z2], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    b3 = tf.get_variable("b3", [1,1], initializer = tf.zeros_initializer())
    ### END CODE HERE ###

    Utrain = tf.constant(utrain)

    STDtrain = tf.constant(stdtrain)
    
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3,
                  "Utrain": Utrain,
                  "STDtrain": STDtrain,
                 }
    return parameters
def get_parameters(n_x,l_z):
    
    l_z2 = l_z + l_z/2
    ### START CODE HERE ### (approx. 6 lines of code)
    W1 = tf.get_variable("W1", [l_z,n_x])
    b1 = tf.get_variable("b1", [l_z,1])
    W2 = tf.get_variable("W2", [l_z2,l_z])
    b2 = tf.get_variable("b2", [l_z2,1])
    W3 = tf.get_variable("W3", [1,l_z2])
    b3 = tf.get_variable("b3", [1,1])
    ### END CODE HERE ###

    
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3
                 }
    return parameters
    
def forward_propagation(X, parameters):
    print("FP")
    
    # Retrieve the parameters from the dictionary "parameters"
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']
    
    ### START CODE HERE ### (approx. 5 lines)              # Numpy Equivalents:
    Z1 = tf.add(tf.matmul(W1,X),b1)                                              # Z1 = np.dot(W1, X) + b1
    A1 = tf.nn.relu(Z1)                                              # A1 = relu(Z1)
    Z2 = tf.add(tf.matmul(W2,A1),b2)                                              # Z2 = np.dot(W2, a1) + b2
    A2 = tf.nn.relu(Z2)                                              # A2 = relu(Z2)
    Z3 = tf.add(tf.matmul(W3,A2),b3)                                              # Z3 = np.dot(W3,Z2) + b3
    ### END CODE HERE ###
    
    return Z3

def compute_cost(Z3, Y, parameters):
    """
    Computes the cost
    
    Arguments:
    Z3 -- output of forward propagation (output of the last LINEAR unit), of shape (6, number of examples)
    Y -- "true" labels vector placeholder, same shape as Z3
    
    Returns:
    cost - Tensor of the cost function
    """
    
    W1 = parameters['W1']
    W2 = parameters['W2']
    W3 = parameters['W3']
    
    # to fit the tensorflow requirement for tf.nn.softmax_cross_entropy_with_logits(...,...)
    
    
    #m = Y.shape[0] 
    ### START CODE HERE ### (1 line of code)
    #cost = np.mean(np.square(Z3-Y))
    cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=Y,logits=Z3))
    regularizer = tf.nn.l2_loss(W1)+tf.nn.l2_loss(W2)+tf.nn.l2_loss(W3)
    cost = tf.reduce_mean(cost + 0.002* regularizer)
    ### END CODE HERE ###
    #print("COST = " + cost)
    
    return cost

def printParams(parameters,sess, save):
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']
    Utrain = parameters['Utrain']
    STDtrain = parameters['STDtrain']
    print ("W1 = " + str(W1.eval(session=sess)))
    print ("b1 = " + str(b1.eval(session=sess)))
    print ("W2 = " + str(W2.eval(session=sess)))
    print ("b2 = " + str(b2.eval(session=sess)))
    print ("W3 = " + str(W3.eval(session=sess)))
    print ("b3 = " + str(b3.eval(session=sess)))
    print ("Utrain = " + str(Utrain.eval(session=sess)))
    print ("STDtrain = " + str(STDtrain.eval(session=sess)))
    if save:
        saver = tf.train.Saver({"W1": W1,
                               "b1": b1,
                               "W2": W2,
                               "b2": b2,
                               "W3": W3,
                               "b3": b3})
        saver.save(sess, '/Users/mlnoone/tensorflow/proj/thrombo/filename.chkp')



def model(X_train, Y_train, X_test, Y_test, L_Z, learning_rate = 0.0001,
          num_epochs = 1500, print_cost = True):
    """
    Implements a three-layer tensorflow neural network: LINEAR->RELU->LINEAR->RELU->LINEAR->SOFTMAX.
    
    Arguments:
    X_train -- training set, of shape (input size = 12288, number of training examples = 1080)
    Y_train -- test set, of shape (output size = 6, number of training examples = 1080)
    X_test -- training set, of shape (input size = 12288, number of training examples = 120)
    Y_test -- test set, of shape (output size = 6, number of test examples = 120)
    learning_rate -- learning rate of the optimization
    num_epochs -- number of epochs of the optimization loop
    minibatch_size -- size of a minibatch
    print_cost -- True to print the cost every 100 epochs
    
    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """
    
    ops.reset_default_graph()     
    (n_x, m) = X_train.shape                          # (n_x: input size, m : number of examples in the train set)
    n_y = Y_train.shape[0]                            # n_y : output size
    costs = []                                        # To keep track of the cost
    costs_test = []                                        # To keep track of the cost
    
    
    X, Y = create_placeholders(n_x, n_y)

    parameters = initialize_parameters(n_x,L_Z,utrain,stdtrain)
    Z3 = forward_propagation(X, parameters)
    cost = compute_cost(Z3, Y, parameters)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate).minimize(cost)
   
    print("GVI")
    init = tf.global_variables_initializer()
    
    
    sess = tf.Session()
    sess.run(init)
    
    #printParams(parameters,sess,False)
        
        # Do the training loop
    for epoch in range(num_epochs):
      _ , e_cost = sess.run([optimizer, cost], feed_dict={X: X_train, Y: Y_train})
       
      cost_test = sess.run(cost, feed_dict={X: X_test, Y: Y_test})
      
      # Print the cost every epoch
      if print_cost == True and epoch % 100 == 0:
       print ("Cost after epoch %i: %f - %f" % (epoch, e_cost, cost_test))
      if print_cost == True and epoch % 10 == 0:
          costs.append(e_cost)
          costs_test.append(cost_test)
    #params =  parameters.eval()     
    print("After training")
    printParams(parameters,sess, True)
    sess.close()
                
        # plot the cost
    plt.plot(np.squeeze(costs),'b')
    #plt.plot(np.squeeze(costs_test),'r')
    plt.ylabel('Cost')
    plt.xlabel('Iterations (per tens)')
    plt.title("Learning rate = " + str(learning_rate))
    plt.show()

    # lets save the parameters in a variable
    return parameters

def predict(X_train, Y_train, X_test, Y_test, utrain, stdtrain, l_z):
    
    ops.reset_default_graph() 
    (n_x, m) = X_test.shape
    n_y = Y_test.shape[0]
    
    l_z2 = l_z + l_z/2
    W1 = tf.get_variable("W1", [l_z,n_x])
    b1 = tf.get_variable("b1", [l_z,1])
    W2 = tf.get_variable("W2", [l_z2,l_z])
    b2 = tf.get_variable("b2", [l_z2,1])
    W3 = tf.get_variable("W3", [1,l_z2])
    b3 = tf.get_variable("b3", [1,1])
    Utrain = tf.constant(utrain)

    STDtrain = tf.constant(stdtrain)
    
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3
                 }
    X, Y = create_placeholders(n_x, n_y)
    #X_norm = (X - parameters['Utrain']) / parameters['STDtrain']
    A3 = tf.round(tf.sigmoid(forward_propagation(X, parameters)))
    #init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    sess = tf.Session()
    #sess.run(init)
    saver.restore(sess, "output.chkp")
    parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2, "W3": W3, "b3": b3, "Utrain": Utrain, "STDtrain": STDtrain}
    #printParams(parameters,sess,False)
    #utrain = parameters['Utrain'].eval(session=sess)
    #stdtrain = parameters['STDtrain'].eval(session=sess)

    result_train = sess.run(A3, feed_dict={X: X_train, Y: Y_train} )
    
    result = sess.run(A3, feed_dict={X: X_test, Y: Y_test} )
    
    
    print_results(result_train, Y_train)
    print_results(result, Y_test)



    
    #plt.plot(result.T, Y_test.T,'go')
    #plt.plot(Y_test.T, Y_test.T, 'rx')
    
    #plt.ylabel('Y')
    #plt.xlabel('result')
    #plt.show()
    sess.close()

def print_results(result, Y_test) :
    print("RESULT = "+str(result))
    print("Y = "+str(Y_test))
    equal = result == Y_test;
    print("EQUAL = "+str(np.sum(equal)))
    truep = np.sum( (result == 1) & (result == Y_test))
    totalp = np.sum(result == 1)
    actualp = np.sum(Y_test == 1)
    print("TOTAL POSITIVE = "+str(totalp))
    print("TRUE POSITIVE = "+str(truep))
    print("ACTUAL POSITIVE = "+str(actualp))
    
    precision = (100* truep) / totalp
    recall = (100 *truep) / actualp
    f1 = (precision * recall) / (precision + recall)
    print("PRECISION "+ str ( precision ) +"%" )
    print("RECALL "+ str ( recall) +"%" )
    print("F1 "+ str ( 2*f1) +"%" )

    print("TOTAL = "+str(Y_test.size))
    print("ACCURACY "+ str (  (100*np.sum(result == Y_test)) / Y_test.size ) +"%" )


print ("X_train_norm shape" + str(np.shape(X_train_norm)))
print ("Y_train_shape" + str(np.shape(Y_train)))
parameters = model(X_train_norm, Y_train, X_test_norm, Y_test, L_Z,  0.15)
predict(X_train_norm, Y_train, X_test_norm, Y_test, utrain, stdtrain, L_Z)

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
