# -*- coding: utf-8 -*-
"""
Deep learning module.
version: 20180506  
@author: Ai-sawan J.
"""
import random
import numpy as np
import matplotlib.pyplot as plt

def linear(W,b,A_prev):
    '''
    compute linear combination for feedforward.
    input: W, weights in shape (n_l,n_l-1)
            b, bias in shape (n_l,1)
            A, output from activation units of previous layer. in shape(n_l-1,m)
    output:Z, input for activation units
    '''
    Z = np.dot(W,A_prev) + b
    return Z

def relu(Z):
    '''
    compute Relu activation function
    input: Z shape of (num hidden units ,num obs)
    output: A for next layer
           
    '''
    #braodcast 0 to the same shape as Z then compare maximum element-wise
    A = np.maximum(0,Z)
    return A

def sigmoid(Z):
    '''
    compute sigmoid activation function
    '''
    A = 1/(1+np.exp(-Z))
    return A
def linear_relu_units(A_prev,W,b):
    '''
    combining calculation of linear step and relu activation step
    input:input of linear step including A_l-1 or X
            Weight and bias
    output: A, output from relu
            unit_caches, correcting linear cache and activation cache
             activation_cache is input Z_l for computing relu derivative in back prop
    '''
    Z = linear(W,b,A_prev)
    A = relu(Z)
    
    return Z, A

def linear_sigmoid_units():
    '''
    compute calculation of linear step and sigmoid activation step
    placeholder for future implementation
    '''
    return None

def linear_relu_forward(X,parameters):
    '''
    combining calculation of all hidden layers. If hidden layer contains diiferent
    activation fuction, cannot use this function
    input: X, parameters
    output A,
    caches list of tuple (Z,A) each layer
    '''
    num_layers = len(parameters)//2
    A_prev = X
    caches = []
    #compute every layer except the last one
    for l in range(0,num_layers-1):
        W = parameters["W"+str(l+1)]
        b = parameters["b"+str(l+1)]
        Z,A_prev = linear_relu_units(A_prev,W,b) #update A_prev for the next layer
        caches.append((Z,A_prev))
        
    return A_prev, caches
            
#can create feedforward adam or momentum, but start with simplest
def feedforward_prop(X,parameters):
    '''
    Feed forward neural network with the number of hidden layers and output layers
    as in parameter shape.
    input: X, input data as in shape (n_x,m)
            Y, target output as in shape (1,m)
            parameters, a dictionary mapping W1,b1,W2,b2,...,WL,bL. 
                Wl shape (n_l,n_l-1). bl shape (n_l,1)
    output: AL, predictions, shape (1,m)
            caches, list of tuple containing Zl and Al , in shape (n_l,m)
    '''
    L = len(parameters)//2
    #hidden layers
    A_prev, caches = linear_relu_forward(X,parameters)
    
    #output layer
    Z, Y_hat  = linear_relu_units(A_prev,parameters["W"+str(L)],parameters["b"+str(L)])
    caches.append((Z,Y_hat))
    
    return Y_hat, caches

def binaryCrossEntropy_cost(Y_hat,Y):
    '''
        calculate binary cross entropy cost function
        input: A, prediction as in shape (1,m)
                Y, target as in shape (1,m)
        output: cost, scalar
    '''
    m = Y.shape[1]
    cost = (-1/m)*(np.dot(Y,np.log(Y_hat))+np.dot((1-Y),np.log(1-Y_hat)))
    return cost
    
def mslog_cost(Y_hat,Y):
    '''
        calculate mean square error between the logarithm of 
        the predicted value and the logarithm of the target values.
        This is simpler than rmslog but give the same objective that reduce
        predictive error.
        input: A, prediction as in shape (1,m)
                Y, target as in shape (1,m)
        output: cost, scalar
    '''
    epsilon = 10**-8 #to prevent log(0) and penalized if predict price close to zero
    m = Y.shape[1]
    cost = (1/(2*m))*np.sum((np.log(Y_hat+epsilon) - np.log(Y+epsilon))**2)
    return cost

def msle_derivative(Y_hat,Y):
    '''
    compute derivative of mean square log loss wrt Y_hat. Initializing value
    for backpropagation.
    (we can just put sqrt(mslog_derivative) to get rmse_log_derivative)
    input: Y_hat, shape (1,m) predictive value of forward propagation
    output: dA, shape(1,m) derivative of msl loss wrt Y_hat
    '''
    
    epsilon = 10**-8
    dA = (np.divide((np.log(Y_hat+epsilon)-np.log(Y+epsilon)),Y_hat+epsilon))
    return dA

def rmse_log(Y_hat,Y):
    '''
    calculate root mean square error between the logarithm of 
    the predicted value and the logarithm of the target values.
    input: A, prediction as in shape (1,m)
    Y, target as in shape (1,m)
    output: rmse (scalar)
    '''
    m = Y.shape[1]
    rmse = np.sqrt((1/m)*np.sum((np.log(Y_hat+1) - np.log(Y+1))**2))

    return rmse

    
def relu_derivative(Z):
    '''
    compute derivative of RELU function wrt. input of RELU function (Z)
    input: Z, shape (n_l,m)
    output g_prime, shape(n_l,m)
    '''
    g_prime = (Z > 0)*1
    return g_prime

def backprop_relu(Z,dA):
    '''
    compute backpropagation at the relu activation unit.
    input: Z, from cache at the same layer
            dA, derivative of L wrt A of current layer
    output dZ, loss function wrt Z
    '''
    dZ = dA*relu_derivative(Z)
    return dZ
           
def backprop_linear(dZ,A_prev,W,b):
    '''
    compute derivative of component of linear combination (W,b,A_l-1) wrt loss function.
    input: dZ, result of backprop of current activation unit
            A, Activation input of current layer
            W,b weight and bias of current layer
            
    output: dW_l, db_l, dA_prev , same shape as in W_l,b_l, and A_l-1
            dA_prev will be feed into backprop of previous layer as dA
    '''
    m = A_prev.shape[1]
    dW = (1/m)*np.dot(dZ,A_prev.T)
    db = (1/m)*np.sum(dZ,axis=1,keepdims=True)
    dA_prev = np.dot(W.T,dZ)
       
    return dW,db,dA_prev

def backprop_relu_linear(Z,dA,A_prev,W,b):
    '''
    compute backprop relu and linear combind.
    input: Z,A from cache,dA of current layer, W,b from parameters
    '''
    dZ = backprop_relu(Z,dA)
    dW,db,dA_prev = backprop_linear(dZ,A_prev,W,b)
    return dW,db,dA_prev

def backprop_mse_relu_linear(X,Y,parameters,caches):
    '''
    Back propagation with mean square error between log predicted values and log target values
    (inthe future should have option for different cost functions)
    input: Y for target values, shape (1,m)
            parameters , weight and bias
            caches, list of tuple (Z_l,A_l) in shape (n_l,m) from forward prop
    output grad, set of dW db all layers
            dA, derivative of loss function wrt input unit. not used for supervise learning
    '''
    
    L = len(parameters)//2
    grads={}
    
    #init Z A of deepest layer (Y_hat)
    Z, A = caches[L-1]
    
    
    #init derivative of loss wrt predictive values
    dA = msle_derivative(A,Y)
    
    #calculate last layer separately, we can change last layer to any activation function
    grads["dW"+str(L)], grads["db"+str(L)], dA = backprop_relu_linear(Z,dA,caches[L-2][1],parameters["W"+str(L)],parameters["b"+str(L)])
    
    #backprop hidden layers
    for l in range(L-1,0,-1):
        if l != 1:
            Z = caches[l-1][0]
            grads["dW"+str(l)], grads["db"+str(l)], dA = backprop_relu_linear(Z,dA,caches[l-2][1],parameters["W"+str(l)],parameters["b"+str(l)])
        else:
            Z = caches[l-1][0]
            grads["dW"+str(l)], grads["db"+str(l)], dA = backprop_relu_linear(Z,dA,X,parameters["W"+str(l)],parameters["b"+str(l)])
    return grads, dA
    
def gradient_decent_update(parameters, grads, learning_rate=0.01):
    '''
    compute update weights through gradient decent algorithm
    input: parameters, weights and bias
            grads, gradient decent from backpropagation
            learning_rate, amount of weight to be updated
    output: parameter, updated weight and bias.
    '''
    L = len(parameters)//2
           
    for l in range(L):
        parameters["W"+str(l+1)] = parameters["W"+str(l+1)] - learning_rate*grads["dW"+str(l+1)]
        parameters["b"+str(l+1)] = parameters["b"+str(l+1)] - learning_rate*grads["db"+str(l+1)]
    
    return parameters

def linear_relu_mse_cost_model(X,Y,dims,learning_rate=0.1,num_iter = 10000):
    '''
    initialize weight, compute forward propagation, cost of each iteration, backward propagation, and update parameters.
    and compute cross validation error.
    input: X, data in shape (n,m)
            Y target values in shape (1,m)
            dims, tuple of NN dimensions
            learning_rate, amount to update parameters
    '''
    costs =[]
    n,m = X.shape
    X,mean_x, var_x = normalize(X)
    parameters = initweight_He(dims)
    
    for i in range(num_iter):
        Y_hat, caches = feedforward_prop(X,parameters)
        cost = mslog_cost(Y_hat,Y)
        grads, _ = backprop_mse_relu_linear(X,Y,parameters,caches)
        parameters = gradient_decent_update(parameters, grads, learning_rate)
        
        if i % 100 == 0:
            costs.append(cost)
        if i % 2000 == 0:
            print("Cost after iteration %i: %f" %(i, cost))
            
    plt.plot(costs[3:])
    plt.ylabel('cost')
    plt.xlabel('iteration (per 100)')
    plt.title("Training cost at Learning rate = " + str(learning_rate))
    plt.show()
    
    y_hat,_ = feedforward_prop(X,parameters)
    rmse = rmse_log(y_hat,Y)
    print("RMS Logarithmic Error:= %f" %(rmse) )
    
    return parameters,mean_x, var_x


def normalize(X):
    '''normalize features of input data to be mean = 0, var =1.
    this helps speed up NN learning. 
    input: X numpy array with shape (num_feature, num obs).
    output: normalized X with the same shape.
    '''
    n_x,m = X.shape
    epsilon = 10**-8
    mean_x = np.mean(X,axis=1,keepdims=True)
    X = X - mean_x
    var_x = (1/m)*np.sum(X**2, axis=1, keepdims=True)
    X = X / np.sqrt(var_x + epsilon)
    return X, mean_x, var_x

def randomize_dataset(X):
    '''
    randomize order of training data 
    input: X shape (n,m), Y(1,m)
    output randomized X Y
    '''
    return None

def initweight_He(dims):
    '''
    randomly initialize parameter with He initialization
    input: dims, dimensions of neural network, tuple (input_dim,hidden_dim,..,output_dim)
    output: parameters for each layer, dictionary of parameter W1,b1,...WL,bL
            Wl is np.array shape (n_l,n_l-1), bl is np.array shape (n_l,1)
    '''
    num_layers = len(dims)
    parameters ={}
    for l in range(1,num_layers):
        parameters["W"+str(l)] = np.random.rand(dims[l],dims[l-1])*np.sqrt(2/(dims[l]+dims[l-1]))
        parameters["b"+str(l)] = np.zeros((dims[l],1))
    
    return parameters
    
def predict(X,parameters):
    '''
    use learning parameters on new examples and produce predictive values
    input X dataset shape (n,m), 
        parameters
    output Y_hat
    '''
    Y_hat, _ = feedforward_prop(X,parameters)
    
    return Y_hat

def plotcost():
    '''
    plot training error and CV error and 
    '''
    return None

def crossvalidation(X,Y,dims,num_folds=3,learning_rate = 1,num_iter=2000):
    '''
    compute crossvalidation error from linear_relu_mse_cost_model
    input:X (n,m)
            Y (1,m)
            dims; tuple dimensions of linear_relu_mse_cost_model
            learning_rate of the model
            num_iter of the model
    output list of cross validation costs
    '''
        
    m = Y.shape[1]
    ls = list(range(m))
    random.shuffle(ls)
    num_part = m//num_folds
    
    cvcosts=[]
    lift = None
    Y_hats = np.array([])
    ycvs = np.array([])
    for n in range(num_folds):
        print("Cross Validation %i" %(n))
        if n != num_folds - 1:
            lb = n*num_part
            ub = (n+1)*num_part
            part = ls[lb:ub]
        else:
            lb = n*num_part
            part = ls[lb:]
        
        xtr = np.delete(X,part,axis=1)
        ytr = np.delete(Y,part,axis=1)
        
        xcv = X[:,part]
        ycv = Y[:,part]
        
        params, mux, varx = linear_relu_mse_cost_model(xtr,ytr,dims,learning_rate=learning_rate,num_iter = num_iter)
        x_test_norm = (xcv - mux)/np.sqrt(varx+(10**-8))
        Y_hat = predict(x_test_norm,params)
        eval_metric = rmse_log(Y_hat,ycv)
        Y_hats = np.append(Y_hats,Y_hat)
        ycvs = np.append(ycvs,ycv)
        cvcosts.append(eval_metric)
        
    ls_array = np.array([ls]).T
    Y_hats = np.reshape(Y_hats,(-1,1))
    ycvs = np.reshape(ycvs,(-1,1))
    lift = np.concatenate((ls_array,Y_hats,ycvs),axis=1)   
    return cvcosts, lift
  
    
    


