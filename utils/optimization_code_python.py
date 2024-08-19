import numpy as np
import math



""" Mini-Batch Gradient Descent  (mejora 1)"""  
import math

def random_mini_batches(X, Y, mini_batch_size=32):

    m = X.shape[1]
    mini_batches = []

    #Paso 1: Creamos versión shuffled del set de entrenamiento
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation]

    #Paso 2: Partir la mezcla en minibatchs de tamaño 'mini_batch_size'
    inc = mini_batch_size
    num_complete_minibatch = math.floor(m / mini_batch_size)

    for k in range(0,num_complete_minibatch):

        mini_batch_X = shuffled_X[:, k*inc : (k+1)*inc]
        mini_batch_Y = shuffled_Y[:, k*inc : (k+1)*inc]

        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    if m % mini_batch_size != 0:

        mini_batch_X = shuffled_X[:, num_complete_minibatch*inc : ]
        mini_batch_Y = shuffled_Y[:, num_complete_minibatch*inc : ]

        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches


#Paso 3: Modelo de la propagación hacia delante

def L_model_forward_dropout(X, parameters, dropout_prob=0, layers_drop=[0, 0, 0, 0, 0]):

    caches = []

    L = len(parameters) // 2
    A = X

    keep_prob = 1 - dropout_prob
    

    for l in range(1, L):
        
        A_prev = A

        A, cache = linear_activation_forward(A_prev, parameters["W"+str(l)], parameters["b"+str(l)], "relu")

        if layers_drop[l] == 1:
            D = np.random.rand(*A.shape)
            D = (D < keep_prob).astype(int)
            A = A * D
            A = A/keep_prob
            
            cache = (cache , D)
        else:
            cache = (cache, None)

        caches.append(cache)       


    #Ultima capa, sigmoid:
    AL, cache = linear_activation_forward(A, parameters["W"+str(L)], parameters["b"+str(L)], "sigmoid")
    caches.append(cache)

    return AL, caches



""" Momentum para Gradient Descent (mejora 2)"""
#When using minibatches for update we make the update just with a subset of examples. Momentum takes in account past gradients to smooth the 
# update.

#Step 1 - Inicializar velocity 'v'
def initilize_velocity(parameters):
    
    L = len(parameters) // 2

    v = {}

    for l in range(1,L+1):

        v["dW"+str(l)] = np.zeros((parameters["W"+str(l)].shape))
        v["db"+str(l)] = np.zeros((parameters["b"+str(l)].shape))

    return v

#Step 2 - Update parameters with momentum
def update_parameters_with_momentum(parameters, grads, v, beta, learning_rate):

    L = len(parameters) // 2

    for l in range(1,L+1):

        #Exponentially Weighted Averages
        v["dW"+str(l)] = beta * v["dW"+str(l)] + (1-beta) * grads["dW"+str(l)]
        v["db"+str(l)] = beta * v["db"+str(l)] + (1-beta) * grads["db"+str(l)]

        #Update of parameters with momentum
        parameters["W"+str(l)] = parameters["W"+str(l)] - learning_rate * v["dW"+str(l)]
        parameters["b"+str(l)] = parameters["b"+str(l)] - learning_rate * v["db"+str(l)]

    return parameters, v


""" Adam Algorithm - Momentum + RMSProp (mejora 3)"""

#Step 1: Initialize with Adam: v --> exponentially weighted average. s --> exponentially weighted average of the squares
def initialize_parameters_adam(parameters):

    L = len(parameters) // 2

    v = {}
    s = {}

    for l in range(1,L+1):

        v["dW" + str(l)] = np.zeros((parameters["W"+str(l)].shape))
        v["db" + str(l)] = np.zeros((parameters["b"+str(l)].shape))
        s["dW" + str(l)] = np.zeros((parameters["W"+str(l)].shape))
        s["db" + str(l)] = np.zeros((parameters["b"+str(l)].shape))
        
    return v, s

#Step 2: Calculates the respective exponentially weighted average of past gradient before and after bias correction and updates the parameters

def update_parameters_with_adam(parameters, grads, v, s, t, beta1, beta2, learning_rate, epsilon):
    
    L = len(parameters) // 2

    v_corrected = {}
    s_corrected = {}

    for l in range(1, L+1):

        #exponential weighted average 'v' and squared 's'
        v["dW" + str(l)] = beta1 * v["dW" + str(l)] + (1-beta1) * grads["dW"+str(l)]
        v["db" + str(l)] = beta1 * v["db" + str(l)] + (1-beta1) * grads["db"+str(l)]

        v_corrected["dW"+str(l)] = v["dW"+str(l)] / (1-np.power(beta1,t))
        v_corrected["db"+str(l)] = v["db"+str(l)] / (1-np.power(beta1,t))

        s["dW"+str(l)] = beta2 * s["dW"+str(l)] + (1 - beta2) * np.power(grads["dW"+str(l)], 2)
        s["db"+str(l)] = beta2 * s["db"+str(l)] + (1 - beta2) * np.power(grads["db"+str(l)], 2)
        
        #with bias correction
        s_corrected["dW" + str(l)] = s["dW"+str(l)] / (1-np.power(beta2,t))
        s_corrected["db" + str(l)] = s["db"+str(l)] / (1-np.power(beta2,t))

        #update
        parameters["W"+str(l)] = parameters["W"+str(l)] - learning_rate * (v_corrected["dW"+str(l)] / (np.sqrt(s_corrected["dW"+str(l)]) + epsilon))
        parameters["b"+str(l)] = parameters["b"+str(l)] - learning_rate * (v_corrected["db"+str(l)] / (np.sqrt(s_corrected["db"+str(l)]) + epsilon))

    return parameters, v, s



""" Cost Function Module """
def bce_compute(AL, Y):

    m = Y.shape[1]

    epsilon = 1e-8
    AL = np.clip(AL, epsilon, 1 - epsilon)  # Avoid log(0) or log(1-0)

    cost = -1/m * np.sum( Y * np.log(AL) + (1 - Y) * np.log(1-AL))

    return np.squeeze(cost)

""" Regularization -- Reduce Overfitting --> L2"""
#Step 1: compute the cost
def compute_cost_with_regularization(AL, Y, parameters, lambd):

    m = Y.shape[1]

    cross_entropy_cost = bce_compute(AL,Y)

    L2_regularization_cost = 0

    for key in parameters:
        if 'W' in key:
            L2_regularization_cost += np.sum(np.square(parameters[key]))


    L2_regularization_cost = (lambd / (2 * m)) * L2_regularization_cost

    cost = cross_entropy_cost + L2_regularization_cost

    return  cost