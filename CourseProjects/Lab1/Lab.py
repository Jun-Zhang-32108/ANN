# -*- coding: utf-8 -*-
# python3/pyhon2
# 2018-11-16
# Lab Assignment 1 for ANN course (DD2437)
# Author: Victor Castillo, Jun Zhang

import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd



"""
Generates two classes of linearly separable data
"""
"""
The multivariate normal, multinormal or Gaussian distribution is a generalization of the one-dimensional normal distribution to higher dimensions. 
Such a distribution is specified by its mean and covariance matrix. These parameters are analogous to the mean (average or “center”) and variance 
(standard deviation, or “width,” squared) of the one-dimensional normal distribution.
https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.multivariate_normal.html
"""

n_class_instances = 100
mean2 = [-5, 5] #mean for class 1
cov2 = [[3, 0], [0, 3]] #covariance for class 1
mean1 = [5, 10] #mean for class 2
cov1 = [[3, 0], [0, 3]] #covariance for class 2

# mean2 = [0, -1.5] #mean for class 1
# cov2 = [[0.25, 0], [0, 0.25]] #covariance for class 1
# mean1 = [0, 1.5] #mean for class 2
# cov1 = [[0.25, 0], [0, 0.25]] #covariance for class 2


def generation_data(hasBias,mean1,cov1,mean2,cov2):
    #x and y position for each instance will create a 3 neuron layer (two features + bias)
    x2, y2 = np.random.multivariate_normal(mean2, cov2, n_class_instances).T #generate random multivariate normal points (100) for class 1 
    x1, y1 = np.random.multivariate_normal(mean1, cov1, n_class_instances).T #generate random multivariate normal points (100) for class 2
    #Here I suggest using different symbols for different classes of data
    plt.figure()
    plt.title('Input Dataset')
    plt.plot(x1, y1, 'v') #class 1 
    plt.plot(x2, y2, 'o') #class 2
    plt.axis('equal')
    plt.grid()
    # plt.show() #Please plot your patterns with different colours per class.


    #add the bias column
    #bias = np.ones(100, dtype=np.int) #Add bias term
    if(hasBias == True):
        bias = np.ones(n_class_instances) #bias lenght = number of instances
        #bias = np.ones(x1.shape)*np.random.rand(1) #Add bias term, a random value below 1

        pos = np.column_stack([x1,y1,bias]) #add bias for positive class (class 1) by column
        neg = np.column_stack([x2,y2,bias]) #add bias for negative class (class 2)
    else:
        pos = np.column_stack([x1,y1]) #add x1 and y1 for positive class (class 1) by column
        neg = np.column_stack([x2,y2]) #add x2 and y2 negative class (class 2)
    all_data = np.concatenate([pos,neg]) #concatenate positive and negative class by row by default

    #print(bias)
    #print(pos)
    #print(neg)
    #print(all_data)
    #print(all_data.shape)

    """
    In problems where binary representation (0/1) is inherent, it is convenient sometimes 
    and practical to rely instead on a symmetric (−1/1) resentation of the patterns.
    """
    #define the positive class labels (target) for class 1
    pos_targets = np.ones(pos.shape[0]) #shape[0] indicates the number of rows
    #define the negative class labels (target) for class 2
    neg_targets = -1 * np.ones(neg.shape[0])
    targets = np.append(pos_targets,neg_targets)

    #add the class labels (target) column to the data
    all_samples = np.column_stack([all_data,targets])
    df_correct = pd.DataFrame(all_samples)
    df_correct.to_csv('Dataset_Unshuffled.csv',header = None, index = None)
    np.random.shuffle(all_samples)

    """
    Although this reordering (shuffling) does not matter for batch learning, it has
    implications for the speed of convergence for sequential (on-line) learning, where
    updates are made on a sample-by-sample basis. 
    """

    #print(targets)
    print("Shape of Dataset:",all_samples.shape)
    # print(all_samples)
    df = pd.DataFrame(all_samples)
    # print(df)
    if(hasBias == True):
        DatasetName = 'Dataset_HasBias.csv'
    else:
        DatasetName = 'Dataset.csv'
    df.to_csv(DatasetName, header = None, index = None)

def generation_data_non_linear(hasBias = True,mean1 = [1.0,0.3],std1 = 0.2,mean2 = [0.0, -0.1],std2 = 0.3, ndata = 100):
    #x and y position for each instance will create a 3 neuron layer (two features + bias)
    # x2, y2 = np.random.multivariate_normal(mean2, cov2, n_class_instances).T #generate random multivariate normal points (100) for class 1 
    # x1, y1 = np.random.multivariate_normal(mean1, cov1, n_class_instances).T #generate random multivariate normal points (100) for class 2
    x1a = np.random.normal(0,1,int(0.5*ndata))*std1 - mean1[0]
    x1b = np.random.normal(0,1,int(0.5*ndata))*std1 + mean1[0]
    y1  = np.random.normal(0,1,ndata)*std1 + mean1[1]
    x2  = np.random.normal(0,1,ndata)*std1 + mean2[0]
    y2  = np.random.normal(0,1,ndata)*std1 + mean2[1]
    x1 = np.concatenate((x1a,x1b))
    print('Length of x1:',len(x1))
    #Here I suggest using different symbols for different classes of data
    plt.figure()
    plt.title('Input Dataset')
    plt.plot(x1, y1, 'v') #class 1 
    plt.plot(x2, y2, 'o') #class 2
    plt.axis('equal')
    plt.grid()
    # plt.show() #Please plot your patterns with different colours per class.


    #add the bias column
    #bias = np.ones(100, dtype=np.int) #Add bias term
    if(hasBias == True):
        bias = np.ones(n_class_instances) #bias lenght = number of instances
        #bias = np.ones(x1.shape)*np.random.rand(1) #Add bias term, a random value below 1

        pos = np.column_stack([x1,y1,bias]) #add bias for positive class (class 1) by column
        neg = np.column_stack([x2,y2,bias]) #add bias for negative class (class 2)
    else:
        pos = np.column_stack([x1,y1]) #add x1 and y1 for positive class (class 1) by column
        neg = np.column_stack([x2,y2]) #add x2 and y2 negative class (class 2)
    all_data = np.concatenate([pos,neg]) #concatenate positive and negative class by row by default


    """
    In problems where binary representation (0/1) is inherent, it is convenient sometimes 
    and practical to rely instead on a symmetric (−1/1) resentation of the patterns.
    """
    #define the positive class labels (target) for class 1
    pos_targets = np.ones(pos.shape[0]) #shape[0] indicates the number of rows
    #define the negative class labels (target) for class 2
    neg_targets = -1 * np.ones(neg.shape[0])
    targets = np.append(pos_targets,neg_targets)

    #add the class labels (target) column to the data
    all_samples = np.column_stack([all_data,targets])
    df_correct = pd.DataFrame(all_samples)
    df_correct.to_csv('Dataset_Unshuffled_NonlinearlySepartable.csv',header = None, index = None)
    np.random.shuffle(all_samples)

    """
    Although this reordering (shuffling) does not matter for batch learning, it has
    implications for the speed of convergence for sequential (on-line) learning, where
    updates are made on a sample-by-sample basis. 
    """

    #print(targets)
    print("Shape of Dataset:",all_samples.shape)
    # print(all_samples)
    df = pd.DataFrame(all_samples)
    # print(df)
    if(hasBias == True):
        DatasetName = 'Dataset_HasBias_NonlinearlySepartable.csv'
    else:
        DatasetName = 'Dataset_NonlinearlySepartable.csv'
    df.to_csv(DatasetName, header = None, index = None)

def sampling(portion1, portion2, ndata, sample_protion , origianl_input_dataset, output_path_name):
    # the number of records to be sampled
    n_sampledata = ndata*2*sample_protion 
    df = pd.read_csv(origianl_input_dataset, header = None)
    df1 = df.iloc[:ndata,:]
    df2 = df.iloc[ndata:,:]
    new_df1 = df1.sample(n=None, frac=(1-portion1), replace=False, weights=None, random_state=None, axis=0)
    new_df2 = df2.sample(n=None, frac=(1-portion2), replace=False, weights=None, random_state=None, axis=0)
    new_df = pd.concat([new_df1,new_df2],ignore_index = True)
    x1 = new_df1.iloc[:,0]
    y1 = new_df1.iloc[:,1]
    x2 = new_df2.iloc[:,0]
    y2 = new_df2.iloc[:,1]
    plt.figure()
    plt.title('Sampled Dataset')
    plt.plot(x1, y1, 'v') #class 1 
    plt.plot(x2, y2, 'o') #class 2
    plt.axis('equal')
    plt.grid()
    new_df = new_df.sample(frac=1)
    new_df.to_csv(output_path_name, header = None, index = None)

def conditional_sampling(relative_portion1, relative_portion2, ndata , sample_protion , origianl_input_dataset, output_path_name):
    # the number of records to be sampled
    n_sampledata = ndata*2*sample_protion 
    df = pd.read_csv(origianl_input_dataset, header = None)
    df1 = df.iloc[:ndata,:]
    df2 = df.iloc[ndata:,:]
    counter1 = 0
    counter2 = 0
    counter = 0
    while((counter1 < n_sampledata*relative_portion1) or (counter2 < n_sampledata*relative_portion2)):
        if(counter > ndata*10):
            print("Error! No enough samples that satisfied the condition to be removed! ")
            exit()
        sample = df1.sample(n=1)
        counter += 1 
        if ((sample.iloc[0,0] < 0) and (counter1 < n_sampledata*relative_portion1)):
            counter1 += 1
            df1.drop(sample.index,inplace = True)
        elif ((sample.iloc[0,0]> 0) and (counter2 < n_sampledata*relative_portion2)):
            counter2 += 1
            df1.drop(sample.index,inplace = True)
        else:
            continue

    new_df = pd.concat([df1,df2],ignore_index = True)
    x1 = df1.iloc[:,0]
    y1 = df1.iloc[:,1]
    x2 = df2.iloc[:,0]
    y2 = df2.iloc[:,1]
    plt.figure()
    plt.title('Conditional Sampled Dataset')
    plt.plot(x1, y1, 'v') #class 1 
    plt.plot(x2, y2, 'o') #class 2
    plt.axis('equal')
    plt.grid()
    new_df.to_csv(output_path_name, header = None, index = None)


# Generation of new input dataset
generation_data(True,mean1,cov1,mean2,cov2)

# Generate nonlinearly separtable dataset
# generation_data_non_linear()

# #Subsample the nonlinearly separtable dataset
# sampling(0.25, 0.25, 100, 0.25, 'Dataset_Unshuffled_NonlinearlySepartable.csv', 'nonlinearly_separatable_subsample_1.csv')
# sampling(0.5, 0.0, 100, 0.25, 'Dataset_Unshuffled_NonlinearlySepartable.csv', 'nonlinearly_separatable_subsample_2.csv')
# sampling(0.0, 0.5, 100, 0.25, 'Dataset_Unshuffled_NonlinearlySepartable.csv', 'nonlinearly_separatable_subsample_3.csv')
# conditional_sampling(0.2,0.8, 100, 0.25, 'Dataset_Unshuffled_NonlinearlySepartable.csv', 'nonlinearly_separatable_subsample_conditional.csv')


def initWeights(dim,samples):
    #np.random.seed(25) #fixed random seed
    w = []
    for i in range(dim):
        w.append(np.random.normal(0,1/samples)) #"i" random numbers, with mean 0 and 0.01 spread (1/number of samples), very small random weights
    return w

#Dimension, one for each feature in input data + bias -> X_train.shape[1], spread (1/number of samples) -> X_train.shape[0]
"""
Before the learning phase can be executed, the weights must be initialised (have
initial values assigned). The normal procedure is to start with small random
numbers drawn from the normal distribution with zero mean. Construct a function 
to create an initial weight matrix by using random number generators built into 
programming/scripting languages. Note that the matrix must have matching dimensions.

The weights are stored in matrix W with as many columns as the dimensionality
of the input patterns and with the number of rows matching the number of the
outputs (dimensionality of the output).
"""


def accuracy(W,X,y):
    predictions = []
    correct = 0
    for i in range(len(y)):        
        pred = np.dot(W,X[i])
        #print(pred)
        if pred>=0:
            predictions.append(1)
        else:
            predictions.append(-1)
    #print(predictions)
    #print(y)
    
    for i in range(len(predictions)):
        if predictions[i]==y[i]:
            correct+=1 
    acc = correct/len(y)
    return acc

def accuracy_separately(W,X,y):
    predictions = []
    #class 1
    correct_1 = 0
    #class 2
    correct_2 = 0
    for i in range(len(y)):        
        pred = np.dot(W,X[i])
        #print(pred)
        if pred>=0:
            predictions.append(1)
        else:
            predictions.append(-1)
    #print(predictions)
    #print(y)
    
    for i in range(len(predictions)):
        if predictions[i]==y[i]:
            if(y[i] == 1):      
                correct_1 += 1
            else:
                correct_2 += 1

    acc1 = correct_1/np.sum(y == 1)
    acc2 = correct_2/np.sum(y == -1)
    return acc1, acc2



def Output_Result(W,X):
    predictions = []
    for i in range(X.shape[0]):        
        pred = np.dot(W,X[i])
        #print(pred)
        if pred>=0:
            predictions.append(1)
        else:
            predictions.append(-1)
    output = np.column_stack([X_train,predictions])
    df_output = pd.DataFrame(output)
    df_output.to_csv('Output_Result.csv',header = None, index = None)



def MeanSquareError(W,X,y):
    return np.sum((np.sum(np.multiply(W,X),axis = 1) - y)**2)/len(y)

def plot_MSE(MSE,title):
    plt.figure()
    plt.plot(range(epochs),MSE, '-',label="train")
    plt.title(title)
    plt.xlabel('Number of Epochs')
    plt.ylabel('MSE')
    plt.grid()
    plt.legend()

#Prints erorrs during each epoch/iteration
def printLearningCurve(trainErr,testErr,epochs,title_for_pic):
    plt.figure()
    plt.plot(range(epochs),trainErr, '-',label="train")
    plt.title(title_for_pic)
    plt.xlabel('Number of Epochs')
    plt.ylabel('Accuracy')
    plt.grid()
    plt.legend()

#Prints erorrs during each epoch/iteration
def printLearningCurve_separtely(trainErr_1,trainErr_2,testErr,epochs,title_for_pic):
    plt.figure()
    plt.plot(range(epochs),trainErr_1, '-',label="class1")
    plt.plot(range(epochs),trainErr_2, '-',label="class2")
    plt.title(title_for_pic)
    plt.xlabel('Number of Epochs')
    plt.ylabel('Accuracy')
    plt.grid()
    plt.legend()
"""
The network outputs corresponding to all input patterns can then be calculated 
by a simple matrix multiplication followed by thresholding at zero
"""
def deltarule_seq(X_train,y_train,W,step,epochs): #sequentially go through each instance in the data
  
  """
  Write your code so that the learning according to the formula above can be flex-
  ibly repeated epochs times (where 20 is a suitable number for a low-dimensional
  perceptron). Try to avoid loops as much as possible at the cost of powerful matrix
  operations (especially multiplications). Make sure that your code works for ar-
  bitrary sizes of input and output patterns and the number of training patterns.
  """
  train_acc_ep = []
  MSE_list = []
  W_list = []
  for ep in range(epochs):
    #print(ep)
    
    for i in range(X_train.shape[0]): #go through each row
      #print (X_train[i])
      #print(W)
      product = np.dot(W,X_train[i])
      #print(product)
      
      error = product - y_train[i]
      #print(error)
      
      dW = -step*np.dot(error,X_train[i]) #delta W, delta rule = n*error*x 
      W = np.add(W,dW) #W(new) = W(old) + delta W
      
      #train_acc = accuracy(W,X_train,y_train)
      #print(train_acc)
      #train_acc_ep.append(train_acc) #call with printLearningCurve(acc_delta_seq,0,epochs*n_class_instances)
      
      #print(W)
    
    #print(W)  
    train_acc = accuracy(W,X_train,y_train)
    MSE_list.append(MeanSquareError(W,X_train,y_train))
    #print(train_acc)
    train_acc_ep.append(train_acc) #printLearningCurve(acc_delta_seq,0,epochs)
    W_list.append(W)
    
    #"""
    #Note: It's behaving strangely if data is shuffled, sometimes it isn't converging unless learning rate step is around 0.00001
    #If we are using on-line training rather than batch training, we should usually make sure we shuffle the order of the training data each epoch.
    train_data = np.column_stack([X_train,y_train])
    np.random.shuffle(train_data)
    
    X_train = train_data[:,0:-1]
    y_train = train_data[:,-1]
    
    #print(train_data.shape)
    #print(train_data)
    #print(X_train.shape)
    #print(y_train.shape)
    #print(y_train)
    #"""
  
  return W,train_acc_ep, W_list, MSE_list

"""
The network outputs corresponding to all input patterns can then be calculated 
by a simple matrix multiplication followed by thresholding at zero
"""
def Perceptron_seq(X_train,y_train,W,step,epochs): #sequentially go through each instance in the data
  
  """
  Write your code so that the learning according to the formula above can be flex-
  ibly repeated epochs times (where 20 is a suitable number for a low-dimensional
  perceptron). Try to avoid loops as much as possible at the cost of powerful matrix
  operations (especially multiplications). Make sure that your code works for ar-
  bitrary sizes of input and output patterns and the number of training patterns.
  """
  train_acc_ep = []
  MSE_list = []
  W_list = []
  for ep in range(epochs):
    #print(ep)
    
    for i in range(X_train.shape[0]): #go through each row
      #print (X_train[i])
      #print(W)
      product = np.dot(W,X_train[i])
      #print(product)
      if(product>=0):
        result = 1
      else:
        result = -1
      error = result - y_train[i]
      #print(error)
      
      dW = -step*np.dot(error,X_train[i]) #delta W, delta rule = n*error*x 
      W = np.add(W,dW) #W(new) = W(old) + delta W
      
      #train_acc = accuracy(W,X_train,y_train)
      #print(train_acc)
      #train_acc_ep.append(train_acc) #call with printLearningCurve(acc_delta_seq,0,epochs*n_class_instances)
      
      #print(W)
    
    #print(W)  
    train_acc = accuracy(W,X_train,y_train)
    MSE_list.append(MeanSquareError(W,X_train,y_train))
    #print(train_acc)
    train_acc_ep.append(train_acc) #printLearningCurve(acc_delta_seq,0,epochs)
    W_list.append(W)
    
    #"""
    #Note: It's behaving strangely if data is shuffled, sometimes it isn't converging unless learning rate step is around 0.00001
    #If we are using on-line training rather than batch training, we should usually make sure we shuffle the order of the training data each epoch.
    train_data = np.column_stack([X_train,y_train])
    np.random.shuffle(train_data)
    
    X_train = train_data[:,0:-1]
    y_train = train_data[:,-1]
    
    #print(train_data.shape)
    #print(train_data)
    #print(X_train.shape)
    #print(y_train.shape)
    #print(y_train)
    #"""
  
  return W,train_acc_ep, W_list, MSE_list


def deltarule_batch(X_train,y_train,W,step,epochs): #apply in batch with all data
  #print(W)
  
  """
  Write your code so that the learning according to the formula above can be flex-
  ibly repeated epochs times (where 20 is a suitable number for a low-dimensional
  perceptron). Try to avoid loops as much as possible at the cost of powerful matrix
  operations (especially multiplications). Make sure that your code works for ar-
  bitrary sizes of input and output patterns and the number of training patterns.
  """
  train_acc_ep = []
  W_list = []
  MSE_list = []

  for ep in range(epochs):
    #product = np.dot(W,np.transpose(X_train)) #multiply W and patterns, transposed to match dimensions, get an array of weighted inputs
    product = np.dot(X_train,W)
    #print(product)
    """
    To get the total weight change   for the entire epoch, i.e. accounting for all 
    training patterns, the weight update contributions from all patterns should be 
    summed. Since we store the patterns as columns in X and T, we get this sum “for free” 
    when the matrixes are multiplied.
    """
    error = product-y_train
    #print(error)
    #print(error.sum())
    
    dW = -step*np.dot(error,X_train)#/len(y_train) #delta W, delta rule = n*error*x .#Note: if not normalized by len(y_train), dW grows too large and doesn't converge unless learning rate step is around 0.000001. It's so big that it will output 100 positive predictions, get around 50% accuracy, flip on next epoch to 100% negative predictions,  get around 50% accuracy and repeat again.
    W = np.add(W,dW) #W(new) = W(old) + delta W
    #print(dW)
    #print(ep)
    # print(W)
    
    train_acc = accuracy(W,X_train,y_train)
    #print(train_acc)
    MSE_list.append(MeanSquareError(W,X_train,y_train))

    train_acc_ep.append(train_acc)
    W_list.append(W)
  
  return W,train_acc_ep, W_list, MSE_list

def deltarule_batch_subsample(X_train,y_train,W,step,epochs): #apply in batch with all data
  #print(W)
  
  """
  Write your code so that the learning according to the formula above can be flex-
  ibly repeated epochs times (where 20 is a suitable number for a low-dimensional
  perceptron). Try to avoid loops as much as possible at the cost of powerful matrix
  operations (especially multiplications). Make sure that your code works for ar-
  bitrary sizes of input and output patterns and the number of training patterns.
  """
  train_acc_ep_1 = []
  train_acc_ep_2 = []
  W_list = []
  MSE_list = []
  for ep in range(epochs):
    #product = np.dot(W,np.transpose(X_train)) #multiply W and patterns, transposed to match dimensions, get an array of weighted inputs
    product = np.dot(X_train,W)
    #print(product)
    """
    To get the total weight change   for the entire epoch, i.e. accounting for all 
    training patterns, the weight update contributions from all patterns should be 
    summed. Since we store the patterns as columns in X and T, we get this sum “for free” 
    when the matrixes are multiplied.
    """
    error = product-y_train
    #print(error)
    #print(error.sum())
    
    dW = -step*np.dot(error,X_train)#/len(y_train) #delta W, delta rule = n*error*x .#Note: if not normalized by len(y_train), dW grows too large and doesn't converge unless learning rate step is around 0.000001. It's so big that it will output 100 positive predictions, get around 50% accuracy, flip on next epoch to 100% negative predictions,  get around 50% accuracy and repeat again.
    W = np.add(W,dW) #W(new) = W(old) + delta W
    #print(dW)
    #print(ep)
    # print(W)
    
    train_acc_1, train_acc_2= accuracy_separately(W,X_train,y_train)
    #print(train_acc)
    MSE_list.append(MeanSquareError(W,X_train,y_train))

    train_acc_ep_1.append(train_acc_1)
    train_acc_ep_2.append(train_acc_2)   
    W_list.append(W)
  
  return W,train_acc_ep_1, train_acc_ep_2, W_list, MSE_list

def test_subsample(inputDataset,n_class_instances,step,epochs):

    print('Input Dataset:',inputDataset)
    input_df = pd.read_csv(str(inputDataset)+'.csv',header = None)
    #split the data into train set and test set (50/50 X_train/X_test), and labels/targets (y_train/y_test)
    all_samples = np.array(input_df)
    X_train = all_samples[:n_class_instances,:3]#from row 0 to 100 and columns: x position, y position and bias, inclusive #AKA patterns
    y_train = all_samples[:n_class_instances,3] #labels from row 0 to 100 #aka targets
    X_test = all_samples[n_class_instances:,:3] #from row 100 and columns: x position, y position and bias, inclusive
    y_test = all_samples[n_class_instances:,3] #labels from row 100
    #if step is too large (0.1 or 0.001), batch will end up just jumping from all positive predictions 
    # #to all negative predictions, with accuracy around 50%, but not converge, but sequential 
    # #can converge fast being more stochastic
    # step = 0.0001 #n, The step length or learning rate η should be set to some suitable small value like 0.001. , test also with 0.0001 and 0.000001
    # #if step is very small, needs many epochs to converge, but sequential and batch behave almost exactly the same
    # epochs = 300 #iterations, (where 20 is a suitable number for a low-dimensional perceptron). , test also with 200, 2000 and 2000
    W=initWeights(X_train.shape[1],X_train.shape[0])
    print('initial weigths: ',W)
    print("Delta rule Batch")
    W_delta_batch,acc_delta_batch_1, acc_delta_batch_2, W_list_batch, MSE_batch = deltarule_batch_subsample(X_train,y_train,W,step,epochs)
    print(W_delta_batch)

    printLearningCurve_separtely(acc_delta_batch_1,acc_delta_batch_2,0,epochs,'Accuracy_Batch_Training_'+inputDataset)
    # plot_MSE(MSE_batch,'MSE_Batch_Training_'+inputDataset)

    print("Batch Test Data Accuracy")
    test_acc_batch = accuracy(W_delta_batch,X_test,y_test)
    print(test_acc_batch)
    index =['batch_x','batch_y', 'batch_bias']
    W_output_DF = pd.DataFrame(W_list_batch)
    W_output_DF.to_csv('W_output_'+inputDataset+'_.csv',header = index,index = None)

step = 0.0001
epochs = 300
# test_subsample('Dataset_HasBias_NonlinearlySepartable', n_class_instances, step, epochs)
# test_subsample('nonlinearly_separatable_subsample_1', n_class_instances, step, epochs)
# test_subsample('nonlinearly_separatable_subsample_2', n_class_instances, step, epochs)
# test_subsample('nonlinearly_separatable_subsample_3', n_class_instances, step, epochs)
# test_subsample('nonlinearly_separatable_subsample_conditional', n_class_instances, step, epochs)


input_df = pd.read_csv('Dataset_HasBias.csv',header = None)
    #split the data into train set and test set (50/50 X_train/X_test), and labels/targets (y_train/y_test)
# print(input_df)
all_samples = np.array(input_df)
# print(all_samples)
# plt.figure()
# plt.title('Input Dataset')
# plt.plot(all_samples[:n_class_instances,0], all_samples[:n_class_instances,1] , 'v') #class 1 
# plt.plot(all_samples[n_class_instances:,0] ,all_samples[n_class_instances:,1] , 'o') #class 2
# plt.axis('equal')
# plt.grid()
X_train = all_samples[:n_class_instances,:3] #from row 0 to 100 and columns: x position, y position and bias, inclusive #AKA patterns
y_train = all_samples[:n_class_instances,3] #labels from row 0 to 100 #aka targets
    #print(X_train)
    #print(X_train.shape)
    #print(y_train)
    #print(y_train.shape)


X_test = all_samples[n_class_instances:,:3] #from row 100 and columns: x position, y position and bias, inclusive
y_test = all_samples[n_class_instances:,3] #labels from row 100
    #print(X_test)
    #print(X_test.shape)
    #print(y_test)
    #print(y_test.shape)

# #if step is too large (0.1 or 0.001), batch will end up just jumping from all positive predictions 
# # #to all negative predictions, with accuracy around 50%, but not converge, but sequential 
# # #can converge fast being more stochastic
# # step = 0.0001 #n, The step length or learning rate η should be set to some suitable small value like 0.001. , test also with 0.0001 and 0.000001
# # #if step is very small, needs many epochs to converge, but sequential and batch behave almost exactly the same
# # epochs = 300 #iterations, (where 20 is a suitable number for a low-dimensional perceptron). , test also with 200, 2000 and 2000
W=initWeights(X_train.shape[1],X_train.shape[0])
print("Delta rule Batch")
W_delta_batch,acc_delta_batch, W_list_batch, MSE_batch = deltarule_batch(X_train,y_train,W,step,epochs)
print(W_delta_batch)
#print(acc_delta_batch)

printLearningCurve(acc_delta_batch,0,epochs,'Accuracy_Batch_Training')
plot_MSE(MSE_batch,'MSE_Batch_Training')



print("Batch Test Data Accuracy")
test_acc_batch = accuracy(W_delta_batch,X_test,y_test)
print(test_acc_batch)
index =['batch_x','batch_y','batch_bias']
W_output_DF = pd.DataFrame(W_list_batch)
W_output_DF.to_csv('W_output_withBias_new.csv',header = index,index = None)



# print("Delta rule Sequential")
# W_delta_seq,acc_delta_seq, W_list_seq, MSE_seq = deltarule_seq(X_train,y_train,W,step,epochs)
# print(W_delta_seq)
# #print(acc_delta_seq)
# W_output_Matrix = np.column_stack([W_list_batch,W_list_seq])
# index =['batch_x','batch_y','batch_bias','seq_x','seq_y','seq_bias']
# W_output_DF = pd.DataFrame(W_output_Matrix)
# W_output_DF.to_csv('W_output.csv',header = index,index = None)

# #printLearningCurve(acc_delta_seq,0,epochs*n_class_instances)
# printLearningCurve(acc_delta_seq,0,epochs,'Accuracy_Sequential_Training')
# plot_MSE(MSE_seq,'MSE_Sequential_Training')



# print("Sequential Test Data Accuracy")
# test_acc_seq = accuracy(W_delta_seq,X_test,y_test)
# print(test_acc_seq)


# print("Perceptron Sequential")
# W_Per_seq,acc_Per_seq, W_list_Per_seq, MSE_Per = Perceptron_seq(X_train,y_train,W,step,epochs)
# print(W_Per_seq)
# # W_output_Matrix = np.column_stack([W_list_batch,W_list_seq])
# index =['Per_x','Per_y','Per_bias']
# W_Per_DF = pd.DataFrame(W_list_Per_seq)
# W_Per_DF.to_csv('W_Per_output.csv',header = index,index = None)

# printLearningCurve(acc_Per_seq,0,epochs,'Accuracy_Percetron_Training')
# plot_MSE(MSE_Per,'MSE_Percetron_Training')

# # Test the trained network using training set
# print("Percetron Train Data Accuracy")
# train_acc_per = accuracy(W_Per_seq,X_train,y_train)
# # Output_Result(W_Per_seq,X_train)
# print(train_acc_per)


# # Test the trained network using test set
# print("Percetron Test Data Accuracy")
# test_acc_per = accuracy(W_Per_seq,X_test,y_test)
# print(test_acc_per)



plt.show()