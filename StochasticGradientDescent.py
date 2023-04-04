import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

dataframe = pd.read_csv(r'C:\Users\HP\OneDrive\Desktop\Deep Project 1\Housing Data.csv') #importing data   
data = dataframe.values
X, y = data[:, :-1], data[:, -1]          #splitting data
print(X.shape, y.shape)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30,random_state=0)       #train test split
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

class NNetwork:          #Neural Network Class
    weightsL1 = None         #contains weights between input layer and hidden layer
    weightsL2 = None         #contains weights between hidden layer and output layer
    bias = None              #contains weights of biases
    Loss = []                #contains losses of on going epoch
    netL1 = None             #contains net values of hidden layer neurons
    netL2 = None             #contains net values of output layer neurons   i.e 1
    out1 = None              #contains output of first neuron of hidden layer
    out2 = None              #contains output of second neuron of hidden layer
    out3 = None              #contains output of third neuron of hidden layer
    outL1 = None             #contains output values of hidden layer neurons in an array
    outL2 = None             #contains output value of output layer neuron
    lossavg  = []            #contains average loss of all epochs
    def __init__(self,Layers,X_train,y_train,X_test,y_test):
        self.Layers=Layers
        self.X_train = X_train
        self.m,n = X_train.shape
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        np.random.seed(0)                            #assigning weights
        self.weightsL1 = np.random.rand(3,13)
        self.weightsL2 = np.random.rand(1,3)
        self.bias = np.random.rand(2,1)
       
        
    def Relu(self,x):                                #activation function
        if x<=0:
            return 0
        else:
            return x
    
    def relu_derivative(self,z):                    #derivative of activation function
        if(z>0):
            return 1
        else:
            return 0
        
    def forwardpass(self,i,TrainFlag):              #forward pass function
        
        #for training
        
        if TrainFlag == 1:
            self.netL1 = np.dot(self.weightsL1,X_train[i]) + self.bias[0]
            # self.netL1 = np.round(self.netL1,8)
            self.out1 = self.Relu(self.netL1[0])
            self.out2 = self.Relu(self.netL1[1])
            self.out3 = self.Relu(self.netL1[2])
            self.outL1 = [self.out1,self.out2,self.out3]
            self.netL2 = np.dot(self.weightsL2,self.outL1) + self.bias[1]
            #self.netL2 = np.round(self.netL2,8)
            self.outL2 = self.Relu(self.netL2[0])
            
        #for testing
            
            
        if TrainFlag == 0:
            self.netL1 = np.dot(self.weightsL1,X_test[i]) + self.bias[0]
            # self.netL1 = np.round(self.netL1,8)
            self.out1 = self.Relu(self.netL1[0])
            self.out2 = self.Relu(self.netL1[1])
            self.out3 = self.Relu(self.netL1[2])
            self.outL1 = [self.out1,self.out2,self.out3]
            self.netL2 = np.dot(self.weightsL2,self.outL1) + self.bias[1]
            #self.netL2 = np.round(self.netL2,8)
            self.outL2 = self.Relu(self.netL2[0])
    
    def calculateLoss(self,i):         #function to calculate loss
        self.Loss.append(((self.y_train[i]-self.outL2)*(self.y_train[i]-self.outL2))/2)       #mean square loss is used
        
    def avgloss(self,epochno):
        self.lossavg.append(sum(self.Loss)/len(self.Loss))
        
        
        
    def backpropag(self,k):
        # for input Layer
        
        for i in range(0,13):
            delta = (-(self.y_train[k]-self.outL2)*self.relu_derivative(self.outL2)*self.weightsL2[0][0])*self.relu_derivative(self.netL1[0])*self.X_train[k][i]
            self.weightsL1[0][i] = self.weightsL1[0][i] + (-0.00000001*delta)
        for i in range(0,13):
            delta = (-(self.y_train[k]-self.outL2)*self.relu_derivative(self.outL2)*self.weightsL2[0][1])*self.relu_derivative(self.netL1[1])*self.X_train[k][i]
            self.weightsL1[1][i] = self.weightsL1[1][i] + (-0.00000001*delta)
        for i in range(0,13):
            delta = (-(self.y_train[k]-self.outL2)*self.relu_derivative(self.outL2)*self.weightsL2[0][2])*self.relu_derivative(self.netL1[2])*self.X_train[k][i]
            self.weightsL1[2][i] = self.weightsL1[2][i] + (-0.00000001*delta)
            
        # for hidden layer
        
        for i in range(0,3):
            delta = (-(self.y_train[k]-self.outL2)*self.relu_derivative(self.outL2))
            self.weightsL2[0][i] = self.weightsL2[0][i] + (-0.00000001*delta)
            
    def emptylossarray(self):
        self.Loss.clear()
        
n1 = NNetwork(3,X_train,y_train,X_test,y_test)  #creating a neural network


#training loop
n1.lossavg.clear()
for k in range(0,150):
    for i in range (0,354):
        n1.forwardpass(i,True)
        n1.calculateLoss(i)
        n1.backpropag(i)
    n1.avgloss(k-1)
    print("Epoch # "+str(k+1)+" Loss = "+str(n1.lossavg[k])+" \n")
    n1.emptylossarray()

#training graph

plt.plot(range(1,151),n1.lossavg)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training Graph")
plt.show()

#testing loop
n1.lossavg.clear()
n1.emptylossarray()
for i in range(0,152):
    print("Input # "+str(i+1)+" \n")
    n1.forwardpass(i,False)
    print("actual output = "+str(n1.y_train[i])+" \n")
    print("predicted output = "+str(n1.outL2)+" \n")
    n1.calculateLoss(i)

#testing graph
plt.plot(range(0,152),n1.Loss)
plt.xlabel("Inputs")
plt.ylabel("Loss")
plt.title("Testing Graph")
plt.show()
