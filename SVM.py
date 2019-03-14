#To help us perform math operations
import numpy as np
#to plot our data and model visually
from matplotlib import pyplot as plt
%matplotlib inline

#Define data

#Input data
#[x,y,bias]
X = np.array([
    [-2,4,-1],
    [4,1,-1],
    [1,6,-1],
    [2,4,-1],
    [6,2,-1]
])

#Associated output labels - First 2 examples are labled '-1' and last 3 '1'
y = np.array([-1,-1,1,1,1])

#Plot these examples on 2D graph
#For each example
for d, sample in enumerate(X):
    #Plot the negative smaples 
    if d < 2:
        plt.scatter(sample[0],sample[1], s=120,  marker='_', linewidths=2)
    #Plot the positive samples
    else:
        plt.scatter(sample[0], sample[1], s=120, marker='+', linewidths=2)

#Print a possible hyperplane, that is seperating the two classes.
#Choose two points and draw the line between them (naive guess)
plt.plot([-2,0.6],[6,0.5])

#Perform stochastic gradient descent to learn the seperating hyperplane
#sgd (stochastic gradient descent)
def svm_sgd_plot(X,Y):
    #Initialize our SVMs weight vector with zeros (3 values)
    w = np.zeros(len(X[0]))
    #The learning rate
    eta = 1
    #How manu iterations to train
    epochs = 100000
    #Store misclassifications to plot how they change
    errors=[]

    #Training with gradient desent
    for epoch in range(1,epochs):
        error = 0
        for i, x in enumerate(X):
            #Misclassification
            if (Y[i]*np.dot(X[i],w)) < 1:
                w = w + eta *( X[i] * Y[i] + (-2 *(1/epoch)* w) )
                error = 1
            else:
                #Correct classification
                #Update weights
                w = w + eta *(-2 * (1/epoch)* w) 
        errors.append(error)

for d, sample in enumerate(X):
    #Plot the negative smaples 
    if d < 2:
        plt.scatter(sample[0],sample[1], s=120,  marker='_', linewidths=2)
    #Plot the positive samples
    else:
        plt.scatter(sample[0], sample[1], s=120, marker='+', linewidths=2)

#Add test samples
plt.scatter(2,2, s=120, marker='_', linewidths=2, color='yellow')
plt.scatter(4,3, s=120, marker='+', linewidths=2, color='blue')

#Print the hyperplane calculated by svm_sgd
x2 = [w[0],w[1],-w[1],w[0]]
x3 = [w[0],w[1],w[1],-w[0]]

x2x3 = np.array([x2,x3])
X,Y,U,V = zip(*x2x3)
ax = plt.gca()
ax.quiver(X,Y,U,V, scale=1, color='blue')

w = svm_sgd_plot(x,y)