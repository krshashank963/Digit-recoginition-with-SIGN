# Digit-recoginition-with-SIGN
 Training a Convolutional Neural Network on the dataset and tyring to get considerably High Accuracy. (90%++) to detect the digit with help of SIGN
 
Implementing with keras library 

Lets look at some parameters of keras library:

units: output dimensions of node

kernel_initializer: to initialize weights

activation: activation function, we use relu

input_dim: input dimension that is number of pixels in our images (4096 px)

optimizer: we use adam optimizer

Adam is one of the most effective optimization algorithms for training neural networks.

Some advantages of Adam is that relatively low memory requirements and usually works well even with little tuning of hyperparameters

loss: Cost function is same. By the way the name of the cost function is cross-entropy cost function that we use previous parts.

J=−1m∑i=0m(y(i)log(a[2](i))+(1−y(i))log(1−a[2](i)))(6)

metrics: it is accuracy.

cross_val_score: use cross validation.
epochs: number of iteration

 
 
 
