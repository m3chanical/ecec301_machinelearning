import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report,confusion_matrix

'''

This is a Dark/Light example modified by Carl.

First, ran Dark/Light example 3 with no modifications and observed the output. Original MLPClassifier:

MLPClassifier(activation='relu', alpha=0.0001, batch_size='auto', beta_1=0.9,
       beta_2=0.999, early_stopping=False, epsilon=1e-08,
       hidden_layer_sizes=4, learning_rate='constant',
       learning_rate_init=0.01, max_iter=200, momentum=0.9,
       nesterovs_momentum=True, power_t=0.5, random_state=None,
       shuffle=True, solver='lbfgs', tol=0.0001, validation_fraction=0.1,
       verbose=False, warm_start=False)

*** 1) Next, add 'e' to the np.array in the first part, and modify further code to suit. ***
Essentially not much change.
New output:

The length of X_with_repeats is:  1056
MLPClassifier(activation='relu', alpha=0.0001, batch_size='auto', beta_1=0.9,
       beta_2=0.999, early_stopping=False, epsilon=1e-08,
       hidden_layer_sizes=4, learning_rate='constant',
       learning_rate_init=0.01, max_iter=200, momentum=0.9,
       nesterovs_momentum=True, power_t=0.5, random_state=None,
       shuffle=True, solver='lbfgs', tol=0.0001, validation_fraction=0.1,
       verbose=False, warm_start=False)
Here are the model's predictions after training:
[1 1 1 1 1 1 1 0 1 1 1 1 1 0 1 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 1
 0 1 0 1 0 1 1 1 1 1 1 1 1 1 1 0 1 0 1 1 1 1 1 1 0 1 1 1 1 1 0 1 0 0 1 1 1
 1 1 1 1 1 1 1 1 1 1 0 1 0 1 1 0 1 1 1 0 1 1 0 1 1 1 1 0 0 1 1 1 1 1 0 1 1
 1 1 1 1 0 1 0 1 1 0 0 1 1 1 1 0 1 0 1 0 1 1 1 0 1 1 1 0 1 1 1 1 1 0 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1 0 1 0 1
 1 1 0 1 0 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1 1 1 1 0 1 1 1 1 1 0 1 1 1 1 1 1 0
 1 0 1 1 0 1 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 0 1 0 1 1 1 1 1 1 0 1
 1 0 0 1 1]


CONFUSION MATRIX
[[ 49   0]
 [  0 215]]
The number of misclassified dark or light images is:  0


CLASSIFICATION REPORT
             precision    recall  f1-score   support

          0       1.00      1.00      1.00        49
          1       1.00      1.00      1.00       215

avg / total       1.00      1.00      1.00       264

*** 3) Next, changed code back, and modified the mlpclassifier to have 10 layers ***
mlp = MLPClassifier(hidden_layer_sizes=10, solver='lbfgs', learning_rate_init=0.01)
And then 100, then 1000. There was very little change to the output. 

*** 4) Changing solver to 'sgd' (which the API documentation refers to as the stochastic gradient descent.) produced an
interesting effect. stochastic gradient descent didn't work so well for this type of dataset. the time of execution 
wasn't measured, but it took significantly longer to finish.*** 

CONFUSION MATRIX
[[ 4 19]
 [ 0 45]]
The number of misclassified dark or light images is:  19


CLASSIFICATION REPORT
             precision    recall  f1-score   support

          0       1.00      0.17      0.30        23
          1       0.70      1.00      0.83        45

avg / total       0.80      0.72      0.65        68

 *** 5) removed the line to make boolean dark/light decision and instead retain the sum that produced values 1-4. 
 The classifier performed equally well in this case. 

CONFUSION MATRIX
[[ 5  0  0  0  0  0]
 [ 0 47  0  0  0  0]
 [ 0  0 72  0  0  0]
 [ 0  0  0 79  0  0]
 [ 0  0  0  0 55  0]
 [ 0  0  0  0  0  6]]
The number of misclassified dark or light images is:  0


CLASSIFICATION REPORT
             precision    recall  f1-score   support

          0       1.00      1.00      1.00         5
          1       1.00      1.00      1.00        47
          2       1.00      1.00      1.00        72
          3       1.00      1.00      1.00        79
          4       1.00      1.00      1.00        55
          5       1.00      1.00      1.00         6

avg / total       1.00      1.00      1.00       264
'''

# START WITH EXAMPLE 1 -->  then EXAMPLE 2 -->  then EXAMPLE 3

# Step 1 - Create a numpy array of measurements X.
# Instead of 30, there will be only 4 features - one for each element in the 2x2 array of light and dark cells.

zero_one = [0, 1]
X = [np.array([a, b, c, d, e]) for a in zero_one for b in zero_one for c in zero_one for d in zero_one for e in zero_one]
# for x in X: print x

X_with_repeats = []
for x in X:
    X_with_repeats.append(x)
    for y in X:
        X_with_repeats.append(y)

print "The length of X_with_repeats is: ", len(X_with_repeats)
X = np.array(X_with_repeats)  # convert to a 2D np array


# Step 2 - Create a numpy array y with the targets or dark/light classification.

y = [sum(x) for x in X]
# print y

def dark_or_light(n):
    if n in [0, 1]: return 0
    else:
        return 1

#y = [dark_or_light(n) for n in y]
# print y

# * * * * * NEW STUFF FOR EXAMPLE 2 STARTS HERE
# Let's split our data into training and testing sets, this is done easily with SciKit Learn's
# train_test_split function from model_selection:

X_train, X_test, y_train, y_test = train_test_split(X, y)

# print "\nHere is the X_train data:\n", X_train

# print "\nHere is the X_test data:\n", X_test

# 2B * * * * * * * * * *  Data Preprocessing * * * * * * * * * * * *

"""
Data Preprocessing
    The neural network may have difficulty converging before the maximum number of iterations allowed if the data is not normalized. 
    Multi-layer Perceptron is sensitive to feature scaling, so it is highly recommended to scale your data. 
    Note that you must apply the same scaling to the test set for meaningful results. 
    There are a lot of different methods for normalization of data, we will use the built-in StandardScaler for standardization.
"""

#
# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
# # Fit only to the training data
# scaler.fit(X_train)
#
# print "\n\nWhat type of thing is scaler? ", type(scaler)
# print scaler
#
# # Now apply the transformations to the data:
# X_train = scaler.transform(X_train)
# X_test = scaler.transform(X_test)

# print "\nHere is the transformed X_train data:\n", X_train
# print "\nHere is the transformed X_test data:\n", X_test


# * * * * * NEW STUFF FOR EXAMPLE 32 STARTS HERE

# * * * * * * * * * * * * * * * NEW * * * * * * * * * * * * * * * * * *
# 3A * * * * * * * * * *  Train the Model * * * * * * * * * * * *

"""
Training the model
    Now it is time to train our model. SciKit Learn makes this incredibly easy, by using estimator objects. 
    In this case we will import our estimator (the Multi-Layer Perceptron Classifier model) 
    from the neural_network library of SciKit-Learn!

Next we create an instance of the model, there are a lot of parameters you can choose to define and customize here, 
    we will only define the hidden_layer_sizes. 
    For this parameter you pass in a tuple consisting of the number of neurons you want at each layer, 
    where the nth entry in the tuple represents the number of neurons in the nth layer of the MLP model. 
    There are many ways to choose these numbers;  - - explore!!
"""

mlp = MLPClassifier(hidden_layer_sizes=1000, solver='lbfgs', learning_rate_init=0.001)

# Now that the model has been made we can fit the training data to our model,
# remember that this data has already been processed and scaled:

fitted_model = mlp.fit(X_train,y_train)

print fitted_model

# You can see the output that shows the default values of the other parameters in the model.
# Play around with them and discover what effects they have on your model!

"""  3B
Predictions and Evaluation
    Now that we have a model it is time to use it to get predictions! 
    We can do this simply with the predict() method off of our fitted model:
"""

predictions = mlp.predict(X_test)
print "Here are the model's predictions after training:\n", predictions

"""
    Now we can use SciKit-Learn's built in metrics such as a classification report 
    and confusion matrix to evaluate how well our model performed:
"""

print "\n\nCONFUSION MATRIX"

Confusion_Matrix = confusion_matrix(y_test, predictions) # it's a numpy array

print(Confusion_Matrix)

number_misclassified = Confusion_Matrix[0][1] + Confusion_Matrix[1][0]

print "The number of misclassified dark or light images is: ", number_misclassified
print "\n\nCLASSIFICATION REPORT"
print(classification_report(y_test, predictions))




