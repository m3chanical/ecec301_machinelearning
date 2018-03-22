'''
This is a Dark/Light example modified by Yigit.

ECEC 301- Project: Machine Learning Part 2 by Yigit Can Alparslan
Here, I added a timer to this dark_or_light_example3 code, modified the parameters of MLP Classifier
object ten times and reported the execution time, the number of misclassified dark or light images, and of course
I also specified which variable I modified.

Base Test (This is my reference) I change one variable at a given time of this base test, and then report the results.

*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

tested: mlp = MLPClassifier(activation='relu', alpha=0.0001, batch_size='auto', beta_1=0.9,
       beta_2=0.999, early_stopping=False, epsilon=1e-08,
       hidden_layer_sizes=4, learning_rate='constant',
       learning_rate_init=0.01, max_iter=200, momentum=0.9,
       nesterovs_momentum=True, power_t=0.5, random_state=None,
       shuffle=True, solver='lbfgs', tol=0.0001, validation_fraction=0.1,
       verbose=False, warm_start=False)
Time of Execution: 2.10217345158 seconds
The number of misclassified dark or light images is:  0

*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

Variation 1) solver:'lbfgs' -->  'adam'
Tested: mlp = MLPClassifier(hidden_layer_sizes=(4), solver='adam', learning_rate_init=0.01)
Time of execution: 2.49953750499 seconds
The number of misclassified dark or light images is:  0



solver={'lbfgs', 'sgd', 'adam'}, default 'adam'}

The default solver 'adam' works pretty well on relatively
 |      large datasets (with thousands of training samples or more) in terms of
 |      both training time and validation score.
 |      For small datasets, however, 'lbfgs' can converge faster and perform
 |      better.


To explore this, I played with the solver .I chose hidden layer size to be 4.
The solver for weight optimization is changed to adam. The time of execution is 0.351841252357 
seconds with solver chosen to be 'adam' instead of 'lbfgs' . In both cases, none of the cells was misclassified.

*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

Variation 2) hidden_layer_size: 4-->1
tested: mlp = MLPClassifier(hidden_layer_sizes=(1), solver='lbfgs', learning_rate_init=0.01)
Time of Execution: 2.33176418657 seconds
The number of misclassified dark or light images is: 0

*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*


Variation 3) warm_start=False--> True
tested: mlp = MLPClassifier(hidden_layer_sizes=(1), solver='lbfgs', learning_rate_init=0.01)
Time of Execution: 2.06082276817 seconds seconds
The number of misclassified dark or light images is: 0

*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

Variation 4) shuffle=True-->False
tested: mlp = MLPClassifier(hidden_layer_sizes=(1), solver='lbfgs', learning_rate_init=0.01,shuffle=False)
Time of Execution: 2.0157706582 seconds
The number of misclassified dark or light images is: 0

*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

Variation 5) learning_rate 0.01-->0.00001
tested: mlp = MLPClassifier(hidden_layer_sizes=(1), solver='lbfgs', learning_rate_init=0.00001)
Time of Execution: 2.05389065907 seconds
The number of misclassified dark or light images is: 0

*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

Variation 6) hidden_layer_sizes:4-->2
tested: mlp = MLPClassifier(hidden_layer_sizes=(2), solver='adam', learning_rate_init=0.01)
Time of Execution: 2.35492633259 seconds seconds
The number of misclassified dark or light images is: 0

*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

Variation 7) max_iter=200(default)-->50
tested: mlp = MLPClassifier(hidden_layer_sizes=(4), solver='adam', learning_rate_init=0.01,max_iter=50)
Time of Execution:  2.39854787932 seconds
The number of misclassified dark or light images is: 16

*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*


Variation 8) max_iter=200(default)-->50 AND hidden_layer_size=2
tested:MLPClassifier(hidden_layer_sizes=(2), solver='adam', learning_rate_init=0.01,max_iter=50)
Time of execution:2.13787923949 seconds
The number of misclassified dark of light images: 18

*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

Variation 9)activation='relu'(default)-->'tanh'
tested: MLPClassifier(hidden_layer_sizes=(4), solver='adam', learning_rate_init=0.01,activation='tanh')
Time of execution:2.07538247036 seconds
The number of misclassified dark of light images:0

*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

Variation 10) learning_rate='constant'(default)-->'adaptive'
tested: mlp = MLPClassifier(hidden_layer_sizes=(4), solver='lbfgs', learning_rate_init=0.01,learning_rate='adaptive')
Time of execution:2.29382529751 seconds
The number of misclassified dark of light images: 0

*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
'''

# START WITH EXAMPLE 1 -->  then EXAMPLE 2 -->  then EXAMPLE 3
# Step 1 - Create a numpy array of measurements X.
# Instead of 30, there will be only 4 features - one for each element in the 2x2 array of light and dark cells.

import numpy as np
zero_one = [0, 1]
X = [ np.array([a, b, c, d])  for a in zero_one   for b in zero_one  for c in zero_one  for d in zero_one  ]
# for x in X: print x

X_with_repeats = []
for x in X:
    X_with_repeats.append(x)
    for y in X:
        X_with_repeats.append(y)

print "The length of X_with_repeats is: ", len(X_with_repeats)
X = np.array(X_with_repeats)  # convert to a 2D np array


# Step 2 - Create a numpy array y with the targets or dark/light classification.

y = [sum(x)  for x in X]
# print y

def  dark_or_light(n):
    if n in [0, 1]: return 0
    else:
        return 1

y = [ dark_or_light(n) for n in y  ]
# print y


# * * * * * NEW STUFF FOR EXAMPLE 2 STARTS HERE

# Let's split our data into training and testing sets, this is done easily with SciKit Learn's
# train_test_split function from model_selection:

from sklearn.model_selection import train_test_split
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
"""
from sklearn.neural_network import MLPClassifier



"""
Next we create an instance of the model, there are a lot of parameters you can choose to define and customize here, 
    we will only define the hidden_layer_sizes. 
    For this parameter you pass in a tuple consisting of the number of neurons you want at each layer, 
    where the nth entry in the tuple represents the number of neurons in the nth layer of the MLP model. 
    There are many ways to choose these numbers;  - - explore!!
"""


mlp = MLPClassifier(hidden_layer_sizes=(4), solver='lbfgs', learning_rate_init=0.01)


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

from sklearn.metrics import classification_report,confusion_matrix
print "\n\nCONFUSION MATRIX"
Confusion_Matrix = confusion_matrix(y_test, predictions) # it's a numpy array
print(Confusion_Matrix)

number_misclassified = Confusion_Matrix[0][1] + Confusion_Matrix[1][0]
print "The number of misclassified dark or light images is: ", number_misclassified

print "\n\nCLASSIFICATION REPORT"

print(classification_report(y_test,predictions))



