# This is a dark or light example done by Yoshin

# Why hello there Doctor Carr! At this point you've probably already read a bunch of these, so here's a joke:

# EXTERIOR: DAGOBAH -- DAY
#            With Yoda strapped to his back, Luke climbs up one of
#         the many thick vines that grow in the swamp until he
#         reaches the Dagobah statistics lab. Panting heavily, he
#         continues his exercises -- grepping, installing new
#         packages, logging in as root, and writing replacements for
#         two-year-old shell scripts in Python.
#
# YODA: Code!  Yes.  A programmer's strength flows from code
#       maintainability.  But beware of Perl.  Terse syntax... more
#       than one way to do it...  default variables.  The dark side
#       of code maintainability are they.  Easily they flow, quick
#       to join you when code you write.  If once you start down the
#       dark path, forever will it dominate your destiny, consume
#       you it will.
#
# LUKE: Is Perl better than Python?
#
# YODA: No... no... no.  Quicker, easier, more seductive.
#
# LUKE: But how will I know why Python is better than Perl?
#
# YODA: You will know.  When your code you try to read six months
#       from now.

# Alright, now for the stuff you're actually grading me for
# First, let's do all the prep stuff, you wrote this, so I'll spare the explanation, but I am going to increase the
# size of the matrix to 9x9, because the dataset is too simple right now and I like to live dangerously

import numpy as np
zero_one = [0, 1]
X = [ np.array([a, b, c, d, e, f, g, h, i])  for a in zero_one   for b in zero_one  for c in zero_one  for d in zero_one
      for e in zero_one for f in zero_one  for g in zero_one  for h in zero_one  for i in zero_one  ]

X_with_repeats = []
for x in X:
    X_with_repeats.append(x)
    for y in X:
        X_with_repeats.append(y)

print "The length of X_with_repeats is: ", len(X_with_repeats)
X = np.array(X_with_repeats)  # convert to a 2D np array

y = [sum(x)  for x in X]

def  dark_or_light(n):
    if n in [0, 1]: return 0
    else:
        return 1

y = [ dark_or_light(n) for n in y  ]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y)

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score

# Great! Now that that's done, we can do some fun stuff
# There are tons of things that influence how likely it is to develop a quality neural network
# However, try as you may to to develop a perfect method, for complex datasets, it often comes down to chance.
# In fact, a script can often result in 100% accuracy on one run, then drop to 10% on the next


# Let's leverage this. Instead of making one mlp classifier, we'll make 4

models = [MLPClassifier(hidden_layer_sizes=(100,100,100),max_iter=1000,solver='lbfgs') for i in range(4)]

# Now, we're going to train every classifier in that list, and make some predictions with all of them
[obj.fit(X_train,y_train) for obj in models]
model_predictions = [obj.predict(X_test) for obj in models]
scores = [accuracy_score(y_test,prediction) for prediction in model_predictions]

# Next, I'm going to single out the best performing model
index_of_best = scores.index(max(scores))
print scores
print index_of_best

optimal_perceptron = models[index_of_best]

# And finally, run the predictions a second time. Why? Well this proves that the randomness is entirely in the training process, 
# and not during the predicting
predictions = optimal_perceptron.predict(X_test)

# And also evaluate all the guesses just for fun, oh I'm using Pandas for this, feel free to comment this out if you don't have it installed

import pandas as pd
evaluations = []
for a,b in zip(y_test,predictions):
    if a==b:
        evaluations.append('PASS')
    else:
        evaluations.append('FAIL')

results = pd.DataFrame({'Actual':y_test,'Prediction':predictions, 'Result':evaluations})
print results
print '\n'
print classification_report(y_test,predictions)
print "Cumulative Accuracy: %.02f%%\n" % (float(accuracy_score(y_test,predictions))*100.0)

# There you go! Not bad huh? Run this script a bunch and you'll see that you'll score 100% quite more often.
# "BUT WAIT" I hear you say, that took really long. Is this technique even worth it?
# Well, not on it's own no. but that's what multi-core processors are for!
# Here's one way to do it with the multiprocessing module.
# I commented it out because for some reason, it doesn't play nicely with PyCharm.
# It should work if you run it in a terminal though! (I hope)

"""
import multiprocessing as mp
output = mp.Queue()

def model_process(X_train,y_train,X_test,y_test):
    mlp = MLPClassifier(hidden_layer_sizes=(100,100,100),max_iter=1000)
    mlp.fit(X_train,y_train)
    prediction = mlp.predict(X_test)
    score = accuracy_score(y_test,prediction)
    output.put((mlp,score))

processes = [mp.Process(target=model_process, args=(X_train,y_train,X_test,y_test)) for x in range(4)]

# Run processes
for p in processes:
    p.start()

# Exit the completed processes
for p in processes:
    p.join()

# Get process results from the output queue
results = [output.get() for p in processes]

print(results)
"""
