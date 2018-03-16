import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
import pickle

wine = pd.read_csv('wine.data', names = ["Cultivator", "Alchol", "Malic_Acid", "Ash", "Alcalinity_of_Ash", "Magnesium", "Total_phenols", "Falvanoids", "Nonflavanoid_phenols", "Proanthocyanins", "Color_intensity", "Hue", "OD280", "Proline"])


# Set up data and labels
X = wine.drop('Cultivator',axis=1)
y = wine['Cultivator']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y)

# Normalize data
scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Generate list of mlps
models = [MLPClassifier(hidden_layer_sizes=(100,100,100),max_iter=1000) for i in range(10)]

# Generate Multiple Models and Choose the best
[obj.fit(X_train,y_train) for obj in models]
model_predictions = [obj.predict(X_test) for obj in models]
scores = [accuracy_score(y_test,prediction) for prediction in model_predictions]
index_of_best = scores.index(max(scores))
print scores
print index_of_best

optimal_perceptron = models[index_of_best]
predictions = optimal_perceptron.predict(X_test)

# Evaluate
evaluations = []
for a,b in zip(y_test.values,predictions):
    if a==b:
        evaluations.append('PASS')
    else:
        evaluations.append('FAIL')


#print(confusion_matrix(y_test,predictions))
results = pd.DataFrame({'Actual':y_test.values,'Prediction':predictions, 'Result':evaluations})
print results
print '\n'
print classification_report(y_test,predictions)
print "Cumulative Accuracy: %.02f%%\n" % (float(accuracy_score(y_test,predictions))*100.0)

# Give the option to save the generated neural network
while True:
    choice = raw_input('Save network? (y/n) >> ')
    if choice == 'y':
        with open('wine_detection_network.pickle', 'wb') as f:
            pickle.dump(optimal_perceptron, f)
        break
    elif choice == 'n':
        break
    else:
        print "invalid choice, please try again >> "




