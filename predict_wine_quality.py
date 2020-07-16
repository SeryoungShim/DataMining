import numpy as np
import csv
from sklearn import tree
from sklearn import svm
from sklearn import linear_model
from sklearn import neighbors

answer=1
wine = []
data = np.genfromtxt('winequality-red.csv',
                     dtype = np.float32,
                     delimiter = ";",
                     skip_header = 1)
X = data[:,0:11]
Y = data[:, 11]

def decisionTree() :
    classifier = tree.DecisionTreeClassifier(random_state = 0)
    print("1. Decision tree:",end = ' ')
    predict(classifier)

def SVM() :
    classifier = svm.SVC(random_state = 0)
    print("2. Support vector machine:",end = ' ')
    predict(classifier)

def logisticRegression() :
    classifier = linear_model.LogisticRegression(random_state = 0)
    print("3. Logistic regression:", end = ' ')
    predict(classifier)

def kNeighbors() :
    classifier = neighbors.KNeighborsClassifier(n_neighbors = 5)
    print("4. k-NN classifier:", end = ' ')
    predict(classifier)

def predict(classifier) :
    classifier = classifier.fit(X,Y)
    predicted_class = classifier.predict(val)
    print(int(predicted_class[0]))


while answer!='2' :
    val = []
    print(" [ Student ID: 1715437 ]")
    print(" [ Name: 심세령 ]")
    print()
    print("1. Predict wine quality")
    print("2. Quit")
    print()
    answer = input(">>")
    print()

    if(answer=='1'):
        print("Input the values of a wine:")
        val.append(float(input("1. fixed acidity: ")))
        val.append(float(input("2. volatile acidity: ")))
        val.append(float(input("3. citric acid: ")))
        val.append(float(input("4. residual sugar: ")))
        val.append(float(input("5. chlorides: ")))
        val.append(float(input("6. free sulfur dioxide: ")))
        val.append(float(input("7. total sulfur dioxide: ")))
        val.append(float(input("8. density: ")))
        val.append(float(input("9. pH: ")))
        val.append(float(input("10. sulphates: ")))
        val.append(float(input("11. alcohol: ")))
        print()
        val = np.array([val])
        decisionTree()
        SVM()
        logisticRegression()
        kNeighbors()
        
    else:
        break

