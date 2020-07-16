import numpy as np
import csv
from sklearn import tree
from sklearn import svm
from sklearn import linear_model
from sklearn import neighbors
from sklearn import metrics
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans


answer='1'
wine = []
data = np.genfromtxt('winequality-red.csv',
                     dtype = np.float32,
                     delimiter = ";",
                     skip_header = 1)
X = data[:,0:11]
Y = data[:, 11]

def decisionTree() :
    classifier = tree.DecisionTreeClassifier(random_state = 0)
    if(answer=='1'):
        print("1. Decision tree:",end = ' ')
    elif(answer=='2'):
        print("Decision tree:")
    predict(classifier)
def SVM() :
    classifier = svm.SVC(random_state = 0)
    if(answer=='1'):
        print("2. Support vector machine:",end = ' ')
    elif(answer=='2'):
        print("Support vector machine:")
    predict(classifier)
def logisticRegression() :
    classifier = linear_model.LogisticRegression(random_state = 0)
    if(answer=='1'):
        print("3. Logistic regression:", end = ' ')
    elif(answer=='2'):
        print("Logistic regression:")
    predict(classifier)
def kNeighbors() :
    classifier = neighbors.KNeighborsClassifier(n_neighbors = 5)
    if(answer=='1'):
        print("4. k-NN classifier:", end = ' ')
    elif(answer=='2'):
        print("k-NN classifier:")
    predict(classifier)
    
def predict(classifier) :
    classifier = classifier.fit(X,Y)
    if(answer=='1'):
        predicted_class = classifier.predict(val)
        print(int(predicted_class[0]))
    elif(answer=='2') :
        predict = classifier.predict(X)
        evaluate(Y,predict)
        
def evaluate(y_true, y_pred):
    print("1. Confusion matrix")
    cm = metrics.confusion_matrix(y_true, y_pred)
    print(cm)
    acc = metrics.accuracy_score(y_true, y_pred)
    print("2. Accuracy: ",acc)
    pre = metrics.precision_score(y_true, y_pred, average=None)
    print("3. Precision: ",pre)
    recall = metrics.recall_score(y_true, y_pred, average=None)
    print("4. Recall: ", recall)
    f1 = metrics.f1_score(y_true, y_pred, average=None)
    print("5. F-measure: ",f1)
    print()


def cluster(num_cluster) :
    print("<hierarchical clustering>")
    h_cluster(num_cluster)
    print("<k-means clustering>")
    k_cluster(num_cluster)
def h_cluster(num_cluster) :
    hier = AgglomerativeClustering(n_clusters=num_cluster)
    hier.fit(X)
    array=[]
    for k in range(num_cluster) :
        array.append(0)
    for i in range(len(hier.labels_)) :
        array[hier.labels_[i]] = array[hier.labels_[i]] + 1
    clusterPrint(num_cluster,array)
def k_cluster(num_cluster) :
    k_mean = KMeans(n_clusters=num_cluster, random_state=0)
    k_mean.fit(X)
    array=[]
    for k in range(num_cluster) :
        array.append(0)
    for i in range(len(k_mean.labels_)) :
        array[k_mean.labels_[i]] = array[k_mean.labels_[i]] + 1
    clusterPrint(num_cluster,array)
    
def clusterPrint(num_cluster, array) :
    for i in range(num_cluster) :
        print("Cluster ", i, ": ",array[i])
    print()
    
def functions() :
    decisionTree()
    SVM()
    logisticRegression()
    kNeighbors()


while answer!='4' :
    val = []
    print(" [ Student ID: 1715437 ]")
    print(" [ Name: 심세령 ]")
    print()
    print("1. Predict wine quality")
    print("2. Evaluate wine prediction models")
    print("3. Cluster wines")
    print("4. Quit")
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
        functions()
        print()
        
    elif (answer=='2'):
        functions()
        print()
    
    elif(answer=='3'):
        num_cluster = int(input("Input the number of clusters: "))
        print("The number of wines in each cluster:")
        print()
        cluster(num_cluster)
        
        
    else:
        break
