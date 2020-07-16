import sys
import csv
import numpy as np
import pandas as pd
import random
from sklearn.preprocessing import LabelEncoder


def preprocess(train, test):
    for dataset in [train, test]:
        #Sex
        sex_mapping={"male" : 0, "female" : 1}
        dataset['Sex'] = dataset['Sex'].map(sex_mapping)
        
        #Embarked
        dataset['Embarked'].fillna('S', inplace=True)
        dataset['Embarked'] = LabelEncoder().fit_transform(dataset['Embarked'])
        
        #Fare
        dataset['Fare'] = dataset['Fare'].fillna(dataset['Fare'].mean())
        dataset['Fare'] = dataset['Fare']//5

        #Name
        dataset['Name'] = dataset.Name.str.extract(' ([A-Za-z]+)\.')
        dataset['Name'] = dataset['Name'].replace(['Mrs','Mme'], 4)
        dataset['Name'] = dataset['Name'].replace(['Miss','Mlle', 'Ms'], 3)
        dataset['Name'] = dataset['Name'].replace('Master', 2)
        dataset['Name'] = dataset['Name'].replace('Ms', 1)
        dataset['Name'] = dataset['Name'].replace(['Capt', 'Col', 'Countess', 'Don','Dona', 'Dr', 'Jonkheer', 'Lady','Major', 'Rev', 'Sir', 'Mr'], 0)

        #Age
        '''
        dataset.loc[(dataset.Age.isnull())&(dataset.Sex==0),'Age'] = 28
        dataset.loc[(dataset.Age.isnull())&(dataset.Sex==1),'Age'] = 31
        '''
        dataset['Age'] = dataset['Age'].fillna(28)
        dataset['Age'] = dataset['Age'].astype(int)
        dataset['Age'] = dataset['Age']//5
        
    #feature selection
    train_X = train.drop(['Cabin', 'Ticket', 'Survived', 'PassengerId'], axis=1)
    train_T = train['Survived']
    train_X['bias'] = pd.Series(np.zeros(train_X.shape[0])+1)    #bias
    train_x = np.array(train_X.values)
    train_t = np.array(train_T.values)
    #test data
    test_X = test.drop(['Cabin', 'Ticket', 'PassengerId'], axis=1)
    test_X['bias'] = pd.Series(np.zeros(train_X.shape[0])+1)     #bias
    test_x = np.array(test_X.values)
    
    return (train_x, train_t, test_x)

#gradient_descent, update weight
def gradient(X, Y, learning_rate, weight, it):              # X:train_data Y:label
    parameters = X.shape[1]
    h = predict_pb(X, weight)
    dJ = np.dot(h-Y, X)/X.shape[0]                          #compute the partial derivative w.r.t wi

    for i in range(parameters):                        
        weight[i] = weight[i] - learning_rate*dJ[i]         #update wi

        if(it%1000==0):
            print("[%d]error[%d]: %f" % (it, i, dJ[i]))
            if(i==parameters-1):
                print("--------------------------------")
                if(it%1000==0):
                    print(weight)

    return weight

#probability using logistic function
def predict_pb(X, weight):
    h = np.dot(X, weight.transpose())           #z
    h = 1/(1+np.exp(-h))                        #sigmoid
    return h

#predict test data
def test_predict(X, weight):
    h = predict_pb(X, weight)
    h = np.round(h).astype(int)
    return h

#------------------------------------------------
learning_rate = 0.0001
H = []

#open csv file
train = pd.read_csv(sys.argv[1])
test = pd.read_csv(sys.argv[2])

#preprocessing
(train_x, train_t, test_x) = preprocess(train, test)

#weight initialize
#weight = np.array([random.uniform(-1,1) for i in range(train_x.shape[1])])
weight = np.array([ 0.11, 0.93, 0.82, 0.32, 0.95, 0.99, -0.08, 0.07, 0.43])

#train
for i in range(100000):
    weight = gradient(train_x, train_t, learning_rate, weight, i)


H = test_predict(test_x, weight)

test["Survived"] = pd.Series(H)

'''
print(test['PassengerId', 'Survived])
'''

#csv file
dataframe = pd.DataFrame({"PassengerId" : test["PassengerId"],
                        "Survived" : test["Survived"]})
dataframe.to_csv('1715437.csv', index=False)
