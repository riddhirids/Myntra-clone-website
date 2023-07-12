# importing basic libraries
import pandas as pd
import numpy as np

# loading data
train = pd.read_csv('./train.csv', low_memory= False)
test = pd.read_csv('./test.csv', low_memory = False)

# dropping name as it's not required
train.drop('Name', axis = 1, inplace = True)
test.drop('Name', axis = 1, inplace = True)

# dropping these columns because they contain a high percentage of NULL values
train.drop('PassengerId', axis = 1, inplace = True)
train.drop('Cabin', axis = 1, inplace = True)
train.drop('Ticket', axis = 1, inplace = True)
test.drop('PassengerId', axis = 1, inplace = True)
test.drop('Cabin', axis = 1, inplace = True)
test.drop('Ticket', axis = 1, inplace = True)

# replacing null with mean
train['Age'].fillna(train['Age'].mean(), inplace = True)
test['Age'].fillna(test['Age'].mean(), inplace = True)
test['Fare'].fillna(test['Fare'].mean(), inplace = True)

# droping null values in training set
train = train.dropna()

# we have finite number of Pclass, sex, Embarked point, therefore converting them to integer data using pandas dummies
Pclass = pd.get_dummies(train['Pclass'], drop_first = True)
Pclass1 = pd.get_dummies(test['Pclass'], drop_first = True)
Sex = pd.get_dummies(train['Sex'], drop_first = True)
Sex1 = pd.get_dummies(test['Sex'], drop_first = True)
Embarked = pd.get_dummies(train['Embarked'], drop_first = True)
Embarked1 = pd.get_dummies(test['Embarked'], drop_first = True)

# adding dummies to training and testing data
train = pd.concat([train, Pclass, Sex, Embarked], axis = 1)
test = pd.concat([test, Pclass1, Sex1, Embarked1], axis = 1)

# dropping previous columns of Pclass, Sex, Embarked
train.drop(['Pclass', 'Sex', 'Embarked'], axis = 1, inplace = True)
test.drop(['Pclass', 'Sex', 'Embarked'], axis = 1, inplace = True)

# separating features and labels
y = train['Survived']
x = train.drop(['Survived'], axis = 1)

# using gradient boosting classifier as it gives maximum accuracy when tested on 20% of training set
from sklearn.ensemble import GradientBoostingClassifier
clf = GradientBoostingClassifier()
clf.fit(x, y)
pred = clf.predict(test)

# printing predictions in csv file
sam = pd.read_csv('./gender_submission.csv', low_memory=False)
sam['Survived'] = pred
sam.to_csv('pred.csv', index = False)
