# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score

def oneHotEncode(X, i):
    labelencoder_X = LabelEncoder()
    X[:, i] = labelencoder_X.fit_transform(X[:, i])
    onehotencoder = OneHotEncoder(categorical_features = [i])
    X = onehotencoder.fit_transform(X).toarray()    
    
    return X

def pre_processes_y(y):
    for i in range(len(y)):
        if(int(y[i]) > 1):
            y[i] = str(1)
    
    return y

def classification(classifier, dataset, names, 
                   sep = ';', X_values = [],  y_values = [],
                   missingData_index = [], oneHotEncode_index = [], must_pre_processes_y = True):
  
    #Reading the dataset
    df = pd.read_csv(dataset, sep = sep, names = names)
    
    if(X_values == []):
        X = df.iloc[:, :-1].values
    else:
        X = df.iloc[:, X_values].values
        
    if(y_values == []):
        y = df.iloc[:, -1].values
    else:   
        y = df.iloc[:, y_values].values
        
    #Function to turn diagnostic into binary 
    if(must_pre_processes_y):
        y = pre_processes_y(y)
    
    # Taking care of missing data
    if not(missingData_index == []):
        imputer = Imputer(missing_values = 'NaN', strategy = 'most_frequent', axis = 0)
        imputer = imputer.fit(X[:, missingData_index])
        X[:, missingData_index] = imputer.transform(X[:, missingData_index])
    
    #Encoding nominal data [2,9,17]    
    if not(oneHotEncode_index == []):
        for encode in oneHotEncode_index:
            X = oneHotEncode(X, encode)
    
    #Feature Scaling
    sc_X = StandardScaler()
    X = sc_X.fit_transform(X)
    
    #Separating Training and Test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state = 1)
    
    #Fitting classifier
    classifier.fit(X_train, y_train)
    
    #Predicting with Classifier
    y_pred = classifier.predict(X_test)
    
    #Printing Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    
    #Printing Score from K-fold cross validation test
    scores = cross_val_score(classifier, X, y, scoring='accuracy', cv=10)
   
    return [cm, scores, classifier]