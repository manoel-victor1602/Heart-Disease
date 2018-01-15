# -*- coding: utf-8 -*-

from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from utils import pre_processes_y, change
import itertools
import matplotlib.pyplot as plt
import numpy as np

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def oneHotEncode(X, i):
    labelencoder_X = LabelEncoder()
    X[:, i] = labelencoder_X.fit_transform(X[:, i])
    onehotencoder = OneHotEncoder(categorical_features = [i])
    X = onehotencoder.fit_transform(X).toarray()    
    
    return X

def feature_target_split(df,
                         X_values = [],  y_values = [],):
    if(X_values == []):
        X = df.iloc[:, :-1].values
    else:
        X = df.iloc[:, X_values].values
        
    if(y_values == []):
        y = df.iloc[:, -1].values
    else:   
        y = df.iloc[:, y_values].values
    
    return [X, y]

def missingData(X,
                missingData_index = []):
    
    imputer = Imputer(missing_values = 'NaN', strategy = 'most_frequent', axis = 0)
    imputer = imputer.fit(X[:, missingData_index])
    X[:, missingData_index] = imputer.transform(X[:, missingData_index])

    return X

def feature_Scaling(X):
    sc_X = StandardScaler()
    X = sc_X.fit_transform(X)

    return X

def train_test_classify(classifier, X, y,
                        test_size = 0.2, random_state = 1,
                        scoring = 'accuracy', cv = 10):
    
    #Separating Training and Test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state = random_state)
    
    #Fitting classifier
    classifier.fit(X_train, y_train)
    
    #Predicting with Classifier
    y_pred = classifier.predict(X_test)
    
    #Printing Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    
    #Printing Score from K-fold cross validation test
    scores = cross_val_score(classifier, X, y, scoring=scoring, cv=cv)
    
    return [cm, scores, classifier]

'''
Obs: Must_pre_process_y argument inserted for specific problems related to the dataset 
'''
def classification(classifier, df,
                   X_values = [],  y_values = [],
                   missingData_index = [], oneHotEncode_index = [],
                   must_pre_processes_y = True):
    
    X, y = feature_target_split(df)
        
    #Function to turn diagnostic into binary 
    if(must_pre_processes_y):
        y = pre_processes_y(y)
    
    # Taking care of missing data
    if not(missingData_index == []):
        X = missingData(X,
                        missingData_index)
        
    #Encoding nominal data [2,9,17]    
    if not(oneHotEncode_index == []):
        for encode in oneHotEncode_index:
            X = oneHotEncode(X, encode)
    
    #Feature Scaling
    X = feature_Scaling(X)
    
    cm, scores, classifier = train_test_classify(classifier, X, y)
    
    return [cm, scores, classifier]