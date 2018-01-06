import pandas as pd

def oneHotEncode(X, i):
    labelencoder_X = LabelEncoder()
    X[:, i] = labelencoder_X.fit_transform(X[:, i])
    onehotencoder = OneHotEncoder(categorical_features = [i])
    X = onehotencoder.fit_transform(X).toarray()    
    
    return X

#Turn y into binary
def pre_processes_y(y):
    for i in range(len(y)):
        if(int(y[i]) > 1):
            y[i] = str(1)
    
    return y

#Changing Values from y_cleveland
def change(y):
    for i in range(len(y)):
        if(y[i] == 0):
            y[i] = 1
        else:
            y[i] = 2
            
    return y

names = ['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal','num']

#Reading Training set
cleveland_dataset = pd.read_csv('processed_cleveland_data.csv', sep=';', names = names)
X_cleveland = cleveland_dataset.iloc[:, :-1].values
y_cleveland = cleveland_dataset.iloc[:, 13].values

#Function to turn diagnostic into binary 
y_cleveland = pre_processes_y(y_cleveland)

#Function to adapt the cleveland's values to being like statlog's values
y_cleveland = change(y_cleveland)

# Taking care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'most_frequent', axis = 0)
imputer = imputer.fit(X_cleveland[:, [11,12]])
X_cleveland[:, [11,12]] = imputer.transform(X_cleveland[:, [11,12]])

#Encoding X's nominal data [2,6,12]
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
X_cleveland = oneHotEncode(X_cleveland, 2)
X_cleveland = oneHotEncode(X_cleveland, 9)
X_cleveland = oneHotEncode(X_cleveland, 17)

#Applying Feature Scaling to statlog's independent features
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_cleveland = sc_X.fit_transform(X_cleveland)

#Reading Test set (Statlog's dataset)
statlog_dataset = pd.read_csv('data.statlog.csv', sep=';', names = names)
X_statlog = statlog_dataset.iloc[:, :-1].values
y_statlog = statlog_dataset.iloc[:, 13].values

#Encoding nominal data [2,6,12]
X_statlog = oneHotEncode(X_statlog, 2)
X_statlog = oneHotEncode(X_statlog, 9)
X_statlog = oneHotEncode(X_statlog, 17)

#Feature Scaling
X_statlog = sc_X.fit_transform(X_statlog)

#Building Classifier
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier()
classifier.fit(X_cleveland, y_cleveland)

#Predicting with Classifier
y_pred = classifier.predict(X_statlog)

#Printing Confusion Matrix
from sklearn.metrics import confusion_matrix
confusion_matrix(y_statlog, y_pred)