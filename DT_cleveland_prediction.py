import pandas as pd

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

names = ['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal','num']

dataset = pd.read_csv('processed_cleveland_data.csv', sep=';', names = names)
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 13].values

#Function to turn diagnostic into binary 
y = pre_processes_y(y)

# Taking care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'most_frequent', axis = 0)
imputer = imputer.fit(X[:, [11,12]])
X[:, [11,12]] = imputer.transform(X[:, [11,12]])

#Encoding nominal data [2,6,12]
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
X = oneHotEncode(X, 2)
X = oneHotEncode(X, 9)
X = oneHotEncode(X, 17)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X = sc_X.fit_transform(X)

#Separating Training and Test sets
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state = 1)

#Building Classifier
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier()
classifier.fit(X_train, y_train)

#Predicting with Classifier
y_pred = classifier.predict(X_test)

#Printing Confusion Matrix
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, y_pred)

#Verifying statistical correlation
import statsmodels.formula.api as sm

it = [i for i in range(13)]
X_opt = X[:, it]

regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()