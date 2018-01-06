import pandas as pd

def oneHotEncode(X, i):
    labelencoder_X = LabelEncoder()
    X[:, i] = labelencoder_X.fit_transform(X[:, i])
    onehotencoder = OneHotEncoder(categorical_features = [i])
    X = onehotencoder.fit_transform(X).toarray()    
    
    return X

names = ['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal','num']

dataset = pd.read_csv('data.statlog.csv', sep=';', names = names)
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, 13].values

#Encoding nominal data [2,6,12]
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
X = oneHotEncode(X, 1)
X = oneHotEncode(X, 8)
X = oneHotEncode(X, 16)

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

X_opt = X

regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()