import pandas as pd
from templates import classification

names = ['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal','num']

df = pd.read_csv('cleveland_statlog_data.csv', sep=';', names = names)

missingData_index = [11,12]
oneHotEncode_index = [2,9,17]

from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier()

cm, scores, classifier = classification(classifier, df,
                                       missingData_index=missingData_index,
                                       oneHotEncode_index=oneHotEncode_index,
                                       must_pre_processes_y=False)

print("%.2f" %(scores.mean()*100)+ "%")