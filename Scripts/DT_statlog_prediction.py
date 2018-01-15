import pandas as pd
from Scripts.templates import classification
from sklearn.tree import DecisionTreeClassifier

classifier = DecisionTreeClassifier()
    
names = ['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal','num']

dataset = pd.read_csv('Datasets/data.statlog.csv', sep=';', names=names)

oneHotEncode_index = [2,9,17]

cm, scores, classifier = classification(classifier, dataset, names, 
                                        oneHotEncode_index=oneHotEncode_index,
                                        must_pre_processes_y=False)

print("%.2f" %(scores.mean()*100)+ "%")
print(cm)