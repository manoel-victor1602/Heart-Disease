from templates import classification
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier()

dataset = 'processed_cleveland_data.csv'
    
names = ['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal','num']

missingData_index = [11,12]
oneHotEncode_index = [2,9,17]

cm, scores, classifier = classification(classifier, dataset, names, 
                                        missingData_index=missingData_index, 
                                        oneHotEncode_index=oneHotEncode_index)

print("%.2f" %(scores.mean()*100)+ "%")