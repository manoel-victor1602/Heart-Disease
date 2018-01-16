from Scripts.templates import classification, plot_confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd

classifier = KNeighborsClassifier()

names = ['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal','num']

class_names = ['Healthy','Sick']

dataset = pd.read_csv('Datasets/processed_cleveland_data.csv', sep=';', names=names)

missingData_index = [11,12]
oneHotEncode_index = [2,9,17]

cm, scores, classifier = classification(classifier, dataset, names, 
                                        missingData_index=missingData_index, 
                                        oneHotEncode_index=oneHotEncode_index)

plot_confusion_matrix(cm, class_names,
                      save=True, name='KNN_cleveland_prediction.png')

print("%.2f" %(scores.mean()*100) + "%")