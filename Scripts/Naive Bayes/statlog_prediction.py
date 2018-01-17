from Scripts.templates import classification, plot_confusion_matrix
import pandas as pd

#The csv's columns names
names = ['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal','num']

#Class names to plot in Confusion Matrix
class_names = ['Healthy','Sick']

#The dataset
dataset = pd.read_csv('Datasets/data.statlog.csv', sep=';', names=names)

#If handling with processed_cleveland_data.csv
missingData_index = []

#Both cleveland and statlog has categorical data in these indexes
oneHotEncode_index = [2,9,17]

#Create classifier
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()

#Making Confusion Matrix, Calculating Scores and Fitting the classifier 
cm, scores, classifier = classification(classifier, dataset, names, 
                                        missingData_index=missingData_index, 
                                        oneHotEncode_index=oneHotEncode_index,
                                        must_pre_processes_y=False)

#Plotting Confusion Matrix
plot_confusion_matrix(cm, class_names,
                      save=True, name='NB_statlog_prediction.png') 

#Printing the mean of the scores
print("%.2f" %(scores.mean()*100) + "%")