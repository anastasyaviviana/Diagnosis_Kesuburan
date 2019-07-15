import pandas as pd
import numpy as np
import category_encoders as ce
df=pd.read_csv('fertility.csv')
df.columns=['season','age','cd','acc','si','fever','alcohol','smoking','sitting','diagnosis']
# print(df)

# print(df.isnull().sum())

# labelling var_y
from sklearn.preprocessing import LabelEncoder
label=LabelEncoder()
df['diagnosis']=label.fit_transform(df['diagnosis'])
# print(label.classes_)   # ['Altered' 'Normal']

df['cd']=label.fit_transform(df['cd'])
# print(label.classes_) #['no' 'yes']

df['acc']=label.fit_transform(df['acc'])
# print(label.classes_) #['no' 'yes']

df['si']=label.fit_transform(df['si'])
# print(label.classes_) #['no' 'yes']
df['fever']=label.fit_transform(df['fever'])
# print(label.classes_) #['less than 3 months ago' 'more than 3 months ago' 'no']
df['alcohol']=label.fit_transform(df['alcohol'])
# print(label.classes_) #['every day' 'hardly ever or never' 'once a week' 'several times a day''several times a week']
df['smoking']=label.fit_transform(df['smoking'])
# print(label.classes_)  ['daily' 'never' 'occasional']


var_x=df.drop(['season','diagnosis'],axis=1)
var_y=df['diagnosis']
# print(var_x)

#splitting sklearn model selection
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(var_x,var_y,test_size=0.1)

# logistic regression
from sklearn.linear_model import LogisticRegression
modellog=LogisticRegression(solver='liblinear',multi_class='auto')      
modellog.fit(xtrain,ytrain)
# print('Score logistic = {}%'.format(modellog.score(xtest,ytest)*100))

#EXTREME RANDOM FOREST
from sklearn.ensemble import ExtraTreesClassifier
modelex=ExtraTreesClassifier(n_estimators=100)
modelex.fit(xtrain,ytrain)
# print('Score extreme random forest = {}%'.format(modelex.score(xtest,ytest)*100))

#SVM
from sklearn.svm import SVC
modelsvm=SVC(gamma='auto')
modelsvm.fit(xtrain,ytrain)
# print('Score svm = {}%'.format(modelsvm.score(xtest,ytest)*100))

#KETERANGAN VARIABEL X
# Childish diseases (no=0, yes=1)
# Accident or serious trauma (no=0, yes=1)
# Surgical intervention (no=0, yes=1)
# High fevers in the last year (less than 3 months ago=0,more than 3 months ago=1,no=2)
# Frequency of alcohol consumption (every day=0,hardly ever or never=1,once a week=2,several times a day=3,several times=4,a week=5)
# Smoking habit (daily=0,never=1,occasional=2) 


#---------------------
# prediction for Arin
#---------------------

# logistic model
hasil = modellog.predict([[29,0,0,0,2,0,0,5]])
if hasil[0]==1:
    hasil='NORMAL'
else:
    hasil='ALTERED'
print('\nArin, prediksi kesuburan: ' + hasil + ' (Logistic Regression)')

# extreme random forest
hasil = modelex.predict([[29,0,0,0,2,0,0,5]])
if hasil[0]==1:
    hasil='NORMAL'
else:
    hasil='ALTERED'
print('Arin, prediksi kesuburan: ' + hasil + ' (Extreme Random Forest)') 

# svm
hasil= modelsvm.predict([[29,0,0,0,2,0,0,5]])
if hasil[0]==1:
    hasil='NORMAL'
else:
    hasil='ALTERED'
print('Arin, prediksi kesuburan: ' + hasil + ' (SVM)') 

#-------------------
#prediction of bebi
#-------------------

# logistic model
hasil = modellog.predict([[31,0,1,0,2,5,1,24]])
if hasil[0]==1:
    hasil='NORMAL'
else:
    hasil='ALTERED'
print('\nBebi, prediksi kesuburan: ' + hasil + ' (Logistic Regression)')

# Extreme Random Forest
hasil= modelex.predict([[31,0,1,0,2,5,1,24]])
if hasil[0]==1:
    hasil='NORMAL'
else:
    hasil='ALTERED'
print('Bebi, prediksi kesuburan: ' + hasil + ' (Extreme Random Forest)')

# SVM
hasil = modelsvm.predict([[31,0,1,0,2,5,1,24]])
if hasil[0]==1:
    hasil='NORMAL'
else:
    hasil='ALTERED'
print('Bebi, prediksi kesuburan: ' + hasil + ' (SVM)')

#--------------------
# prediction for Caca
#--------------------

# logistic model
hasil= modellog.predict([[25,1,0,1,0,1,1,7]])
if hasil[0]==1:
    hasil='NORMAL'
else:
    hasil='ALTERED'
print('\nCaca, prediksi kesuburan: ' + hasil + ' (Logistic Regression)')

# Extreme Random Forest
hasil = modelex.predict([[25,1,0,1,0,1,1,7]])
if hasil[0]==1:
    hasil='NORMAL'
else:
    hasil='ALTERED'
print('Caca, prediksi kesuburan: ' + hasil + ' (Extreme Random Forest)')

# SVM
hasil = modelsvm.predict([[25,1,0,1,0,1,1,7]])
if hasil[0]==1:
    hasil='NORMAL'
else:
    hasil='ALTERED'
print('Caca, prediksi kesuburan: ' + hasil + ' (SVM)')

#---------------------
# prediction for Dini
#---------------------

# logistic model
hasil= modellog.predict([[28,0,1,1,2,1,0,24]])
if hasil[0]==1:
    hasil='NORMAL'
else:
    hasil='ALTERED'
print('\nDini, prediksi kesuburan: ' + hasil + ' (Logistic Regression)')

# Extreme Random Forest
hasil= modelex.predict([[28,0,1,1,2,1,0,24]])
if hasil[0]==1:
    hasil='NORMAL'
else:
    hasil='ALTERED'
print('Dini, prediksi kesuburan: ' + hasil+ ' (Extreme Random Forest)')

# SVM
hasil = modelex.predict([[28,0,1,1,2,1,0,24]])
if hasil[0]==1:
    hasil='NORMAL'
else:
    hasil='ALTERED'
print('Dini, prediksi kesuburan: ' +hasil + ' (SVM))')

#---------------------
# prediction for Enno
#---------------------

# logistic model
hasil= modellog.predict([[42,1,0,0,2,1,1,8]])
if hasil[0]==1:
    hasil='NORMAL'
else:
    hasil='ALTERED'
print('\nEnno, prediksi kesuburan: ' + hasil + ' (Logistic Regression)')

# Extreme Random Forest
hasil= modelex.predict([[42,1,0,0,2,1,1,8]])
if hasil[0]==1:
    hasil='NORMAL'
else:
    hasil='ALTERED'
print('Enno, prediksi kesuburan: ' + hasil + ' (Extreme Random Forest)')

# SVM
hasil= modelsvm.predict([[42,1,0,0,2,1,1,8]])
if hasil[0]==1:
    hasil='NORMAL'
else:
    hasil='ALTERED'
print('Enno, prediksi kesuburan: ' + hasil + ' (SVM)')


