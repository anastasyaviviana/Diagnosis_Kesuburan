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
# print(label.classes_) # ['Altered' 'Normal']

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
# print(label.classes_) #['daily' 'never' 'occasional']


var_x=df.drop(['season','diagnosis'],axis=1)
var_y=df['diagnosis']
# print(var_x)

# print(var_y.value_counts())
    #Data dengan diagnosis altered hanya berjumlah 12, sedangkan diagnosis normal berjumlah 88
    #sehingga tidak perlu splitting data

# logistic regression
from sklearn.linear_model import LogisticRegression
modellog=LogisticRegression(solver='liblinear',multi_class='auto')      
modellog.fit(var_x,var_y)
# print('Score logistic = {}%'.format(modellog.score(xtest,ytest)*100))

#EXTREME RANDOM FOREST
from sklearn.ensemble import ExtraTreesClassifier
modelex=ExtraTreesClassifier(n_estimators=100)
modelex.fit(var_x,var_y)
# print('Score extreme random forest = {}%'.format(modelex.score(xtest,ytest)*100))

#SVM
from sklearn.svm import SVC
modelsvm=SVC(gamma='auto')
modelsvm.fit(var_x,var_y)
# print('Score svm = {}%'.format(modelsvm.score(xtest,ytest)*100))

#-----------------------------------------------------------------------------------------#
# KETERANGAN VARIABEL X :
# Childish diseases (no=0, yes=1)
# Accident or serious trauma (no=0, yes=1)
# Surgical intervention (no=0, yes=1)
# High fevers in the last year (less than 3 months ago=0,more than 3 months ago=1,no=2)
# Frequency of alcohol consumption (every day=0,hardly ever or never=1,once a week=2,several times a day=3,several times=4,a week=5)
# Smoking habit (daily=0,never=1,occasional=2) 
#------------------------------------------------------------------------------------------#

#data predict
name=['Arin','Bebi','Caca','Dini','Enno']
age=[29,31,25,28,42]
cd=[0,0,1,0,1]
acc=[0,1,0,1,0]
si=[0,1,0,1,0]
fever=[2,2,0,2,2]
alcohol=[0,5,1,1,1]
smoking=[0,1,1,0,1]
sitting=[5,24,7,24,8]

dfpredict=pd.DataFrame({
    'name':np.array(name),
    'age':np.array(age),
    'cd':np.array(cd),
    'acc':np.array(acc),
    'si':np.array(si),
    'fever':np.array(fever),
    'alcohol':np.array(alcohol),
    'smoking':np.array(smoking),
    'sitting':np.array(sitting)
})

print(dfpredict)
dfPredictX = dfpredict.drop(['name'], axis='columns')

method=['Logistic Regression','Extreme Random Forest','SVM']

for j in range(5):
    
    hasil_log=modellog.predict(dfPredictX)[0]
    if hasil_log==1:
        hasil_log='NORMAL'
    else:
        hasil_log='ALTERED'
    print('{}, prediksi kesuburan : {} (Logistic Regression)'.format(dfpredict['name'].iloc[j],hasil_log))
        
    hasil_log=modelex.predict(dfPredictX)[0]
    if hasil_log==1:
        hasil_log='NORMAL'
    else:
        hasil_log='ALTERED'
    print('{}, prediksi kesuburan : {} (Extreme Random Forest)'.format(dfpredict['name'].iloc[j],hasil_log))
        
    hasil_log=modelsvm.predict(dfPredictX)[0]
    if hasil_log==1:
        hasil_log='NORMAL'
    else:
        hasil_log='ALTERED'
    print('{}, prediksi kesuburan : {} (SVM)\n'.format(dfpredict['name'].iloc[j],hasil_log))

