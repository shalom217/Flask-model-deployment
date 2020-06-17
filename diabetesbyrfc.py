# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 22:38:15 2020

@author: shalom
"""

from numpy import *
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns


import pickle

df = pd.read_csv("C:/Users/shalo/Desktop/ML stuffs/datasets/diabetes2.csv")

#EDA by pandas profiling
from pandas_profiling import ProfileReport
### To Create the Simple report quickly
#profile = ProfileReport(df, title='Pandas Profiling Report', explorative=True)
#profile.to_file("dia_output.html")

#seperating the dependent and independent features
x=df.drop('Outcome',axis=1)
y=df['Outcome']

#checking the balance of dataset
LABELS = ["Normal", "Diabeteic"]#1= diabetic,0=normal
sns.countplot(x='Outcome',data=df)
plt.title("Class Distribution")
plt.xticks(range(2), LABELS)
plt.xlabel("Class")
plt.ylabel("Frequency")
#hence data is imbalanced we do oversampling


from imblearn.over_sampling import RandomOverSampler
os =  RandomOverSampler()
x, y = os.fit_sample(x, y)

from collections import Counter#to check before and after
print('Original dataset shape {}'.format(Counter(y)))
print('Resampled dataset shape {}'.format(Counter(y_os_res)))

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(x, y,test_size=0.3,random_state=0)


#scaling the features
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.fit_transform(X_test)


#features importance
from sklearn.ensemble import ExtraTreesClassifier
model=ExtraTreesClassifier()
model.fit(X_train,y_train)
Series1=pd.Series(model.feature_importances_,index=x.columns)
Series1.plot(kind='barh')
plt.show()
#so all feature (except glucose) have more or less equal importance so we keep all the features


from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier()

#hyperparameter tuning
from sklearn.model_selection import RandomizedSearchCV
params={
    'max_depth': [80, 90, 100, 110],
    "criterion" : ['gini','entropy'],
    'bootstrap': [True,False],
    
    'max_features': [2, 3,4,5],
    'min_samples_leaf': [3, 4, 5,6],
    'min_samples_split': [8, 10, 12,15],
    'n_estimators': [100, 200, 300, 1000]
    
}

ran=RandomizedSearchCV(rfc, params,n_iter=5,scoring='roc_auc',n_jobs=-1,cv=5,verbose=3)
ran.fit(x, y)
print('ran:',ran.best_params_)
print('ran:',ran.best_estimator_)

rfc=RandomForestClassifier(bootstrap=False, ccp_alpha=0.0, class_weight=None,
                       criterion='gini', max_depth=110, max_features=5,
                       max_leaf_nodes=None, max_samples=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=3, min_samples_split=8,
                       min_weight_fraction_leaf=0.0, n_estimators=500,
                       n_jobs=None, oob_score=False, random_state=None,
                       verbose=0, warm_start=False)


 

rfc.fit(X_train,y_train)
pre=rfc.predict(X_test)

from sklearn.metrics import accuracy_score, classification_report,confusion_matrix
aa=accuracy_score(y_test,pre)
print(aa)
print(confusion_matrix(y_test,pre))
print(classification_report(y_test,pre))


from sklearn.model_selection import cross_val_score
acc=cross_val_score(rfc,x, y,cv=10)
print(acc.mean())


#evaluating the model
data1=[[0,118,84,47,230,45.8,0.551,31]]
data1=scaler.fit_transform(data1)
#data=asarray(data)
prediction1 = rfc.predict(data1)


pickle.dump(rfc, open('diaBalanced_tuned_scaled.pkl','wb'))#saving the model

#evaluating the model again
path=r"C:\Users\shalo\Desktop\ML stuffs\projects\diabetesCLASSIFIER\RandomFC\diaBalanced_tuned_notscaled.pkl"
model = pickle.load(open(path, 'rb'))
data1=[[0,118,84,47,230,45.8,0.551,31]]
prediction = model.predict(data1)
