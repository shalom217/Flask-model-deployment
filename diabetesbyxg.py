# -*- coding: utf-8 -*-
"""
Created on Sun Jun  7 20:31:40 2020

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


from xgboost import XGBClassifier
classifier=XGBClassifier()

#hyperparameter tuning
from sklearn.model_selection import RandomizedSearchCV
params={
 "learning_rate"    : [0.05, 0.10, 0.15, 0.20, 0.25, 0.30 ] ,
 "max_depth"        : [ 3, 4, 5, 6, 8, 10, 12, 15],
 "min_child_weight" : [ 1, 3, 5, 7 ],
 "gamma"            : [ 0.0, 0.1, 0.2 , 0.3, 0.4 ],
 "colsample_bytree" : [ 0.3, 0.4, 0.5 , 0.7 ]
    
}

ran=RandomizedSearchCV(classifier, params,n_iter=5,scoring='roc_auc',n_jobs=-1,cv=5,verbose=3)
ran.fit(x, y)
print('ran:',ran.best_params_)
print('ran:',ran.best_estimator_)


classifier=XGBClassifier(base_score=0.5, booster=None, colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=0.4, gamma=0.2, gpu_id=-1,
              importance_type='gain', interaction_constraints=None,
              learning_rate=0.15, max_delta_step=0, max_depth=12,
              min_child_weight=3, missing=nan, monotone_constraints=None,
              n_estimators=100, n_jobs=0, num_parallel_tree=1,
              objective='binary:logistic', random_state=0, reg_alpha=0,
              reg_lambda=1, scale_pos_weight=1, subsample=1, tree_method=None,
              validate_parameters=False, verbosity=None)#check with or without fine tuned params
classifier.fit(X_train,y_train)

y_pred=classifier.predict(X_test)

from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print(accuracy_score(y_test,y_pred))

from sklearn.model_selection import cross_val_score
acc=cross_val_score(classifier,x, y,cv=10)
print(acc.mean())


pickle.dump(classifier, open('diabyxgb_balanced_tuned_scaled.pkl','wb'))#saving the model

#evaluating the model
data2=[[7,106,92,18,0,22.7,0.235,48]]
#data2=asarray(data2)
data2=pd.DataFrame(data=data2,columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])
#data2=scaler.fit_transform(data2)
prediction2 = classifier.predict(data2)
print(prediction2)


'''
columns=[]
columns=df.columns.tolist()
columns.pop()
X_train=pd.DataFrame(data=X_train,columns=columns)
X_test=pd.DataFrame(data=X_test,columns=columns)