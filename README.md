# Flask-model-deployment
The disease of diabetes has become very common today.
A report was published which tells more than 100 million U.S. adults are now living with diabetes or prediabetes.

![alt text](https://github.com/shalom217/Flask-model-deployment/blob/master/Screenshot_2.png)



So it becomes a primary need for anyone to make the regular checkups for the diabetes.
This classifier classifies a person is diabetic or not based on patient health and this service is hosted directly on web.
Model takes some values related to the patient are 1)Number of times pregnant, 2)Plasma glucose concentration a 2 hours in an oral glucose tolerance test, 3)Diastolic blood pressure (mm Hg), 4)Triceps skin fold thickness (mm), 5)2-Hour serum insulin (mu U/ml),6)Body mass index (weight in kg/(height in m)^2), 7)Diabetes pedigree function, 8)Age and will tell patient is diabetic or not.
Model is bulit using Random Forest Classifier for model creation and being hosted on web using flask.
Model is also built with XGB and with both RFC and XGB some variations are used.
Download the files and use by your own.
# Dataset
https://www.kaggle.com/uciml/pima-indians-diabetes-database

# Accuracy:
I used diffrent variation in this like ----
1.) Balanced and imbalanced dataset,
2.) Scaled or unscaled data,
3.) With hyperparameter tuning or not.
So accuracy was ranging from 74 to 85 percent.
The best with XGB is 84% with balanced and unscaled data with hyperparameter tuning, and
The best with Randomforest was 85% with balanced and unscaled data with hyperparameter tuning.
 

# Model deployment
Check out the model deployment over heroku platform:. https://diabetesnew.herokuapp.com/

![alt text](https://github.com/shalom217/Flask-model-deployment/blob/master/flask_api_dia.png)
