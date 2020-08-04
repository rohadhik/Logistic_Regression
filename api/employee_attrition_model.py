# Import Libraries
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from sklearn import metrics
import pickle

# load data set
hr_df = pd.read_csv('../datasets-11142-15488-HR_comma_sep.csv')

# convert the category data to numerical
cat_hr_df = pd.get_dummies(hr_df, columns=['Department','salary'])

# assign features and target
X = cat_hr_df.drop('left', axis=1)
y = cat_hr_df.left


# oversample the lower class
oversample = SMOTE()
X_smote, y_smote = oversample.fit_resample(X, y)

# split train and test
X_smote_train, X_smote_test, y_smote_train, y_smote_test = train_test_split(X_smote,y_smote,train_size=0.75)

# Fit logistic regression model
logistic_model_smote = LogisticRegression(class_weight='balanced',max_iter=1000)
logistic_model_smote = logistic_model_smote.fit(X_smote_train, y_smote_train)

# predict the target of test data
y_smote_pred = logistic_model_smote.predict(X_smote_test)


# Save the Modle to file in the current working directory
Pkl_Filename = "Pickle_LR_Model.pkl"
with open(Pkl_Filename, 'wb') as file:
    pickle.dump(logistic_model_smote, file)


# Print evaluation Matrix
accuracy_score = metrics.accuracy_score(y_smote_test, y_smote_pred)

confusion_matrix = metrics.confusion_matrix(y_smote_test, y_smote_pred)

classification_reports = metrics.classification_report(y_smote_test, y_smote_pred)