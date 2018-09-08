
'''
This prediction engine builds and trains a model on user demographic data and heart_rate to predict the probability of a stroke

'''
#visualization libraries
import matplotlib
import seaborn as sns

# library imports
import numpy as np
import pandas as pd
import scipy as sc

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split #training and testing data split
from sklearn.svm import SVR
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import VarianceThreshold
from sklearn import preprocessing
from sklearn import utils
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from datetime import datetime, timedelta
from sklearn.metrics import roc_auc_score
 
import time

# Load Train and Test CSV
def train_model():

	headerNames = ["id","gender","age","hypertension","heart_disease","ever_married","work_type",
               "residence_type","avg_glucose_level","bmi","smoking_status","stroke"]
	prefix = "../dataset/"

	# ID cannot be used for prediction 
	# hence setting index_col = 0 takes care of removing ID field from dataset in both train and test dataframes.
	traindf = pd.read_csv(prefix + "train.csv", header=None, delim_whitespace=False,  names=headerNames, index_col=0,) 
	testdf = pd.read_csv(prefix + "test.csv", header=None, delim_whitespace=False,  names=headerNames, index_col=0,)
	testdf = testdf.drop('stroke', axis=1)

	#print(traindf.shape)
	#print(testdf.shape)
	#traindf.head(10)

	# gender, age, BMI to heart rate data
	headerNames = ["gender","age","height","weight","hr_max"]
	
	# ID cannot be used for prediction 
	# hence setting index_col = 0 takes care of removing ID field from dataset in both train and test dataframes.
	hrratetraindf = pd.read_csv(prefix + "demog-max-hrrate.csv", header=None, delim_whitespace=False,  names=headerNames, ) #index_col=0, 
	hrratetraindf['weight'] = hrratetraindf['weight'].astype(float)

	hrratetraindf['height'] = hrratetraindf['height'].astype(float)
	hrratetraindf.loc[hrratetraindf['height'] > 10, 'height'] = hrratetraindf['height']/100
	hrratetraindf['BMI'] = hrratetraindf['weight'] / (hrratetraindf['height'] * hrratetraindf['height'])

	#print(hrratetraindf.shape)
	hrratetraindf.head(100)

	# Set of Unique Values for stroke - it is a binary classification problem
	#print(traindf['gender'].unique())
	#print(traindf['ever_married'].unique())
	#print(traindf['work_type'].unique())
	#print(traindf['residence_type'].unique())
	#print(traindf['smoking_status'].unique())
	#print(traindf['stroke'].unique())

	traindf.columns

	# Train Data Stats
	traindf.describe()

	# stats of categorical features
	traindf.describe(include=['O'])

	# for starters, fill every nan value with mean column values across the dataset.
	#traindf = traindf.dropna() 
	#testdf = testdf.dropna() 

	#fill NaN values with 0.0 for training and test
	traindf['bmi'].fillna(0.0, inplace=True) 
	testdf['bmi'].fillna(0.0, inplace=True) 

	#traindf['gender'].fillna(traindf['gender'].dropna().mean(), inplace=True)

	#print(traindf.shape)
	#print(testdf.shape)

	# Feature Engineering - Convert Categorical Data to Numeric > gender
	# convert  to numeric

	#Train Data
	traindf['gender_numeric']  = 0.0 # default value
	traindf.loc[traindf['gender'] == 'Male', 'gender_numeric'] = 1.0
	traindf.loc[traindf['gender'] == 'Female', 'gender_numeric'] = 2.0
	traindf.loc[traindf['gender'] == 'Other', 'gender_numeric'] = 3.0
	traindf = traindf.drop('gender', axis=1)

	#Test Data
	testdf['gender_numeric']  = 0.0 # default value
	testdf.loc[testdf['gender'] == 'Male', 'gender_numeric'] = 1.0
	testdf.loc[testdf['gender'] == 'Female', 'gender_numeric'] = 2.0
	testdf.loc[testdf['gender'] == 'Other', 'gender_numeric'] = 3.0
	testdf = testdf.drop('gender', axis=1)

	# Feature Engineering - Convert Categorical Data to Numeric > ever_married
	# convert ever_married to numeric

	#Train Data
	traindf['ever_married_numeric']  = 0.0 # default value
	traindf.loc[traindf['ever_married'] == 'No', 'ever_married_numeric'] = 1.0
	traindf.loc[traindf['ever_married'] == 'Yes', 'ever_married_numeric'] = 2.0
	traindf = traindf.drop('ever_married', axis=1)

	#Test Data
	testdf['ever_married_numeric']  = 0.0 # default value
	testdf.loc[testdf['ever_married'] == 'No', 'ever_married_numeric'] = 1.0
	testdf.loc[testdf['ever_married'] == 'Yes', 'ever_married_numeric'] = 2.0
	testdf = testdf.drop('ever_married', axis=1)

	# Feature Engineering - Convert Categorical Data to Numeric > work_type
	# convert work_type to numeric
	#['children' 'Private' 'Never_worked' 'Self-employed' 'Govt_job']

	#Train Data
	traindf['work_type_numeric']  = 0.0 # default value
	traindf.loc[traindf['work_type'] == 'children', 'work_type_numeric'] = 1.0
	traindf.loc[traindf['work_type'] == 'Private', 'work_type_numeric'] = 2.0
	traindf.loc[traindf['work_type'] == 'Never_worked', 'work_type_numeric'] = 3.0
	traindf.loc[traindf['work_type'] == 'Self-employed', 'work_type_numeric'] = 4.0
	traindf.loc[traindf['work_type'] == 'Govt_job', 'work_type_numeric'] = 5.0
	traindf = traindf.drop('work_type', axis=1)

	#Test Data
	testdf['work_type_numeric']  = 0.0 # default value
	testdf.loc[testdf['work_type'] == 'children', 'work_type_numeric'] = 1.0
	testdf.loc[testdf['work_type'] == 'Private', 'work_type_numeric'] = 2.0
	testdf.loc[testdf['work_type'] == 'Never_worked', 'work_type_numeric'] = 3.0
	testdf.loc[testdf['work_type'] == 'Self-employed', 'work_type_numeric'] = 4.0
	testdf.loc[testdf['work_type'] == 'Govt_job', 'work_type_numeric'] = 5.0
	testdf = testdf.drop('work_type', axis=1)

	# Feature Engineering - Convert Categorical Data to Numeric > residence_type
	# convert residence_type to numeric
	#['Rural' 'Urban']

	#Train Data
	traindf['residence_type_numeric']  = 0.0 # default value
	traindf.loc[traindf['residence_type'] == 'Rural', 'residence_type_numeric'] = 1.0
	traindf.loc[traindf['residence_type'] == 'Urban', 'residence_type_numeric'] = 2.0
	traindf = traindf.drop('residence_type', axis=1)

	#Test Data
	testdf['residence_type_numeric']  = 0.0 # default value
	testdf.loc[testdf['residence_type'] == 'Rural', 'residence_type_numeric'] = 1.0
	testdf.loc[testdf['residence_type'] == 'Urban', 'residence_type_numeric'] = 2.0
	testdf = testdf.drop('residence_type', axis=1)


	# In[14]:



	# Feature Engineering - Convert Categorical Data to Numeric > smoking_status
	# convert smoking_status to numeric
	#[nan 'never smoked' 'formerly smoked' 'smokes']

	#Train Data
	traindf['smoking_status_numeric']  = 0.0 # default value
	traindf.loc[traindf['smoking_status'] == 'never smoked', 'smoking_status_numeric'] = 1.0
	traindf.loc[traindf['smoking_status'] == 'formerly smoked', 'smoking_status_numeric'] = 2.0
	traindf.loc[traindf['smoking_status'] == 'smokes', 'smoking_status_numeric'] = 3.0
	traindf = traindf.drop('smoking_status', axis=1)

	#Test Data
	testdf['smoking_status_numeric']  = 0.0 # default value
	testdf.loc[testdf['smoking_status'] == 'never smoked', 'smoking_status_numeric'] = 1.0
	testdf.loc[testdf['smoking_status'] == 'formerly smoked', 'smoking_status_numeric'] = 2.0
	testdf.loc[testdf['smoking_status'] == 'smokes', 'smoking_status_numeric'] = 3.0
	testdf = testdf.drop('smoking_status', axis=1)
	#print(testdf['smoking_status_numeric'].unique())


	# In[15]:


	# convert integer based columns to float
	traindf['hypertension'] = traindf['hypertension'].astype(float)
	traindf['heart_disease'] = traindf['heart_disease'].astype(float)
	traindf['stroke'] = traindf['stroke'].astype(float)

	testdf['hypertension'] = testdf['hypertension'].astype(float)
	testdf['heart_disease'] = testdf['heart_disease'].astype(float)

	# removing glucose level - not collecting data real time
	traindf = traindf.drop('avg_glucose_level', axis=1)
	testdf = testdf.drop('avg_glucose_level', axis=1)


	fig=plt.gcf()
	traindf.hist(figsize=(18, 16), alpha=0.5, bins=50)
	#plt.show()
	fig.savefig('histograms.png')

	sns.heatmap(traindf.corr(),annot=True,cmap='RdYlGn',linewidths=0.2) #data.corr()-->correlation matrix
	fig=plt.gcf()
	fig.set_size_inches(20,16)
	#plt.show()
	fig.savefig('Correlation_before.png')


	sns.heatmap(traindf.corr(),annot=True,cmap='RdYlGn',linewidths=0.2) #data.corr()-->correlation matrix
	fig=plt.gcf()
	fig.set_size_inches(20,16)
	#plt.show()
	fig.savefig('Correlation_after.png')


	# extract features from training set - all columns except 'stroke'
	train_features = traindf.loc[:, traindf.columns != 'stroke']
	print(train_features.columns)

	# extract label from training set - Approved
	train_label = traindf.loc[:, traindf.columns == 'stroke']
	train_label.columns

	# check for null valued columns
	#print("Train Data -any null ?? ")
	#print(traindf.columns[traindf.isnull().any()].tolist())
	#print("Test Data -any null ?? ")
	#print(testdf.columns[testdf.isnull().any()].tolist())
	#check test columns
	#print(testdf.columns)


	## Prediction model 
	#print(hrratetraindf.columns)
	hr_train_features = hrratetraindf.loc[:, hrratetraindf.columns != 'hr_max']
	hr_train_features= hr_train_features.drop('height',axis=1)
	hr_train_features= hr_train_features.drop('weight',axis=1)

	hr_train_features['gender'] = hr_train_features['gender'].astype(float)
	hr_train_features['age'] = hr_train_features['age'].astype(float)

	#print(hr_train_features.columns)
	#print(hr_train_features.head(10))
	# extract label from training set - Approved
	hr_train_label = hrratetraindf.loc[:, hrratetraindf.columns == 'hr_max']
	hr_train_label['hr_max'] = hr_train_label['hr_max'].astype(float)
	#print(hr_train_label.columns)
	
	hr_train_label['hr_max'].fillna(hr_train_label['hr_max'].dropna().mean(), inplace=True)

	# check for null valued columns
	#print("Train Data -any null ?? ")
	#print(hr_train_features.columns[hr_train_features.isnull().any()].tolist())
	#print("Label Data -any null ?? ")
	#print(hr_train_label.columns[hr_train_label.isnull().any()].tolist())

	# determine hr_max using hrratedata
	from sklearn.ensemble import RandomForestRegressor
	from sklearn.pipeline import make_pipeline
	from sklearn.preprocessing import StandardScaler
	from sklearn.decomposition import PCA
	hr_model = make_pipeline(StandardScaler(with_std=True, with_mean=True),  RandomForestRegressor(max_depth=5, n_estimators=98, max_features=2,
	                                                        max_leaf_nodes=7,min_samples_split=15, criterion='mse'))

	hr_model.fit(hr_train_features, hr_train_label)
	hr_train_pred = hr_model.predict(hr_train_features)
	#print(hr_train_pred)
	print ("HR RMSE :: " , np.sqrt(mean_squared_error(hr_train_label, hr_train_pred))) # Training RMSE

	## predict hr rate for original training data and plug it into training data
	print("train columns ",train_features.columns)
	hr_train_features = train_features
	hr_train_features= hr_train_features.drop('hypertension', axis=1)
	hr_train_features= hr_train_features.drop('heart_disease', axis=1)
	hr_train_features= hr_train_features.drop('ever_married_numeric', axis=1)

	hr_train_features= hr_train_features.drop('work_type_numeric', axis=1)
	hr_train_features= hr_train_features.drop('smoking_status_numeric', axis=1)
	hr_train_features= hr_train_features.drop('residence_type_numeric', axis=1)

	print("hr train columns ",hr_train_features.columns)

	train_features['predicted_hr_max']=hr_model.predict(hr_train_features)

	traindf['hr'] = train_features['predicted_hr_max']
	traindf.loc[traindf['stroke'] == 0.0, 'hr'] = train_features['predicted_hr_max'] - 70; # normal is less than 120


	train_features['hr'] = traindf['hr']
	train_features = train_features.drop('predicted_hr_max', axis=1)
	#print(train_features.columns)
	print(train_features.columns)

	#traindf.head(10)


	#Train the model with best parameters of RF
	# best params for RF using randomizedCV
	# {StandardScaler(with_std=True), PCA(n_components=10), RandomForestClassifier(max_depth=5, n_estimators=85, min_samples_split=2) #best 0.783
	#BEST PARAMETERS >>>  {'n_estimators': 87, 'min_samples_split': 5, 'max_depth': 6}
	from sklearn.pipeline import make_pipeline
	from sklearn.preprocessing import StandardScaler
	from sklearn.decomposition import PCA
	model = make_pipeline(StandardScaler(with_std=True, with_mean=True),  RandomForestClassifier(max_depth=2, n_estimators=98, max_features=2,
	                                                        max_leaf_nodes=7,min_samples_split=15, criterion='entropy'))
	#
	model.fit(train_features, train_label)
	train_pred = model.predict(train_features)
	print(metrics.accuracy_score(train_label, train_pred)) # Training Accuracy Score
	print (np.sqrt(mean_squared_error(train_label, train_pred))) # Training RMSE
	print(roc_auc_score(train_label, train_pred)) # AUC-ROC values


	#print(train_features.columns)
	#print(testdf.columns)print
	return model

def predict(model, input_dict):
	'''
	Pass the reference of the model
	and a single user data to make the prediction
	'''
	print(input_dict)
	testdata = pd.DataFrame([input_dict])
	#Predict with test data - predict probabilities
	test_pred = model.predict_proba(testdata) #test features are all in testdf

	print("****************************************************************************************")
	print("Predicted Output [probability of No Stroke, probability of Stroke]  >>>>>>>>> ",test_pred) # Predicted Values
	print("****************************************************************************************")

	return test_pred


'''
	Sample test run 
'''
if __name__=="__main__":
	'''input ={'age':'30.0', 'hypertension': '0.0', 'heart_disease':'0.0', 'bmi':'26.5', 'gender_numeric':1.0,
   	'ever_married_numeric':'1.0', 'work_type_numeric':'1.0', 'residence_type_numeric':'1.0',
   	'smoking_status_numeric':'0.0', 'hr':'120.4'}'''
	input = {'age':30.0, 'hypertension': 0.0, 'heart_disease':0.0, 'bmi':26.5, 'gender_numeric':1.0,'ever_married_numeric':1.0, 'work_type_numeric':1.0, 'residence_type_numeric':1.0,'smoking_status_numeric':0.0, 'hr':220.4}
	model = train_model()
	predict(model, input)
