# Code File :
# Project : Fight Fare Prediction
# Import the required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as  sns

# Import the Train & Test Dataset
train_data = pd.read_excel(r"C:\Users\Sai\Desktop\P3\Data_Train_lyst6947.xlsx")
test_data = pd.read_excel(r"C:\Users\Sai\Desktop\P3\Test_set_lyst5257.xlsx")

# To display all the values of columns
pd.set_option('display.max_columns', None)

# To display topmost '5' values
train_data.head()

# To display details of features
train_data.info()

# To display the count of classes/values present inside a feature
train_data['Arrival_min'].unique()

# Check for any NUll values present in the dataset
train_data.isnull().sum()

# Drops NaN values, if any present in the dataset
train_data.dropna(inplace=True)

# Exploratory Data Analysis
# To extract day & month from 'Date_of_Journey' feature & drop this feature
train_data['Journey_day'] = pd.to_datetime(train_data['Date_of_Journey'], format = '%d/%m/%Y').dt.day
train_data['Journey_month'] = pd.to_datetime(train_data['Date_of_Journey'], format = '%d/%m/%Y').dt.month
train_data.drop(['Date_of_Journey'], axis = 1, inplace = True)
train_data.columns

# Similary to extract hour, minute from 
# 'Dep_Time'
train_data['Dep_hour'] = pd.to_datetime(train_data['Dep_Time']).dt.hour
train_data['Dep_min'] = pd.to_datetime(train_data['Dep_Time']).dt.minute
train_data.drop(['Dep_Time'],axis=1,inplace=True)

# 'Arrival_Time'
train_data['Arrival_hour'] = pd.to_datetime(train_data['Arrival_Time']).dt.hour
train_data['Arrival_min'] = pd.to_datetime(train_data['Arrival_Time']).dt.minute
train_data.drop(['Arrival_Time'], axis=1,inplace=True)

# 'Duration'
duration = list(train_data['Duration'])
for i in range(len(duration)):
    if len(duration[i].split()) != 2: # To check whether it has hours or minutes
        if 'h' in duration[i]:
            duration[i] = duration[i].strip() + ' 0m'
        else:
            duration[i] = '0h ' + duration[i]
duration_hours = []
duration_mins = []
for i in range(len(duration)):
    duration_hours.append(int(duration[i].split(sep = 'h')[0])) # Extracts hours
    duration_mins.append(int(duration[i].split(sep= 'm')[0].split()[-1])) # Extracts minutes

# Adding above features to train_data
train_data['Duration_hour'] = duration_hours
train_data['Duration_mins'] = duration_mins
train_data.drop(['Duration'],axis=1,inplace=True)

# Data visualization:
plt.bar(train_data['Airline'], train_data['Price'])
plt.title('Airlines vs Fare Price');plt.xticks(rotation=90)
plt.xlabel('Airline')
plt.ylabel('Price')

# Handling categorical data:
# Onehot encoding : data in order
# Label encoding : data not in order
train_data['Dep_hour'].value_counts()
from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()
train_data['Airline'] = label_encoder.fit_transform(train_data['Airline'])
train_data['Source']= label_encoder.fit_transform(train_data['Source'])
train_data['Destination'] = label_encoder.fit_transform(train_data['Destination'])

# 'Additional_Info' has almost near zero variance feature
# 'Route' and 'Total_Stops' are gives the same infromation
train_data.drop(['Additional_Info','Route','Airline','Source','Destination'],axis=1,inplace=True)
train_data['Total_Stops'] = train_data['Total_Stops'].replace({'1 stop':1,'2 stops':2,'3 stops':3,'4 stops':4,'non-stop':0})

# Final training data
data_train = train_data 

# Rearranging columns
titles = list(data_train.columns)
titles[0],titles[1] = titles[1],titles[0]
data_train = data_train[titles]
  
# Working on Test Data set:
# Drops NaN values, if any present in the dataset
test_data.dropna(inplace=True)

# Check for any NUll values present in the dataset
test_data.isnull().sum()

# Exploratory Data Analysis
# To extract date, time from 'Date_of_Journey' feature & drop this feature
test_data['Journey_day'] = pd.to_datetime(test_data['Date_of_Journey'], format = '%d/%m/%Y').dt.day
test_data['Journey_month'] = pd.to_datetime(test_data['Date_of_Journey'], format = '%d/%m/%Y').dt.month

# Similary to extract hour, minute from 
# 'Dep_Time'
test_data['Dep_hour'] = pd.to_datetime(test_data['Dep_Time']).dt.hour
test_data['Dep_min'] = pd.to_datetime(test_data['Dep_Time']).dt.minute

# 'Arrival_Time'
test_data['Arrival_hour'] = pd.to_datetime(test_data['Arrival_Time']).dt.hour
test_data['Arrival_min'] = pd.to_datetime(test_data['Arrival_Time']).dt.minute

# 'Duration'
duration = list(test_data['Duration'])
for i in range(len(duration)):
    if len(duration[i].split()) != 2: # To check whether it has hours or minutes
        if 'h' in duration[i]:
            duration[i] = duration[i].strip() + ' 0m'
        else:
            duration[i] = '0h ' + duration[i]
duration_hours = []
duration_mins = []
for i in range(len(duration)):
    duration_hours.append(int(duration[i].split(sep = 'h')[0])) # Extracts hours
    duration_mins.append(int(duration[i].split(sep= 'm')[0].split()[-1])) # Extracts minutes
# Adding above features to train_data
test_data['Duration_hour'] = duration_hours
test_data['Duration_mins'] = duration_mins

# Handling categorical data:
# Onehot encoding : data in order
# Label encoding : data not in order
test_data['Airline'] = label_encoder.fit_transform(test_data['Airline'])
test_data['Source']= label_encoder.fit_transform(test_data['Source'])
test_data['Destination'] = label_encoder.fit_transform(test_data['Destination'])

# 'Additional_Info' has almost near zero variance feature
# 'Route' and 'Total_Stops' are gives the same infromation
test_data.drop(['Duration','Date_of_Journey','Dep_Time','Arrival_Time','Additional_Info','Route','Airline','Source','Destination'],axis=1,inplace=True)
test_data['Total_Stops'] = test_data['Total_Stops'].replace({'1 stop':1,'2 stops':2,'3 stops':3,'4 stops':4,'non-stop':0})

# Final testing data
data_test = test_data

# Scaling the input features:
def norm_func(i):
    x = (i - i.min())/(i.max() - i.min())
    return (x)



# Split into X,y datasets
X = norm_func(data_train.iloc[:,1:])
y = data_train.iloc[:,0]

# Feature Selection
plt.figure(figsize=(18,18))
sns.heatmap(train_data.corr(),annot=True,cmap='RdYlGn')
plt.show()

# Feature Extraction using ExtraTreeRegressor
from sklearn.ensemble import ExtraTreesRegressor
ftr_selection = ExtraTreesRegressor()
ftr_selection.fit(X,y)

# Visualize the important features
plt.figure(figsize=(12,8))
features_impt = pd.Series(ftr_selection.feature_importances_,index=X.columns)
features_impt.nlargest(20).plot(kind='barh')
plt.show() 

# Model Building
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42)

from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor()
model.fit(X_train,y_train)

y_prediction = model.predict(X_test)

# Evaluation Metrics for test, train data
model.score(X_test, y_prediction) 
model.score(X_train, y_train)

from sklearn import metrics
print('MAE: ', metrics.mean_absolute_error(y_test, y_prediction))
print('MSE: ', metrics.mean_squared_error(y_test, y_prediction))
print('RMSE: ',np.sqrt(metrics.mean_squared_error(y_test, y_prediction)))

X.columns
data_train.columns

# Saving the model
import pickle
pickle.dump(model,open(r'C:\Users\Sai\Desktop\P3\model.pkl','wb'))

# Load the model from disk
model = pickle.load(open(r'C:\Users\Sai\Desktop\P3\model.pkl','rb'))

# Predicting the output for the test data points
predicted_values = pd.DataFrame(data_train.iloc[0:1,1:])
output = [model.predict(predicted_values)]
output
