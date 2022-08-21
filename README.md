# flight_fare_prediction
flight_fare_prediction
# checkin
* Project Title: “Flight_Fare_prediction”
* Project Mangement Methodology used: CRISP ML (Q)
* Scope of the Project: In this Business Problem, based on the past flight booking information details of the customer i.e., dataset given by the client, using the data  analysis & visualization, data wrangling & build the model that predicts the customer who is going to be check in to the hotel room.
*********************************************************************************************************************************************************************** 
* Step (1) - Business Understanding & Data Understanding:
* Business Objective: To predict whether customer will check in to the Hotel room or not.
* Business Constriant: To choose most significant features.
* Data Collection & Data Types:   
*            **Column**           **Count**     **Non-Null**       **Dtype**  
          0   Airline               10683         non-null           object  
          1   Date_of_Journey       10683         non-null           object
          2   Source                10683         non-null           object
          3   Destination           10683         non-null           object
          4   Route                 10682         non-null           object
          5   Dep_Time              10683         non-null           object
          6   Arrival_Time          10683         non-null           object
          7   Duration              10683         non-null           object
          8   Total_Stops           10682         non-null           object
          9   Additional_Info       10683         non-null           object
          10  Price                 10683         non-null           int64  
            
           *  RangeIndex: 10683 entries, 0 to 10682, Data columns (total 11 columns)
           *  dtypes: dtypes: int64(1), object(10)
***********************************************************************************************************************************************************************
* Step (2) - Data Preprocessing/EDA/Feature Engineering:
* Data Preprocessing:
* 1) Dummy variable creation: Using Label Encoding
* 2) Checking for zero variance features: With thresholdlimit=0
* 3) Handling missing values: Used median imputation
* 4) Replacing negative values with '0'
* 5) Scaling the input features using MinMax scaler
* 6) Extracting Month,Date,Hour & Min on Time format features
* Feature Selection: Features are extracted using Extra Tree Regressor
* EDA: Checked Muilticollinearity between input features using correlation matrix(heat map) & drop the correlated features
***********************************************************************************************************************************************************************
* Step (3) - Data Mining/Model Building & it's evaluation:
* Splitted the dataset in 1) Training data set into a)Train data set(80%) b)Validation data set(20%) & 2)Test data set
* Used Random Forest Regressor
* Evaluation Metrics: Accuracy score(On Train data: 1,Test Data:0.95)
*********************************************************************************************************************************************************************** 
* Step (4) - Model Deployment:
* Created the model file in pickle format & done randomly on testing data points & after done the deployment   
* Backend server: Flask,Frontend:HTML,CSS,Cloud Deployment:Heroku
*********************************************************************************************************************************************************************** 
* Step (5) - Attachments/Links
* GitHub: https://github.com/prahasithsai/flight_fare_prediction
* Heroku deployment:https://flight-price-preydiction.herokuapp.com/
* Used Libraries: pandas, numpy,matplotlib,seaborn,sklearn,flask.
  
