import os
import gc
import datetime
import re

import pandas as pd
pd.options.display.max_rows = 2000
pd.options.display.max_columns = 100

from statistics import mean

import numpy as np

import seaborn as sns
sns.set(font_scale=1.2)
sns.set_style("whitegrid")

import matplotlib.pyplot as plt
from matplotlib import __version__ as plt_version

from functools import reduce

import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import ConfusionMatrixDisplay

from sklearn import naive_bayes #Naive Bayes

from statsmodels.tsa.seasonal import seasonal_decompose

from sklearn.naive_bayes import MultinomialNB

import calendar as cl
from calendar import monthrange

from sklearn import __version__ as sk_version


from sklearn.model_selection import train_test_split #Split data in testing and training

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet

from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor

from sklearn import metrics

import pydotplus

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import plot_tree
from sklearn.model_selection import train_test_split


from six import StringIO
  
from IPython.display import Image  
from sklearn.tree import export_graphviz


import plotly.express as px
from plotly import __version__ as plotly_version

from tqdm import tqdm
from tqdm import __version__ as tqdm_version

import chart_studio.plotly as py

from plotly import __version__
import plotly.graph_objs as go
from plotly.subplots import make_subplots

import plotly
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot


# In[2]:


def Model_Performance(model,X,y):
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
    mean = X_train.mean()
    stdev = X_train.std()
        
    X_train_st = (X_train - mean)/stdev 
    X_test_st = (X_test - mean)/stdev
    
    model.fit(X_train_st,y_train) 

    y_pred_Train = model.predict(X_train_st) #Predictions
    y_pred_Test = model.predict(X_test_st) #Predictions
    
    Metrics(y_test, y_pred_Test)
    
    Predicted_Plot(y_train, y_pred_Train, y_test, y_pred_Test)
    
    Multiple_Runs(model,X, y)
    
    return


# In[3]:


def Metrics(y_test, y_pred_Test):
    print('Test Metrics:')
    print('R squared:', metrics.r2_score(y_test, y_pred_Test))
    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred_Test))  
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred_Test))  
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred_Test)))
    
    return


# In[4]:


def Predicted_Plot(y_train, y_pred_Train, y_test, y_pred_Test):

    fig, ax = plt.subplots(ncols=2, figsize=(10,4))

    ax[0].scatter(y_train, y_pred_Train)
    ax[0].grid()
    ax[0].set_xlabel('Observed Label')
    ax[0].set_ylabel('Predicted Label')
    ax[0].set_title('Training Set')

    ax[1].scatter(y_test, y_pred_Test)
    ax[1].grid()
    ax[1].set_xlabel('Observed Label')
    ax[1].set_ylabel('Predicted Label')
    ax[1].set_title('Testing Set')
    plt.show()
    
    return


# In[5]:


def Multiple_Runs(model,X, y):

    Train_MSE = [] #Empty list to Store MSEs for training data set
    Test_MSE = []  #Empty list to Store MSEs for testing data set

    Train_R2 = [] #Empty list to Store R2s for training data set
    Test_R2 = []  #Empty list to Store R2s for testing data set

    for i in tqdm(range(100)):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        mean = X_train.mean()
        stdev = X_train.std()
        
        X_train_st = (X_train - mean)/stdev 
        X_test_st = (X_test - mean)/stdev 
    
        model.fit(X_train_st, y_train) #Train the model
   
        y_pred_Train  = model.predict(X_train_st)  #Predictions on training model
        y_pred_Test   = model.predict(X_test_st)   #Predictions on testing model
    
        train_R2 = metrics.r2_score(y_train, y_pred_Train) #Obtaining the metrics
        test_R2  = metrics.r2_score(y_test, y_pred_Test)
    
        train_MSE = metrics.mean_squared_error(y_train, y_pred_Train)
        test_MSE  = metrics.mean_squared_error(y_test, y_pred_Test)
    
        Train_MSE.append(train_MSE) #Storing the metrics in the lists
        Test_MSE.append(test_MSE) 
    
        Train_R2.append(train_R2) #Storing the metrics in the lists
        Test_R2.append(test_R2)  
    
    print('Train MSE median:', np.median(Train_MSE))
    print('Test MSE median:', np.median(Test_MSE))

    print('\nTrain_R2 median:', np.median(Train_R2))
    print('Test_R2 median:', np.median(Test_R2))

    fig, ax = plt.subplots(ncols=2, figsize=(10,4))

    ax[0].boxplot([Train_MSE, Test_MSE])
    ax[0].set_xticks([1,2],minor = False)                   #setting boxplot names
    ax[0].set_xticklabels(['Train','Test'], minor = False)  #setting boxplot names
    ax[0].grid()
    ax[0].set_title('Mean Squared Error')

    ax[1].boxplot([Train_R2, Test_R2])
    ax[1].set_xticks([1,2],minor = False)
    ax[1].set_xticklabels(['Train','Test'], minor = False)
    ax[1].grid()
    ax[1].set_title('R squared')

    plt.show()

    print('Train MSE standard deviation:', np.std(Train_MSE))
    print('Test MSE standard deviation: ', np.std(Test_MSE))

    print('\nTrain_R2 standard deviation:', np.std(Train_R2))
    print('Test_R2 standard deviation: ', np.std(Test_R2))


# In[6]:


#lag function
#Lag Function
def lag_variable(variable,n_lags):
    """
    Input: Pandas Dataframe
    Output:Same dataframe with their columns lags "n_lags"
    """
    data=pd.DataFrame()
    variables_name=variable.columns.values
    for i in range(1,(n_lags+1)):
        for j in variables_name:
            name=str(j)+'lag_'+ str(i)
            variable[name]=variable[j].shift(i)
    #data = variable.dropna()  # Esto me elimina data vieja que puede ser usada. 
    data = variable
    return data


# ### Importing the data

# In[7]:


#Unemployment Data From US
unem = pd.read_csv("C:/Users/colet/Downloads/Adelphi/Optimization & Prescriptive Models/Project Data/unemployment_report.csv")
unem.index = pd.to_datetime(unem['DATE'])
unem.drop("DATE", axis = 1, inplace = True)
unem.dropna()
#unem.tail()

#GDP data
GDP = pd.read_csv("C:/Users/colet/Downloads/Adelphi/Optimization & Prescriptive Models/Project Data/GDP.csv")
GDP.index = pd.to_datetime(GDP['Date'])
GDP.drop("Date", axis = 1, inplace = True)
GDP.dropna()
#GDP.tail()

#inflation data
inf = pd.read_csv("C:/Users/colet/Downloads/Adelphi/Optimization & Prescriptive Models/Project Data/inflation_monthly.csv")
inf.index = pd.to_datetime(inf['DATE'])
inf.drop("DATE", axis = 1, inplace = True)
inf.dropna()
#inf.tail()

#construction permits
cp = pd.read_csv("C:/Users/colet/Downloads/Adelphi/Optimization & Prescriptive Models/Project Data/construction.csv")
cp.index = pd.to_datetime(cp['DATE'])
cp.drop("DATE", axis = 1, inplace = True)
cp.dropna()
#cp.tail()


#population data
pop = pd.read_csv("C:/Users/colet/Downloads/Adelphi/Optimization & Prescriptive Models/Project Data/population_growth_rate.csv")
pop.index = pd.to_datetime(pop['Date'])
pop.drop("Date", axis = 1, inplace = True)
pop.dropna()
#pop.tail()

#EFFR data
EFFR = pd.read_csv("C:/Users/colet/Downloads/Adelphi/Optimization & Prescriptive Models/Project Data/EFFR.csv")
EFFR.index = pd.to_datetime(EFFR['DATE'])
EFFR.drop("DATE", axis = 1, inplace = True)
EFFR.dropna()
#EFFR.tail()


# ### Creating data that is datetime and not datetime

# In[8]:


#Datetime Data. Doesn't have t in data. 
s_dt = pd.read_csv("C:/Users/colet/Downloads/Adelphi/Optimization & Prescriptive Models/Scaled_sales_Mansfield_WH_SW_Alto_2022.csv")
s_dt = pd.to_datetime(s_dt['Date'])
s_dt = s_dt.drop(columns = ["t"])
s_dt.head()


# In[9]:


#Not Datetime Data, has t in data, no date in data
s = pd.read_csv("C:/Users/colet/Downloads/Adelphi/Optimization & Prescriptive Models/Scaled_sales_Mansfield_WH_SW_Alto_2022.csv")
s = s.drop(columns = ["Date", "EFFR_NA", "Inflation_NA", "GDP_NA", "Unem_NA", "Permit_NA", "Pop_NA", "Orders_NA"])
s.head()


# In[10]:


#Not Datetime Data with the NA data, with t in data, no date in data
s_with_NA = pd.read_csv("C:/Users/colet/Downloads/Adelphi/Optimization & Prescriptive Models/Scaled_sales_Mansfield_WH_SW_Alto_2022.csv")
s_with_NA = s_with_NA.drop(columns = ["Date"])
s_with_NA.head()


# ### Finding best lagged variables, before anomaly detection

# In[11]:


#MinMaxing and Standardizing the datetime data
sMinMax = (s-s.min())/(s.max()-s.min())
sSTD = (s-s.mean())/s.std()


# In[12]:


#Setting X and y
X = sSTD
y = s.Orders
y.head()


# In[13]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=5)


# In[14]:


#Training the multiple regression model 

LinearReg = LinearRegression() #Creates the function
LinearReg.fit(X_train, y_train) #Train the model

y_pred_Train = LinearReg.predict(X_train) #Predictions on training model
y_pred_Test  = LinearReg.predict(X_test)  #Predictions on testing model

print('Intercept:',LinearReg.intercept_)
print('Coefficients:', LinearReg.coef_)


# In[15]:


s_vars = s.drop(columns=['Orders', 't'])
s_vars.head()


# In[16]:


#lets lag
s_vars_lag = lag_variable(s_vars,12)
s_vars_lag.head()


# In[17]:


s_vars_lag = s_vars_lag.dropna()


# In[18]:


s_total_vars = reduce(lambda left,right: pd.merge(left,right,left_index=True, right_index=True, how='left'),
                 [s_vars_lag,s])
s_total_vars.head()


# In[19]:


corr = s_total_vars.corr()['Orders'].sort_values(ascending=False)
print(corr[corr <= -0.3])
#print(corr[corr <= -0.1])
print(len(corr))
#print(total.columns)
#print(total.shape)
#print(corr)


# In[20]:


s_vars_keep = s_total_vars[["Unemlag_3","Permitlag_3","GDPlag_3","Inflationlag_2","Pop_Growthlag_2","EFFRlag_12","Orders"]].copy()
s_vars_keepT = s_total_vars[["t", "Unemlag_3","Permitlag_3","GDPlag_3","Inflationlag_2","Pop_Growthlag_2","EFFRlag_12","Orders"]].copy()
s_vars_keep = s_vars_keep.dropna()
s_vars_keepT.head()


# ##### For the lagged variables before taking out the anomalies, we have Unem3, Permit3, GDP3, Inflation2, Pop2, EFFR12. 

# ## Finding anomalies within orders and the variables, then finding most correlated variables

# ### Start by finding the anomalies

# In[21]:


import numpy as np
print('Numpy:', np.__version__)

import matplotlib.pyplot as plt
from matplotlib import __version__ as plt_v
print('Matplotlib:', np.__version__)


# In[22]:


def outliers_detection(model, name, Y):
    clf = model
    clf.fit(Y)
    
    outliers = clf.predict(Y)
    
    Y_outliers = Y[np.where(outliers==1)]
    X_outliers = X[np.where(outliers==1)]
    
    Y_inliers = Y[np.where(outliers==0)]
    X_inliers = X[np.where(outliers==0)]
    print(X_outliers)
    
    anomaly_score = model.decision_function(Y)
    plt.plot(anomaly_score)
    plt.ylabel = ('Anomaly_score')
    plt.show()
    
    plt.scatter(X_outliers, Y_outliers, edgecolor='black',color='red', label='outliers')
    plt.scatter(X_inliers, Y_inliers, edgecolor='black',color='green', label='inliers')
    plt.title(name)
    plt.legend()
    plt.grid()
#    plt.ylabel('Y') #Weird error
    plt.xlabel('X')
    plt.show()
    
    return(X_outliers)


# #### Finding anomalies in orders

# In[23]:


#Datetime Data
s_all = pd.read_csv("C:/Users/colet/Downloads/Adelphi/Optimization & Prescriptive Models/Scaled_sales_Mansfield_WH_SW_Alto_2022.csv")
#Only keeping orders
s_orders = s_all.drop(columns = ["Date","t", "Pop_Growth", "EFFR", "Inflation", "GDP", "Unem", "Permit", "Orders_NA", "Pop_NA", "EFFR_NA", "Inflation_NA", "GDP_NA", "Unem_NA", "Permit_NA"])
#Only keeping t
s_t = s_all.drop(columns = ["Date","Orders", "Pop_Growth", "EFFR", "Inflation", "GDP", "Unem", "Permit", "Orders_NA", "Pop_NA", "EFFR_NA", "Inflation_NA", "GDP_NA", "Unem_NA", "Permit_NA"])


# In[24]:


new_array = np.array(s_orders.values)
X = np.arange(0,60)

Y = new_array.reshape(-1, 1) #Changing the shape of the data. Not the scale!!!


# In[25]:


from pyod.models.knn import KNN
model = KNN()
KNN_Outliers = outliers_detection(model, 'KNN', Y)


# In[26]:


plt.plot(s_orders)


# #### Finding anomalies in Population Growth

# In[27]:


#Only keeping pop
s_pop = s_all.drop(columns = ["Date","t", "Orders", "EFFR", "Inflation", "GDP", "Unem", "Permit", "Orders_NA", "Pop_NA", "EFFR_NA", "Inflation_NA", "GDP_NA", "Unem_NA", "Permit_NA"])


# In[28]:


new_array = np.array(s_pop.values)
X = np.arange(0,60)

Y = new_array.reshape(-1, 1) #Changing the shape of the data. Not the scale!!!


# In[29]:


from pyod.models.ocsvm import OCSVM
model = OCSVM()
OCSVM_Outliers = outliers_detection(model, 'OCSVM', Y)


# In[30]:


plt.plot(s_pop)


# #### Finding anomalies in EFFR

# In[31]:


#Only keeping pop
s_effr = s_all.drop(columns = ["Date","t", "Orders", "Pop_Growth", "Inflation", "GDP", "Unem", "Permit", "Orders_NA", "Pop_NA", "EFFR_NA", "Inflation_NA", "GDP_NA", "Unem_NA", "Permit_NA"])


# In[32]:


new_array = np.array(s_effr.values)
X = np.arange(0,60)

Y = new_array.reshape(-1, 1) #Changing the shape of the data. Not the scale!!!


# In[33]:


from pyod.models.cof import COF
model = COF()
COF_Outliers = outliers_detection(model, 'COF', Y)


# In[34]:


plt.plot(s_effr)


# #### Finding anomalies in inflation

# In[35]:


#Only keeping pop
s_inf = s_all.drop(columns = ["Date","t", "Orders", "Pop_Growth", "EFFR", "GDP", "Unem", "Permit", "Orders_NA", "Pop_NA", "EFFR_NA", "Inflation_NA", "GDP_NA", "Unem_NA", "Permit_NA"])


# In[36]:


new_array = np.array(s_inf.values)
X = np.arange(0,60)

Y = new_array.reshape(-1, 1) #Changing the shape of the data. Not the scale!!!


# In[37]:


from pyod.models.ocsvm import OCSVM
model = OCSVM()
OCSVM_Outliers = outliers_detection(model, 'OCSVM', Y)


# In[38]:


plt.plot(s_inf)


# #### Finding anomalies in GDP

# In[39]:


#Only keeping pop
s_gdp = s_all.drop(columns = ["Date","t", "Orders", "Pop_Growth", "EFFR", "Inflation", "Unem", "Permit", "Orders_NA", "Pop_NA", "EFFR_NA", "Inflation_NA", "GDP_NA", "Unem_NA", "Permit_NA"])


# In[40]:


new_array = np.array(s_gdp.values)
X = np.arange(0,60)

Y = new_array.reshape(-1, 1) #Changing the shape of the data. Not the scale!!!


# In[41]:


from pyod.models.knn import KNN
model = KNN()
KNN_Outliers = outliers_detection(model, 'KNN', Y)


# In[42]:


plt.plot(s_gdp)


# #### Finding anomalies in Unemployment

# In[43]:


#Only keeping pop
s_unem = s_all.drop(columns = ["Date","t", "Orders", "Pop_Growth", "EFFR", "Inflation", "GDP", "Permit", "Orders_NA", "Pop_NA", "EFFR_NA", "Inflation_NA", "GDP_NA", "Unem_NA", "Permit_NA"])


# In[44]:


new_array = np.array(s_unem.values)
X = np.arange(0,60)

Y = new_array.reshape(-1, 1) #Changing the shape of the data. Not the scale!!!


# In[45]:


from pyod.models.mad import MAD
model = MAD()
MAD_Outliers = outliers_detection(model, 'MAD', Y)


# In[46]:


plt.plot(s_unem)


# #### Finding anomalies in Permits

# In[47]:


#Only keeping pop
s_permit = s_all.drop(columns = ["Date","t", "Orders", "Pop_Growth", "EFFR", "Inflation", "GDP", "Unem", "Orders_NA", "Pop_NA", "EFFR_NA", "Inflation_NA", "GDP_NA", "Unem_NA", "Permit_NA"])


# In[48]:


new_array = np.array(s_permit.values)
X = np.arange(0,60)

Y = new_array.reshape(-1, 1) #Changing the shape of the data. Not the scale!!!


# In[49]:


from pyod.models.lmdd import LMDD
model = LMDD()
LMDD_Outliers = outliers_detection(model, 'LMDD', Y)


# In[50]:


plt.plot(s_permit)


# ### Dropped all anomaly values found for each variable, now using missingvaluefiller to replace

# In[51]:


import pandas as pd
import darts
from darts import TimeSeries


# In[52]:


s_na = pd.read_csv("C:/Users/colet/Downloads/Adelphi/Optimization & Prescriptive Models/Scaled_sales_Mansfield_WH_SW_Alto_2022.csv")
s_na.head()


# #### Filling Pop data

# In[53]:


#Making the data into a time series dataset, splitting the data into training and validation. 

s_na_series = TimeSeries.from_dataframe(s_na, "Date", "Pop_NA")
train, val = s_na_series[:40], s_na_series[40:]


# In[54]:


#Importing data with missing values

series2 = TimeSeries.from_dataframe(s_na, "Date", "Pop_NA")
train, val = s_na_series[:40], s_na_series[40:]
series2.plot()


# In[55]:


#Using Darts MissingValuesFiller, it fills the missing values fairly accurately. 

from darts.dataprocessing.transformers import MissingValuesFiller

filler = MissingValuesFiller()
filled = filler.transform(series2, method='quadratic')

filled.plot()


# In[56]:


print(filled[38:55])


# #### Filling EFFR data

# In[57]:


#Making the data into a time series dataset, splitting the data into training and validation. 

s_na_series = TimeSeries.from_dataframe(s_na, "Date", "EFFR_NA")
train, val = s_na_series[:40], s_na_series[40:]


# In[58]:


#Importing data with missing values

series2 = TimeSeries.from_dataframe(s_na, "Date", "EFFR_NA")
train, val = s_na_series[:40], s_na_series[40:]
series2.plot()


# In[59]:


#Using Darts MissingValuesFiller, it fills the missing values fairly accurately. 

from darts.dataprocessing.transformers import MissingValuesFiller

filler = MissingValuesFiller()
filled = filler.transform(series2)

filled.plot()


# In[60]:


print(filled[50:60])


# #### Filling inflation data

# In[61]:


#Making the data into a time series dataset, splitting the data into training and validation. 

s_na_series = TimeSeries.from_dataframe(s_na, "Date", "Inflation_NA")
train, val = s_na_series[:40], s_na_series[40:]


# In[62]:


#Importing data with missing values

series2 = TimeSeries.from_dataframe(s_na, "Date", "Inflation_NA")
train, val = s_na_series[:40], s_na_series[40:]
series2.plot()


# In[63]:


#Using Darts MissingValuesFiller, it fills the missing values fairly accurately. 

from darts.dataprocessing.transformers import MissingValuesFiller

filler = MissingValuesFiller()
filled = filler.transform(series2)

filled.plot()


# In[64]:


print(filled[52:57])


# #### Filling GDP Data

# In[65]:


#Making the data into a time series dataset, splitting the data into training and validation. 

s_na_series = TimeSeries.from_dataframe(s_na, "Date", "GDP_NA")
train, val = s_na_series[:40], s_na_series[40:]


# In[66]:


#Importing data with missing values

series2 = TimeSeries.from_dataframe(s_na, "Date", "GDP_NA")
train, val = s_na_series[:40], s_na_series[40:]
series2.plot()


# In[67]:


#Using Darts MissingValuesFiller, it fills the missing values fairly accurately. 

from darts.dataprocessing.transformers import MissingValuesFiller

filler = MissingValuesFiller()
filled = filler.transform(series2)

filled.plot()


# In[68]:


print(filled[54:60])


# #### Filling Unemployment Data

# In[69]:


#Making the data into a time series dataset, splitting the data into training and validation. 

s_na_series = TimeSeries.from_dataframe(s_na, "Date", "Unem_NA")
train, val = s_na_series[:40], s_na_series[40:]


# In[70]:


#Importing data with missing values

series2 = TimeSeries.from_dataframe(s_na, "Date", "Unem_NA")
train, val = s_na_series[:40], s_na_series[40:]
series2.plot()


# In[71]:


#Using Darts MissingValuesFiller, it fills the missing values fairly accurately. 

from darts.dataprocessing.transformers import MissingValuesFiller

filler = MissingValuesFiller()
filled = filler.transform(series2)

filled.plot()


# In[72]:


print(filled[44:49])


# #### Filling Permit Data

# In[73]:


#Making the data into a time series dataset, splitting the data into training and validation. 

s_na_series = TimeSeries.from_dataframe(s_na, "Date", "Permit_NA")
train, val = s_na_series[:40], s_na_series[40:]


# In[74]:


#Importing data with missing values

series2 = TimeSeries.from_dataframe(s_na, "Date", "Permit_NA")
train, val = s_na_series[:40], s_na_series[40:]
series2.plot()


# In[75]:


#Using Darts MissingValuesFiller, it fills the missing values fairly accurately. 

from darts.dataprocessing.transformers import MissingValuesFiller

filler = MissingValuesFiller()
filled = filler.transform(series2)

filled.plot()


# In[76]:


print(filled[52:60])


# # Running the correlations again with the new variables

# We need to create one big dataframe with all the new variables so that we can compare it with the new sales data without the outliers.

# # Variables No Outliers Legend
# * data_fill = orders and variables (autofilled the removed values)
# * orders_fill = orders (autofilled the removed values)
# * variables_fill = variables (autofilled the removed values)
# 

# In[82]:


data_fill = pd.read_csv("C:/Users/colet/Downloads/Adelphi/Optimization & Prescriptive Models/Project Data/Mansfield_NA_filled.csv")
data_fill.head()


# In[83]:


#Orders -> autofilled 
orders_fill = data_fill[['Orders_NA']].copy()
orders_fill


# In[84]:


#Variables -> autofilled
variables_fill = data_fill.drop(columns=['Date', 't','Orders','Orders_NA','Pop_Growth','EFFR','Inflation','GDP','Unem','Permit'])
variables_fill


# In[85]:


#Lagging the autofilled variables
filled_variables_lag = lag_variable(variables_fill,12)
filled_variables_lag.head()


# In[86]:


#We are collecting everything under one frame so that it is easier to see the correlations
lagged_var_and_orders = reduce(lambda left,right: pd.merge(left,right,left_index=True, right_index=True, how='left'),
                 [orders_fill,filled_variables_lag])
lagged_var_and_orders.head()


# In[87]:


corr = lagged_var_and_orders.corr()['Orders_NA'].sort_values(ascending=False)
print(corr[corr >= -0.4])
print(corr[corr <= -0.45])
print(len(corr))
print(lagged_var_and_orders.columns)
print(lagged_var_and_orders.shape)
#print(corr)


# ## Most Correlated Lags to Orders
# * EFFR_NAlag_11
# * GDP_NAlag_2
# * Inflation_NAlag_1
# * Permit_NAlag_2
# * Population_no_lag
# * Unemployment_lag_2
# 
# These will be under "bestvar_" without the NA values.

# In[88]:


bestvar = lagged_var_and_orders[["EFFR_NAlag_11","GDP_NAlag_2","Inflation_NAlag_1","Permit_NAlag_2","Pop_NA","Unem_NAlag_2","Orders_NA"]].copy()


# In[89]:


plt.figure(figsize=(10,9))
mask1 = np.triu(np.ones_like(bestvar.corr(), dtype=np.bool))
sns.heatmap(data = bestvar.corr(), annot= True, linewidths=0.5, cmap='Blues', mask=mask1,  annot_kws={"fontsize":25})


# # APPLIED MACHINE LEARNING MODELS

# Okay so If we want to see predictions with the applied machine learning models, we would have to use at least 3 month lags, so for that we will be creating a separate storage.

# In[91]:


varforaml = lagged_var_and_orders[["EFFR_NAlag_11","GDP_NAlag_3","Inflation_NAlag_3","Permit_NAlag_3","Pop_NAlag_3","Unem_NAlag_3","Orders_NA"]].copy()
varforaml


# In[92]:


#MIN MAX
varforaml_minmax = (varforaml-varforaml.min())/(varforaml.max()-varforaml.min())

#STANDARDIZE
varforaml_std = (varforaml-varforaml.mean())/varforaml.std()
varforaml_std.head(12)


# In[ ]:


varforaml_std_nona = varforaml_std.dropna() #we are dropping NA values from the lagged variables


# In[116]:


varforaml_std_nona.head()


# ### Splitting data and setting xtest and xtrain

# In[93]:


feature_cols = ["EFFR_NAlag_11","GDP_NAlag_3","Inflation_NAlag_3","Permit_NAlag_3","Pop_NAlag_3","Unem_NAlag_3"]

X = varforaml_std_nona[feature_cols] # Features
y = varforaml_std_nona.iloc[:,-1] # Target variable

y.head()
#drop na values


# In[94]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)


# ## MULTIPLE LINEAR REGRESSION

# In[95]:


LinearReg = LinearRegression()  
fit = LinearReg.fit(X_train, y_train)

y_pred_Train = LinearReg.predict(X_train) #Predictions on training model
y_pred_Test  = LinearReg.predict(X_test)  #Predictions on testing model

#yhat = fit.predict(0,60)
#s['MultReg'] = yhat 

print('Intercept:',LinearReg.intercept_)
print('Coefficients:', LinearReg.coef_)


# In[96]:


#Two plots togheter

fig, ax = plt.subplots(ncols=2, figsize=(10,4))

ax[0].scatter(y_train, y_pred_Train)
ax[0].set_ylim(0,1.5)
ax[0].set_xlim(0,1.5)
ax[0].grid()
ax[0].set_xlabel('y')
ax[0].set_ylabel('yhat')
ax[0].set_title('Training Set')


ax[1].scatter(y_test, y_pred_Test)
ax[1].set_ylim(0,1.5)
ax[1].set_xlim(0,1.5)
ax[1].grid()
ax[1].set_xlabel('y')
ax[1].set_ylabel('yhat')
ax[1].set_title('Testing Set')
plt.show()


# In[97]:


print('Training Metrics:')
print('R squared:', metrics.r2_score(y_train, y_pred_Train))
print('Mean Absolute Error:', metrics.mean_absolute_error(y_train, y_pred_Train))  
print('Mean Squared Error:', metrics.mean_squared_error(y_train, y_pred_Train))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_train, y_pred_Train)))

print('\nTesting Metrics:')
print('R squared:', metrics.r2_score(y_test, y_pred_Test))
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred_Test))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred_Test))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred_Test)))


# In[98]:


Train_MSE = [] #Empty list to Store MSEs for training data set
Test_MSE = []  #Empty list to Store MSEs for testing data set

Train_R2 = [] #Empty list to Store R2s for training data set
Test_R2 = []  #Empty list to Store R2s for testing data set

for i in range(1000):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    LinearReg = LinearRegression() #Creates the function
    LinearReg.fit(X_train, y_train) #Train the model
    
    y_pred_Train  = LinearReg.predict(X_train)  #Predictions on training model
    y_pred_Test   = LinearReg.predict(X_test)   #Predictions on testing model
    
    train_R2 = metrics.r2_score(y_train, y_pred_Train) #Obtaining the metrics
    test_R2  = metrics.r2_score(y_test, y_pred_Test)
    
    train_MSE = metrics.mean_squared_error(y_train, y_pred_Train)
    test_MSE  = metrics.mean_squared_error(y_test, y_pred_Test)
    
    Train_MSE.append(train_MSE) #Storing the metrics in the lists
    Test_MSE.append(test_MSE) 
    
    Train_R2.append(train_R2) #Storing the metrics in the lists
    Test_R2.append(test_R2)  
    
print('Train MSE median:', np.median(Train_MSE))
print('Test MSE median:', np.median(Test_MSE))

print('\nTrain_R2 median:', np.median(Train_R2))
print('Test_R2 median:', np.median(Test_R2))


# ## PREDICTIONS FOR MULTIPLE LINEAR REGRESSION

# In[99]:


#1 Month - change variable values
LinearReg.predict([[-0.56759,1.671448,1.980216,1.801152,-1.413848,-1.430423
]])


# In[100]:


#2 Month - change variable values
LinearReg.predict([[-0.559861,1.671448,1.901339,1.782279,-1.413848,-0.156453
]])


# In[101]:


#3 Month - change variable values
LinearReg.predict([[-0.552131,1.671448,1.861901,1.843262,-1.413848,-1.430423
]])


# ## GRADIENT BOOST

# In[102]:


feature_cols = ["EFFR_NAlag_11","GDP_NAlag_3","Inflation_NAlag_3","Permit_NAlag_3","Pop_NAlag_3","Unem_NAlag_3"]
varforaml_std_nona = varforaml_std.dropna() #we are dropping NA values from the lagged variables

X = varforaml_std_nona[feature_cols] # Features
y = varforaml_std_nona.iloc[:,-1] # Target variable

#drop na values


# In[103]:


train_mse = []
train_mse = []
pred_mse = []
K = list(range(1,200,10))

for k in tqdm(K):
    
    model = GradientBoostingRegressor(n_estimators=k) #Number of trees in the forest
    model.fit(X_train, y_train.ravel()) #ravel flattens the array

    y_pred_Train = model.predict(X_train) #Predictions
    y_pred_Test = model.predict(X_test) #Predictions
        
    train_mse.append(metrics.mean_squared_error(y_train, y_pred_Train))
    pred_mse.append(metrics.mean_squared_error(y_test, y_pred_Test))
       
plt.plot(K,train_mse,'b')
plt.plot(K,pred_mse,'r')
plt.xlabel('K')
plt.ylabel('MSE')
plt.xticks(K)
plt.show()


# In[104]:


model = GradientBoostingRegressor(n_estimators=100)
Model_Performance(model,X,y)


# In[105]:


GBR = GradientBoostingRegressor(n_estimators=100)
GBR.fit(X_train,y_train)

Features = varforaml_std_nona.columns[0:6]

Feature_importances = pd.Series(GBR.feature_importances_, index=Features)
Feature_importances.plot.bar()


# ## PREDICTIONS FOR GRADIENT BOOST

# In[106]:


#1 Month - change variable values
GBR.predict([[-0.56759,1.671448,1.980216,1.801152,-1.413848,-1.430423]])


# In[107]:


#2 Month - change variable values
GBR.predict([[-0.559861,1.671448,1.901339,1.782279,-1.413848,-0.156453]])


# In[108]:


#3 Month - change variable values
GBR.predict([[-0.552131,1.671448,1.861901,1.843262,-1.413848,-1.430423]])


# ## DECISION TREE

# In[109]:


feature_cols = ["EFFR_NAlag_11","GDP_NAlag_3","Inflation_NAlag_3","Permit_NAlag_3","Pop_NAlag_3","Unem_NAlag_3"]
varforaml_std_nona = varforaml_std.dropna() #we are dropping NA values from the lagged variables

X = varforaml_std_nona[feature_cols] # Features
y = varforaml_std_nona.iloc[:,-1] # Target variable

y.head()
#drop na values


# In[110]:


#Learning curves
train_R2 =[]
test_R2=[]

for depth in tqdm(range(1,50)):
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2) 
    dtree = DecisionTreeRegressor(max_depth=depth)
    dtree.fit(X_train,y_train)
    y_pred_Train = dtree.predict(X_train) #Predictions
    y_pred_Test = dtree.predict(X_test) #Predictions
    train_R2.append(metrics.r2_score(y_train,y_pred_Train))
    test_R2.append(metrics.r2_score(y_test, y_pred_Test))

plt.plot(train_R2)
plt.plot(test_R2)
plt.ylabel('R2')
plt.xlabel('depth')
plt.show()


# In[111]:


print('Training Metrics:')
print('R squared:', metrics.r2_score(y_train, y_pred_Train))
print('Mean Absolute Error:', metrics.mean_absolute_error(y_train, y_pred_Train))  
print('Mean Squared Error:', metrics.mean_squared_error(y_train, y_pred_Train))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_train, y_pred_Train)))

print('\nTesting Metrics:')
print('R squared:', metrics.r2_score(y_test, y_pred_Test))
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred_Test))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred_Test))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred_Test)))


# ## PREDICTIONS FOR DECISION TREE

# In[112]:


#1 Month
dtree.predict([[-0.56759,1.671448,1.980216,1.801152,-1.413848,-1.430423]])


# In[113]:


#2 Month
dtree.predict([[-0.559861,1.671448,1.901339,1.782279,-1.413848,-0.156453]])


# In[114]:


#3 Month
dtree.predict([[-0.552131,1.671448,1.861901,1.843262,-1.413848,-1.430423]])


# # PREDICTIVE AND NEURAL NETWORK MODELS

# ## Reinstating Most correlated lags found above

# In[128]:


varforpm = lagged_var_and_orders[["EFFR_NAlag_11","GDP_NAlag_2","Inflation_NAlag_1","Permit_NAlag_2","Pop_NA","Unem_NAlag_2","Orders_NA"]].copy()
varforpm


# In[130]:


#Dropping NA's
varforpm = varforpm.dropna() #we are dropping NA values from the lagged variables
varforpm.head()


# In[185]:


varforpm.head()


# In[131]:


#STANDARDIZE
varforpm_std = (varforpm-varforpm.mean())/varforpm.std()
varforpm_std.head(12)


# In[132]:


t_col = s[["t"]].copy()
#t_col.head()


# In[135]:


#Creating std df with t
varforpm_std_t = pd.concat([varforpm_std_nona, t_col], axis=1, join='inner')
#varforpm_std_t


# In[136]:


#Creating df not std with t
varforpm_with_t_nostd = pd.concat([varforpm, t_col], axis=1, join='inner')
#varforpm_with_t_nostd


# ## Running SARIMA and SARIMAX Models

# In[138]:


from statsmodels.tools.eval_measures import rmse, meanabs, rmspe

from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.statespace.varmax import VARMAX


# In[139]:


model = SARIMAX(varforpm_std['Orders_NA'], order=(1, 1, 2), seasonal_order=(1, 1, 2, 12)).fit() #Pink Messages
print(model.summary())

yhat = model.predict(0,60)
varforpm_with_t_nostd['SARIMA'] = yhat
varforpm_with_t_nostd.tail()

varforpm_with_t_nostd.plot.line('t', ['Orders_NA', 'SARIMA'])

plt.xticks(rotation=90)
plt.show()


# In[141]:


print('\n\nNext three months forecast:',)
model.predict(61,63)


# In[143]:


RMSE = rmse(varforpm_with_t_nostd.loc[1:,'Orders_NA'], varforpm_with_t_nostd.loc[1:,'SARIMA'])
MAD = meanabs(varforpm_with_t_nostd.loc[1:,'Orders_NA'], varforpm_with_t_nostd.loc[1:,'SARIMA'])
RMSPE= rmspe(varforpm_with_t_nostd.loc[1:,'Orders_NA'], varforpm_with_t_nostd.loc[1:,'SARIMA'])

print('RMSE:', RMSE)
print('MAD:',  MAD)
print('RMSPE:',RMSPE)


# In[145]:


varforpm_with_t_nostd.head()


# In[149]:


model = SARIMAX(varforpm_std_t['Orders_NA'], exog=varforpm_std_t[["EFFR_NAlag_11","GDP_NAlag_2","Inflation_NAlag_1","Permit_NAlag_2","Pop_NA","Unem_NAlag_2"]], order=(1, 1, 2), seasonal_order=(0, 1, 1, 12)).fit() #Pink Menewdropnewdropagenewdrop
print(model.summary()) #exog=newdrop[['EX1', 'EX2']]

yhat = model.predict(0,48)
varforpm_with_t_nostd['SARIMAX'] = yhat
#varforpm_with_t_nostd.head()

varforpm_with_t_nostd.plot.line('t', ['Orders_NA', 'SARIMAX'])
plt.xticks(rotation=90)
plt.show()

RMSE = rmse(varforpm_with_t_nostd.loc[1:,'Orders_NA'], varforpm_with_t_nostd.loc[1:,'SARIMAX'])
MAD = meanabs(varforpm_with_t_nostd.loc[1:,'Orders_NA'], varforpm_with_t_nostd.loc[1:,'SARIMAX'])
RMSPE= rmse(varforpm_with_t_nostd.loc[1:,'Orders_NA'], varforpm_with_t_nostd.loc[1:,'SARIMAX'])

print('RMSE:', RMSE)
print('MAD:',  MAD)
print('RMSPE:',RMSPE)


# In[156]:


varforpm_with_t_nostd.head()


# ## Holt Winters Model

# In[197]:


from statsmodels.tsa.holtwinters import ExponentialSmoothing

model = ExponentialSmoothing(varforpm_std_t['Orders_NA'],trend = 'add', seasonal='add', seasonal_periods=12).fit()

print(model.summary())

yhat = model.predict(0,60)
varforpm_with_t_nostd['HW'] = yhat

varforpm_with_t_nostd.plot.line('t', ['Orders_NA', 'HW'])
plt.show()

RMSE  = rmse(varforpm_with_t_nostd.loc[1:,'Orders_NA'], varforpm_with_t_nostd.loc[1:,'HW'])
MAD   = meanabs(varforpm_with_t_nostd.loc[1:,'Orders_NA'], varforpm_with_t_nostd.loc[1:,'HW'])
RMSPE = rmspe(varforpm_with_t_nostd.loc[1:,'Orders_NA'], varforpm_with_t_nostd.loc[1:,'HW'])

print('RMSE:', RMSE)
print('MAD:',  MAD)
print('RMSPE:',RMSPE)


# In[198]:


print('\n\nNext three months forecast:',)
model.predict(61,63)


# In[210]:


from sklearn.metrics import r2_score
print('R2 Score', r2_score(varforpm_with_t_nostd['Orders_NA'], varforpm_with_t_nostd['HW']))


# In[203]:


varforpm_with_t_nostd.head(30)


# In[209]:


from sklearn.metrics import r2_score
print('R2 Score', r2_score(varforpm_std_t['Orders_NA'], varforpm_with_t_nostd['SARIMAX']))


# In[ ]:




