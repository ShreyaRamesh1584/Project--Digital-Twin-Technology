#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


import sys

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")


# In[4]:


import pandas as pd
df= pd.read_csv(r'C:\Users\shreya.ramesh\Downloads\Superstore_dataset.csv', encoding = 'utf-8', encoding_errors= 'replace')
df.head(2)


# In[5]:


column_names=df.columns.values


# In[6]:


#Since all columns are not needed for sales forecasting the following can be deleted
df=df[column_names]


# In[7]:


df.head(2)


# In[8]:


df.info()


# In[9]:


df.drop(['Row ID', 'Order ID', 'Customer ID', 'Product ID', 'Postal Code', 'Ship Mode' ], axis=1, inplace=True)


# In[10]:


df


# In[11]:


#Data has 52000 rows, too big a data so for the purpose of smooth running of the model decide to continue with the first 10000 drop the rest
df=df.iloc[0:10000,:]
print(df['Segment'].unique())


# In[12]:


#Basic Exploratory Data Analysis
plt.figure(figsize=(20,8))
sns.boxplot("State", "Sales", data=df)
plt.title("State wise Sales")


# In[13]:


plt.figure(figsize=(20,8))
sns.boxplot("Region", "Sales", data=df)
plt.title("Region wise Sales")
plt.ylim(0,500)


# In[14]:


df[df['Region']== "Africa"].describe()


# In[15]:


plt.figure(figsize=(10,4))
sns.boxplot("Order Priority", "Sales", data=df)
plt.title("Relationship between order priority and sales")
plt.ylim(0,18000)


# In[16]:


df[df["Order Priority"]=="Medium"].describe()


# In[17]:


plt.figure(figsize=(15,7))
sns.boxplot("Order Priority", "Profit", data=df)
plt.title("Relationship between order priority and profit")


# In[18]:


plt.figure(figsize=(20,8))
sns.boxplot("Country", "Sales", data=df)
plt.title("Relationship between country and sales")
plt.ylim(0,200)


# In[19]:


plt.figure(figsize=(20,8))
sns.boxplot("Market", "Sales", data=df)
plt.title("Relationship between Market and Sales")
plt.ylim(0,7500)


# In[20]:


df


# In[21]:


plt.figure(figsize=(20,8))
sns.boxplot("Category", "Sales", data=df)
plt.title("Relationship between category and sales")
plt.ylim(0,7500)


# In[22]:


plt.figure(figsize=(20,8))
sns.boxplot("Sub-Category", "Sales", data=df)
plt.title("Relationship between sub category and sales")
plt.ylim(0,5000)


# In[23]:


df["Country"].unique()


# In[24]:


print(df["Sales"].max())
print(df["Sales"].min())


# In[25]:


plt.figure(figsize=(10,5))
plt.hist(x="Sales", data=df, bins=10)
plt.ylim(0,20000)


# In[26]:


df["Sales"].describe()


# In[27]:


plt.figure(figsize=(10,5))
sns.distplot(df["Sales"], bins = 30)
#sns.distplot(df["Profit"])


# In[28]:


#column_names=df[df.columns.unique()]
column_names=["No. of unique columns in the table"]
pd.DataFrame(df.nunique(axis=0), columns=column_names)


# In[29]:


#Checking and removing duplicates
df.duplicated().sum()
#no duplicates found


# In[30]:


df.columns.values


# In[31]:


#Correlation heatmap graph
Categorical=['Region', 'Country', 'Market', 'City', 'State', 'Segment', 'Category', 'Sub-Category', 'Order Priority', 'Product Name']


# In[32]:


numerical=[item for item in df.columns.to_list() if item not in Categorical]


# In[33]:


corr_data=df[numerical]
corr=corr_data.corr(method='pearson')
plt.close()
corr_plot=sns.heatmap(corr, annot=True, cmap="RdYlGn")
fig=plt.gcf()
#plt.gcf() is used to get the figure, so it gets saved in the variable fig
fig.set_size_inches(10,8)
#set_size_inches is used to set the size of the figure 
plt.xticks(fontsize=10, rotation=-30)
plt.yticks(fontsize=10)
plt.title("Correlation heatmap graph")
plt.show()


# In[34]:


#Regression Plot
plt.figure(figsize=(10,8))
sns.set_style("whitegrid")
sns.lmplot("Quantity", "Sales", data=df, hue="Category")
plt.ylim(0,5000)
#df


# In[35]:


plt.figure(figsize=(10,8))
sns.lmplot("Discount", "Sales", data=df, hue="Category")
plt.ylim(0,7500)
sns.set_style("whitegrid")


# In[37]:


plt.figure(figsize=(10,8))
sns.lmplot("Profit", "Sales", data=df, hue="Category")
plt.ylim(0,7500)
#sns.set_style("whitegrid")


# In[38]:


plt.figure(figsize=(10,8))
sns.lmplot("Shipping Cost", "Sales", data=df, hue="Category")
plt.ylim(0,7500)
sns.set_style("whitegrid")


# In[39]:


#Bar plot
plt.figure(figsize=(20,8))
sns.barplot("Country", "Sales", data=df)
plt.ylim(0,500)


# In[40]:


#df
plt.figure(figsize=(20,8))
sns.barplot("City", "Sales", data=df)


# In[41]:


plt.figure(figsize=(10,3))
sns.barplot("Market", "Sales", data=df)


# In[42]:


plt.figure(figsize=(20,8))
sns.barplot("State", "Sales", data=df)


# In[43]:


#Categorical=['Region', 'Country', 'Market', 'City', 'State', 'Segment', 'Category', 'Sub-Category', 'Order Priority', 'Product Name']
plt.figure(figsize=(10,3))
sns.barplot("Segment", "Sales", data=df)


# In[44]:


plt.figure(figsize=(10,3))
sns.barplot("Category", "Sales", data=df)


# In[45]:


plt.figure(figsize=(20,3))
sns.barplot("Sub-Category", "Sales", data=df)


# In[46]:


plt.figure(figsize=(10,3))
sns.barplot("Order Priority", "Sales", data=df)


# In[49]:


plt.figure(figsize=(20,8))
sns.catplot("Market", kind="count", data=df)
plt.xlabel("Market Region")
plt.ylabel("Sales Count")
plt.title("Sales by Market Region")
plt.show()


# In[50]:


f1=pd.read_csv(r'C:\Users\shreya.ramesh\Downloads\Superstore_dataset.csv', encoding='utf-8', encoding_errors= 'replace')
f1=f1[0:10000]
#f1


# plt.figure(figsize=(10,8))
# top10countries=f1.groupby('Country')['Sales'].count().sort_values(ascending=False)
# top10countries=top10countries[:10]
# top10countries.plot(kind='bar')

# In[51]:


#Top 10 States 
plt.figure(figsize=(10,5))
top10states=f1.groupby('State')['Sales'].count().sort_values(ascending=False)
#it by default takes ascending so you convert to false to take the descending value 
top10states=top10states[:10]
top10states.plot(kind='bar')
#Calfornia is the state with the most sales


# In[52]:


plt.figure(figsize=(20,5))
top10countries=f1.groupby('Country')['Sales'].count().sort_values(ascending=False)
top10countries=top10countries[:10]
top10countries.plot(kind='bar')
#United States is the country with the most sales


# In[53]:


plt.figure(figsize=(20,4))
top10cities=f1.groupby('City')['Sales'].count().sort_values(ascending=False)
top10cities=top10cities[:10]
top10cities.plot(kind='bar')
#New York is the city with the most sales


# In[54]:


#Top 10 Segments
plt.figure(figsize=(8,2))
segment_wise_analysis=f1.groupby('Segment')['Sales'].count().sort_values(ascending=False)
segment_wise_analysis.plot(kind='bar')
plt.title('Segment wise Sales analysis')


# In[55]:


#Region wise total sales
plt.figure(figsize=(10,3))
Region_wise_sales=f1.groupby('Region')['Sales'].count().sort_values(ascending=False)
Region_wise_sales.plot(kind='bar')
plt.xticks(rotation=-30)
plt.title('Region wise sales analysis')
#Central Region has maximum sales


# In[56]:


#Ship mode wise number of sales
plt.figure(figsize=(10,6))
sns.catplot(x='Ship Mode', kind='count', data=f1, )
# sns.catplot(x='Ship')
plt.xlabel('Ship Mode')
plt.ylabel('Sales')
plt.show()


# In[57]:


f1[f1['Ship Mode']=='Standard Class'].describe()


# In[58]:


#Pair Plot 
plt.figure(figsize=(20,8))
sns.pairplot(df)


# In[59]:


#Changing the datatype of Order Date
f1['Order Date']=pd.to_datetime(f1['Order Date'])
f1['Ship Date']=pd.to_datetime(f1['Ship Date'])


# In[60]:


f1['Order Date'].describe()


# In[61]:


f1['year']=pd.DatetimeIndex(f1['Order Date']).year
#adding a column to f1
plt.figure(figsize=(15,5))
sns.lineplot(x='year', y='Sales', data=f1)


# In[62]:


#Checking for duplicates and dropping if any found
f1.duplicated().sum()
#no duplicates found 


# In[63]:


#Removing all non-essential features from the dataframe
df.Region.unique()


# In[64]:


df2=pd.get_dummies(df[['Region', 'Sales']], drop_first=True)


# In[65]:


df2.head()


# In[66]:


plt.figure(figsize=(10,5))
sns.heatmap(df2.corr() ,annot=True)


# In[67]:


df3=pd.get_dummies(df[['Market', 'Sales']], drop_first=True)
df3.head()


# In[68]:


plt.figure(figsize=(10,4))
sns.heatmap(df3.corr(), annot=True)


# In[69]:


df4=pd.get_dummies(df[['Category', 'Sales']])
df4.head()


# In[70]:


sns.heatmap(df4.corr(), annot=True)


# In[71]:


df5=pd.get_dummies(df[['Sub-Category', 'Sales']])
plt.figure(figsize=(10,5))
sns.heatmap(df5.corr(), annot=True)


# In[72]:


df6=pd.get_dummies(df[['Country', 'Sales']])
sns.heatmap(df6.corr(), annot=True)


# In[73]:


#An analysis of sales by market and region shows a negative correlation, therefore a seperate analysis for country, city and state is not needed.
#An analysis of sales by category and sub-category again shows the correlation, which can be extrapolated to product name
#Hence dropping all the columns not needed
df.drop(['City', 'State', 'Country', 'Product Name', 'Market', 'Region'], axis=1, inplace=True)
#the inplace parameter will make the changes to the original dataframe. If not true, it will assume the default value of false which will then drop the columns only in the copy of the dataframe not in the original dataframe


# In[74]:


#we can also drop Order Date, Ship Date, Customer Name as they have no impact on sales 
df.drop(['Customer Name', 'Order Date', 'Ship Date'], axis=1, inplace=True)


# In[75]:


#Removing outliers by replacing the outlier values using the Flooring and Capping method
plt.figure(figsize=(10,5))
sns.boxplot('Category','Sales', data=df)
plt.ylim(0,500)


# In[76]:


df.isnull().sum()


# In[77]:


#Analysing the quantity sold for outliers
sns.boxplot('Quantity', data=df)
#We cannot remove outliers in quantity simply, because depending on the discount given, there could be an increase in quantity sold


# In[78]:


#Analysing the discount feature for outliers 
sns.boxplot('Discount', data=df)
#Here we can see some unusual discounts crossing 50%. But this could be discounts offered to clear out remaining stock too


# In[79]:


df.Discount.describe()


# In[80]:


plt.hist(x='Discount', data=df)


# In[81]:


#Analysing the profits for outliers
sns.boxplot('Profit', data=df)
plt.xlim(-2000,10000)


# In[82]:


plt.hist('Profit', data=df)


# In[83]:


df[df.Profit>=3000]


# In[84]:


df.drop(8898, inplace=True)


# In[85]:


#Removing the negative outliers in the profit too 
Q1=df['Profit'].quantile(q=0.25)
Q3=df['Profit'].quantile(q=0.75)
IQR=Q3-Q1
#IQR=Inter Quantile Range
upper_limit=Q3+(1.5*IQR)
lower_limit=Q1-(1.5*IQR)
print(upper_limit, lower_limit)


# In[86]:


df['Profit']=np.where(df['Profit']<-1200, -54.918000000000006, df['Profit'])


# In[87]:


df.head(2)


# In[88]:


sns.scatterplot('Profit', 'Sales', data=df)


# In[89]:


#Shipping Cost Feature
sns.boxplot('Shipping Cost', data=df)


# In[90]:


plt.hist(x='Shipping Cost', data=df )


# In[91]:


sns.scatterplot('Shipping Cost', 'Sales', data=df)


# In[92]:


df[df['Shipping Cost']>300].head(5)
#It can be observed that shipping cost is more for the corporate segment, which are essential, so no changes are made here 


# In[93]:


sns.boxplot('Sales', data=df)


# In[94]:


df[df.Sales>8000]
#The sales of above 8000 are for machines, so they cannot be considered as outliers


# In[95]:


#Creating another dataset with threshold standard deviation kept at less than 3
exep=df.copy()
exep.head()


# In[96]:


from scipy import stats
import numpy as np
z=np.abs(stats.zscore(exep[['Discount', 'Sales', 'Shipping Cost', 'Profit', 'Quantity']]))
print(z)


# In[97]:


threshold=3
exep_1=exep[(z<3).all(axis=1)]
exep_1.shape


# In[98]:


#We have created two dataframes 
#One is to know the impact of more sold products with higher discounts and higher sales [df]
#Second is optimized taking threshold standard deviation as 3 
#Check the data for formatting issues and clean the data where needed
df.info()


# In[99]:


sns.countplot('Segment', data=df)
plt.title('Segment wise number of sales')


# In[100]:


sns.countplot('Category', data=df)
plt.title('Category wise number of sales')


# In[101]:


sns.countplot('Sub-Category', data=df)
plt.title('Sub-Category wise number of sales')
plt.xticks(rotation=-90)


# In[102]:


sns.countplot('Order Priority', data=df)
plt.title('Order Priority wise number of sales')


# In[103]:


segment_wise_profits=df.groupby('Segment')['Profit'].sum().sort_values(ascending=False)
segment_wise_profits.plot(kind='bar')
plt.ylabel('Profits')
plt.title('Segment wise profits')
#Consumer segment made the maximum profit


# In[104]:


Category_wise_profits=df.groupby('Category')['Profit'].sum().sort_values(ascending=False)
Category_wise_profits.plot(kind='bar')
plt.ylabel('Profits')
plt.title('Category wise profits')


# In[105]:


Sub_Category_wise_profits=df.groupby('Sub-Category')['Profit'].sum().sort_values(ascending=False)
Sub_Category_wise_profits.plot(kind='bar')
plt.ylabel('Profits')
plt.title('Sub-Category wise profits')


# In[106]:


plt.bar('Order Priority', 'Profit', data=df)


# In[107]:


sns.barplot('Order Priority', 'Profit', data=df)
#gives a mean profit of each priority 
#Shows profit generated per sales is highest for critical order priority


# In[110]:


sns.barplot('Sub-Category', 'Profit', data=df)
plt.xticks(rotation=-90)
#Highest mean profits are generated for the copier sub-category 


# In[111]:


sns.catplot(x="Segment", col="Category", data=df, kind="count")
#Office supplies are more sold in comparison to other category


# In[112]:


#Building a Regression model
df=pd.get_dummies(df,drop_first=True)
df.head(2)


# In[113]:


X=df.drop('Sales', axis=1)
#it is the independent variable so we drop sales from that 


# In[114]:


Y=df['Sales']
#We put sales in the dependent variable, and all the other parameters that influence the sales in the independent variable 


# In[115]:


from sklearn.model_selection import train_test_split
X_train, X_test,Y_train, Y_test = train_test_split(X,Y, test_size=0.3, random_state=0)


# In[116]:


#Training the linear regression model on the training set
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train, Y_train)


# In[117]:


#Predicting the values
Y_pred=regressor.predict(X_test)
np.set_printoptions(precision=2)
#The precision factor will round it off to that values, essentialy decides the number of decimal places in a floating point variable
Y_pred


# In[118]:


import sklearn.metrics


# In[119]:


mse=sklearn.metrics.mean_squared_error(Y_test,Y_pred)
mse


# In[120]:


mae=sklearn.metrics.mean_absolute_error(Y_test,Y_pred)
mae


# In[121]:


#Find the root mean squared error
import math
rmse=math.sqrt(mse)
rmse


# In[122]:


evaluate=pd.DataFrame({'Actual':Y_test.values.flatten(), 'Predicted':Y_pred.flatten()})
evaluate.head(5)


# In[123]:


#plotting it on a bar plot
evaluate.head(20).plot(kind='bar')


# In[124]:


#Calculating the r2 score 
from sklearn.metrics import r2_score
r2_score=r2_score(Y_test, Y_pred)
r2_score


# In[125]:


def mean_absolute_percentage_error(y_true, y_predict):
    y_true, y_predict= np.array(y_true), np.array(y_predict)
    return np.mean(np.abs((y_true - y_predict) / y_true)) * 100


# In[126]:


mean_absolute_percentage_error(Y_test, Y_pred)


# In[127]:


#K-fold cross validation method for defining accuracy of the model 
from sklearn.model_selection import cross_val_score
lr=LinearRegression()
np.mean(cross_val_score(lr, X, Y, cv=10))


# In[128]:


#Now using exep dataframe with threshold standard deviation value of 3, checking the accuracy of the linear regression model with this dataframe
exep_1=pd.get_dummies(exep_1, drop_first=True)
X=exep_1.drop('Sales', axis=1)
X.head(2)


# In[129]:


Y=exep_1['Sales']
Y.head(2)


# In[130]:


X_train, X_test, Y_train, Y_test=train_test_split(X, Y, test_size=0.3, random_state=0)


# In[131]:


regressor=LinearRegression()
regressor.fit(X_train, Y_train)


# In[132]:


Y_pred=regressor.predict(X_test)
#Comparing test and predicted values
evaluate=pd.DataFrame({'Actual':Y_test.values.flatten(), 'Predicted':Y_pred.flatten()})
evaluate.head(5)


# In[133]:


#r2 score
from sklearn.metrics import r2_score
score=r2_score(Y_test, Y_pred)
score


# In[134]:


mse=sklearn.metrics.mean_squared_error(Y_test, Y_pred)
mse


# In[135]:


mae=sklearn.metrics.mean_absolute_error(Y_test, Y_pred)
mae


# In[136]:


def mean_absolute_percentage_error(y_true, y_predicted):
    y_true, y_predicted=np.array(y_true), np.array(y_predicted)
    return np.mean(np.abs((y_true-y_predicted)/y_true))*100


# In[137]:


mean_absolute_percentage_error(Y_test, Y_pred)


# In[140]:


from sklearn.model_selection import cross_val_score 
np.mean(cross_val_score(lr, X_train, Y_train, cv=10))


# In[ ]:


#Checking the errors in both datasets of df and exep_1, it has been observed that exep_1 has less erros in comparison to df, so for further models we will use exep_1


# In[141]:


#Running hyper parameter tuning
from sklearn.model_selection import GridSearchCV
parameters={'fit_intercept':[True,False], 'normalize':[True,False], 'copy_X':[True, False]}
exep_1_grid=GridSearchCV(regressor, parameters, cv=10)
exep_1_grid.fit(X_train, Y_train)
print('r2 score is:', exep_1_grid.best_score_)
print("Residual sum of squares: %.2f"
              % np.mean((exep_1_grid.predict(X_test) - Y_test) ** 2))


# In[142]:


#Building Random Forest Regression Model
from sklearn.ensemble import RandomForestRegressor


# In[147]:


rfr=RandomForestRegressor(n_estimators=100, random_state=0)
rfr.fit(X_train, Y_train)


# In[145]:


#Cross Validation Score
from sklearn.model_selection import cross_val_score
rfr=RandomForestRegressor()
np.mean(cross_val_score(rfr, X_train, Y_train, cv=10))


# In[148]:


rfr_y_pred=rfr.predict(X_test)


# In[149]:


from sklearn.metrics import mean_squared_error
rfr_mse=mean_squared_error(Y_test, rfr_y_pred)
rfr_mse


# In[150]:


#Finding mean absolute error
from sklearn.metrics import mean_absolute_error
rfr_mae=mean_absolute_error(rfr_y_pred, Y_test)
rfr_mae


# In[151]:


rfr_mape=mean_absolute_percentage_error(rfr_y_pred, Y_test)
rfr_mape


# In[152]:


from sklearn.metrics import r2_score
rfr_r2_score=r2_score(rfr_y_pred, Y_test)
rfr_r2_score


# In[153]:


#Hyperparameter tuning of the model using RandomizedSearchCV
from sklearn.model_selection import RandomizedSearchCV
parameters={'n_estimators':[150, 200, 250, 270, 280, 300], 'max_depth':[4.8, 4.9, 5, 5.1, 5.2]}
rfr_grid_search=RandomizedSearchCV(estimator=rfr, cv=10, n_jobs=-1, n_iter=10, param_distributions=parameters)
rfr_grid_search=rfr_grid_search.fit(X_train, Y_train)
rfr_best_score=rfr_grid_search.best_score_
rfr_best_parameter=rfr_grid_search.best_params_
print(rfr_best_score, rfr_best_parameter)
#Here the rfr_best_score and rfr_best_parameter will reflect the best score that was obtained and the parameters against which that score was obtained.


# In[154]:


from sklearn.model_selection import RandomizedSearchCV
parameters={'n_estimators':[300, 350], 'max_depth':[5, 5.01, 5.02, 5.03]}
rfr_grid_search=RandomizedSearchCV(estimator=rfr, cv=10, n_jobs=-1, n_iter=10, param_distributions=parameters)
rfr_grid_search=rfr_grid_search.fit(X_train, Y_train)
rfr_best_score=rfr_grid_search.best_score_
rfr_best_parameter=rfr_grid_search.best_params_
print(rfr_best_score, rfr_best_parameter)


# In[155]:


from sklearn.model_selection import RandomizedSearchCV
parameters={'n_estimators':[280, 290, 300, 310, 320, 330], 'max_depth':[5, 5.1, 5.2, 5.3]}
rfr_grid_search=RandomizedSearchCV(estimator=rfr, cv=10, n_jobs=-1, n_iter=10, param_distributions=parameters)
rfr_grid_search=rfr_grid_search.fit(X_train, Y_train)
rfr_best_score=rfr_grid_search.best_score_
rfr_best_parameter=rfr_grid_search.best_params_
print(rfr_best_score, rfr_best_parameter)


# In[156]:


rfr_predict_values=rfr_grid_search.predict(X_test)
print(len(rfr_predict_values))


# In[157]:


evaluate_rfr=pd.DataFrame({'Actual':Y_test.values.flatten(), 'Predicted':rfr_predict_values.flatten()})
evaluate_rfr.head(5)


# In[158]:


rfr_r2_score=r2_score(Y_test, rfr_predict_values)
print(rfr_r2_score)


# In[159]:


#Building a decision tree model
from sklearn.tree import DecisionTreeRegressor
dtr=DecisionTreeRegressor(random_state=0)
dtr=dtr.fit(X_train, Y_train)


# In[161]:


from sklearn.model_selection import cross_val_score
Accuracy=cross_val_score(dtr, X_train, Y_train, cv=10)


# In[162]:


print("Accuracy(MEAN): {:.2f} %".format(Accuracy.mean()*100))
print("Accuracy(STD): {:.2f} %".format(Accuracy.std()*100))


# In[163]:


dtr_y_pred=dtr.predict(X_test)
dtr_y_train_pred=dtr.predict(X_train)


# In[164]:


evaluate=pd.DataFrame({'Actual':Y_test.values.flatten(), 'Predicted':dtr_y_pred.flatten()})
evaluate.head(5)


# In[165]:


#Finding the mean squared error, root mean squared error and mean absolute error
from sklearn.metrics import mean_squared_error
mse_dtr=mean_squared_error(Y_test, dtr_y_pred)
print(mse)


# In[166]:


from sklearn.metrics import mean_absolute_error
mae_dtr=mean_absolute_error(Y_test, dtr_y_pred)
print(mae)


# In[167]:


#r2 score calculation
from sklearn.metrics import r2_score
dtr_score=r2_score(Y_test, dtr_y_pred)
print(dtr_score)


# In[168]:


import math 
rmse_dtr=math.sqrt(mse_dtr)
print(rmse_dtr)


# In[169]:


path=dtr.cost_complexity_pruning_path(X_train, Y_train)
alphas=path['ccp_alphas']
alphas


# In[170]:


#Hyper Parameter tuning of Decision Tree Model 
parameters={'splitter':['best', 'random'], 'max_depth':[3,5], 'ccp_alpha':[0.1,0.2,0.3,0.4]}
dtr_grid_search=RandomizedSearchCV(param_distributions=parameters, cv=10, n_jobs=1, n_iter=10, estimator=dtr)
dtr_grid_search=dtr_grid_search.fit(X_train, Y_train)
dtr_best_parameter=dtr_grid_search.best_params_
dtr_best_score=dtr_grid_search.best_score_
print(dtr_best_score,dtr_best_parameter)


# In[171]:


#r2 score of the model
parameters={'splitter':['best', 'random'], 'max_depth':[5.4, 5.45, 5.46, 5.49, 5.5], 'ccp_alpha':[0.222,0.223,0.224,0.225, 0.227,0.228]}
dtr_grid_search=RandomizedSearchCV(param_distributions=parameters, cv=10, n_jobs=1, n_iter=10, estimator=dtr)
dtr_grid_search=dtr_grid_search.fit(X_train, Y_train)
dtr_best_parameter=dtr_grid_search.best_params_
dtr_best_score=dtr_grid_search.best_score_
print(dtr_best_score,dtr_best_parameter)


# In[172]:


dtr_y_pred=dtr_grid_search.predict(X_test)


# In[173]:


evaluate=pd.DataFrame({'Actual':Y_test.values.flatten(),'Predicted':dtr_y_pred.flatten()})
evaluate.head()


# In[174]:


dtr_pred_r2score=r2_score(Y_test, dtr_y_pred)
print(dtr_pred_r2score)


# In[175]:


#Building a support vector regression model
from sklearn.svm import SVR
svr=SVR(kernel='rbf')
svr.fit(X_train, Y_train)


# In[176]:


#K-fold cross validation
Accuracy_svr=cross_val_score(svr, X_train, Y_train, cv=10)
print('Mean Accuracy Percentage is:',Accuracy_svr.mean()*100)


# In[177]:


svr_y_predict=svr.predict(X_test)


# In[178]:


## Mean squared error and mean absolute error
mse=mean_squared_error(Y_test,svr_y_predict)
mse


# In[179]:


mae=mean_absolute_error(Y_test,svr_y_predict)
mae


# In[180]:


#Root mean square error
rmse_svr=math.sqrt(mse)
rmse_svr


# In[181]:


r2_score_svr=r2_score(Y_test, svr_y_predict)
r2_score_svr


# In[209]:


#Hyperparameter tuning of SVR model
parameter=[{'C':[1,10,100,1000], 'kernel':['rbf'], 'gamma':[0.1,0.2,0.3,0.4]}]
svr_grid_search=RandomizedSearchCV(svr, param_distributions=parameter, n_iter=10, cv=10, n_jobs=-1)


# In[210]:


svr_grid_search=svr_grid_search.fit(X_train, Y_train)


# In[211]:


svr_best_score=svr_grid_search.best_score_
svr_best_parameter=svr_grid_search.best_params_
print(svr_best_score, svr_best_parameter)


# In[212]:


svr_y_predict_values=svr_grid_search.predict(X_test)


# In[214]:


parameter=[{'C':[10,20,30,1000], 'kernel':['rbf'], 'gamma':[0.1,0.01, 0.001]}]
svr_grid_search=RandomizedSearchCV(svr, param_distributions=parameter, n_iter=10, cv=10, n_jobs=-1)


# In[215]:


svr_grid_search=svr_grid_search.fit(X_train, Y_train)


# In[216]:


svr_best_score=svr_grid_search.best_score_
svr_best_parameter=svr_grid_search.best_params_
print(svr_best_score, svr_best_parameter)


# In[218]:


parameter=[{'C':[20,25, 27, 30, 35], 'kernel':['rbf'], 'gamma':[0.001, 0.0001, 0.00001]}]
svr_grid_search=RandomizedSearchCV(svr, param_distributions=parameter, n_iter=10, cv=10, n_jobs=-1)


# In[219]:


svr_grid_search=svr_grid_search.fit(X_train, Y_train)


# In[220]:


svr_best_score=svr_grid_search.best_score_
svr_best_parameter=svr_grid_search.best_params_
print(svr_best_score, svr_best_parameter)


# In[222]:


svr_y_predict=svr_grid_search.predict(X_test)


# In[224]:


svr_r2score=r2_score(Y_test, svr_y_predict)
print(svr_r2score)


# In[225]:


#Based on the four models trained, the r2 scores for all 4 models are as seen here:
#Linear Regression: 0.7625
#Random Forest Regression: 0.8332
#Decision Tree Model: 0.7998
#Support Vector Regression model: 0.790069

#Based on the following, Random Forest Regression gives the best prediction of sales, and can be used for future predictions. 

