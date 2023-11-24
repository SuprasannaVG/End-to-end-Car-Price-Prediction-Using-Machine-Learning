#!/usr/bin/env python
# coding: utf-8

# # End-to-end Car Price Prediction Using Machine Learning 
# 

# Name: Suprasanna V Gunaga
#     
# Stream: BE Information Science and Engineering, RNSIT Bengaluru (2021-2025).
#     
# 

# This notebook looks into using various Python-based machine learning and data science libraries in an attempt to build a machine learning model capable of predicting car price based on car attributes.
# 
# 
# 
# 

# In[24]:


import pandas as pd
import numpy as np
import xgboost


# In[25]:


data=pd.read_csv("CSV files/car datap.csv")


# In[26]:


data.head()


# In[27]:


data.shape


# In[28]:


data.info()


# In[29]:


data.isna().sum()


# In[30]:


data.describe()


# # Data Preprocessing

# In[31]:


data.head(1)


# In[32]:


import datetime


# In[33]:


date_time=datetime.datetime.now()
date_time


# In[34]:


date_year=date_time.year


# In[35]:


data["Age"]=date_time.year-data["Year"]
data.head()


# In[36]:


data.drop("Year",axis=1,inplace=True)


# In[37]:


data.head()


# # Outlier Removing

# In[38]:


import seaborn as sns


# In[39]:


sns.boxplot(data["Selling_Price"])


# In[40]:


sorted(data["Selling_Price"],reverse=True)


# In[41]:


data=data[~(data["Selling_Price"]>=33.0) & (data["Selling_Price"]<=35.0)]


# In[42]:


data.shape


# # Converting Object type to Numerical Type

# In[43]:


data.head(1)


# In[44]:


data["Fuel_Type"].unique()


# In[45]:


for label,content in data.items():
    if not pd.api.types.is_numeric_dtype(content):
          print(label)
  


# In[46]:


#This will turn all of the string value into category values
for label, content in data.items():
    if pd.api.types.is_string_dtype(content):
        data[label]=content.astype("category").cat.as_ordered()


# In[47]:


data.info()


# In[48]:


pd.Categorical(data["Fuel_Type"]).codes


# In[49]:


# Turn categorical variables into numbers and missing
for label,content in data.items():
    if not pd.api.types.is_numeric_dtype(content):
        # Add binary column to indicate whether sample had missing value
       data[label+"_is_missing"]=pd.isnull(content)
        # Turn categories into numbers and add +1
       data[label]=pd.Categorical(content).codes+1


# In[50]:


data.head()


# In[51]:


data.info()


# In[52]:


X=data.drop(["Car_Name","Selling_Price","Car_Name_is_missing","Fuel_Type_is_missing","Seller_Type_is_missing","Transmission_is_missing"],axis=1)
y=data['Selling_Price']


# In[53]:


X


# In[54]:


y


# # Splitting the dataset into Training set and Test set

# In[55]:


from sklearn.model_selection import train_test_split


# In[56]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=42)


# # Importing the models

# In[57]:


from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor


# # Model Training

# In[58]:


lr=LinearRegression()
lr.fit(X_train,y_train)

rf=RandomForestRegressor()
rf.fit(X_train,y_train)

xgb=GradientBoostingRegressor()
xgb.fit(X_train,y_train)

xg=XGBRegressor()
xg.fit(X_train,y_train)


# # Prediction on Test Data

# In[59]:


y_pred1=lr.predict(X_test)
y_pred2=rf.predict(X_test)
y_pred3=xgb.predict(X_test)
y_pred4=xg.predict(X_test)


# # Evaluating The Algorithm

# In[60]:


from sklearn import metrics


# In[61]:


score1=metrics.r2_score(y_test,y_pred1)
score2=metrics.r2_score(y_test,y_pred2)
score3=metrics.r2_score(y_test,y_pred3)
score4=metrics.r2_score(y_test,y_pred4)


# In[62]:


print(score1,score2,score3,score4)
print("Max score is",max(score1,score2,score3,score4))


# In[63]:


final_data=pd.DataFrame({"Models":["LR","RF","GBR","XG"],
             "R2_Score":[score1,score2,score3,score4]})
final_data


# In[66]:


final_data.plot(kind="bar",figsize=(10,6),color=["lightblue"]);


# # Save the Model

# In[65]:


xg=XGBRegressor()
xg_final=xg.fit(x,y)


# In[ ]:


import joblib


# In[ ]:


joblib.dump(xg_final,"Car_Price_Predictor")


# In[ ]:


model=joblib.load('Car_Price_Predictor')


# # Prediction on New Data

# In[ ]:


import pandas as pd
data_new = pd.DataFrame({
    'Present_Price':5.59,
    'Kms_Driven':27000,
    'Fuel_Type':0,
    'Seller_Type':0,
    'Transmission':0,
    'Owner':0,
    'Age':8},
     index=[0])


# In[ ]:


model.predict(data_new)


# In[ ]:


from tkinter import *
import pandas as pd

import joblib
def show_entry_fields():
    

    p1=float(e1.get()) 
    p2=float(e2.get()) 
    p3=float(e3.get())
    p4=float(e4.get())
    p5=float(e5.get())
    p6=float(e6.get())
    p7=float(e7.get())
    model=joblib.load('Car_Price_Predictor')
    data_new = pd.DataFrame({
    'Present_Price':p1,
    'Kms_Driven':p2,
    'Fule_Type':p3,
    'Seller_Type':p4,
    'Transmission':p5,
    'Owner':p6,
    'Age':p7},
     index=[0])
    result=model.predict(data_new)
    Label(master, text="Car Purchase amount").grid(row=8)
    Label(master, text=result).grid(row=10)
    print("Car Purchase amount", result[0])





master=Tk()

master.title("Car Price Prediction Using Machine Learning")

label=Label(master, text="Car Price Prediction Using Machine Learning",
        bg="black", fg = "white"). \
               grid(row=0,columnspan=2)


Label(master, text="Present_Price").grid(row=1)
Label(master, text="Kms_Driven").grid(row=2)
Label (master, text="Fuel_Type").grid(row=3)
Label(master, text="Seller Type").grid(row=4) 
Label(master, text="Transmission").grid(row=5)
Label(master, text="Owner").grid(row=6)
Label(master, text="Age").grid(row=7) 

e1 = Entry(master)
e2 = Entry(master)
e3 = Entry(master)
e4 = Entry(master)
e5 = Entry(master)
e6=  Entry(master)
e7 = Entry(master)
e1.grid(row=1, column=1)
e2.grid(row=2, column=1)
e3.grid(row=3, column=1)
e4.grid(row=4, column=1)
e5.grid(row=5, column=1)
e6.grid(row=6, column=1)
e7.grid(row=7, column=1)

Button(master, text='Predict', command=show_entry_fields).grid()

mainloop()


# In[ ]:





# In[ ]:




