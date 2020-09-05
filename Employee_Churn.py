#!/usr/bin/env python
# coding: utf-8

# In[497]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# In[337]:


train=pd.read_csv("Train.csv")
test=pd.read_csv("Test.csv")


# In[338]:


print(train.shape)
print(test.shape)


# In[339]:


combined_data=pd.concat([train,test],axis=0,sort=False)
combined_data=combined_data.reset_index(drop=True)


# In[342]:


test.head()


# In[345]:


train.corr()


# In[346]:


reg_data=combined_data[["Employee_ID","Gender","Relationship_Status","Time_of_service","Age"]]


# In[348]:


reg_data=reg_data[reg_data["Time_of_service"].notna()]


# In[349]:


reg_data.isnull().sum()


# In[350]:


reg_data_dum=pd.get_dummies(reg_data,columns=["Gender","Relationship_Status"])


# In[351]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# In[352]:


reg_train=reg_data_dum[reg_data_dum["Age"].notnull()]


# In[353]:


reg_test=reg_data_dum[reg_data_dum["Age"].isnull()]


# In[354]:


reg_train_X=reg_train.loc[:, reg_train.columns != 'Age']


# In[355]:


reg_train_Y=reg_train["Age"]


# In[356]:


reg_test_X=reg_test.loc[:, reg_test.columns != 'Age']


# In[357]:


reg_train.shape, reg_test_X.shape


# In[358]:


reg_model = LinearRegression()


# In[359]:


reg_train_X=reg_train_X.drop(['Employee_ID'], axis = 1) 
reg_test_X=reg_test_X.drop(["Employee_ID"],axis=1)


# In[360]:


reg_train_X.isnull().sum()


# In[361]:


reg_model.fit(reg_train_X,reg_train_Y)


# In[362]:


age=reg_model.predict(reg_test_X)


# In[363]:


df_Age = pd.DataFrame(data=age,  columns=["Age"])


# In[364]:


df_age=round(df_Age)


# In[365]:


reg_test["Age"]=df_age["Age"]


# In[366]:


reg_test.head()


# In[368]:


reg_test=reg_test.reset_index()


# In[369]:


reg_test["Age"]=df_age["Age"]


# In[370]:


reg_test=reg_test.set_index("index")


# In[371]:


indexes=reg_test.index.values


# In[372]:


combined_data["Age"].update(reg_test["Age"])

combined_data.isnull().sum()
# In[382]:


combined_data.describe()


# In[392]:


combined_data.isnull().sum()


# In[384]:


combined_data['Time_of_service'] = combined_data['Time_of_service'].fillna((combined_data['Time_of_service'].median()))


# In[388]:


combined_data['Pay_Scale'] = combined_data['Pay_Scale'].fillna((combined_data['Pay_Scale'].median()))


# In[390]:


combined_data['Work_Life_balance'] = combined_data['Work_Life_balance'].fillna((combined_data['Work_Life_balance'].median()))


# In[391]:


combined_data['VAR2'] = combined_data['VAR2'].fillna((combined_data['VAR2'].median()))
combined_data['VAR4'] = combined_data['VAR4'].fillna((combined_data['VAR4'].median()))


# In[396]:


combined_data.columns


# In[415]:


combined_clean=pd.get_dummies(combined_data,columns=["Gender","Relationship_Status","Education_Level","Hometown","Unit",'Decision_skill_possess','Compensation_and_Benefits'])


# In[418]:


combined_clean.columns


# In[513]:


dt_train_x=combined_clean.iloc[:,combined_clean.columns!=("Attrition_rate")]
dt_test_x=combined_clean.iloc[:,combined_clean.columns!="Attrition_rate"]


# In[514]:


dt_test_x.shape


# In[515]:


dt_train_x=dt_train_x.iloc[:7000,1:51]
dt_train_y=combined_clean.iloc[:7000,combined_clean.columns=="Attrition_rate"]
sub_test_x=dt_test_x.iloc[7000:,1:51]
dt_test_y=combined_clean.iloc[7000:,combined_clean.columns=="Attrition_rate"]


# In[531]:


sub_test_x.head()


# In[518]:


X_train, X_test, y_train, y_test = train_test_split(dt_train_x, dt_train_y, test_size=0.2, random_state=123)


# In[521]:


xgbr = xgb.XGBRegressor(verbosity=0) 
xgbr.fit(X_train,y_train)


# In[522]:


xgbr.score(X_train, y_train)


# In[523]:


y_pred=xgbr.predict(X_test)


# In[524]:


rmse=mean_squared_error(y_test,y_pred)


# In[525]:


submission_score=100*max(0,1-rmse)
submission_score


# In[526]:


submission_file=xgbr.predict(sub_test_x)


# In[529]:


sub=pd.DataFrame(submission_file)


# In[530]:


sub.to_csv("predicted.csv")

