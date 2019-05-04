#!/usr/bin/env python
# coding: utf-8

# ## モジュールのインポート

# In[1]:


import pandas as pd
import numpy as np
import sys


# In[2]:


sys.setrecursionlimit(10000)


# In[3]:


train = pd.read_csv("train_house_price.csv")
test = pd.read_csv("test_house_price.csv")
train.shape, test.shape


# ## 欠損値を把握する関数

# In[4]:


def grasp_lack(df):
    for column in df.columns:
        lack = df[column].isnull().sum()
        print(column,":",lack)


# ## trainの欠損値を補完

# In[5]:


train.MasVnrArea = train.MasVnrArea.fillna(train.MasVnrArea.median())
train.GarageYrBlt = train.GarageYrBlt.fillna(train.GarageYrBlt.median())
train.MasVnrType = train.MasVnrType.fillna("None")
train.BsmtQual = train.BsmtQual.fillna("Gd")
train.BsmtCond = train.BsmtCond.fillna("TA")
train.BsmtExposure = train.BsmtExposure.fillna("No")
train.BsmtFinType1 = train.BsmtFinType1.fillna("Unf")
train.BsmtFinType2 = train.BsmtFinType2.fillna("Unf")
train.Electrical = train.Electrical.fillna("SBrkr")
train.GarageType = train.GarageType.fillna("Attchd")
train.GarageFinish = train.GarageFinish.fillna("Unf")
train.GarageQual = train.GarageQual.fillna("TA")
train.GarageCond = train.GarageCond.fillna("TA")


# ## testの欠損値を補完

# In[6]:


test.GarageArea = test.GarageArea.fillna(test.GarageArea.median())
test.GarageCars = test.GarageCars.fillna(test.GarageCars.median())
test.GarageYrBlt = test.GarageYrBlt.fillna(test.GarageYrBlt.median())
test.BsmtHalfBath = test.BsmtHalfBath.fillna(test.BsmtHalfBath.median())
test.BsmtFullBath = test.BsmtFullBath.fillna(test.BsmtFullBath.median())
test.TotalBsmtSF = test.TotalBsmtSF.fillna(test.TotalBsmtSF.median())
test.BsmtUnfSF  = test.BsmtUnfSF.fillna(test.BsmtUnfSF.median())
test.BsmtFinSF1 = test.BsmtFinSF1.fillna(test.BsmtFinSF1.median())
test.BsmtFinSF2 = test.BsmtFinSF2.fillna(test.BsmtFinSF2.median())
test.MasVnrArea = test.MasVnrArea.fillna(test.MasVnrArea.median())
test.MSZoning = test.MSZoning.fillna("RL")
test.Utilities = test.Utilities.fillna("AllPub")
test.Exterior1st = test.Exterior1st.fillna("VinylSd")
test.Exterior2nd = test.Exterior2nd.fillna("VinylSd")
test.MasVnrType = test.MasVnrType.fillna("None")
test.BsmtQual = test.BsmtQual.fillna("TA")
test.BsmtCond = test.BsmtCond.fillna("TA")
test.BsmtExposure = test.BsmtExposure.fillna("No")
test.BsmtFinType1 = test.BsmtFinType1.fillna("GLQ")
test.BsmtFinType2 = test.BsmtFinType2.fillna("Unf")
test.KitchenQual = test.KitchenQual.fillna("TA")
test.Functional = test.Functional.fillna("Typ")
test.GarageType = test.GarageType.fillna("Attchd")
test.GarageFinish = test.GarageFinish.fillna("Unf")
test.GarageQual = test.GarageQual.fillna("TA")
test.GarageCond = test.GarageCond.fillna("TA")
test.SaleType = test.SaleType.fillna("WD")


# ## train,testの欠損値が大きいデータを削除

# In[7]:


del train["LotFrontage"]
del train["Alley"]
del train["FireplaceQu"]
del train["PoolQC"]
del train["Fence"]
del train["MiscFeature"]

del test["LotFrontage"]
del test["Alley"]
del test["FireplaceQu"]
del test["PoolQC"]
del test["Fence"]
del test["MiscFeature"]


# ## train,testにis_trainカラムを追加

# In[8]:


train["is_train"] = 1
test["is_train"] = 0
train.shape, test.shape


# ## train,testを結合させる
# * pd.concat([df1, df2])

# In[9]:


train_dropped = train.drop(["SalePrice"], axis=1)
train_test_combined = pd.concat([train_dropped, test], axis=0)
train_test_combined.shape


# ## dtypesを全て表示する関数

# In[10]:


def print_dtypes(df):
    for column in df.columns:
        dtype = df[column].dtypes
        print(column, ":", dtype)


# ## category型に変換する

# In[11]:


object_column_list = ["MSZoning","Street","LotShape","LandContour","Utilities","LotConfig","LandSlope","Neighborhood",
                      "Condition1","Condition2","BldgType","HouseStyle","RoofStyle","RoofMatl","Exterior1st","Exterior2nd",
                      "MasVnrType","ExterQual","ExterCond","Foundation","BsmtQual","ExterCond","Foundation","BsmtQual","BsmtCond",
                      "BsmtExposure","BsmtFinType1","BsmtFinType2","Heating","HeatingQC","CentralAir","Electrical","KitchenQual",
                      "Functional","GarageType","GarageFinish","GarageQual","GarageCond","PavedDrive","SaleType","SaleCondition"]

for object_column in object_column_list:
    train_test_combined[object_column] = train_test_combined[object_column].astype("category")


# ## category型からint型に変換する

# In[12]:


for object_columns in train_test_combined.columns:
    if train_test_combined[object_columns].dtypes != "int64":
        if train_test_combined[object_columns].dtypes != "int8":
            if train_test_combined[object_columns].dtypes != "float64":
                train_test_combined[object_columns] = train_test_combined[object_columns].cat.codes


# In[13]:


#train_test_combined.nunique()


# ## df_trainとdf_testに再び分割する

# In[14]:


df_train = train_test_combined.loc[train_test_combined["is_train"] == 1]
df_test = train_test_combined.loc[train_test_combined["is_train"] == 0]

df_train = df_train.drop(["is_train"], axis=1)
df_test = df_test.drop(["is_train"], axis=1)

df_train.shape, df_test.shape


# ## df_trainにSalePriceカラムを戻す

# In[15]:


df_train["SalePrice"] = train["SalePrice"]


# ## ↳ここまでで、データの整形が完了

# In[16]:


train_features = df_train.drop(["SalePrice"], axis=1)
train_target = df_train.SalePrice


# In[17]:


test_features = df_test


# In[18]:


from sklearn.ensemble import RandomForestRegressor


# In[19]:


my_model = RandomForestRegressor()
my_pred = my_model.fit(train_features, train_target)


# In[20]:


my_answer = my_pred.predict(test_features)
my_answer


# ## 予測データをcsvに書き出す
# * pd.DataFrame(データの配列、インデックスの配列、列名の配列)

# In[21]:


#Idを取得
Id = np.array(test["Id"]).astype(int)

#Id.shape, my_answer.shape

#my_predとIdをDataFrameへ落とし込む
my_solution = pd.DataFrame(my_answer, Id, columns=["SalePrice"])

#my_SalePrice_pred.csvとして書き出し
my_solution.to_csv("my_SalePrice_pred.csv", index_label = ["Id"])


# ## ----------------------------------------------------------------------------

# In[30]:


from sklearn.model_selection import train_test_split


# In[31]:


my_train, my_test = train_test_split(df_train, train_size=0.8)


# In[32]:


my_train.shape, my_test.shape


# In[33]:


my_train_features = my_train.drop(["SalePrice"], axis=1)
my_train_target = my_train.SalePrice

my_test_features = my_test.drop(["SalePrice"], axis=1)
my_test_target = my_test.SalePrice


# In[34]:


my_other_model = RandomForestRegressor()
my_other_pred = my_other_model.fit(my_train_features, my_train_target)
my_other_pred


# In[35]:


my_other_answer = my_other_pred.predict(my_test_features)


# In[44]:


my_test_target.values, my_other_answer


# In[66]:


from sklearn import metrics

list_1 = list(my_test_target.values)
list_2 = []

for i in list(my_other_answer):
    list_2.append(int(i))

#print(list_1)
#print(list_2)


# In[64]:


#print(metrics.classification_report(list_1, list_2))
#print(metrics.confusion_matrix(list_1, list_2))

