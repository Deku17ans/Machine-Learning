#!/usr/bin/env python
# coding: utf-8

# ## pandasを用いてcsvファイルを読み込む
#   

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


train = pd.read_csv("train_my_friends_data.csv")
test = pd.read_csv("test_my_friends_data.csv")


# In[3]:


train.head()


# In[4]:


test.head()


# In[5]:


train.describe()


# ## 文字列のデータを数字に変える
# * school:juniorhigh=0 high=1 university=2
# * affinity:sobad=0 bad=1 normal=2 good=3 verygood=4
# * sex:male=0 female=1
# * last_school:high=0 university=1
# * fun:no=0 common=1 yes=2
# * noisy:no=0 common=1 yes=2
# * clever:no=0 common=1 yes=2

# In[6]:


train["school"][train["school"] == "juniorhigh" ] = 0
train["school"][train["school"] == "high"] = 1
train["school"][train["school"] == "university"] = 2
train["affinity"][train["affinity"] == "sobad"] = 0
train["affinity"][train["affinity"] == "bad"] = 1
train["affinity"][train["affinity"] == "normal"] = 2
train["affinity"][train["affinity"] == "good"] = 3
train["affinity"][train["affinity"] == "verygood"] = 4 
train["sex"][train["sex"] == "male"] = 0
train["sex"][train["sex"] == "female"] = 1
train["last_school"][train["last_school"] == "high"] = 0
train["last_school"][train["last_school"] == "university"] = 1
train["fun"][train["fun"] == "no"] = 0
train["fun"][train["fun"] == "common"] = 1
train["fun"][train["fun"] == "yes"] = 2
train["noisy"][train["noisy"] == "no"] = 0
train["noisy"][train["noisy"] == "common"] = 1
train["noisy"][train["noisy"] == "yes"] = 2
train["clever"][train["clever"] == "no"] = 0
train["clever"][train["clever"] == "common"] = 1
train["clever"][train["clever"] == "yes"] = 2

train.head(10)


# ## テストデータも変える

# In[7]:


test["school"][test["school"] == "juniorhigh" ] = 0
test["school"][test["school"] == "high"] = 1
test["school"][test["school"] == "university"] = 2
test["affinity"][test["affinity"] == "sobad"] = 0
test["affinity"][test["affinity"] == "bad"] = 1
test["affinity"][test["affinity"] == "normal"] = 2
test["affinity"][test["affinity"] == "good"] = 3
test["affinity"][test["affinity"] == "verygood"] = 4 
test["sex"][test["sex"] == "male"] = 0
test["sex"][test["sex"] == "female"] = 1
test["last_school"][test["last_school"] == "high"] = 0
test["last_school"][test["last_school"] == "university"] = 1
test["fun"][test["fun"] == "no"] = 0
test["fun"][test["fun"] == "common"] = 1
test["fun"][test["fun"] == "yes"] = 2
test["noisy"][test["noisy"] == "no"] = 0
test["noisy"][test["noisy"] == "common"] = 1
test["noisy"][test["noisy"] == "yes"] = 2
test["clever"][test["clever"] == "no"] = 0
test["clever"][test["clever"] == "common"] = 1
test["clever"][test["clever"] == "yes"] = 2

test.head()


# In[8]:


from sklearn import tree


# In[9]:


train_target = train["affinity"].values
train_features = train[["school","sex","last_school","fun","noisy","clever"]].values
test_target = test["affinity"].values
test_features = test[["school","sex","last_school","fun","noisy","clever"]].values


# In[10]:


#print(train)
print(test)


# In[11]:


train_target = train_target.astype("int")
train_features = train_features.astype("int")
test_target = test_target.astype("int")
test_features = test_features.astype("int")


# ## 下記部分でValueError: Unknown label type: 'unknown'
# * dtypeで中身がどんなタイプになっているかを確認する

# In[12]:


my_tree = tree.DecisionTreeClassifier()
my_tree.fit(train_features,train_target)


# ## 決定木は一番上に重要な説明変数が来る

# In[13]:


prediction = my_tree.predict(test_features)


# In[15]:


print(prediction)
print(test_target)


# In[ ]:




