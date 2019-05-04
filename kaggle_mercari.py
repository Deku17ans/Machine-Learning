#!/usr/bin/env python
# coding: utf-8

# ## ライブラリのインポート

# In[1]:


import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from IPython.display import display
from sklearn import metrics
from sklearn.model_selection import train_test_split
pd.set_option("display.float_format", lambda x:"%.5f" % x)


# ## Pandasでデータフレームを読み込む
# * int64:64ビットの符号付整数

# In[2]:


types_dict_train = {"train_id":"int64", "item_condition_id":"int8", "price":"float64", "shipping":"int8"}
types_dict_test = {"test_id":"int64", "item_condition_id":"int8", "shipping":"int8"}


# * tsvファイルからPandas DataFrameへ読み込み
#     * tsvファイルとはタブで区切られたファイルのことである
#   
# * delimiterは値と値を区切る文字を指定する

# In[3]:


train = pd.read_csv("mercari_train.tsv", delimiter="\t", low_memory=True, dtype=types_dict_train)
test = pd.read_csv("mercari_test.tsv", delimiter="\t", low_memory=True, dtype=types_dict_test)


# In[4]:


train.head()


# In[5]:


test.head()


# In[6]:


train.shape, test.shape


# ## 各データの情報
# * train_id / test _id – ユーザー投稿のID
# * name – 投稿のタイトル。タイトルに価格に関する情報がある場合はメルカリが事前に削除をして[rm]と置き換えています。
# * item_condition_id – ユーザーが指定した商品の状態
# * category_name – 投稿カテゴリー
# * brand_name – ブランドの名前
# * price – 訓練データのみ。実際に売られた価格。米ドル表示。今回のチャレンジの予測ターゲットとなります。
# * shipping – 送料のフラグ。「1」は販売者負担。「0」は購入者負担。
# * item_description – ユーザーが投稿した商品説明の全文。タイトルと同様に価格情報がある場合は[rm]と置き換えられています。

# ## 統計量の確認
# * pd.option_context()はwithブロック内で一時的に設定を変更する
# * display(df)とすればいつものように出力される
# * describeの引数
#     * include="all"とすればすべての型が対象となる
# * transpose()で表の縦軸と横軸を入れ替えることができる

# In[7]:


#def display_all(df):
    #with pd.option_context("display.max_rows", 1000):
        #with pd.option_context("display.max_columns", 1000):
            #display(df)
 
# trainの基本統計量を表示
#display_all(train.describe(include='all').transpose())


# In[8]:


#display_all(train.describe(include="all"))
train.describe(include="all").transpose()


# ## 性別や血液型、郵便番号などの名義尺度、サイズや階層、評価など順序に意味のある順序尺度はカテゴリデータと呼ばれる
# * 「train["train_id"]」と「train.train_id」は同じ結果が得られる

# In[9]:


# trainのカテゴリ名、商品説明、投稿タイトル、ブランド名のデータタイプを「category」へ変換する
train.category_name = train.category_name.astype("category")
train.item_description = train.item_description.astype("category")
train.name = train.name.astype("category")
train.brand_name = test.brand_name.astype("category")

# testのカテゴリ名、商品説明、投稿タイトル、ブランド名のデータタイプを「category」へ変換する
test.category_name = test.category_name.astype('category')
test.item_description = test.item_description.astype('category')
test.name = test.name.astype('category')
test.brand_name = test.brand_name.astype('category')
 
#train.dtypes, test.dtypes


# ## ユニーク値を調べる
# * IDはしっかりとユニーク値となっている

# In[10]:


train.apply(lambda x: x.nunique())


# In[11]:


test.apply(lambda x: x.nunique())


# ## 欠損データの確認
# * NAN(Not A Number)は数字ではないデータ
# * isnull()は欠損データがあるとTrueで返ってくる

# In[12]:


#trainの欠損データの個数
train.isnull().sum()
#trainの欠損データの% train.shape[0]は(1482535, 8)の一番目を表している
#train.isnull().sum()/train.shape[0]


# In[13]:


#testの欠損データの個数
test.isnull().sum()


# ## メルカリデータの事前処理
# * 文字列のデータをPandasの関数を使って数値へと変換する。そのためtrainとtestで別々に処理を行わず、連結して処理を行う

# In[14]:


#それぞれのカラム名を変更する
train = train.rename(columns = {"train_id":"id"})
test = test.rename(columns = {"test_id":"id"})

#train.head()
#test.head()


# In[15]:


#両方のセットに「is_train」のカラムを追加
# 1:trainのデータ、0:testのデータ
train["is_train"] = 1
test["is_train"] = 0


# ## dropの引数axis=0の時、行を削除する。axis=1の時列を削除する

# In[16]:


# trainのprice以外のデータをtestと連結 concat関数はデータを連結するときに使う
train_test_combine = pd.concat([train.drop(['price'], axis=1),test],axis=0)
train_test_combine.head()
#train_test_combine.shape


# In[17]:


#train_test_combineのデータタイプをcategoryの変換
train_test_combine.category_name = train_test_combine.category_name.astype("category")
train_test_combine.item_description = train_test_combine.item_description.astype("category")
train_test_combine.name = train_test_combine.name.astype("category")
train_test_combine.brand_name = train_test_combine.brand_name.astype("category")

train_test_combine.dtypes

#文字列を「.cat.codes」で数値へ変換する
train_test_combine.name = train_test_combine.name.cat.codes
train_test_combine.category_name = train_test_combine.category_name.cat.codes
train_test_combine.brand_name = train_test_combine.brand_name.cat.codes
train_test_combine.item_description = train_test_combine.item_description.cat.codes

train_test_combine.head()


# In[18]:


train_test_combine.dtypes


# * 「is_train」のフラグでcombineからtestとtrainに切り分ける

# In[19]:


df_test = train_test_combine.loc[train_test_combine["is_train"] == 0]
df_train = train_test_combine.loc[train_test_combine["is_train"] == 1]

#is_trainのカラムをtrainとtestのデータフレームから落とす
df_test = df_test.drop(["is_train"], axis=1)
df_train = df_train.drop(["is_train"],axis=1)

df_test.shape, df_train.shape


# In[20]:


#df_trainにpriceを戻す
df_train["price"] = train.price
df_train.head()


# In[21]:


df_train["price"] = df_train["price"].apply(lambda x:np.log(x) if x>0 else x)
df_train.head()


# ## ↳ここまででデータの処理が完了

# In[25]:


# x ＝ price以外の全ての値、y = price（ターゲット）で切り分ける
# xを訓練データ、yを教師データとする
x_train, y_train = df_train.drop(['price'], axis=1), df_train.price
 
# モデルの作成
m = RandomForestRegressor()
m.fit(x_train, y_train)
 
# スコアを表示
m.score(x_train, y_train)

