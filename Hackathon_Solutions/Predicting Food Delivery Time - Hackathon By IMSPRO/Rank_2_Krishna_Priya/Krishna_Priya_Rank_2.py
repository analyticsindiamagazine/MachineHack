#!/usr/bin/env python
# coding: utf-8
Predicting Food Delivery Time - Hackathon by IMS Proschool


Size of training set: 11,094 records

Size of test set: 2,774 records

Hackathon closes on December 10, 2019

Evaluation Metric:

The final score will be calculated based on accuracy or the number of true predictions using the confusion matrix.

FEATURES:

Restaurant: A unique ID that represents a restaurant.
Location: The location of the restaurant.
Cuisines: The cuisines offered by the restaurant.
Average_Cost: The average cost for one person/order.
Minimum_Order: The minimum order amount.
Rating: Customer rating for the restaurant.
Votes: The total number of customer votes for the restaurant.
Reviews: The number of customer reviews for the restaurant.
Delivery_Time: The order delivery time of the restaurant. (Target Classes) 
# In[1]:


import pandas as pd
import numpy as np


# In[2]:


train = pd.read_excel('Data_Train.xlsx')
test = pd.read_excel('Data_Test.xlsx')
data = train.append(test, ignore_index=True,sort=False)
print(train.shape,test.shape,data.shape)


# In[3]:


data.head()


# In[4]:


data.isnull().sum()


# In[5]:


data.nunique()


# In[6]:


data.info()


# In[7]:


data['Delivery_Time'].value_counts()


# In[9]:


data['Cuisines'] = data['Cuisines'].str.lower()
data['Cuisines'] = data['Cuisines'].str.replace(' ','')
Cuisines_list = data['Cuisines'].str.split(',')
from collections import Counter
Cuisines_counter = Counter(([a for b in Cuisines_list.tolist() for a in b]))
Cuisines_counter


# In[10]:


# Cuisines_list_set = set([a for b in Cuisines_list.tolist() for a in b])
# for Cuisines in Cuisines_list_set:
#     print(Cuisines, ": ", train[train['Cuisines'].str.contains(Cuisines)]['Delivery_Time'].median())


# In[11]:


data['Cuisines'] = data['Cuisines'].str.replace('rolls','fastfood')
data['Cuisines'] = data['Cuisines'].str.replace('burger','fastfood')
data['Cuisines'] = data['Cuisines'].str.replace('wraps','fastfood')
data['Cuisines'] = data['Cuisines'].str.replace('streetfood','fastfood')
data['Cuisines'] = data['Cuisines'].str.replace('momos','fastfood')
data['Cuisines'] = data['Cuisines'].str.replace('sandwich','fastfood')
data['Cuisines'] = data['Cuisines'].str.replace('fingerfood','fastfood')
data['Cuisines'] = data['Cuisines'].str.replace('barfood','fastfood')
data['Cuisines'] = data['Cuisines'].str.replace('rawmeats','fastfood')
data['Cuisines'] = data['Cuisines'].str.replace('hotdogs','fastfood')
data['Cuisines'] = data['Cuisines'].str.replace('cafe','fastfood')
data['Cuisines'] = data['Cuisines'].str.replace('pizza','fastfood')

data['Cuisines'] = data['Cuisines'].str.replace('icecream','desserts')
data['Cuisines'] = data['Cuisines'].str.replace('mithai','desserts')
data['Cuisines'] = data['Cuisines'].str.replace('bakery','desserts')
data['Cuisines'] = data['Cuisines'].str.replace('bubbletea','desserts')
data['Cuisines'] = data['Cuisines'].str.replace('mishti','desserts')
data['Cuisines'] = data['Cuisines'].str.replace('paan','desserts')
data['Cuisines'] = data['Cuisines'].str.replace('frozenyogurt','desserts')

data['Cuisines'] = data['Cuisines'].str.replace('italian','european')
data['Cuisines'] = data['Cuisines'].str.replace('german','european')
data['Cuisines'] = data['Cuisines'].str.replace('spanish','european')
data['Cuisines'] = data['Cuisines'].str.replace('steak','european')
data['Cuisines'] = data['Cuisines'].str.replace('mediterranean','european')
data['Cuisines'] = data['Cuisines'].str.replace('brazilian','european')
data['Cuisines'] = data['Cuisines'].str.replace('belgian','european')
data['Cuisines'] = data['Cuisines'].str.replace('french','european')
data['Cuisines'] = data['Cuisines'].str.replace('portuguese','european')
data['Cuisines'] = data['Cuisines'].str.replace('african','european')
data['Cuisines'] = data['Cuisines'].str.replace('greek','european')

data['Cuisines'] = data['Cuisines'].str.replace('mexican','american')
data['Cuisines'] = data['Cuisines'].str.replace('bbq','american')
data['Cuisines'] = data['Cuisines'].str.replace('roastchicken','american')
data['Cuisines'] = data['Cuisines'].str.replace('charcoalchicken','american')
data['Cuisines'] = data['Cuisines'].str.replace('tex-mex','american')
data['Cuisines'] = data['Cuisines'].str.replace('southamerican','american')

data['Cuisines'] = data['Cuisines'].str.replace('arabian','middleeast')
data['Cuisines'] = data['Cuisines'].str.replace('kebab','middleeast')
data['Cuisines'] = data['Cuisines'].str.replace('lebanese','middleeast')
data['Cuisines'] = data['Cuisines'].str.replace('afghan','middleeast')
data['Cuisines'] = data['Cuisines'].str.replace('iranian','middleeast')
data['Cuisines'] = data['Cuisines'].str.replace('middleeastern','middleeast')
data['Cuisines'] = data['Cuisines'].str.replace('turkish','middleeast')
data['Cuisines'] = data['Cuisines'].str.replace('israeli','middleeast')

data['Cuisines'] = data['Cuisines'].str.replace('chinese','chinese')

data['Cuisines'] = data['Cuisines'].str.replace('kerala','regional')
data['Cuisines'] = data['Cuisines'].str.replace('bihari','regional')
data['Cuisines'] = data['Cuisines'].str.replace('Lucknowi','regional')
data['Cuisines'] = data['Cuisines'].str.replace('mangalorean','regional')
data['Cuisines'] = data['Cuisines'].str.replace('bengali','regional')
data['Cuisines'] = data['Cuisines'].str.replace('andhra','regional')
data['Cuisines'] = data['Cuisines'].str.replace('assamese','regional')
data['Cuisines'] = data['Cuisines'].str.replace('maharashtrian','regional')
data['Cuisines'] = data['Cuisines'].str.replace('chettinad','regional')
data['Cuisines'] = data['Cuisines'].str.replace('parsi','regional')
data['Cuisines'] = data['Cuisines'].str.replace('odia','regional')
data['Cuisines'] = data['Cuisines'].str.replace('tamil','regional')
data['Cuisines'] = data['Cuisines'].str.replace('northeastern','regional')
data['Cuisines'] = data['Cuisines'].str.replace('bohri','regional')
data['Cuisines'] = data['Cuisines'].str.replace('goan','regional')
data['Cuisines'] = data['Cuisines'].str.replace('gujarati','regional')
data['Cuisines'] = data['Cuisines'].str.replace('rajasthani','regional')
data['Cuisines'] = data['Cuisines'].str.replace('naga','regional')
data['Cuisines'] = data['Cuisines'].str.replace('awadhi','regional')
data['Cuisines'] = data['Cuisines'].str.replace('kashmiri','regional')
data['Cuisines'] = data['Cuisines'].str.replace('malwani','regional')

data['Cuisines'] = data['Cuisines'].str.replace('thai','seafood')
data['Cuisines'] = data['Cuisines'].str.replace('konkan','seafood')
data['Cuisines'] = data['Cuisines'].str.replace('srilankan','seafood')
data['Cuisines'] = data['Cuisines'].str.replace('poké','seafood')

#data['Cuisines'] = data['Cuisines'].str.replace('thai','asian')
data['Cuisines'] = data['Cuisines'].str.replace('indonesian','asian')
data['Cuisines'] = data['Cuisines'].str.replace('japanese','asian')
data['Cuisines'] = data['Cuisines'].str.replace('burmese','asian')
data['Cuisines'] = data['Cuisines'].str.replace('sushi','asian')
data['Cuisines'] = data['Cuisines'].str.replace('cantonese','asian')
data['Cuisines'] = data['Cuisines'].str.replace('tibetan','asian')
data['Cuisines'] = data['Cuisines'].str.replace('malaysian','asian')
data['Cuisines'] = data['Cuisines'].str.replace('vietnamese','asian')
data['Cuisines'] = data['Cuisines'].str.replace('korean','asian')
data['Cuisines'] = data['Cuisines'].str.replace('bangladeshi','asian')
data['Cuisines'] = data['Cuisines'].str.replace('nepalese','asian')


data['Cuisines'] = data['Cuisines'].str.replace('tea','beverages')
data['Cuisines'] = data['Cuisines'].str.replace('juices','beverages')
data['Cuisines'] = data['Cuisines'].str.replace('coffee','beverages')

data['Cuisines'] = data['Cuisines'].str.replace('hyderabadi','biryani')
data['Cuisines'] = data['Cuisines'].str.replace('lucknowi','biryani')

data['Cuisines'] = data['Cuisines'].str.replace('indian','northindian')
data['Cuisines'] = data['Cuisines'].str.replace('modernindian','northindian')
data['Cuisines'] = data['Cuisines'].str.replace('modernnorthindian','northindian')
data['Cuisines'] = data['Cuisines'].str.replace('northindian','northindian')
data['Cuisines'] = data['Cuisines'].str.replace('northnorthindian','northindian')


data['Cuisines'] = data['Cuisines'].str.replace('southnorthindian','southindian')

data['Cuisines'] = data['Cuisines'].str.replace('salad','healthyfood')


# In[12]:


data['Cuisines'] = data['Cuisines'].str.lower()
data['Cuisines'] = data['Cuisines'].str.replace(' ','')
Cuisines_list = data['Cuisines'].str.split(',')
from collections import Counter
Cuisines_counter = Counter(([a for b in Cuisines_list.tolist() for a in b]))
Cuisines_counter


# In[13]:


# for Cuisines in Cuisines_counter.keys():
#     data[Cuisines] = 0
#     data.loc[data['Cuisines'].str.contains(Cuisines), Cuisines] = 1
# del data['Cuisines']


# In[14]:


data['Delivery_Time'] = data['Delivery_Time'].str.replace(' minutes','')
data['Delivery_Time']=pd.to_numeric(data['Delivery_Time'])


# In[15]:


data['City']=data.Location.str.rpartition(',')[2]
data['Locality']=data.Location.str.rpartition(',')[0]


# In[16]:


data.Locality[data.City.str.contains('Delhi University-GTB Nagar')]='Delhi University-GTB Nagar'
data.Locality[data.City.str.contains('Mumbai Central')]='Mumbai Central'
data.Locality[data.City.str.contains('Majestic')]='Majestic'
data.Locality[data.City.str.contains('Delhi Cantt.')]='Delhi Cantt'
data.Locality[data.City.str.contains('Pune University')]='Pune University'
data['Locality'] = data['Locality'].str.strip()
data['Locality'] = data['Locality'].str.lower()


# In[17]:


data.City[data.City.str.contains('Delhi University-GTB Nagar')]='Delhi'
data.City[data.City.str.contains('Mumbai CST Area')]='Mumbai'
data.City[data.City.str.contains('Mumbai Central')]='Mumbai'
data.City[data.City.str.contains('India Gate')]='Delhi'
data.City[data.City.str.contains('Delhi Cantt.')]='Delhi'
data.City[data.City.str.contains('Maharashtra')]='Pune'
data.City[data.City.str.contains('Pune University')]='Pune'
data.City[data.City.str.contains('Gurgoan')]='Gurgaon'
data.City[data.City.str.contains('Electronic City')]='Bangalore'

# data.City[data.City.str.contains('Whitefield')]='Bangalore'
# data.City[data.City.str.contains('Marathalli')]='Bangalore'

data['City'] = data['City'].str.strip()
data['City'] = data['City'].str.lower()
del data['Location']


# In[18]:


data1=data.copy()


# In[19]:


data['Average_Cost'] = data['Average_Cost'].str.replace("[^0-9]","")
data['Average_Cost'] = data['Average_Cost'].str.strip()
data['Average_Cost']=pd.to_numeric(data['Average_Cost'])

# data.Average_Cost.fillna(data.groupby(['City','Locality'])['Average_Cost'].transform('mean'), inplace=True)

data['Minimum_Order'] = data['Minimum_Order'].str.replace("[^0-9]","")
data['Minimum_Order'] = data['Minimum_Order'].str.strip()
data['Minimum_Order']=pd.to_numeric(data['Minimum_Order'])


# In[20]:


data.Rating = data.Rating.replace("NEW",np.nan)
data.Rating = data.Rating.replace("-",np.nan)
data.Rating = data.Rating.replace("Opening Soon",np.nan)
data.Rating = data.Rating.replace("Temporarily Closed",np.nan)
data.Rating = data.Rating.astype('float')
# data.Rating.fillna(data.groupby(['City','Locality'])['Rating'].transform('mean'), inplace=True)
#data.Rating.fillna(0, inplace=True)


# In[21]:


data.Votes = data.Votes.replace("-",np.nan)
data.Votes = data.Votes.astype('float')
# data.Votes.fillna(data.groupby(['City','Locality'])['Votes'].transform('mean'), inplace=True)
#data.Votes.fillna(0, inplace=True)


# In[22]:


data.Reviews = data.Reviews.replace("-",np.nan)
data.Reviews = data.Reviews.astype('float')
# data.Reviews.fillna(data.groupby(['City','Locality'])['Reviews'].transform('mean'), inplace=True)
#data.Reviews.fillna(0, inplace=True)


# In[23]:


print(data.shape)


# In[24]:


data['total_div_of_reviews'] = data['Votes']/data['Reviews']
data['total_sum_of_reviews'] = data['Votes']*data['Reviews']

data['total_div_of_ratings'] = data['Votes']/data['Rating']
data['total_sum_of_ratings'] = data['Votes']*data['Rating']

data['total_div_of_Minimum_Order'] = data['Votes']/data['Minimum_Order']
data['total_sum_of_Minimum_Order'] = data['Votes']*data['Minimum_Order']

data['total_div_of_Average_Cost'] = data['Votes']/data['Average_Cost']
data['total_sum_of_Average_Cost'] = data['Votes']*data['Average_Cost']

data['total_div_of_ratings_Reviews'] = data['Rating']/data['Reviews']
data['total_sum_of_ratings_1'] = data['Rating']*data['Reviews']

data['total_div_of_Minimum_Order_1'] = data['Rating']/data['Minimum_Order']
data['total_sum_of_Minimum_Order_1'] = data['Rating']*data['Minimum_Order']

data['total_div_of_Average_Cost_1'] = data['Rating']/data['Average_Cost']
data['total_sum_of_Average_Cost_1'] = data['Rating']*data['Average_Cost']

data['total_div_of_reviews_Minimum_Order_1'] = data['Reviews']/data['Minimum_Order']
data['total_sum_of_reviews_Minimum_Order_1'] = data['Reviews']*data['Minimum_Order']

data['total_div_of_reviews_Average_Cost_1'] = data['Reviews']/data['Average_Cost']
data['total_sum_of_reviews_Average_Cost_1'] = data['Reviews']*data['Average_Cost']

data['total_div_of_Minimum_Order_Average_Cost_1'] = data['Average_Cost']/data['Minimum_Order']
data['total_sum_of_Minimum_Order_Average_Cost_1'] = data['Average_Cost']*data['Minimum_Order']


# In[25]:


data1['total_div_of_reviews']  = data['total_div_of_reviews']
data1['total_div_of_ratings']  = data['total_div_of_ratings']
data1['total_sum_of_Average_Cost']  = data['total_sum_of_Average_Cost']
data1['total_div_of_Average_Cost_1']  = data['total_div_of_Average_Cost_1']
data1['total_div_of_Minimum_Order_1']  = data['total_div_of_Minimum_Order_1']
data1['total_sum_of_Minimum_Order']  = data['total_sum_of_Minimum_Order']
data1['total_div_of_Average_Cost']  = data['total_div_of_Average_Cost']
data1['total_div_of_ratings_Reviews']  = data['total_div_of_ratings_Reviews']
data1['total_div_of_reviews_Average_Cost_1']  = data['total_div_of_reviews_Average_Cost_1']
data1['total_sum_of_Average_Cost_1']  = data['total_sum_of_Average_Cost_1']
data1['total_sum_of_reviews_Minimum_Order_1']  = data['total_sum_of_reviews_Minimum_Order_1']
data1['total_div_of_Minimum_Order']  = data['total_div_of_Minimum_Order']
data1['total_sum_of_ratings_1']  = data['total_sum_of_ratings_1']
data1['total_sum_of_reviews']  = data['total_sum_of_reviews']
data1['total_sum_of_ratings']  = data['total_sum_of_ratings']
data1['total_sum_of_reviews_Average_Cost_1']  = data['total_sum_of_reviews_Average_Cost_1']
data1['total_sum_of_Minimum_Order_1']  = data['total_sum_of_Minimum_Order_1']
data1['total_div_of_reviews_Minimum_Order_1']  = data['total_div_of_reviews_Minimum_Order_1']
data1['total_div_of_Minimum_Order_Average_Cost_1']  = data['total_div_of_Minimum_Order_Average_Cost_1']
data1['total_sum_of_Minimum_Order_Average_Cost_1']  = data['total_sum_of_Minimum_Order_Average_Cost_1']


# In[26]:


data1.shape


# In[27]:


import seaborn as sns
import matplotlib.pyplot as plt
colormap = plt.cm.RdBu
plt.figure(figsize=(13,13))
sns.heatmap(data1.corr(),linewidths=0.1,square=True, cmap=colormap, linecolor='white', annot=True)


# In[28]:


train_x=data1[~data1['Delivery_Time'].isnull()]
train_y=train_x['Delivery_Time']
test_x=data1[data1['Delivery_Time'].isnull()]

del train_x['Delivery_Time']
del test_x['Delivery_Time']


# In[29]:


from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(train_x, train_y, test_size = 0.2, random_state = 18121995, stratify=train_y)


# # CATBOOST

# In[30]:


categorical_features_indices = np.where(X_train.dtypes =='object')[0]
categorical_features_indices


# # Stratified 5-Fold

# In[31]:


import catboost
def make_classifier():
    model = catboost.CatBoostClassifier(
        iterations=30000,
        random_state=18121995,
        learning_rate=0.01,
        loss_function='MultiClass',
        early_stopping_rounds=200,
    )
    return model


# In[32]:


import os, sys, datetime
from time import time
import catboost
from sklearn.model_selection import StratifiedKFold

start_time = time()

NFOLDS = 5
folds = StratifiedKFold(n_splits=NFOLDS, shuffle=False, random_state=18121995)
models = []
scores = []
for fold, (train_ids, test_ids) in enumerate(folds.split(train_x, train_y)):
    print('● Fold :', fold+1)
    model = make_classifier()
    model.fit(train_x.loc[train_ids], train_y.loc[train_ids], 
              eval_set=(train_x.loc[test_ids], train_y.loc[test_ids]),
              use_best_model=False,
              verbose=500,
              cat_features=categorical_features_indices)    
    models.append(model)
    print('\n')

print('finished in {}'.format( 
    str(datetime.timedelta(seconds=time() - start_time))))


# # Max Voting

# In[37]:


from scipy import stats

predictions = []
for model in models:
    predictions.append(model.predict(train_x).astype(int))
predictions = np.concatenate(predictions, axis=1)
df = pd.DataFrame(predictions)

vote = stats.mode(predictions, axis=1)[0].reshape(-1)
df['vote'] = vote
df['y'] = train_y
df


# In[38]:


predictions = []
for model in models:
    predictions.append(model.predict(test_x))
predictions = np.concatenate(predictions, axis=1)
# Voting
predictions = stats.mode(predictions, axis=1)[0].reshape(-1)
print(predictions.shape)


# In[39]:


a=pd.DataFrame()
# predcb=predcb.astype(int)
# predcb=predcb.flatten()
a['Delivery_Time']=np.round(predictions).astype('int')
a['Delivery_Time'] = a['Delivery_Time'].replace({30:'30 minutes', 45:'45 minutes', 65:'65 minutes', 120:'120 minutes', 20:'20 minutes', 80:'80 minutes', 10:'10 minutes'})
a.to_excel('cb_all_cat_1.0.xlsx', index=False)


# # Feature Importance

# In[40]:


sorted(zip(model.feature_importances_,train_x),reverse=True)


# In[41]:


import matplotlib.pyplot as plt
feat_importances=pd.Series(model.feature_importances_,index=train_x.columns)
feat_importances.nsmallest(1000).plot(kind='barh')
plt.show()


# In[ ]:




