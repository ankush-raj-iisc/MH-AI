#!/usr/bin/env python
# coding: utf-8

# # 0. Import necessary libraries

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import math
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


# # 1. Read the input file

# In[2]:


ratings = pd.read_csv('ratings_Electronics.csv',names=['userId', 'productId','Rating','timestamp'])


# In[3]:


#Check data snapshot to see if everything looks fine, timestamp column needs to be dropped later on
ratings.head()


# In[4]:


#Number of rows is 7.82MM and number of columns is 4
ratings.shape


# In[5]:


#Check the datatypes
#userID, productID are object while Rating is float, timestamp is integer
ratings.dtypes


# In[6]:


#The dataset is utilizing almost 240MB of disk space due to 7.82MM rows and 4 columns
#There will be memory issues unless we make the dataset more dense
ratings.info()


# In[7]:


#Countplot of the ratings, maximum user-products have got rating as 5 
sns.countplot(data=ratings, x='Rating');
#ratings["Rating"].value_counts().sort_values(ascending=False).plot(kind="bar")


# In[8]:


#Find the minimum and maximum ratings - It is between 1 and 5
print('Minimum rating is: %d' %(ratings.Rating.min()))
print('Maximum rating is: %d' %(ratings.Rating.max()))


# In[9]:


#Check for missing values - There are no missing values, so no imputation required
print('Number of missing values across columns: \n',ratings.isna().sum())


# In[10]:


#Number of products (~476K) is less than number of users(~4.2MM), so item-item colaborative filtering would make sense
#instead of user-user colaborative filtering

print("Electronic Data Summary")
print("="*100)
print("\nTotal # of Ratings :",ratings.shape[0])
print("Total # of Users   :", len(np.unique(ratings.userId)))
print("Total # of Products  :", len(np.unique(ratings.productId)))
print("\n")
print("="*100)


# In[11]:


#Dropping the Timestamp column
ratings.drop(['timestamp'], axis=1,inplace=True)


# In[12]:


ratings


# In[13]:


#Check and find the max ratings given by user for a particular item
max_ratings = ratings.groupby(['userId','productId'])['Rating'].max().sort_values(ascending=False)
max_ratings.head()


# In[14]:


#Check and find the min ratings given by user for a particular item
min_ratings = ratings.groupby(['userId','productId'])['Rating'].min().sort_values(ascending=False)
min_ratings.head()


# In[15]:


#From above min and max calculation, we see that the ratings are identical for the sample
#However for consistency let us remove duplicates if any just to be sure
ratings.drop_duplicates(inplace=True)


# In[16]:


#Analysis of how many product rating given by a particular user 
no_of_rated_products_per_user = ratings.groupby(by='userId')['Rating'].count().sort_values(ascending=False)
no_of_rated_products_per_user.head()


# In[17]:


#We have certain users who have rated only 1 product and few users have rated upto 520 products
#However the number of rated products per user is fairly skewed seeing the 5 point summary
#Max is 520 and 75% percentile is at 2
no_of_rated_products_per_user.describe().astype(int).T


# In[18]:


#Boxplot shows that we have few users who rate many items (appearing in outliers) but majority rate very few items
sns.boxplot(data=no_of_rated_products_per_user);


# In[19]:


#Let us look at the quantile view to understand where the ratings are concentrated
quantiles = no_of_rated_products_per_user.quantile(np.arange(0,1.01,0.01), interpolation='higher')


# In[20]:


#We can see that all the ratings are clustered at the top end of the quantile
#Basically the outliers that we saw earlier are reflected here in the peak
plt.figure(figsize=(10,10))
plt.title("Quantiles and their Values")
quantiles.plot()
# quantiles with 0.05 difference
plt.scatter(x=quantiles.index[::5], y=quantiles.values[::5], c='red', label="quantiles with 0.05 intervals")
# quantiles with 0.25 difference
plt.scatter(x=quantiles.index[::25], y=quantiles.values[::25], c='green', label = "quantiles with 0.25 intervals")
plt.ylabel('# ratings by user')
plt.xlabel('Value at the quantile')
plt.legend(loc='best')
plt.show()


# # 2. Take a subset of the dataset to make it less sparse/ denser.

# Keep the users only who has given 50 or more number of ratings 

# In[21]:


#We have 1,540 users who have rated more than or equal to 50 products
print('\n # of rated product more than 50 per user : {}\n'.format(sum(no_of_rated_products_per_user >= 50)) )


# In[22]:


#Getting the new dataframe which contains users who has given 50 or more ratings
new_df=ratings.groupby("userId").filter(lambda x:x['Rating'].count() >=50)


# In[23]:


#Products also have skewed ratings with majority of the products having very few ratings
no_of_ratings_per_product = new_df.groupby(by='productId')['Rating'].count().sort_values(ascending=False)

fig = plt.figure(figsize=plt.figaspect(.5))
ax = plt.gca()
plt.plot(no_of_ratings_per_product.values)
plt.title('# RATINGS per Product')
plt.xlabel('Product')
plt.ylabel('# ratings per product')
ax.set_xticklabels([])

plt.show


# In[24]:


#Boxplot shows that we have few products with large number of ratings, but majority have very low ratings
sns.boxplot(data=no_of_ratings_per_product);


# In[25]:


#Let us look at the quantile view to understand where the ratings are concentrated
quantiles = no_of_ratings_per_product.quantile(np.arange(0,1.01,0.01), interpolation='higher')


# In[26]:


#We can see that all the ratings are clustered at the top end of the quantile
#This reflects our finding above in the boxplot
plt.figure(figsize=(10,10))
plt.title("Quantiles and their Values")
quantiles.plot()
# quantiles with 0.05 difference
plt.scatter(x=quantiles.index[::5], y=quantiles.values[::5], c='red', label="quantiles with 0.05 intervals")
# quantiles with 0.25 difference
plt.scatter(x=quantiles.index[::25], y=quantiles.values[::25], c='green', label = "quantiles with 0.25 intervals")
plt.ylabel('# ratings by user')
plt.xlabel('Value at the quantile')
plt.legend(loc='best')
plt.show()


# In[27]:


#Average rating of the product across users
new_df.groupby('productId')['Rating'].mean().head()


# In[28]:


new_df.groupby('productId')['Rating'].mean().sort_values(ascending=False).head()


# In[29]:


#Total no of rating for product
new_df.groupby('productId')['Rating'].count().sort_values(ascending=False).head()


# In[30]:


ratings_mean_count = pd.DataFrame(new_df.groupby('productId')['Rating'].mean())


# In[31]:


ratings_mean_count['rating_counts'] = pd.DataFrame(new_df.groupby('productId')['Rating'].count())


# In[32]:


#Products which have high rating have fewer user reviews as seen below
ratings_mean_count.head()


# In[33]:


#The maximum number of ratings received for a product is 206
ratings_mean_count['rating_counts'].max()


# In[34]:


#Majority of the products have received 1 rating only and it is a right skewed distribution
plt.figure(figsize=(8,6))
#plt.rcParams['patch.force_edgecolor'] = True
ratings_mean_count['rating_counts'].hist(bins=100)


# In[35]:


#We see a left skewed distribution for the ratings
#There are clusters at each of the points 1,2,3,4,5 as that is where the means are concentrated
plt.figure(figsize=(8,6))
plt.rcParams['patch.force_edgecolor'] = True
ratings_mean_count['Rating'].hist(bins=100)


# In[36]:


#From the joint plot below it seems that popular products (higher ratings) tend to be rated more frequently
#To make people more engaged (bottom of the chart) we can start by recommending them based on popularity based system and then
#slowly graduate them to collaborative system once we have sufficient number of data points to giver personlized recommendation
plt.figure(figsize=(8,6))
plt.rcParams['patch.force_edgecolor'] = True
sns.jointplot(x='Rating', y='rating_counts', data=ratings_mean_count, alpha=0.4)


# In[37]:


#PDF and CDF for the number of ratings per product
#PDF is left skewed as majority of the products have very few ratings
ax1 = plt.subplot(121)
sns.kdeplot(no_of_ratings_per_product, shade=True, ax=ax1)
plt.xlabel('No of ratings by product')
plt.title("PDF")

ax2 = plt.subplot(122)
sns.kdeplot(no_of_ratings_per_product, shade=True, cumulative=True,ax=ax2)
plt.xlabel('No of ratings by product')
plt.title('CDF')

plt.show()


# In[38]:


no_of_ratings_per_user = new_df.groupby(by='userId')['Rating'].count().sort_values(ascending=False)


# In[39]:


#PDF and CDF for the number of ratings per user
#PDF is left skewed as majority of the users have given very few ratings
ax1 = plt.subplot(121)
sns.kdeplot(no_of_ratings_per_user, shade=True, ax=ax1)
plt.xlabel('No of ratings by user')
plt.title("PDF")

ax2 = plt.subplot(122)
sns.kdeplot(no_of_ratings_per_user, shade=True, cumulative=True,ax=ax2)
plt.xlabel('No of ratings by user')
plt.title('CDF')

plt.show()


# In[40]:


#Below is the bar graph showing product list of top 30 most popular products
popular_products = pd.DataFrame(new_df.groupby('productId')['Rating'].count())
most_popular = popular_products.sort_values('Rating', ascending=False)
most_popular.head(30).plot(kind = "bar")


# # 3. Split the data randomly into train and test dataset.

# ( For example, split it in 70/30 ratio)

# In[41]:


#Split the data into 70% train and 30% test
train_data, test_data = train_test_split(new_df, test_size = 0.30, random_state=0)
print(train_data.head(5))


# # 4. Popularity Based Method

# In[42]:


#Split the data into 70% train and 30% test
train_data, test_data = train_test_split(new_df, test_size = 0.30, random_state=0)
print(train_data.head(5))


# In[43]:


#Count of user_id for each unique product as recommendation score 
train_data_grouped = train_data.groupby('productId').agg({'userId': 'count'}).reset_index()
train_data_grouped.rename(columns = {'userId': 'score'},inplace=True)
train_data_grouped.head()


# In[44]:


#Sort the products on recommendation score 
train_data_sort = train_data_grouped.sort_values(['score', 'productId'], ascending = [0,1])     
#Generate a recommendation rank based upon score 
train_data_sort['Rank'] = train_data_sort['score'].rank(ascending=0, method='first') 
#Get the top 5 recommendations 
popularity_recommendations = train_data_sort.head(5) 
popularity_recommendations 


# In[45]:


# Use popularity based recommender model to make predictions for a user
# As we note this list will be same for all the users
def recommend(user_id):     
    user_recommendations = popularity_recommendations 
          
    #Add user_id column for which the recommendations are being generated 
    user_recommendations['userId'] = user_id 
      
    #Bring user_id column to the front 
    cols = user_recommendations.columns.tolist() 
    cols = cols[-1:] + cols[:-1] 
    user_recommendations = user_recommendations[cols] 
          
    return user_recommendations 


# In[46]:


# This list is user choice, since this is popularity based recommendation method irrespective of user 
# same products will be suggested
find_recom = ['A15BHBF0L0HV1F','A3VVJIZXLL1QFP','AFHY3XJJ6NCAI','A2WPY1SNQPCC00','AJMJREC90WJVP']   
for i in find_recom:
    print("Here is the recommendation for the userId: %s\n" %(i))
    print(recommend(i))    
    print("\n") 


# In[47]:


train_data_sort.head()
#print(pred)


# In[48]:


test_data.head()


# In[49]:


#Calculating the RMSE of the popularity based recommendation system
#Rating present in the test data is the actual rating (Act_rating)
test_data2 = test_data.copy()
#ratings.drop(['timestamp'], axis=1,inplace=True)
test_data2.drop(['userId'],axis=1,inplace=True)
test_data2.rename(columns = {'Rating':'Act_rating'}, inplace = True)


# In[50]:


test_data2.head()


# In[51]:


#Count of user_id for each unique product as recommendation score 
train_data_grouped2 = train_data.groupby('productId').agg({'Rating': 'sum'}).reset_index()
train_data_grouped2.rename(columns = {'Rating': 'Sum_rating'},inplace=True)
train_data_grouped2.head()


# In[52]:


train_data_inner = pd.merge(train_data_grouped2, train_data_sort)


# In[53]:


train_data_inner.head()


# In[54]:


#Obtain the average rating of the product across users
train_data_inner["Avg_Rating"] = train_data_inner["Sum_rating"]/train_data_inner["score"]


# In[55]:


train_data_inner.head()


# In[56]:


#Merge the train data having average rating with the test data having actual rating at product level
test_data_inner = pd.merge(train_data_inner, test_data2)


# In[57]:


test_data_inner.head()


# In[58]:


#Now the merged data has both actual rating (Act_rating) and predicted rating (Avg_rating)
#Now RMSE can be calculated
test_data_inner.head()


# In[59]:


#RMSE for popularity based recommender system is 1.09
mse = mean_squared_error(test_data_inner["Act_rating"], test_data_inner["Avg_Rating"])
rmse = math.sqrt(mse)
print("RMSE for popularity based recommendation system:", rmse)


# In[60]:


ratings.head()


# In[63]:


# pip install turicreate


# In[64]:


# #Importing turicreate
# #This package takes SFrame instead of dataframe so typecasting accordingly
# import turicreate
# train_data2 = turicreate.SFrame(train_data)
# test_data2 = turicreate.SFrame(test_data)


# In[65]:


# python3 -m venv myenv


# In[66]:


import pandas as pd
from sklearn.metrics import mean_squared_error
from math import sqrt

# Assuming train_data and test_data are pandas DataFrames and have been loaded appropriately

# Compute the mean rating for each product
mean_ratings = train_data.groupby('productId')['Rating'].mean().reset_index()
mean_ratings.rename(columns={'Rating': 'meanRating'}, inplace=True)

# Merge the mean ratings with the test data to get predicted ratings
test_data_with_preds = test_data.merge(mean_ratings, on='productId', how='left')

# For users not present in test data or products not present, we might need to handle NaNs if any
test_data_with_preds['meanRating'].fillna(test_data_with_preds['Rating'].mean(), inplace=True)  # Example strategy

# Now, let's simulate recommendations for the provided users by selecting top 5 products
# Note: This method does not use 'userId' since popularity-based recommendations are not personalized
top_products = mean_ratings.nlargest(5, 'meanRating')
for user in ['AKM1MP6P0OYPR', 'A2CX7LUOHB2NDG', 'A2NWSAGRHCP8N5', 'A2WNBOD3WNDNKT', 'A1GI0U4ZRJA8WN']:
    print(f"Recommendations for {user}:")
    print(top_products)

# Compute RMSE on the test data
rmse = sqrt(mean_squared_error(test_data_with_preds['Rating'], test_data_with_preds['meanRating']))
print(f"RMSE: {rmse}")


# In[68]:


# Function to make recommendations for a user
def recommend_items(user_id, item_similarity_df, user_item_matrix, k=5):
    # Get the items the user has rated
    rated_items = user_item_matrix.loc[user_id].dropna().index
    # Sum of similarity scores for items rated by the user
    sim_scores = item_similarity_df[rated_items].sum(axis=1)
    # Remove items already rated by the user
    sim_scores = sim_scores.drop(index=rated_items)
    # Get top k items
    top_items = sim_scores.nlargest(k).index.tolist()
    return top_items

# Example recommendations for the specified users
user_ids = ['AKM1MP6P0OYPR', 'A2CX7LUOHB2NDG', 'A2NWSAGRHCP8N5', 'A2WNBOD3WNDNKT', 'A1GI0U4ZRJA8WN']
for user_id in user_ids:
    print(f"Recommendations for {user_id}: {recommend_items(user_id, item_similarity_df, user_item_matrix)}")

# Compute RMSE (optional, as direct RMSE calculation might not be straightforward for item-item recommender)
# Here, you would compare predicted ratings (based on similar items) to actual ratings in test_data if necessary.


# In[69]:


import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error
from math import sqrt


# In[70]:


# Assuming train_data and test_data are pandas DataFrames

# Create a user-item matrix for training data
user_item_matrix = train_data.pivot_table(index='userId', columns='productId', values='Rating').fillna(0)


# In[71]:


# Compute item-item cosine similarity
item_similarity = cosine_similarity(user_item_matrix.T)
item_similarity_df = pd.DataFrame(item_similarity, index=user_item_matrix.columns, columns=user_item_matrix.columns)


# In[72]:


user_item_matrix = user_item_matrix.astype(float)  # Ensure all data is float


# In[73]:


from scipy.sparse import csr_matrix

# Convert the DataFrame to a sparse matrix
user_item_matrix_sparse = csr_matrix(user_item_matrix.values)

# Compute cosine similarity on the transposed sparse matrix
item_similarity = cosine_similarity(user_item_matrix_sparse.T)


# In[74]:


from scipy.sparse.linalg import svds
from scipy.sparse import csr_matrix

# Assuming user_item_matrix_sparse is already a csr_matrix
U, sigma, Vt = svds(user_item_matrix_sparse, k=50)  # Reduce to 50 dimensions
item_factors = Vt.T  # Transpose Vt to get item factors

# Compute cosine similarity on reduced dimensions
item_similarity_reduced = cosine_similarity(item_factors)


# In[81]:


get_ipython().system('pip install scikit-surprise')


# In[83]:


from surprise import SVD, Dataset, Reader
from surprise.model_selection import cross_validate, train_test_split
from surprise import accuracy
from surprise.model_selection import PredefinedKFold
import pandas as pd


# In[84]:


# Assuming train_data and test_data are pandas DataFrames with columns ['userId', 'productId', 'Rating']

# Load the data into surprise's format
reader = Reader(rating_scale=(train_data['Rating'].min(), train_data['Rating'].max()))
data = Dataset.load_from_df(train_data[['userId', 'productId', 'Rating']], reader)


# In[85]:


# Split data into training and test set
trainset = data.build_full_trainset()


# In[86]:


# Define the SVD algorithm
algo = SVD(n_factors=8, n_epochs=50, lr_all=0.005, reg_all=0.02)


# In[87]:


# Train the algorithm on the trainset
algo.fit(trainset)


# In[88]:


# Convert test set into surprise format
testset = [tuple(x) for x in test_data[['userId', 'productId', 'Rating']].values]


# In[89]:


# Predict ratings for the testset
predictions = algo.test(testset)


# In[90]:


# Calculate RMSE
accuracy.rmse(predictions)


# In[91]:


# To recommend items for specific users, we need to construct a function since Surprise does not provide direct recommendations
def get_top_n_recommendations(predictions, n=5):
    # First map the predictions to each user.
    top_n = {}
    for uid, iid, true_r, est, _ in predictions:
        if uid not in top_n:
            top_n[uid] = []
        top_n[uid].append((iid, est))

    # Then sort the predictions for each user and retrieve the k highest ones.
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]

    return top_n


# In[92]:


# Get top 5 recommendations for specified users
users = ['AKM1MP6P0OYPR', 'A2CX7LUOHB2NDG', 'A2NWSAGRHCP8N5', 'A2WNBOD3WNDNKT', 'A1GI0U4ZRJA8WN']
top_n = get_top_n_recommendations(predictions, n=5)

# Print the recommended items for each user
for uid in users:
    print(f"Top 5 recommendations for {uid}: {[iid for (iid, _) in top_n.get(uid, [])]}")


# In[93]:


from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split
from surprise import accuracy
import pandas as pd

# Assuming train_data and test_data are already defined pandas DataFrames with columns ['userId', 'productId', 'Rating']

# Define the reader with the rating scale
reader = Reader(rating_scale=(train_data['Rating'].min(), train_data['Rating'].max()))

# Load the dataset from the dataframe
data = Dataset.load_from_df(train_data[['userId', 'productId', 'Rating']], reader)

# Build the full trainset
trainset = data.build_full_trainset()

# Define the SVD algorithm
algo = SVD(n_factors=8, n_epochs=50, lr_all=0.005, reg_all=0.02)

# Train the algorithm on the trainset
algo.fit(trainset)

# Get a list of all product IDs
all_product_ids = trainset.to_raw_iid(range(trainset.n_items))

def get_top_n_recommendations(predictions, n=5):
    top_n = {}
    for uid, iid, true_r, est, _ in predictions:
        if uid not in top_n:
            top_n[uid] = []
        top_n[uid].append((iid, est))

    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]
    return top_n

# Predict ratings for all pairs (user, item) that are NOT in the training set
testset = trainset.build_anti_testset()

predictions = algo.test(testset)

top_n = get_top_n_recommendations(predictions, n=5)

# Print the recommended items for each user
for uid in users:
    user_recommendations = top_n.get(uid, [])
    print(f"Top 5 recommendations for {uid}: {[iid for (iid, _) in user_recommendations]}")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




