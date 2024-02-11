# Development-and-Implementation-of-an-Advanced-Recommendation-System-for-E-commerce.
Different recommendation models are used by online retailers such as Amazon and Flipkart to offer varying options to users. Currently, Amazon uses item-to-item collaborative filtering, which generates real-time, high-quality suggestions even for large data sets.
# Attribute Information:
1. userId : Every user identified with a unique id
2. productId : Every product identified with a unique id \n
3. Rating : Rating of the corresponding product by the corresponding user
4. timestamp : Time of the rating ( ignore this column for this exercise)
# Learning Outcomes:
● Exploratory Data Analysis
● Creating a Recommendation system using real data
● Collaborative filtering
# Objective:
Build a recommendation system to recommend products to customers based on the their previous ratings for other products.
# Steps and tasks:
1. Read and explore the given dataset. (Rename column/add headers, plot histograms, find data characteristics)
2. Take a subset of the dataset to make it less sparse/ denser. ( For example, keep the users only who has given 50 or more number of ratings )
3. Split the data randomly into train and test dataset. ( For example, split it in 70/30 ratio)
4. Build Popularity Recommender model.
5. Build Collaborative Filtering model.
6. Evaluate both the models. ( Once the model is trained on the training data, it can be used to compute the error (RMSE) on predictions made on the test data.
7. Get top - K ( K = 5) recommendations. Since our goal is to recommend new products for each user based on his/her habits, we will recommend 5 new products.
8. Summarise your insights.

# Due to the large size od data sets(350 MB, 1M datapoints), I couldn't upload the CSV file.
