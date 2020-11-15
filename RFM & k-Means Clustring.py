#!/usr/bin/env python
# coding: utf-8

# 
# 
# ## Customer Segmentation (RFM & k-Means Clustering)
# 

# 
# #### Abstract

# This notebook aims at analyzing the content of customer dataset that have been collected from Kaggle (https://www.kaggle.com/sergeymedvedev/customer_segmentation) to answer the question, "Who are the most loyal customers to the businesses?". Suppose that we have a company that selling some of the product, and we want to know how well does the sells performance of the product. We have the data that we can analyze, but what kind of analysis that we can perform? Well, we can segment customers based on their buying behavior on the market. In fact, customer segmentation is a method of dividing customers into groups or clusters on the basis of common characteristics. Businesses around the world would like to understand customer purchasing behavior, so customer segmentation allows them to create personalized offers for each individual group and precisely target the customers who have specific needs and desires. With creating clusters companies might identify new market segments on which they can focus more, as it might be more lucrative. 

# #### Table of Contents
# -  Introduction - [Go to the Introduction](#Introduction)   
# -  About The Dataset - [Go to the About The Dataset](#About-The-Dataset)
# -  Importing Packages - [Go to the Importing Packages](#Importing-Packages)
# -  Importing Data - [Go to the Importing Data](#Importing-Data)
# -  Exploring & Cleaning Data - [Go to the Exploring & Cleaning Data](#Exploring-&-Cleaning-Data)
# -  Customer Segmentation using RFM Modeling - [Go to the Customer Segmentation using RFM Modeling](#Customer-Segmentation-using-RFM-Modeling)
# -  Visualization - [Go to the Visualization](#Visualization)
# -  k-Means Clustring - [Go to the k-Means Clustering](#kMeans-Clustering)
#   - Data Prepration - [Go to the Data Prepration](#Data-Prepration)
#   - Standardization - [Go to the Standardization](#Standardization)
# -  Results and Discussion - [Go to the Results and Discussion](#Results-and-Discussion)
# -  Conclusion and Summary - [Go to the Conclusion and Summary](#Conclusion-and-Summary)
# -  References - [Go to the References](#References)
# 

#  **<a id='Introduction'>1. Introduction**</a>

# In the Retail sector, the various chain of hypermarkets generate an exceptionally large amount of data. This data is generated on a daily basis across the stores. This extensive database of customers transactions needs to be analyzed for designing profitable strategies.
# All customers have different kinds of needs and with the increase in customer base and transaction, it is not easy to understand the requirement of each customer. Identifying potential customers can improve the marketing campaign, which ultimately increases the sales. Segmentation can play a better role in grouping those customers into various segments.
# In this notebook customer segmentation was implemented by utilizing RFM analysis and for each customer the RFM scores were calculated, followed by applying unsupervised ML technique (k-Means) to add customers into different groups or clusters based on their RFM scores. It shows all the steps to create the clusters of customers according to their behavior. These clusters were created for most loyal customers as well as the customers who are on the verge of churning out. 

# **<a id='About-The-Dataset'>2. About The Dataset**</a>

# The dataset has been collected from Kaggle (https://www.kaggle.com/sergeymedvedev/customer_segmentation), consisting of 541910 records of customers and 8 attributes. The dataset contains transaction dates between 01/12/2010 and 9/12/2011.<br>
# 
# Attribute Information:<br>
# - **InvoiceNo** - Invoice number. The number that is uniquely assigned to each transaction. (6 digit Numerical)
# - **StockCode** - Product code. The number that is uniquely assigned to each distinct product. (5 digit Numerical) 
# - **Description** - Product type. (Categorical)
# - **Quantity** - The quantities of each product per transaction. (Numerical)
# - **InvoiceDate** - Invoice date and time. Day and time of generated transaction. (String Date)
# - **UnitPrice** - Unit price. Product price per unit. (Numerical)
# - **CustomerID** - Customer number. The number that is uniquely assigned to each customer. (5 digit Numerical) 
# - **Country** - Country name. The name of the country where a customer resides. (Categorical) 

# **<a id='Importing-Packages'>3. Importing Packages**</a>

# In[210]:


# Import Libraries
get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


# **<a id='Importing-Data'>3. Importing Data**</a>

# In[211]:


# Import Retail data containing transaction from 01/12/2010 to 09/12/2011

cust_data = pd.read_csv('CustomerData.csv', encoding='unicode_escape')  
cust_data.head()


#  **<a id='Exploring-&-Cleaning-Data'>4. Exploring & Cleaning Data**</a> 

# In[212]:


cust_data.info()


# Validating records using shape function

# In[213]:


# Number of rows and columns in the dataset
cust_data.shape


# In this dataset as you can see there is a country column which has repeated values for the UK. We observe that country column has similar values for one particular country, so this column is a good candidate to group the customers based on, in order to see the corresponding distribution for different countries.   

# In[214]:


# Customer Distribution by Country

country_cust_data = cust_data[['Country','CustomerID']].drop_duplicates() # dropping duplicates
country_cust_data.groupby(['Country'])['CustomerID'].aggregate('count').reset_index().sort_values('CustomerID', ascending=False) # counting customers for each country


# As we can see the majority of the customers are from the UK, so we keep the data from the UK only and we filter out the rest of data from other countries.  

# In[215]:


# Keeping only UK data
cust_data = cust_data.query("Country == 'United Kingdom'").reset_index(drop=True)


# In[216]:


# Checking for missing values in the dataset
cust_data.isnull().sum(axis=0) # axis=0; Checking the columns


# There are some missing values in Description and CustomerID columns. For now we can ignore the missing values of the Description column.

# In[217]:


# Removing misssing values from CustomerID column, can ignore missing values in Description column
cust_data = cust_data[pd.notnull(cust_data['CustomerID'])]
cust_data.shape


# In[218]:


# Checking if there are any negative values in the Quantity column, as quantity can't have negative value
cust_data.Quantity.min()


# In[219]:


# Checking if there are any negative values in UnitPrice column, as UnitPrice can't have negative value
cust_data.UnitPrice.min()


# We need to clean the data and filter out all the negative values of the Quantity column.

# In[220]:


# Filtering out records with negative values
cust_data = cust_data[(cust_data['Quantity'] > 0)]
cust_data.shape


# Converting date values from string to date format to do date operation related to Recency calculation.

# In[221]:


# Converting the string date field to datetime format
cust_data['InvoiceDate'] = pd.to_datetime(cust_data['InvoiceDate'])
cust_data['InvoiceDate']


# Calculating total amount related to Monetary calculation.

# In[222]:


# adding new TotalAmount column to the dataframe
cust_data['TotalAmount'] = cust_data['Quantity'] * cust_data['UnitPrice']
cust_data['TotalAmount']


# In[223]:


cust_data.shape


# In[224]:


cust_data.head()


# **<a id='Customer-Segmentation-using-RFM-Modeling'>5. Customer Segmentation using RFM Modeling**</a> 
# 
# RFM (Recency, Frequency, Monetary) analysis is a behavior-based approach grouping customers into segments. It groups the customers on the basis of their previous purchase transactions. How recently, how often, and how much did a customer buy.

# In[225]:


# Recency: Latest Date (last invoice Data)
# Frequency: Count of invoice Number of transaction(s)
# Monetary: Sum of Total Amount of purchaed by each customer

import datetime as dt

# Setting latest date 2011-12-10 as last date was 2011-12-09
Latest_Date = dt.datetime(2011,12,10) # using this latest date to calculate the number of days from the recent purchases

# Creating RFM Modeling scores for each customer
# Calculating Recancy, Frequency and Monetary 
RFMScores = cust_data.groupby('CustomerID').agg({'InvoiceDate': lambda x: (Latest_Date - x.max()).days, 'InvoiceNo': lambda x: len(x), 'TotalAmount': lambda x: x.sum()})

# Converting InvoiceDate into type int to do the mathematical operation in an easy way
RFMScores['InvoiceDate'] = RFMScores['InvoiceDate'].astype(int)

# Renaming column names to Recency, Frequency anf Monetary
RFMScores.rename(columns={'InvoiceDate': 'Recency',
                          'InvoiceNo': 'Frequency',
                          'TotalAmount': 'Monetary'}, inplace=True)
RFMScores.reset_index().head()


# **<a id='Visualization'>6. Visualization**</a> 

# In[226]:


# Descritive Statistics (Recency)
RFMScores.Recency.describe()


# In[227]:


# Recency distribution plot
x = RFMScores['Recency']
ax = sns.distplot(x)


# In[228]:


# Descritive Statistics (Frequency)
RFMScores.Frequency.describe()


# In[229]:


# Frequency distribution plot
x = RFMScores['Frequency']
ax = sns.distplot(x)


# In[230]:


# Descritive Statistics (Monetary)
RFMScores.Monetary.describe()


# In[231]:


# Monetary distribution plot
x = RFMScores['Monetary']
ax = sns.distplot(x)


# As we can see data is right skewed for Recency, Frequency and Monetary. Since in skewed data, the tail region may act as an outlier for the statistical model and we know that outliers adversely affect the model’s performance especially regression-based models, so before they can be used they need to be normalized. 

# Creating quantiles (0.25, 0.50, 0.75) to subdivide the entire data set into four group based on Recency, Frequency and Monetary values we are calculated.

# In[232]:


# Splitting into four segments using quantiles
quantiles = RFMScores.quantile(q=[0.25,0.50,0.75])

# Adding the quantiles into dictionary
quantiles = quantiles.to_dict()  


# In[233]:


quantiles


# Defining RScoring() and FMScoring() functions in order to create segments which will be directed by values 1, 2, 3, and 4, with assigning 1 to the lowest value of Recency (lower the value, the better it is - customer is more engaged with a specific brand) and assigning 1 to the highest value of Frequency and Monetary (higher the value, the better it is).

# In[234]:


# Creating R, F and M segments 
def RScoring(x,p,d):
    if x <= d[p][0.25]:
        return 1
    elif x<= d[p][0.50]:
        return 2
    elif x<= d[p][0.75]:
        return 3
    else:
        return 4
    
def FMScoring(x,p,d):
    if x<= d[p][0.25]:
        return 4
    elif x<= d[p][0.50]:
        return 3
    elif x<= d[p][0.75]:
        return 2
    else:
        return 1


# In[235]:


# Adding R, F, and M segment value columns in the dataset to show R, F and M segment values.
RFMScores['R'] = RFMScores['Recency'].apply(RScoring, args=('Recency', quantiles,))
RFMScores['F'] = RFMScores['Frequency'].apply(FMScoring, args=('Frequency', quantiles,))
RFMScores['M'] = RFMScores['Monetary'].apply(FMScoring, args=('Monetary', quantiles,))
RFMScores.head()


# Creating RFMGroup string columns for each customer by using R, F and M values from the dataset to easily check what particular segment or group a customer belongs to.
# 
# Adding another new column called RFMScore by summing up the values of R, F, and M values which gives the score to a customer's loyalty or engagement. The lower this RFM score is the more loyal customers would be, as well as more engaged s/he would be with the brand. 

# In[236]:


# Adding RFMGroup value column to the dataframe showing combined concatenated score of RFM
RFMScores['RFMGroup'] = RFMScores.R.map(str)+RFMScores.F.map(str)+RFMScores.M.map(str)

# Adding RFMScore value column to the dataframe showin total sum of RFMGrop values.
RFMScores['RFMScore'] = RFMScores[['R', 'F', 'M']].sum(axis=1)
RFMScores.head()


# Assigning loyalty levels to each customer based on RFMScore. To do so, first, we create four levels (Platinum, Gold, Silver and Bronze) and find out each customer belongs to which of these groups. If the customer is in the Platinum group s/he is the most valuable and loyal customer. If the customer belongs to the Bronze group s/he hasn't purchased from quite long and may be on the verge of churning out.

# Next we are calculating the score cuts based on panda's qcut() method. 
# 
# The simplest use of qcut is to define the number of quantiles and let pandas figure out how to divide up the data. In the below, we tell pandas to create 4 equal sized groupings of the data.
# 
# One of the challenges with this approach is that the bin labels are not very easy to explain to an end user. For instance, if we wanted to divide our customers into 4 groups we can explicitly label the bins to make them easier to interpret.

# In[237]:


# Assigning loyalty level to each customer
Loyalty_Level = ['Platinum', 'Gold', 'Silver', 'Bronze']
Score_Cuts = pd.qcut(RFMScores.RFMScore, q=4, labels=Loyalty_Level)
RFMScores['RFM_Loyalty_Level'] = Score_Cuts.values
RFMScores.reset_index().head()


# The above result explicitly defined the range of quantiles and defined the labels (labels=Loyalty_Level) to use when representing the bins. As expected, we now have an equal distribution of customers across the 4 bins and the results are displayed in an easy to understand manner.
# So, this is the way we can assign the loyalty levels to each customer.

# In[238]:


# Validating the data for RFMGroup = 111
RFMScores[RFMScores['RFMGroup'] == '111'].sort_values('Monetary', ascending=False).reset_index().head(10)


# Customers with RFMGroup 111 have loyalty levels as Platinum. So, first, the data was filtered on the basis of the RFM group equals to 111 which is the best group, then sorted in descending order based on Monetary value. You can see customers with the highest Monetary value in Platinum group have been shown at the top. Based on this RFM modeling what should be the marketing strategy? Customers with the RFM group of 111 are the best customers and we can try to cross sell other products of our brand and also encourage them to sign up for loyalty programs to get some elite benefits like free same-day shipping priority access to newly launched products, etc. On the other hand, for customers with RFM group 444 companies may try to offer some rewards or coupons to trigger their spending.  

# Visualizing RFM modeling based on clusters.

# In[239]:


## Importing Libraries

import chart_studio as cs
import plotly.offline as po
import plotly.graph_objs as gobj


# In[240]:


## We can see all observations on the graph
graph = RFMScores.query("Monetary < 50000 and Frequency < 2000")


# In[241]:


## Plotting Recency vs Frequency
plot_data = [
    gobj.Scatter(
        x=graph.query("RFM_Loyalty_Level == 'Bronze'")['Recency'],
        y=graph.query("RFM_Loyalty_Level == 'Bronze'")['Frequency'],
        mode='markers', # each data point is represented as marker point
        name='Bronze',
        marker= dict(size=10, line=dict(width=1), color='blue', opacity=0.8)),
    
     gobj.Scatter(
         x=graph.query("RFM_Loyalty_Level == 'Silver'")['Recency'],
         y=graph.query("RFM_Loyalty_Level == 'Silver'")['Frequency'],
         mode='markers',
         name='Silver',
         marker= dict(size=9, line=dict(width=1), color='yellow', opacity=0.5)),
    
     gobj.Scatter(
         x=graph.query("RFM_Loyalty_Level == 'Gold'")['Recency'],
         y=graph.query("RFM_Loyalty_Level == 'Gold'")['Frequency'],
         mode='markers',
         name='Gold',
         marker= dict(size=8, line=dict(width=1), color='green', opacity=0.9)),
    
     gobj.Scatter(
         x=graph.query("RFM_Loyalty_Level == 'Platinum'")['Recency'],
         y=graph.query("RFM_Loyalty_Level == 'Platinum'")['Frequency'],
         mode='markers',
         name='Platinum',
         marker= dict(size=7, line=dict(width=1), color='red', opacity=0.9)),]

plot_layout = gobj.Layout(
             yaxis={'title': "Frequency"},
             xaxis={'title': "Recency"},
             title='Segments',
             autosize=False,
             width=600,
             height=400)
fig = gobj.Figure(data=plot_data, layout=plot_layout)
#po.plot(fig)
fig.show()


# In[242]:


# Plotting Recency vs Monetary

plot_data = [
    gobj.Scatter(
        x=graph.query("RFM_Loyalty_Level == 'Bronze'")['Recency'],
        y=graph.query("RFM_Loyalty_Level == 'Bronze'")['Monetary'],
        mode='markers',
        name='Bronze',
        marker= dict(size=10, line=dict(width=1), color='blue', opacity=0.8)),
    
    gobj.Scatter(
        x=graph.query("RFM_Loyalty_Level == 'Silver'")['Recency'],
        y=graph.query("RFM_Loyalty_Level == 'Silver'")['Monetary'],
        mode='markers',
        name='Silver',
        marker= dict(size=9, line=dict(width=1), color='yellow', opacity=0.5)),
    
    gobj.Scatter(
        x=graph.query("RFM_Loyalty_Level == 'Gold'")['Recency'],
        y=graph.query("RFM_Loyalty_Level == 'Gold'")['Monetary'],
        mode='markers',
        name='Gold',
        marker= dict(size=8, line=dict(width=1), color='green', opacity=0.9)),
    
    gobj.Scatter(
        x=graph.query("RFM_Loyalty_Level == 'Platinum'")['Recency'],
        y=graph.query("RFM_Loyalty_Level == 'Platinum'")['Monetary'],
        mode='markers',
        name='Platinum',
        marker= dict(size=7, line=dict(width=1), color='red', opacity=0.9)),]

plot_layout = gobj.Layout(
             yaxis={'title': "Monetary"},
             xaxis={'title': "Recency"},
             title='Segments',
             autosize=False,
             width=600,
             height=400)
fig = gobj.Figure(data=plot_data, layout=plot_layout)
#po.plot(fig)
fig.show()


# In[243]:


graph = RFMScores.query("Monetary < 50000 and Frequency < 2000")


# In[244]:


# Plotting Frequency vs Monetary
plot_data = [
    gobj.Scatter(
        x=graph.query("RFM_Loyalty_Level == 'Bronze'")['Frequency'],
        y=graph.query("RFM_Loyalty_Level == 'Bronze'")['Monetary'],
        mode='markers',
        name='Bronze',
        marker= dict(size=10, line=dict(width=1), color='blue', opacity=0.8)),
    
     gobj.Scatter(
         x=graph.query("RFM_Loyalty_Level == 'Silver'")['Frequency'],
         y=graph.query("RFM_Loyalty_Level == 'Silver'")['Monetary'],
         mode='markers',
         name='Silver',
         marker= dict(size=9, line=dict(width=1), color='yellow', opacity=0.5)),
    
     gobj.Scatter(
         x=graph.query("RFM_Loyalty_Level == 'Gold'")['Frequency'],
         y=graph.query("RFM_Loyalty_Level == 'Gold'")['Monetary'],
         mode='markers',
         name='Gold',
         marker= dict(size=8, line=dict(width=1), color='green', opacity=0.9)),
    
     gobj.Scatter(
         x=graph.query("RFM_Loyalty_Level == 'Platinum'")['Frequency'],
         y=graph.query("RFM_Loyalty_Level == 'Platinum'")['Monetary'],
         mode='markers',
         name='Platinum',
         marker= dict(size=7, line=dict(width=1), color='red', opacity=0.9)),]

plot_layout = gobj.Layout(
             yaxis={'title': "Monetary"},
             xaxis={'title': "Frequency"},
             title='Segments',
             autosize=False,
             width=600,
             height=400)
fig = gobj.Figure(data=plot_data, layout=plot_layout)
#po.plot(fig)
fig.show()


# **<a id='kMeans-Clustering'>7. k-Means Clustering**</a> 

# **<a id='Data-Prepration'>7.1. Data Prepration**</a> 

# We are handling the negative and zero values because these values go negative in finite when they are on log scale as we are going to use log transformation for making this data normally distributed. 

# In[245]:


# To do normalization, we handle negative and zero values to handle infinite numbers during log transformation 

def handle_neg_n_zero(num):
    if num <= 0:
        return 1
    else:
        return num
    
# Apply handle_neg_n_zero function to Recency and Monetary columns
RFMScores['Recency'] = [handle_neg_n_zero(x) for x in RFMScores.Recency]
RFMScores['Monetary'] = [handle_neg_n_zero(x) for x in RFMScores.Monetary]

# Performing log transformation to bring data into normal or near normal distribution
Log_Tfd_Data = RFMScores[['Recency', 'Frequency', 'Monetary']].apply(np.log, axis=1).round(3)


# We need to normalize and scale the data in order to create the clusters out of the data points, because clustering uses distances as a similarity factor and for the skewed data we normalize the data.

# In[246]:


# Data distribution after data normalization for Recency
Recency_Plot = Log_Tfd_Data['Recency']
ax = sns.distplot(Recency_Plot)


# In[247]:


# Data distribution after data normalization for Frequency
Frequency_Plot = Log_Tfd_Data['Frequency']
ax = sns.distplot(Frequency_Plot)


# In[248]:


# Data distribution after data normalization for Monetary
Monetary_Plot = Log_Tfd_Data['Monetary']
ax = sns.distplot(Monetary_Plot)


# The graphs show that the data is kind of normally distributed after normalization by applying log transformation.

# Now we are trying to get the Recency, Frequency and Monetary on the same scale.

# **<a id='Standardization'>7.2. Standardization**</a> 

# In[249]:


# Scaling data 
scaleobj = StandardScaler()
Scaled_Data = scaleobj.fit_transform(Log_Tfd_Data)

# Transform it back to the dataframe
Scaled_data = pd.DataFrame(Scaled_Data, index=RFMScores.index, columns=Log_Tfd_Data.columns)


# The Elbow method involves running the algorithm multiple times over a loop with an increasing number of cluster choices and then plotting a clustering score as a function of the number of clusters. When k increases the centroids are closer to cluster centroids.

# Below elbow method runs k-means clustering on the dataset for a range of values for k (say from 1-15) and then for each value of k computes an average score for all clusters. 

# Utilizing the init parameter, which determines how the initial clusters are placed in step 1 of the algorithm. One of the values we can provide for this parameter is “k-means++”, which is an optimization of the algorithm by choosing initial cluster centers in a smart way to speed up convergence. Also providing a max_iter parameter, which is the maximum number of iterations of the algorithm for each run and we selected as 1000. 

# In[250]:


# Defining Sum of square distance dictionary, holding inertia attribute values to identify the sum of squared distances of samples to the nearest cluster center
sum_of_sq_dist = {}  

# We iterate the values of k from 1 to 15 and calculate the inertia for each value of k in the given range.
# init determines the method for initialization
for k in range(1,15):
    km = KMeans(n_clusters=k, init='k-means++', max_iter=1000)
    km = km.fit(Scaled_Data)
    
    # Storing inertia attributes values in the dictionary sum_of_sq_dist
    sum_of_sq_dist[k] = km.inertia_
    
# Plotting the graph for the sum of square distance values and number of clusters (range of 1 to 15)
sns.pointplot(x=list(sum_of_sq_dist.keys()), y=list(sum_of_sq_dist.values()))
plt.xlabel('Number Of Clusters(k)')
plt.ylabel('Sum Of Square Distance')
plt.title('Elbow Method For Optimal k')
plt.show()
    


# Since our k is the sum of square distance is dramatically decreasing at k=3 of the elbow of this line, we consider 3 as optimal value of k in our case. So, the number of clusters is equal to 3 here.

# In[251]:


# Performing k-Mean clustring model
KMean_clust = KMeans(n_clusters = 3, init='k-means++', max_iter=1000)
KMean_clust.fit(Scaled_Data)

# Finding the clusters for the observation given in the dataset
# Adding cluster value for each customer to the Cluster column
RFMScores['Cluster'] = KMean_clust.labels_
RFMScores.head()


# **<a id='Results-and-Discussion'>8. Results and Discussion**</a> 

# In the k-Means method we only have three clusters as against the RFM model where we had four clusters, Platinum, Gold, Silver and Bronze. We can infer that in the case of k-Means out of four clusters two clusters merge together and the remaining two left as it is. So, groups Silver and Bronze seem to be merged into group number two and Platinum and Gold stay in groups 0 and 1.

# Now we are plotting the data points in the form of clusterings using matplotlib library.

# In[252]:


from matplotlib import pyplot as plt
plt.figure(figsize=(4,4))

# Scatter plot Frequency vs Recency
Colors = ["red", "green", "blue"]

# We will assign a specific color to a specific cluster 
RFMScores['Color'] = RFMScores['Cluster'].map(lambda p: Colors[p])
ax = RFMScores.plot(kind='scatter', x="Recency", y="Frequency", figsize=(8,5), c=RFMScores['Color'])


# In[253]:


RFMScores.head()


# **<a id='Conclusion-and-Summary'>9. Conclusion and Summary**</a>
# 
# As usual in data science, cleaning and feature engineering took 80% of the time, but the resulting clustering was well worth the effort. Customer segmentation can have an incredible impact on a business when done well.
# 
# In this assignment we’ve identified four segments (Platinium, Gold, Silver and Bronze) implementing the RFM clustering model. Three segments was implemented by k-Means algorithm, **High (Platinum)** - Group who buys often, spends more and visited the platform recently (high score of RFM), 
# **Medium (Gold)** - Group which spends less than high group and is not that much frequent to visit the platform (medium score of RFM), and **Low (Silver)** - Group which is on the verge of churning out (low score of RFM).
# 
# Based on this RFM modeling what should be the marketing strategy? Customers with the Platinum RFM loyalty level are the best customers and companies can try to cross sell other products of their brands. Also, they can encourage them to sign up for loyalty programs to get some elite benefits like free same-day shipping priority access to newly launched products, etc. On the other hand, for customers with the Silver RFM loyalty level companies may try to offer some rewards or coupons to trigger their spending. 
# 
# It would be fascinating to explore more of the data, and  segment customers further by product line. For example, offering some seasonal products or promotions may receive very different reactions depending on how conservative or adventurous the recipient of the offer.

# **<a id='References'>10. References**</a>
# 
# - [1] : https://www.datacamp.com/community/tutorials/introduction-customer-segmentation-python
# - [2] : https://www.scikit-yb.org/en/latest/api/cluster/elbow.html
# - [3] : https://towardsdatascience.com/customer-segmentation-in-python-9c15acf6f945
# - [4] : https://towardsdatascience.com/customer-segmentation-in-python-9c15acf6f945

# In[ ]:




