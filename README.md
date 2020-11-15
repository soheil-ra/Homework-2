# Customer Segmentation


<p align="center">
<img src="https://user-images.githubusercontent.com/71153587/99133294-6a897b00-25e7-11eb-9507-ff55f10a6740.png" width="700" height="220" />
</p>

## **Table of Content**<br>

ReadMe contains the following sections:

**1. Overview -** [Overview](https://github.com/soheil-ra/Homework-2#Overview)<br>
**2. Goals -** [Goal](https://github.com/soheil-ra/Homework-2#Goals)<br>
**3. Motivation & Background -** [Motivation & Background](https://github.com/soheil-ra/Homework-2#Motivation-and-Background)<br>
**4. Data -** [Data](https://github.com/soheil-ra/Homework-2#Data)<br>

This assignment contains the following areas:

**5. Dataset -** [Dataset](https://www.kaggle.com/sergeymedvedev/customer_segmentation#Dataset) <br>
**6. Code -** [Code](https://github.com/soheil-ra/Homework-2/blob/main/RFM%20%26%20k-Means%20Clustring.ipynb#Code)<br>
**7. Images -** [Images](https://github.com/soheil-ra/Homework-2/tree/main/Images#Images)<br>

## **Overview**<br>
**What is Customer Segmentation?**<br>
Customer segmentation is the action of breaking the customer base into groups depending on demographic, psychographic, etc. Segmentation is mostly used for marketing, but there are other reasons to segment customers as well. Using customer segmentation in marketing means that we can target the right people with the right messaging about our products. This will increase the success of our marketing campaigns. When we perform customer segmentation, we find similar characteristics in each customer's behaviour and needs. Then, those are generalized into groups to satisfy demands with various strategies.
<br>

## **Goals**<br>
The goal is to do customer segmentation analysis by looking at information that we have at our disposal and analyzing it for customer segment trends.<br>

I try to achive the followings for this assignment:<br>

**1. Preparing Data -** This section includes, Cleaning, Exploring and Visualizing  data.<br>
**2. Proposing Methods & Experiments -** To perform customer segmentation I will utilize RFM modeling to calculate the RFM scores for each customer and then will apply unsupervised ML technique (k-Means) to group the customers into different segments based on calculated RFM scores.<br>
 
RFM stands for Recency, Frequency and Monetary.<br>

**Recency (R) -** Who have purchased recently? Number of days since last purchase (least recency).<br>
**Frequency (F) -** Who has purchased frequently? It means the total number of purchases ( high frequency).<br>
**Monetary (M) -** Who have high purchase amount? It means the total money customer spent (high monetary value).<br>

## **Motivation and Background**<br>
Businesses around the world would like to understand customer purchasing behavior with the goal of maximizing the value (revenue and/or profit) from each customer. it is critical to know in advance how any particular marketing action will influence the customer. Accurate customer segmentation allows marketers to engage with each customer in the most effective way and by segmenting them they will decide how to relate to customers in each segment in order to maximize the value of each customer to the business. A customer segmentation analysis allows marketers to identify discrete groups of customers with a high degree of accuracy based on demographic, behavioral and other indicators. Ideally, such “action-centric” customer segmentation will not focus on the short-term value of a marketing action, but rather the long-term customer lifetime value (CLV) impact that such a marketing action will have. Thus, it is necessary to group, or segment, customers according to their CLV.
<br>

<p align="center">
<img src="https://user-images.githubusercontent.com/71153587/99128713-8ab23d80-25d9-11eb-8705-b461aa030db0.jpg"  />
</p>

## **Data**
The dataset have been collected from Kaggle (https://www.kaggle.com/sergeymedvedev/customer_segmentation), consisting of 541910 records of customers and 8 attributes. The dataset contains transaction dates between 01/12/2010 and 9/12/2011.<br>

Attribute Information:<br>
**1. InvoiceNo -** Invoice number. The number that is uniquely assigned to each transaction. (6 digit Numerical) <br>
**2. StockCode -** Product code. The number that is uniquely assigned to each distinct product. (5 digit Numerical) <br>
**3. Description -** Product type. (Categorical) <br>
**4. Quantity -** The quantities of each product per transaction. (Numerical) <br>
**5. InvoiceDate -** Invoice date and time. Day and time of generated transaction. (String Date) <br>
**6. UnitPrice -** Unit price. Product price per unit. (Numerical) <br>
**7. CustomerID -** Customer number. The number that is uniquely assigned to each customer. (5 digit Numerical) <br>
**8. Country -** Country name. The name of the country where a customer resides. (Categorical) <br>
<br>


<pre>
Contributors : <a href=https://github.com/soheil-ra>Soheila Rahmani</a>
</pre>

<pre>
Languages    : Python
Tools/IDE    : Anaconda
Libraries    : pandas, numpy, matplotlib, seaborn, sklearn
</pre>

<pre>
Duration     : November 2020
</pre>
