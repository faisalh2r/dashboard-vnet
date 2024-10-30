# Business Analytics Web Dashboard & Customer Segmentation with RFM and K-Means Clustering

This application is an interactive dashboard that allows users to upload and visualize customer data and company income data. The application is built using Streamlit and is intended to help analyze data more easily and quickly. This project demonstrates customer segmentation using the RFM (Recency, Frequency, Monetary) method and K-Means clustering. It analyzes a ISP sales dataset to identify customer segments for targeted marketing strategies.

## Dataset

The dataset used in this project is a incomes dataset and customers from Victory Network Indonesia.

## Features

- 📊 Dashboard Customers
- 📊 Dashboard Incomes
- 👥 Customers Segmentation
- 🏷 Clean Data

## How to Run

1. Clone the repository.
2. Install the required libraries using pip install `-r requirements.txt`.
3. Run the Python script (Streamlit) to perform the analysis and segmentation.
4. After the dashboard is displayed upload the dataset you want to visualize.

## Results

The project identifies several customer segments:

- **Loyal:** Customers who have subscribed recently, frequently, and for high monetary value (high frequency and monetary, low recency). Frequent subscription and good payment history, High average transaction value
- **Promissing:** Customers who have not subscribed recently, infrequently, and for low monetary value (medium frequency and monetary). Just started subscribing or not very often, but has the potential to become a regular customer. Has interest in the product but has not yet shown full commitment.
- **Need Attention:** Former high-value customers who have stopped purchasing (low frequency, low monetary, high recency). Low transaction frequency or tend to subscribe infrequently. Risk to churn or move to competitors

## Potential Marketing Strategies

- **Loyal:** Special rewards and benefits such as exclusive discounts, given points to be exchanged for attractive prizes, referral programs that when recommending services to others will get special rewards, feedback to evaluate aspects that can be improved, invite to special events, and given service offers that match their preferences or that they might like.
- **Promissing:** Send regular reminders and promotions to keep them interested in subscribing, Offer additional products or bundled packages that suit their needs, to increase long-term commitment, Offer free demos or trials for higher plans, so they can experience greater benefits.
- **Need Attention:** Make an effort to re-engage them by understanding their reasons for switching, Contact them personally to find out the reasons for their dissatisfaction or disinterest, Offer special discounts or comeback promos to entice them back to use the service, Provide packages that are more flexible or suited to their needs so that they do not feel burdened by costs or commitments, Send reminders about the service and the benefits they can get by continuing to use the product, Send a short survey to understand what may be unsatisfactory, and follow up on the results for improvement.
