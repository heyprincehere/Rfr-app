# -*- coding: utf-8 -*-
"""Machine Learning model on Streamli app.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1RUl2ePdzDGFqxg3cHrpj1mPNGlFGakLr
"""

import streamlit as st

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv(r"online_retail_dataset.csv",encoding='latin1')
df.head()

df.isnull().sum()

df['Description'].fillna('Unknown', inplace=True)
df = df.dropna(subset=['CustomerID'])
print(df.isnull().sum())

# Find duplicate rows
duplicates = df[df.duplicated()]
print(duplicates)

original_row_count = df.shape[0]
print(original_row_count)

# Remove duplicates from the dataset
df= df.drop_duplicates()

# Display the number of rows before and after removing duplicates to confirm changes
cleaned_row_count = df.shape[0]
print(cleaned_row_count)
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], format='%d-%m-%Y %H:%M')

from scipy.stats import zscore

z_scores = df.select_dtypes(include=['float64', 'int64']).apply(zscore)
df = df[(z_scores.abs() < 2).all(axis=1)]
print(df.shape)

summary_stats = df.describe()
manual_stats = {
    "count": df.count(),
    "mean": df.mean(numeric_only=True),
    "std": df.std(numeric_only=True),
    "min": df.min(numeric_only=True),
    "25%": df.quantile(0.25, numeric_only=True),
    "50%": df.median(numeric_only=True),
    "75%": df.quantile(0.75, numeric_only=True),
    "max": df.max(numeric_only=True),
}

# Convert manual_stats to a DataFrame for comparison
manual_stats_df = pd.DataFrame(manual_stats)

# Validate that the two outputs match
validation = summary_stats.equals(manual_stats_df)

# Print Results
if validation:
    print("Test Passed: Summary statistics match the manually computed statistics.")
else:
    print("Test Failed: Discrepancies found in summary statistics.")
    print("\nGenerated Statistics:")
    print(summary_stats)
    print("\nManually Computed Statistics:")
    print(manual_stats_df)

df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
list_cols=list(df.columns)
numerical_cols=[]
for i in list_cols:
    if df[i].dtype!='object':
        numerical_cols.append(i)

correlation_matrix = df[numerical_cols].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Extract time-based features
# Extracting weekday name
df['WeekDay'] = df['InvoiceDate'].dt.strftime('%a')

# Extracting day
df['Day'] = df['InvoiceDate'].dt.day

# Extracting month
df['Month'] = df['InvoiceDate'].dt.month

# Extracting Year
df['Year'] = df['InvoiceDate'].dt.year

#Removing -ve values from dataset
df= df[(df['Quantity']>0) & (df['UnitPrice']>0)]

#Derived column
df['TotalAmount'] = df['Quantity']*df['UnitPrice']

# Calculate Monetary (total spending per customer)
monetary = df.groupby('CustomerID')['TotalAmount'].sum().reset_index()
monetary.columns = ['CustomerID', 'Monetary']

# Calculate Frequency (number of transactions per customer)
frequency = df.groupby('CustomerID')['InvoiceNo'].count().reset_index()
frequency.columns = ['CustomerID', 'Frequency']

# Merge the two DataFrames on 'CustomerID'
rfm = pd.merge(monetary, frequency, on='CustomerID',how='inner')

recent_date = max(df['InvoiceDate'])
df['Difference'] = (recent_date - df['InvoiceDate']).dt.days
recency = df.groupby('CustomerID')['Difference'].min()
recency = recency.reset_index()

final_rfm = pd.merge(rfm,recency,on='CustomerID',how='inner')
final_rfm.columns = ['CustomerID','monetary','Frequency','Recency']

attributes = ['monetary','Frequency','Recency']
# Set the figure size
plt.rcParams['figure.figsize'] = [10, 8]

# Create the box plot
sns.boxplot(data=final_rfm[attributes], orient="v", palette="Set2", saturation=1, width=0.7)

# Set the title and labels with styling
plt.title("Outliers Variable Distribution", fontsize=14, fontweight='bold')
plt.ylabel("Range", fontweight="bold")
plt.xlabel("Attributes", fontweight="bold")

# Display the plot
plt.show()

# Function to remove statistical outliers based on IQR
def remove_outliers(final_rfm, column):
    Q1 = final_rfm[column].quantile(0.05)
    Q3 = final_rfm[column].quantile(0.95)
    IQR = Q3 - Q1
    return final_rfm[(final_rfm[column] >= Q1 - 1.5 * IQR) & (final_rfm[column] <= Q3 + 1.5 * IQR)]

# Remove outliers for Amount, Recency, and Frequency
rfm = remove_outliers(final_rfm, 'monetary')
rfm = remove_outliers(final_rfm, 'Recency')
rfm = remove_outliers(final_rfm, 'Frequency')

# Display the updated DataFrame
rfm

from sklearn.preprocessing import StandardScaler
rfm_df = rfm[['monetary', 'Frequency', 'Recency']]

# Instantiate
scaler = StandardScaler()

# fit_transform
rfm_df_scaled = scaler.fit_transform(rfm_df)
rfm_df_scaled.shape

rfm_df_scaled = pd.DataFrame(rfm_df_scaled)
rfm_df_scaled.columns = ['monetary', 'Frequency', 'Recency']
rfm_df_scaled

from sklearn.cluster import KMeans
# Apply K-means clustering
kmeans = KMeans(n_clusters=4, max_iter=50)
kmeans.fit(rfm_df_scaled)

# Assuming rfm_df_scaled is your scaled dataset
# Calculate WCSS for different values of K
wcss = []
range_n_clusters = [2, 3, 4, 5, 6, 7, 8]

for num_clusters in range_n_clusters:
    kmeans = KMeans(n_clusters=num_clusters, max_iter=50, n_init=10, random_state=0)
    kmeans.fit(rfm_df_scaled)
    wcss.append(kmeans.inertia_)

# Plot the Elbow Curve
plt.plot(range_n_clusters, wcss, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# Fit K-Means with optimal number of clusters (replace 4 with your chosen number)
optimal_clusters = 4
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)

rfm['Cluster'] = kmeans.fit_predict(rfm_df_scaled)
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Recency', y='monetary', hue='Cluster', data=rfm, palette='viridis')
plt.title('Customer Segments')
plt.xlabel('Recency')
plt.ylabel('Monetary Value')
plt.legend(title='Cluster')
plt.show()

# Calculate the mean of each feature for each cluster to get the summary
cluster_summary = rfm.groupby('Cluster')[['monetary', 'Frequency', 'Recency']].mean()

# Print the cluster summary
print(cluster_summary)

rfm



# Feature Selection (X) and Target (y)
X = rfm[['Recency', 'Frequency', 'Cluster']]
y = rfm['monetary']

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

"""**Model creation**"""

from sklearn.ensemble import RandomForestRegressor
# Train a Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions
y_pred = rf_model.predict(X_test)

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred) ** 0.5  # Manually take the square root
r2 = r2_score(y_test, y_pred)

print("Model Performance:")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R^2 Score: {r2:.2f}")

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# Predictions on training and test sets
y_train_pred = rf_model.predict(X_train)
y_test_pred = rf_model.predict(X_test)
# Calculate metrics for training set
train_mae = mean_absolute_error(y_train, y_train_pred)
train_rmse = mean_squared_error(y_train, y_train_pred) ** 0.5
train_r2 = r2_score(y_train, y_train_pred)
# Calculate metrics for test set
test_mae = mean_absolute_error(y_test, y_test_pred)
test_rmse = mean_squared_error(y_test, y_test_pred) ** 0.5
test_r2 = r2_score(y_test, y_test_pred)
# Display the regression report
print("Training Performance:")
print(f"Mean Absolute Error (MAE): {train_mae:.2f}")
print(f"Root Mean Squared Error (RMSE): {train_rmse:.2f}")
print(f"R^2 Score: {train_r2:.2f}")
print("\nTest Performance:")
print(f"Mean Absolute Error (MAE): {test_mae:.2f}")
print(f"Root Mean Squared Error (RMSE): {test_rmse:.2f}")
print(f"R^2 Score: {test_r2:.2f}")