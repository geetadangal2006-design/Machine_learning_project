import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
df = pd.read_excel('C:/Users/geeta/Downloads/Mall_Customers_dataset.xlsx')

print(df.head())
print(df.tail())
print(df.sample(5))
print(df.shape)

df.info()
stats = df.describe()
print("Stats : " ,stats)
print(df.describe(include='all'))
# missing values in each column
print(df.isnull().sum())
#total missing values
print("Total Missing value :",df.isnull().sum().sum())
print("duplicated value :",df.duplicated().sum())
X = df[['Annual Income (k$)','Spending Score (1-100)']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i,init='k-means++', random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)
plt.figure(figsize=(8, 6))
plt.plot(range(1, 11), wcss,marker='o',color='b')
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()
kmeans = KMeans(n_clusters=5,init='k-means++', random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)
plt.figure(figsize=(10, 6))
sns.scatterplot(x=df['Annual Income (k$)'], y=df['Spending Score (1-100)'], hue=df['Cluster'],size= df['Age'],palette='viridis',sizes=(50,200),legend ='full')
plt.title('Customer Segments(Income vs  Spending Score with Age)')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.show()
cluster_labels ={
    0: 'High Spenders',
    1: 'Conservative Spenders',
    2: 'Average Spenders',
    3: 'Budget Spenders',
    4: 'Carefree Spenders'
}
df['Customer_Segment'] = df['Cluster'].map(cluster_labels)
print(df[['Annual Income (k$)',
          'Spending Score (1-100)',
          'Cluster',
          'Customer_Segment']].head(10))
df.to_csv('C:/Users/geeta/Downloads/Mall_Customers_dataset.csv',index=False)
print("File saved successfully !")