# Implementation-of-K-Means-Clustering-for-Customer-Segmentation

## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
1.Import dataset and print head,info of the dataset

2.check for null values

3.Import kmeans and fit it to the dataset

4.Plot the graph using elbow method

5.Print the predicted array

6.Plot the customer segments
```

## Program:
```
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

data = pd.read_csv("Mall_Customers.csv")

print(data.head())
print(data.info())
print(data.isnull().sum())

wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init="k-means++", random_state=42)
    kmeans.fit(data.iloc[:, 3:5])
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), wcss, marker='o')
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.title("Elbow Method")
plt.grid(True)
plt.show()

km = KMeans(n_clusters=5, init="k-means++", random_state=42)
y_pred = km.fit_predict(data.iloc[:, 3:5])
data["Cluster"] = y_pred

plt.figure(figsize=(8, 6))
colors = ['red', 'black', 'blue', 'green', 'magenta']
for i in range(5):
    cluster = data[data["Cluster"] == i]
    plt.scatter(cluster["Annual Income (k$)"], cluster["Spending Score (1-100)"], 
                c=colors[i], label=f"Cluster {i}")

plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title("Customer Segments")
plt.legend()
plt.grid(True)
plt.show()
/*
Program to implement the K Means Clustering for Customer Segmentation.
Developed by: LOKESH M
RegisterNumber:  212224040173
*/
```

## Output:

### DATA.HEAD():

<img width="780" height="244" alt="Screenshot 2025-10-06 113831" src="https://github.com/user-attachments/assets/6290685a-ce24-4f2a-a532-3cb2c808224a" />

### DATA.INF0():

<img width="966" height="321" alt="Screenshot 2025-10-06 113841" src="https://github.com/user-attachments/assets/8bca47c3-9850-4db3-bd6b-99da4c1997a5" />

### DATA.ISNULL().SUM():

<img width="648" height="218" alt="Screenshot 2025-10-06 113851" src="https://github.com/user-attachments/assets/bfdac455-70cd-42fe-9dac-1b52e5a14b05" />

### PLOT USING ELBOW METHOD:

<img width="1315" height="608" alt="Screenshot 2025-10-06 113904" src="https://github.com/user-attachments/assets/90f423f2-a2fd-4f1f-b874-9cc6872a46cb" />

### CUSTOMER SEGMENT:

<img width="869" height="620" alt="Screenshot 2025-10-06 114026" src="https://github.com/user-attachments/assets/efe7bd7a-3127-4f0e-80d0-b9e8e61bb538" />





## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.
