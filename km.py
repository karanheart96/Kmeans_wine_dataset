import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
wine_names = ['Class', 'Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash', 'Magnesium', 'Total phenols', \
              'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins', 'Color intensity', 'Hue', 'OD280/OD315',\
              'Proline']
wine_data = pd.read_csv('wine.data', names = wine_names)
wine_df = pd.DataFrame(wine_data)
wine_df.Class = wine_df.Class - 1
print(wine_data.head())
wine_df.plot.scatter(x = 'Alcohol', y = 'OD280/OD315', c= 'Class', figsize=(12,8), colormap='jet')
kmeans = KMeans(n_clusters=3, init = 'random', max_iter = 100, random_state = 5).fit(wine_df.iloc[:,[12,1]])
centroids_df = pd.DataFrame(kmeans.cluster_centers_, columns = list(wine_df.iloc[:,[12,1]].columns.values))
fig, ax = plt.subplots(1, 1)
wine_df.plot.scatter(x = 'Alcohol', y = 'OD280/OD315', c= kmeans.labels_, figsize=(12,8), colormap='jet', ax=ax, mark_right=False)
centroids_df.plot.scatter(x = 'Alcohol', y = 'OD280/OD315', ax = ax,  s = 80, mark_right=False)
print("Cluster Centers: ",kmeans.cluster_centers_)
print("Confusion matrix:")
print(confusion_matrix(wine_data['Class'],kmeans.labels_))
print("Classification report:")
print(classification_report(wine_data['Class'],kmeans.labels_))
print ("Accuracy : ",
           accuracy_score(wine_data['Class'],kmeans.labels_) * 100)

act= wine_data['Class'].tolist()
for i in range(5):
    print("Predicted value : ", kmeans.labels_[i], "Actual value : ", act[i])

