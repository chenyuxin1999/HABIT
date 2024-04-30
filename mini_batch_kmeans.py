from sklearn import datasets
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import MiniBatchKMeans
import torch

label_embedding = torch.load('../examples/all_label_embedding.pt')
kmeans = MiniBatchKMeans(n_clusters=1500,random_state=42,batch_size=1000,max_iter=1000)
y_pred = kmeans.fit(label_embedding)
cluster = kmeans.cluster_centers_
np.save('cluster.npy',cluster)
labels = kmeans.labels_
np.save('cluster_label.npy',labels)

