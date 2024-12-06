
# CARL
import numpy as np
from numpy import unique
from numpy import where
from sklearn.cluster import Birch
from sklearn.metrics import pairwise_distances

def calr(featurepath = 'feature/resnet50_o16_feature_2.txt', M = 500, savepath='r/imagenet_o16_2_kcenter500_indices.csv'):
    features = np.loadtxt(featurepath)
    N = features.shape[0]
    assert N > M , "The number of features must be larger than M!!!"
    BIRCH = Birch(n_clusters=M)
    BIRCH.fit(features)
    predicted_cluster = BIRCH.predict(features)
    selected = []
    for cluster in unique(predicted_cluster):
        row_index = where(predicted_cluster == cluster)[0]
        similarities = 1 - pairwise_distances(features[row_index, :], metric='cosine')
        information_density = np.mean(similarities, axis=1)
        max_index = np.argmax(information_density)
        original_max_index = row_index[max_index]
        selected.append(original_max_index)
    np.savetxt(savepath, np.array(selected)+1, fmt="%d", header="indices", delimiter=",")