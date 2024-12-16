
# CARL

import argparse
import numpy as np
from numpy import unique
from numpy import where
from sklearn.cluster import Birch
from sklearn.metrics import pairwise_distances

parser = argparse.ArgumentParser(description="PyTorch CALR Training")
parser.add_argument(
    "-m",
    "--num-samples",
    type=int,
    help="number of samples",
)
parser.add_argument(
    "-p",
    "--path",
    type=str,
    help="Save path",
)
parser.add_argument(
    "-f",
    "--fpath",
    type=str,
    help="feature path",
)

def calr(featurepath = 'feature/resnet50_o16_feature_2.txt', M = 500, savepath='r/imagenet_o16_2_calr500_indices.csv'):
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

args = parser.parse_args()
calr(args.fpath, args.num_samples, args.path)
print("Good luck!")