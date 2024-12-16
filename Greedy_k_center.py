import numpy as np
from scipy.spatial import distance_matrix
import argparse

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

def greedy_k_center(labeled, unlabeled, amount):

        greedy_indices = []

        # get the minimum distances between the labeled and unlabeled examples (iteratively, to avoid memory issues):
        min_dist = np.min(distance_matrix(labeled[0, :].reshape((1, labeled.shape[1])), unlabeled), axis=0)
        min_dist = min_dist.reshape((1, min_dist.shape[0]))
        for j in range(1, labeled.shape[0], 100):
            if j + 100 < labeled.shape[0]:
                dist = distance_matrix(labeled[j:j+100, :], unlabeled)
            else:
                dist = distance_matrix(labeled[j:, :], unlabeled)
            min_dist = np.vstack((min_dist, np.min(dist, axis=0).reshape((1, min_dist.shape[1]))))
            min_dist = np.min(min_dist, axis=0)
            min_dist = min_dist.reshape((1, min_dist.shape[0]))

        # iteratively insert the farthest index and recalculate the minimum distances:
        farthest = np.argmax(min_dist)
        greedy_indices.append(farthest)
        for i in range(amount-1):
            dist = distance_matrix(unlabeled[greedy_indices[-1], :].reshape((1,unlabeled.shape[1])), unlabeled)
            min_dist = np.vstack((min_dist, dist.reshape((1, min_dist.shape[1]))))
            min_dist = np.min(min_dist, axis=0)
            min_dist = min_dist.reshape((1, min_dist.shape[0]))
            farthest = np.argmax(min_dist)
            greedy_indices.append(farthest)

        return np.array(greedy_indices)

def greedyk(featurepath='feature/resnet50_o16_feature_2.txt', M=500, savepath='r/imagenet_o16_2_kcenter500_indices.csv'):
    '''
    Greedy k center.
    The result will be saved in the savepath.
    '''
    features = np.loadtxt(featurepath)
    N = features.shape[0]
    assert N > M, "The number of features must be larger than M!!!"
    random_initial = np.random.randint(0,N-1,1)
    selected = greedy_k_center(features[random_initial,:], np.delete(features, random_initial, axis = 0), M)
    np.savetxt(savepath, np.array(selected)+1,fmt="%d",header="indices",delimiter=",")
    print("Greedy k center saved in the %s" % savepath)


args = parser.parse_args()
greedyk(args.fpath, args.num_samples, args.path)

print('Have a nice day!')


