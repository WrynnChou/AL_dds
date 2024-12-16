import numpy as np
import pandas as pd
import torch
import argparse

parser = argparse.ArgumentParser(description="PyTorch Prob Training")

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
parser.add_argument(
    "-d",
    "--delta",
    type=float,
    default = 0.0,
    help="distance threshold",
)

class ProbCover:
    def __init__(self, features, uSet, budgetSize, delta):
        self.all_features = features
        self.uSet = uSet
        self.budgetSize = budgetSize
        self.delta = delta
        self.relevant_indices = self.uSet.astype(int)
        self.rel_features = self.all_features[self.relevant_indices]
        self.graph_df = self.construct_graph()

    def construct_graph(self, batch_size=500):
        """
        creates a directed graph where:
        x->y iff l2(x,y) < delta.

        represented by a list of edges (a sparse matrix).
        stored in a dataframe
        """
        xs, ys, ds = [], [], []
        print(f'Start constructing graph using delta={self.delta}')
        # distance computations are done in GPU
        cuda_feats = torch.tensor(self.rel_features).cuda().clone().detach()
        for i in range(len(self.rel_features) // batch_size):
            # distance comparisons are done in batches to reduce memory consumption
            cur_feats = cuda_feats[i * batch_size: (i + 1) * batch_size]
            dist = torch.cdist(cur_feats, cuda_feats)
            mask = dist < self.delta
            # saving edges using indices list - saves memory.
            x, y = mask.nonzero().T
            xs.append(x.cpu() + batch_size * i)
            ys.append(y.cpu())
            ds.append(dist[mask].cpu())

        xs = torch.cat(xs).numpy()
        ys = torch.cat(ys).numpy()
        ds = torch.cat(ds).numpy()

        df = pd.DataFrame({'x': xs, 'y': ys, 'd': ds})
        print(f'Finished constructing graph using delta={self.delta}')
        print(f'Graph contains {len(df)} edges.')
        return df

    def select_samples(self):
        """
        selecting samples using the greedy algorithm.
        iteratively:
        - removes incoming edges to all covered samples
        - selects the sample high the highest out degree (covers most new samples)

        """
        print(f'Start selecting {self.budgetSize} samples.')
        selected = []
        # removing incoming edges to all covered samples from the existing labeled set
        #edge_from_seen = np.isin(self.graph_df.x, np.arange(len(self.lSet)))
        #covered_samples = self.graph_df.y[edge_from_seen].unique()
        covered_samples = torch.tensor([])
        cur_df = self.graph_df[(~np.isin(self.graph_df.y, covered_samples))]
        for i in range(self.budgetSize):
            coverage = len(covered_samples) / len(self.relevant_indices)
            # selecting the sample with the highest degree
            degrees = np.bincount(cur_df.x, minlength=len(self.relevant_indices))
            print(f'Iteration is {i}.\tGraph has {len(cur_df)} edges.\tMax degree is {degrees.max()}.\tCoverage is {coverage:.3f}')
            cur = degrees.argmax()
            # cur = np.random.choice(degrees.argsort()[::-1][:5]) # the paper randomizes selection

            # removing incoming edges to newly covered samples
            new_covered_samples = cur_df.y[(cur_df.x == cur)].values
            assert len(np.intersect1d(covered_samples, new_covered_samples)) == 0, 'all samples should be new'
            cur_df = cur_df[(~np.isin(cur_df.y, new_covered_samples))]

            covered_samples = np.concatenate([covered_samples, new_covered_samples])
            selected.append(cur)

        assert len(selected) == self.budgetSize, 'added a different number of samples'
        activeSet = self.relevant_indices[selected]
        remainSet = np.array(sorted(list(set(self.uSet) - set(activeSet))))

        print(f'Finished the selection of {len(activeSet)} samples.')
        print(f'Active set is {activeSet}')
        return activeSet, remainSet

def upperAverage(mat):
    """
    Calculates the upper average of a given matrix.
    """
    mat = mat.cpu().numpy()
    n = mat.shape[0]
    upper = np.triu(mat, k=-1)
    arr = []
    for i in range(1, n):
        for j in range(i):
            arr.append(upper[j, i])
    arr.sort()
    return arr[1]


if __name__ == "__main__":
    args = parser.parse_args()

    features = np.loadtxt(args.fpath)
    features = torch.from_numpy(features)
    d10 = torch.cdist(features[1:10],features[1:10])
    print(d10)
    delta = args.delta
    if delta == 0.0:
        delta = upperAverage(d10)
    n = features.shape[0]

    all_idx = [i for i in range(n)]
    uSet = np.array(all_idx, dtype=np.ndarray)
    Pc = ProbCover(features = features, uSet=uSet, budgetSize=args.num_samples, delta=delta)
    selected, _ = Pc.select_samples()
    np.savetxt(args.path, selected+1,fmt="%d",header="indices",delimiter=",")

    print(f'Selected samples are saved to {args.path}')


