import sys
import torch
import torchvision.transforms
import torch.nn as nn
from torch.autograd import Variable
import torch.optim
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import adjusted_mutual_info_score
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.utils.linear_assignment_ import linear_assignment
from sklearn.decomposition import PCA
if (sys.version[0] == 2):
    import cPickle as pickle
else:
    import pickle
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from scipy.spatial.distance import cdist
from scipy.sparse.csgraph import minimum_spanning_tree, connected_components
from scipy.sparse import csr_matrix, tril,find, triu, coo_matrix
from scipy.sparse.linalg import eigs
from torch.utils.data.dataset import Dataset
from sklearn.manifold import TSNE
import torch.nn.functional as F
from scipy.sparse.linalg import norm
from torchvision.utils import save_image
from PIL import Image
from sklearn.metrics import silhouette_samples
import itertools
from sklearn.cluster import KMeans, SpectralClustering,AgglomerativeClustering
from sklearn.preprocessing import MinMaxScaler
from plots import *

def find_mknn(X,n_neighbors, dist_meas):
    """

    :param X: dataset
    :param n_neighbours: number of neighbours for mknn calculation
    :return: P - mknn graph (csr matrix), Q - weights car matrix
    """
    samples = X.shape[0]
    batchsize = 10000
    b = np.arange(n_neighbors + 1)
    b = tuple(b[1:].ravel())

    z = np.zeros((samples, n_neighbors))
    weigh = np.zeros_like(z)
    X = np.reshape(X, (X.shape[0], -1))
    # This loop speeds up the computation by operating in batches
    # This can be parallelized to further utilize CPU/GPU resource
    for x in np.arange(0, samples, batchsize):
        start = x
        end = min(x + batchsize, samples)

        w = cdist(X[start:end], X, dist_meas)
        # the first columns will be the indexes of the knn of each sample (the first column is always the same
        # index as the row)
        y = np.argpartition(w, b, axis=1)

        z[start:end, :] = y[:, 1:n_neighbors + 1]
        # the weights are the distances between the two samples
        weigh[start:end, :] = np.reshape(
            w[tuple(np.repeat(np.arange(end - start), n_neighbors)), tuple(
                y[:, 1:n_neighbors + 1].ravel())], (end - start, n_neighbors))
        del (w)

    ind = np.repeat(np.arange(samples), n_neighbors)
    P = csr_matrix((np.ones((samples * n_neighbors)), (ind.ravel(), z.ravel())), shape=(samples, samples))
    Q = csr_matrix((weigh.ravel(), (ind.ravel(), z.ravel())), shape=(samples, samples))
    return P, Q


def connectivity_structure_mknn(X, n_neighbors, dist_meas):
    """
    :param X: the dataset
    :param n_neighbours: the number of closest neighbours taken into account
    :return: matrix E with 1 where the two points are mknn. the matrix is lower triangular (zeros in the top triangular)
     so that each connection will be taken into account only once.
     W is a matrix of the weight of each connection. both are sparse matrices.
    """
    samples = X.shape[0]
    P, Q = find_mknn(X,n_neighbors=n_neighbors, dist_meas=dist_meas)
    Tcsr = minimum_spanning_tree(Q)
    P = P.minimum(P.transpose()) + Tcsr.maximum(Tcsr.transpose())
    index = np.asarray(find(P)).T
    E = csr_matrix((np.ones(len(index)), (index[:, 0], index[:, 1])), [samples, samples])
    connections_per_point = np.sum(E, 0)  # sum of each row
    E = triu(E, k=1)
    a = np.sum(connections_per_point) / samples  # calculating the averge number of connections
    w = \
    np.divide(a, np.sqrt(np.asarray(connections_per_point[0][0, E.row]) * np.asarray(connections_per_point[0][0, E.col])))[0]
    W = coo_matrix((w, (E.row, E.col)), [samples, samples])
    print('number of connections:', len(E.data), 'average connection per point',a)
    return E, W, connections_per_point


def connectivity_structure_Loss_constraints(dataset, encoded_data, mu_gmc2, E, W, num_constraints):
    """

    :param encoded_data:
    :param mu_gmc2:
    :param y:
    :param E:
    :param W:
    :param label_percent: the percentage of labeled connections out of the total connections. the connections are
                          chosen according to the highest clustering loss
    :return: returns a sparse matrix with 1 where there are unlabeled MKNN, -1 where there were false mknn connections,
             and 2 where there were true mknn connection
    """
    y = dataset.train_labels
    len_plot=20
    n_labeled = int(min(num_constraints, len(E.row)))  # number of labeled pairs
    # calculating the clustering loss of all the MKNN pairs:
    encoded_data1 = encoded_data[E.row, ]
    encoded_data2 = encoded_data[E.col, ]
    dist = np.sum((encoded_data1 - encoded_data2) ** 2, 1)
    clust_loss = W.data * mu_gmc2 * dist / (mu_gmc2 + dist)
    # sorting the pairs according to the clustering loss
    idx_w = np.argsort(clust_loss)
    # labeling the n_labeled pairs with the highest clusering loss:
    idx_labeled = idx_w[-1:-(n_labeled+1):-1]
    idx_unlabeled = idx_w[1:-n_labeled+1:1]
    # finding the relevent indexes in all the dataset (not in E.row/col)
    idx_row = E.row[idx_labeled]
    idx_col = E.col[idx_labeled]
    idx_idx_true = y[idx_row]==y[idx_col]     # true (must - link) connections out of labeled ones
    idx_idx_false = y[idx_row]!=y[idx_col]     # false (cannot-ink) connections out of labeled ones
    cl_places = 1*idx_idx_false[0:len_plot]
    # plotting the pairs with the highest loss:
    if len(dataset.train_data.shape)>2:
        len_plot = min(20, n_labeled)
        images_pairs = np.zeros([len_plot, 2, dataset.train_data.shape[1], dataset.train_data.shape[2]])
        images_pairs[:,0,:,:] = np.squeeze(dataset.train_data[idx_row[0:len_plot]])
        images_pairs[:,1,:,:] = np.squeeze(dataset.train_data[idx_col[0:len_plot]])
        plot_cl_pairs_images(np.rot90(np.flip(images_pairs, 3), axes=(2, 3)), cl_places)
    # calculating the new E and W:
    idx_row_true = idx_row[idx_idx_true]
    idx_col_true = idx_col[idx_idx_true]
    idx_row_false = idx_row[idx_idx_false]
    idx_col_false = idx_col[idx_idx_false]
    idx_row_ul = E.row[idx_unlabeled]
    idx_col_ul = E.col[idx_unlabeled]
    E = coo_matrix((np.concatenate([np.ones(len(idx_row_ul)), -1*np.ones(len(idx_row_false)), 2*np.ones(len(idx_row_true))]),
                   (np.concatenate([idx_row_ul, idx_row_false,idx_row_true], axis=0),
                    np.concatenate([idx_col_ul, idx_col_false, idx_col_true,], axis=0))),
                   shape=E.shape)
    # the weight of the unlabeled connections are recalculated according to the number of unlabeled connections each
    # point has, maybe it should also include the labeled true connections
    C_ones = coo_matrix(
        (np.ones(len(idx_row_false)+len(idx_row_true)),
         (np.concatenate([idx_row_false, idx_row_true], axis=0),
          np.concatenate([idx_col_false, idx_col_true], axis=0))),
        shape=E.shape)
    W = W-W.multiply(C_ones)+np.max(W.data)*C_ones
    W = W.tocoo()
    W.data[E.data==-1]=0
    E.data[E.data==-1]=0
    E=triu(E)
    W=triu(W)
    return E, W

def to_img(x, rgb, size):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1+2*rgb, size, size)
    return x

def plot_clusters(X, Y):
    num_labels = max(Y)+1
    plt.figure()
    plt.scatter(X[:, 0], X[:, 1], c=Y[0:5000], cmap=plt.cm.get_cmap("jet", num_labels))
    # plt.colorbar(ticks=range(num_labels))
    plt.clim(-0.5, num_labels-0.5)
    plt.show(block=False)


class PairingDataset(Dataset):
    def __init__(self, dataset, W, E, constraints_type):
        self.transform = dataset.transform
        dataset = dataset.train_data
        self.data1_idx = E.row[E.data==1]
        self.data2_idx = E.col[E.data==1]
        self.weights = W.data[E.data==1]
        self.conn_type = E.data[E.data==1]
        if constraints_type==0 or constraints_type==1:
            # add must-link pairs
            self.data1_idx = np.concatenate([self.data1_idx,E.row[E.data==2]])
            self.data2_idx = np.concatenate([self.data2_idx, E.col[E.data==2]])
            self.weights = np.concatenate([self.weights,W.data[E.data==2]])
            self.conn_type = np.concatenate([self.conn_type,E.data[E.data==2]])
        if constraints_type==0 or constraints_type==-1:
            # add cannot-link pairs
            self.data1_idx = np.concatenate([self.data1_idx,E.row[E.data==-1]])
            self.data2_idx = np.concatenate([self.data2_idx, E.col[E.data==-1]])
            self.weights = np.concatenate([self.weights,W.data[E.data==-1]])
            self.conn_type = np.concatenate([self.conn_type,E.data[E.data==-1]])
        self.weights = torch.from_numpy(self.weights)
        self.weights = self.weights.type(torch.FloatTensor)
        self.data1 = [dataset[i] for i in self.data1_idx]
        self.data2 = [dataset[i] for i in self.data2_idx]

    def __len__(self):
        return len(self.weights)

    def __getitem__(self, idx):
        data1 = self.data1[idx]
        data2 = self.data2[idx]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        if isinstance(data1,torch.ByteTensor):
            data1 = Image.fromarray(data1.numpy(), mode='L')
            data2 = Image.fromarray(data2.numpy(), mode='L')

        if self.transform:
            data1 = self.transform(data1)
            data2 = self.transform(data2)
        return self.data1_idx[idx], self.data2_idx[idx], data1, data2, self.weights[idx], self.conn_type[idx]


def weighted_mse_loss(input, target, weight):
    return torch.sum(weight * torch.sum((input.view(input.size(0),-1) - target.view(target.size(0),-1)) ** 2, 1))


def weighted_bce_loss(input, target, weight):
    return torch.sum(weight*torch.sum(-(target * torch.log(input+10**-12) + (1 - target) * torch.log(1 - input+10**-15)), 1))

def calculate_clusters(clust_method, E, epsilon, final_conn_thresh, u, n_clusters):
    """
    When choosing the MKNN clustering scheme the clusters are calculated according to the following stages:
    1. a connection graph is created according to distances smaller than a threshold
    2. all points which are connected belong to the same cluster
    The other clustering methods are implemented python methods
    :param data:
    :return:
    graph - matrix of connections between pairs of points
    labels - cluster number of each data point (not necessarily the same number as the equivalent gt label)
    num_components - number of clusters
    """
    u_np = u.data.cpu().numpy()
    if clust_method == 'MKNN':
        row_idx = E.row
        col_idx = E.col
        u_np_row = u_np[row_idx, ]
        u_np_col = u_np[col_idx, ]
        # computing connected components:
        diff = np.sqrt(np.sum((u_np_row - u_np_col) ** 2, 1))
        is_conn = diff <= (epsilon * final_conn_thresh)
        graph = coo_matrix((np.ones((2*np.sum(is_conn),)),
                           (np.concatenate([row_idx[is_conn], col_idx[is_conn]], axis=0),
                            np.concatenate([col_idx[is_conn], row_idx[is_conn]], axis=0))),
                           shape=[len(u), len(u)])
        num_components, labels = connected_components(graph, directed=False)
    elif clust_method == 'kmeans':
        kmeans = KMeans(n_clusters=n_clusters)
        labels = kmeans.fit_predict(u_np)
        num_components = n_clusters
        graph = None
    elif clust_method == 'spectral':
        Spectral_Clustering = SpectralClustering(n_clusters=n_clusters, eigen_solver='arpack', affinity="nearest_neighbors")
        labels = Spectral_Clustering.fit_predict(u_np)
        num_components=n_clusters
        graph = None
    elif clust_method == 'agglomerative':
        agglomerative = AgglomerativeClustering(n_clusters=n_clusters)
        labels = agglomerative.fit_predict(u_np)
        num_components = n_clusters
        graph = None
    return graph, labels, num_components

def cluster_acc(y_true, y_pred):
    """
    calculating the accuracy of the clustering.
    since the index of each cluster might be different in y_true and y_pred, this function finds the linear
    assignment which maximizes the accuracy. This means some of the clusters might remain without a matching label.
    :param y_true: ground truth labeling
    :param y_pred: calculated from the model
    :return: the accuracy percentage, ami, nmi and the matrix w of all the combinations of indexes of the original clusters
    and the calculated ones
    """
    assert y_pred.size == y_true.size
    y_true_unique = np.unique(y_true)
    true_cluster_idx = np.nonzero(y_true[:, None] == y_true_unique)[1]
    D = max(y_pred.max()+1, len(y_true_unique)) # number of clusters
    w = np.zeros((D, len(y_true_unique)), dtype=np.int64) # D is in size number of clusters*number of clusters
    for i in range(y_pred.size):
        w[y_pred[i], true_cluster_idx[i]] += 1
    ind = linear_assignment(w.max() - w)
    # calculating the corresponding gt label most fit for each y_pred. since there are usually a lot of clusters,
    # the ones which didn't correspond to a value in the gt will receive the value -1
    y_pred_new = -1 * np.ones(len(y_pred), int)
    for i in range(0, len(y_pred)):
        j = np.argwhere(ind[:, 0] == y_pred[i])
        if j.shape[0] > 0:
            y_pred_new[i] = (ind[j[0], 1])
    acc = sum([w[i, j] for i, j in ind])*1.0/y_pred.size
    ami = adjusted_mutual_info_score(y_true, y_pred)
    nmi = normalized_mutual_info_score(y_true, y_pred)
    return acc, ami, nmi, w, y_pred_new

