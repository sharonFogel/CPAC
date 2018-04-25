import sys
import torch
import torchvision.transforms
import torch.nn as nn
from torch.autograd import Variable
import torch.optim
import numpy as np
from sklearn.cluster import KMeans, SpectralClustering,AgglomerativeClustering
from sklearn.metrics.cluster import adjusted_mutual_info_score
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.utils.linear_assignment_ import linear_assignment
from sklearn.decomposition import PCA
if (sys.version[0] == 2):
    import cPickle as pickle
else:
    import pickle
import matplotlib as mpl

mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.gridspec as gridspec
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
import random
from functions import *
from plots import *
from sklearn.preprocessing import MinMaxScaler, normalize, Normalizer, StandardScaler

class CPAC(nn.Module):
    def __init__(self,
                 autoencoder,
                 learning_rate_ae=0.1,
                 pretrained_weights_path=None,
                 batch_size_ae=256,
                 batch_size_c=128,
                 n_neighbors=10,
                 ):
        super(CPAC, self).__init__()
        self.pretrained_weights_path = pretrained_weights_path
        self.batch_size_ae = batch_size_ae
        self.batch_size_c = batch_size_c
        self.learning_rate_ae = learning_rate_ae
        self.autoencoder = autoencoder.cuda()
        self.optimizer = torch.optim.Adam(self.autoencoder.parameters(), lr=self.learning_rate_ae)
        self.n_neighbors = n_neighbors


    def initialize(self,
                   dataset,
                   autoencoder_weights_save_path=None,
                   layerwise_epochs=150,
                   finetune_epochs=100,
                   plot=False,
                   rgb=False):
        """

        :param dataset:
        :param autoencoder_weights_save_path:
        :param layerwise_pretrain_iters:
        :param finetune_iters:
        :return:
        """

        # Data Loader
        dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=self.batch_size_ae, shuffle=True)

        if self.pretrained_weights_path is None:
            mse_loss = nn.MSELoss()
            print('Running layerwise pretrain')
            self.autoencoder.layerwise_train = True
            self.autoencoder.train(True)  # changing to train mode so that there will be dropout layers
            self.autoencoder.add_layer()
            maxd=self.autoencoder.n_layers
            for d in range(0, maxd):
                self.optimizer = torch.optim.Adam(self.autoencoder.trainable_params, lr=self.learning_rate_ae)
                for epoch in range(layerwise_epochs):
                    for i_batch, (x, y) in enumerate(dataloader):
                        x = Variable(x.type(torch.FloatTensor)).cuda()
                        encoded, decoded = self.autoencoder(x)
                        tot_loss = mse_loss(decoded.view(x.size(0), -1), x.view(x.size(0), -1))
                        # Backprop + Optimize
                        self.optimizer.zero_grad()
                        tot_loss.backward()
                        self.optimizer.step()
                    print('Epoch: ', epoch, '| reconst ae loss:%f ' % (tot_loss.data[0]))
                    if plot and epoch % 10 == 0 and len(dataset.train_data.shape) > 2:
                        decoded = decoded.view(x.data.shape)
                        pic = to_img(decoded.cpu().data, rgb, size=decoded.data.shape[2])
                        save_image(pic, './train_layer_{}_image_epoch{}.png'.format(d, epoch))
                self.autoencoder.add_layer()

            self.autoencoder.layerwise_train = False

            print('Finetuning autoencoder')
            # update encoder and decoder weights:
            self.autoencoder.eval()  # changing to evaluation mode so that there will be no dropout layers
            self.optimizer = torch.optim.Adam(self.autoencoder.parameters(), lr=self.learning_rate_ae)
            for epoch in range(finetune_epochs):
                for i_batch, (x, y) in enumerate(dataloader):
                    x = Variable(x.type(torch.FloatTensor)).cuda()
                    encoded, decoded = self.autoencoder(x)
                    tot_loss = mse_loss(decoded.view(x.size(0), -1), x.view(x.size(0), -1))
                    # Backprop + Optimize
                    self.optimizer.zero_grad()
                    tot_loss.backward()
                    self.optimizer.step()
                print('Epoch: ', epoch, '| reconst ae loss:%f ' % (tot_loss.data[0]))
                if plot and epoch % 10 == 0 and len(dataset.train_data.shape) > 2:
                    decoded = decoded.view(x.data.shape)
                    pic = to_img(decoded.cpu().data, rgb, size=decoded.data.shape[2])
                    save_image(pic, './image_epoch{}.png'.format(d, epoch))


            if autoencoder_weights_save_path is not None:
                print("Saving autoencoder weights to: %s" % autoencoder_weights_save_path)
                torch.save(self.autoencoder.state_dict(), autoencoder_weights_save_path)
        else:
            print('Loading pretrained weights for autoencoder from %s' % self.pretrained_weights_path)
            # original saved file with DataParallel
            state_dict = torch.load(self.pretrained_weights_path)
            # create new OrderedDict that does not contain `module.`
            own_state = self.autoencoder.state_dict()
            for name, param in state_dict.items():
                if name in own_state:
                    own_state[name].copy_(param)

        self.autoencoder.eval()
        self.autoencoder.layerwise_train = False
        for i in range(0, len(list(self.autoencoder.encoder))):
            self.autoencoder.encoder[i][0].train(False)
            self.autoencoder.decoder[i][0].train(False)

    def p_mat(self, q):
        """calculating p_mat - the target distribution of the distances"""
        f = q.sum(0)
        weight = q ** 2 / f
        return torch.t(torch.t(weight) / torch.sum(weight, 1))

    def forward(self, x1, x2):
        """
        forward pass in the clustering and ae layers. The output is the encoded and decoded data of the two
        points and their distance.
        :param x1:
        :param x2:
        :return:
        """
        encoded_data1, decoded_data1 = self.autoencoder(x1)
        encoded_data2, decoded_data2 = self.autoencoder(x2)
        dist = torch.sum((decoded_data1 - decoded_data2) ** 2, 1)
        return dist, encoded_data1, decoded_data1, encoded_data2, decoded_data2

    def cluster(self,
                dataset,
                use_vgg=False,
                epochs_max=100,
                plot=False,
                plot_tsne_bool=False,
                not_calculate_conn_mat=True,
                conn_mat_path=None,
                n_neighbors=10,
                dist_meas='cosine',
                num_constraints=0,
                final_conn_threshold=1,
                clust_method = 'MKNN',
                save_result=True,
                lr_u=None,
                mu_epoch_update=None,
                learning_rate_c=0.001,
                conn_mat_var='Z',
                save_net=False
                ):
        self.clust_method = clust_method
        self.learning_rate_c = learning_rate_c
        x_np = dataset.train_data.reshape(dataset.train_labels.shape[0], -1)
        if not_calculate_conn_mat:
            # Getting back the objects:
            with open(conn_mat_path) as f:  # Python 3: open(..., 'rb')
                self.u, self.E, self.W, connections_per_point, encoded_data_np = pickle.load(f)
        else:
            print('Calculating connection matrix and weights')
            dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=1000, shuffle=False)
            self.u = torch.FloatTensor().cuda()
            x_np = torch.FloatTensor()
            for i_batch, (x, y) in enumerate(dataloader):
                x_np = torch.cat([x_np,x.view(x.size(0), -1).type(torch.FloatTensor)])
                x = Variable(x.type(torch.FloatTensor)).cuda()
                encoded = self.autoencoder(x)[0]
                self.u = torch.cat([self.u, encoded.data])
            x_np = x_np.numpy()
            encoded_data_np = self.u.cpu().numpy()
            if conn_mat_var=='Z':
                scaler = StandardScaler()
                scaler.fit(encoded_data_np)
                encoded_data_np_norm = scaler.transform(encoded_data_np)
                self.E, self.W, connections_per_point = \
                    connectivity_structure_mknn(encoded_data_np_norm, n_neighbors, dist_meas)
            elif conn_mat_var=='X':
                if len(dataset.train_data.shape)>2:
                    minmax_scale = MinMaxScaler().fit(x_np)
                    x_np = minmax_scale.transform(x_np)
                self.E, self.W, connections_per_point = \
                    connectivity_structure_mknn(x_np, n_neighbors, dist_meas)
            if conn_mat_path is not None:
                with open(conn_mat_path, 'w') as f:  # Python 3: open(..., 'wb')
                    pickle.dump([self.u, self.E, self.W, connections_per_point, encoded_data_np], f)

        n_samples = len(encoded_data_np)
        if lr_u == None:
            lr_u = 0.01
            if len(self.E.data/float(n_samples**2)) < 0.002:
                lr_u = 0.04
            if use_vgg:
                lr_u = 0.1
        if mu_epoch_update == None:
            mu_epoch_update = 30
            if use_vgg and len(self.E.data/float(n_samples**2)) > 0.002:
                mu_epoch_update = 10



        if isinstance(dataset.train_labels, np.ndarray):
            y=dataset.train_labels
        else:
            y = dataset.train_labels.numpy()

        self.u = Variable(self.u.cuda(), requires_grad=True)

        # calculate mu for the normalization of the mcclure norm loss between connected points:
        encoded_data1 = encoded_data_np[self.E.row,]
        encoded_data2 = encoded_data_np[self.E.col,]
        dists = np.sqrt(np.sum((encoded_data1 - encoded_data2) ** 2, 1)) + 1e-5
        encoded_len = encoded_data_np.shape[1]
        idx = (dists / np.sqrt(encoded_len)) < 1e-2
        if idx.size:
            dists[(dists / np.sqrt(encoded_len)) < 1e-2] = np.max(dists)
        dists.sort(0)
        shortest = dists[0:min(250, int(np.ceil(0.01 * len(dists))))]
        cl_thresh = (np.max(dists)).astype(torch.FloatTensor)
        self.delta2 = np.mean(shortest)
        self.epsilon = np.mean(dists[0:int(np.ceil(0.01 * len(dists)))])
        mean_encoded = np.mean(encoded_data_np, 0)
        dist_encoded = np.sqrt(np.sum((encoded_data_np - mean_encoded) ** 2, 1))
        self.delta1 = np.mean(dist_encoded)
        mu_gmc2 = 3 * np.max(dists) ** 2
        mu_gmc2 = mu_gmc2.astype(torch.FloatTensor)
        mu_gmc1 = max(2000, 8 * self.delta1)
        self.final_conn_threshold = final_conn_threshold

        # calculate initial accuracy using kmeans:
        y_true_unique = np.unique(y)
        self.n_clusters = len(y_true_unique)
        kmeans = KMeans(n_clusters=self.n_clusters, n_init=20)
        y_pred = kmeans.fit_predict(encoded_data_np)
        acc, ami, nmi, mat_w, y_pred = cluster_acc(y, y_pred)
        print('Initial Accuracy using kmeans' + str(np.round(acc, 5)) +
              ', AMI' + str(np.round(ami, 5)) + ', NMI' + str(np.round(nmi, 5)))

        if num_constraints>0:
            self.E, self.W = connectivity_structure_Loss_constraints(dataset, encoded_data_np, mu_gmc2,
                                                                     self.E, self.W, num_constraints)
            i_cl_row = self.E.row[self.E.data==-1]
            i_cl_col = self.E.col[self.E.data==-1]
            sample_cl = random.sample(xrange(0,len(i_cl_row)), min(len(i_cl_row),20))
            i_cl_row = i_cl_row[sample_cl]
            i_cl_col = i_cl_col[sample_cl]

        # initialize image for TSNE
        idx_tsne = random.sample(range(0, n_samples), min(n_samples, 10000))
        if plot_tsne_bool:
            if num_constraints>0:
                plot_pca_constraints(dataset, encoded_data_np, i_cl_row, i_cl_col, idx_tsne, 0)
            else:
                plot_tsne(y, encoded_data_np, idx_tsne, 0)
                plot_pca(y, encoded_data_np, idx_tsne, 0)

        #converting into data pairs:
        pairs = PairingDataset(dataset, self.W, self.E, constraints_type=1)


        # Data Loader
        data_loader = torch.utils.data.DataLoader(dataset=pairs, batch_size=self.batch_size_c, shuffle=True, pin_memory=True)
        self.optimizer = torch.optim.RMSprop([{'params': self.autoencoder.parameters()}, {'params':[self.u], 'lr':lr_u}],
                                          lr=self.learning_rate_c)
        self.accuracy = []
        self.ami = []
        self.nmi = []
        self.total_loss = []
        self.delta_label = []
        graph, y_pred, num_components = calculate_clusters(self.clust_method, self.E, self.epsilon,
                                                                self.final_conn_threshold, self.u, self.n_clusters)
        self.y_pred = y_pred
        if y is not None:
            acc, ami, nmi, mat_w, y_pred_fit = cluster_acc(y, self.y_pred)
            self.accuracy.append(acc)
            self.ami.append(ami)
            self.nmi.append(nmi)
            print('Initial Accuracy ' + str(np.round(acc, 5)) + ', AMI ' + str(np.round(ami, 5)) +
                  ', NMI ' + str(np.round(nmi, 5)) + ', number of clusters ' + str(num_components))


        # note: compute the largest magnitude eigenvalue instead of the matrix norm as it is faster to compute
        mknn_pairs_places = self.E.data==1
        R = coo_matrix((np.concatenate([self.W.data[mknn_pairs_places], self.W.data[mknn_pairs_places]], axis=0),
                                     (np.concatenate([self.W.row[mknn_pairs_places], self.W.col[mknn_pairs_places]],axis=0)
                                      , np.concatenate([self.W.col[mknn_pairs_places], self.W.row[mknn_pairs_places]],axis=0))),
                                    shape=[n_samples, n_samples])
        D = coo_matrix((np.squeeze(np.asarray(np.sum(R, axis=1))), ((range(n_samples), range(n_samples)))),
                       (n_samples, n_samples))
        eigval_DR = eigs(D - R, k=1, return_eigenvectors=False).real
        lamda = float(np.linalg.norm(encoded_data_np, 2)/(eigval_DR[0]))
        lamda_ml=lamda
        reconst_factor = 1/float(dataset.train_data[0].shape[0]**2)
        rep_factor = 1/float(encoded_len)
        label_percent = num_constraints / float(len(self.E.row))
        plot_tsne_interval = 30
        for epoch in range(0, epochs_max):
            sys.stdout.write('\r')
            # train on batch
            sys.stdout.write('epoch %d, ' % epoch)
            for i_batch, data_batch in enumerate(data_loader):
                idx1, idx2, x1, x2, w, conn_type = data_batch
                x1 = Variable(x1.type(torch.FloatTensor)).cuda()
                x2 = Variable(x2.type(torch.FloatTensor)).cuda()
                # taking the relevant u indexes
                u1 = self.u.index_select(0, Variable(idx1.long(), requires_grad=False).cuda())
                u2 = self.u.index_select(0, Variable(idx2.long(), requires_grad=False).cuda())
                w = Variable(w.type(torch.FloatTensor), requires_grad=False).cuda()
                conn_type = conn_type.cuda()
                w_reconst1 = 1/ np.asarray([connections_per_point[0, i] for i in idx1])
                w_reconst2 = 1/ np.asarray([connections_per_point[0, i] for i in idx2])
                w_reconst1 = Variable(torch.FloatTensor(w_reconst1), requires_grad=False).cuda()
                w_reconst2 = Variable(torch.FloatTensor(w_reconst2), requires_grad=False).cuda()
                _, encoded_data1, decoded_data1, encoded_data2, decoded_data2 = self(x1, x2)
                dist = torch.sum((u1 - u2) ** 2, 1)
                w_mknn_ul = w*Variable((conn_type==1).type(torch.FloatTensor),requires_grad=False).cuda()
                clust_loss = torch.sum(w_mknn_ul * mu_gmc2 * dist / (mu_gmc2 + dist))
                # must-link loss:
                w_ml = w*Variable((conn_type==2).type(torch.FloatTensor),requires_grad=False).cuda()
                ml_loss = 1/(label_percent+1e-7)*torch.sum(w_ml * mu_gmc2 * dist / (mu_gmc2 + dist))
                reconst_loss1 = torch.sum(
                    w_reconst1 * torch.sum((decoded_data1.view(decoded_data1.size(0), -1) - x1.view(x1.size(0), -1)) ** 2, 1))
                reconst_loss2 = torch.sum(
                    w_reconst2 * torch.sum((decoded_data2.view(decoded_data2.size(0), -1) - x2.view(x2.size(0), -1)) ** 2, 1))
                reconst_loss = reconst_loss1 + reconst_loss2
                rep_dists1 = torch.sum((u1 - encoded_data1) ** 2, 1)
                rep_dists2 = torch.sum((u2 - encoded_data2) ** 2, 1)
                rep_loss = torch.sum(w_reconst1 * mu_gmc1 * rep_dists1 / (rep_dists1 + mu_gmc1)) + \
                           torch.sum(w_reconst2 * mu_gmc1 * rep_dists2 / (rep_dists2 + mu_gmc1))
                tot_loss = 1/float(self.batch_size_c)*(rep_factor * (lamda * clust_loss + lamda_ml * ml_loss)
                                                       + rep_factor*rep_loss + reconst_factor*reconst_loss)
                self.optimizer.zero_grad()
                tot_loss.backward()
                self.optimizer.step()
            print('clust loss: %f ml loss: %f reconst ae loss:%f  representation loss:%f total loss:%f' %
                  (1/float(self.batch_size_c)*lamda * rep_factor * clust_loss.data[0],
                   1 / float(self.batch_size_c) * lamda_ml * rep_factor * ml_loss.data[0],
                   1/float(self.batch_size_c) * reconst_factor * reconst_loss.data[0],
                   1 / float(self.batch_size_c) * rep_factor * rep_loss.data[0], tot_loss.data[0]))

            # change mu1 and mu2 every mu_epoch_update:
            if (epoch+1) % mu_epoch_update == 0:
                mu_gmc1 = max(mu_gmc1/2, float(self.delta1/2))
                mu_gmc2 = max(mu_gmc2/2, float(self.delta2/2))

            graph, y_pred, num_components = calculate_clusters(self.clust_method, self.E, self.epsilon,
                                                                self.final_conn_threshold, self.u, self.n_clusters)

            delta_label = ((y_pred != self.y_pred).sum().astype(np.float32) / y_pred.shape[0])
            self.y_pred = y_pred
            if y is not None:
                acc, ami, nmi, mat_w, y_pred_fit = cluster_acc(y, self.y_pred)
                self.accuracy.append(acc)
                self.ami.append(ami)
                self.nmi.append(nmi)
                self.delta_label.append(delta_label)
                self.total_loss.append(float(tot_loss.data.cpu().numpy()))
                print('epoch ' + str(epoch) + ', Accuracy ' + str(np.round(acc, 5)) +
                      ', AMI ' + str(np.round(ami, 5)) + ', NMI ' + str(np.round(nmi, 5)) +
                      ', number of clusters ' + str(num_components))

                if plot_tsne_bool and ((epoch+1) % plot_tsne_interval)==0:
                    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=1000, shuffle=False)
                    encoded_data_np = torch.FloatTensor().cuda()
                    for i_batch, (x, _) in enumerate(dataloader):
                        x = Variable(x.type(torch.FloatTensor)).cuda()
                        encoded = self.autoencoder(x)[0]
                        encoded_data_np = torch.cat([encoded_data_np, encoded.data])
                    encoded_data_np = encoded_data_np.cpu().numpy()
                    if num_constraints>0:
                        plot_pca_constraints(dataset, self.u.data.cpu().numpy(), i_cl_row, i_cl_col, idx_tsne, epoch+1)
                        plot_tsne_constraints(dataset, self.u.data.cpu().numpy(), i_cl_row, i_cl_col, idx_tsne, epoch+1)

                    else:
                        plot_tsne(y,  self.u.data.cpu().numpy(), idx_tsne, epoch+1)
                        plot_pca(y, self.u.data.cpu().numpy(), idx_tsne, epoch+1)

        if plot or plot_tsne_bool:
            plt.ioff()
            plt.show()
        if epoch == (epochs_max-1):
            print('Reached maximum epochs limit. Stopping training.')
        if save_result:
            results_file_name = './dataset_{dataset}_lr{lr}_u{lr_u}_{conn_var}_mu{mu_epoch}_clust{clust_method}_MKNN{n}'. \
                                    format(dataset=dataset.name, lr=self.learning_rate_c, lr_u=lr_u,
                                           mu_epoch=mu_epoch_update, n=n_neighbors,
                                           clust_method=clust_method, conn_var=conn_mat_var) \
                                + (num_constraints>0) * 'num_constraints{num_constraints}'.format(num_constraints=num_constraints)
            with open(results_file_name + '.pkl', 'w') as f:  # Python 3: open(..., 'wb')
                pickle.dump([self.accuracy, self.ami, self.nmi, self.total_loss, self.delta_label,y_pred], f)
            plot_results(acc=self.accuracy, ami=self.ami, nmi=self.nmi, delta_label=self.delta_label,
                         total_loss=self.total_loss, plot_name=results_file_name)
        if save_net:
            net_file_name = './dataset_{dataset}_lr{lr}_u{lr_u}_{conn_var}_mu{mu_epoch}_clust{clust_method}_MKNN{n}'. \
                                    format(dataset=dataset.name, lr=self.learning_rate_c, lr_u=lr_u,
                                           mu_epoch=mu_epoch_update, n=n_neighbors,
                                           clust_method=clust_method, conn_var=conn_mat_var) \
                                + (num_constraints>0) * 'num_constraints{num_constraints}'.format(num_constraints=num_constraints)

            torch.save([self.autoencoder.state_dict(), self.u.data, y_pred_fit], net_file_name)
        return self.y_pred
