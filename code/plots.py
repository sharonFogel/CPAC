import sys
import numpy as np
import torch
import torchvision.transforms
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.gridspec as gridspec
from sklearn.manifold import TSNE
from sklearn import decomposition
if (sys.version[0] == 2):
    import cPickle as pickle
else:
    import pickle
import random

def plot_tsne(y, encoded_data, idx_tsne, epoch):
    #preparing colormap:
    num_labels = int(max(y) + 1)
    cmap = plt.get_cmap('jet')
    mymap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=0, b=1),
        cmap(np.linspace(0, 1, num_labels)))
    Z = [[0, 0], [0, 0]]
    levels = range(0, num_labels + 1, 1)
    CB = plt.contourf(Z, levels, cmap=mymap)

    #calculating tsne
    u_tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300).fit_transform(
    [encoded_data[i] for i in idx_tsne])

    # Plot figure
    plt.figure(1, clear=True)
    plt.scatter(u_tsne[:, 0], u_tsne[:, 1], c=[y[i] for i in idx_tsne],
                cmap=mymap, s=1)
    plt.colorbar(CB, ticks=range(num_labels))
    plt.clim(-0.5, float(num_labels) - 0.5)
    plt.savefig('./tsne_epoch_{}.png'.format(epoch))
    return u_tsne


def plot_pca(y, encoded_data, idx_tsne, epoch):
    #preparing colormap:
    num_labels = int(max(y) + 1)
    cmap = plt.get_cmap('jet')
    mymap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=0, b=1),
        cmap(np.linspace(0, 1, num_labels)))
    Z = [[0, 0], [0, 0]]
    levels = range(0, num_labels + 1, 1)
    CB = plt.contourf(Z, levels, cmap=mymap)

    #calculating pca
    encoded_data = [encoded_data[i] for i in idx_tsne]
    pca = decomposition.PCA(n_components=2)
    pca.fit(encoded_data)
    u_pca = pca.transform(encoded_data)

    # Plot figure
    plt.figure(1, clear=True)
    plt.scatter(u_pca[:, 0], u_pca[:, 1], c=[y[i] for i in idx_tsne],
                cmap=mymap, s=1)
    plt.colorbar(CB, ticks=range(num_labels))
    plt.clim(-0.5, float(num_labels) - 0.5)
    plt.savefig('./pca_epoch_{}.png'.format(epoch))
    return u_pca


def frame_image(img, frame_width, frame_color):
    """
    adding a frame to te image
    :param img:
    :param frame_width:
    :param frame_color:
    :return: framed_img
    """
    b = frame_width # border size in pixel
    ny, nx = img.shape[0], img.shape[1] # resolution / number of pixels in x and y
    framed_img = np.ones((b + ny + b, b + nx + b, 4))
    framed_img[:, :, 0] = frame_color[0]*np.ones((b + ny + b, b + nx + b))
    framed_img[:, :, 1] = frame_color[1]*np.ones((b + ny + b, b + nx + b))
    framed_img[:, :, 2] = frame_color[2]*np.ones((b + ny + b, b + nx + b))
    framed_img[:, :, 3] = frame_color[3] * np.ones((b + ny + b, b + nx + b))
    if img.ndim == 3: # rgb or rgba array
        framed_img[b:-b, b:-b, 0:2] = img
    elif img.ndim == 2: # grayscale image
        framed_img[b:-b, b:-b, 0] = img
        framed_img[b:-b, b:-b, 1] = img
        framed_img[b:-b, b:-b, 2] = img
    return framed_img

def plot_tsne_constraints(dataset, encoded_data, i_cl_row,i_cl_col,idx_tsne, epoch):
    """
    plots tsne of data points with connections between cannot-link constraints
    :param dataset:
    :param encoded_data:
    :param i_cl_row:
    :param i_cl_col:
    :param idx_tsne:
    :param epoch:
    :return:
    """
    if isinstance(dataset.train_labels, np.ndarray):
        y = dataset.train_labels
    else:
        y = dataset.train_labels.numpy()
    num_labels = int(max(y) + 1)
    cmap = plt.get_cmap('jet')
    mymap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=0, b=1),
        cmap(np.linspace(0, 1, num_labels)))
    Z = [[0, 0], [0, 0]]
    levels = range(0, num_labels + 1, 1)
    CB = plt.contourf(Z, levels, cmap=mymap)
    idx_tsne = np.concatenate((idx_tsne, i_cl_row, i_cl_col))
    idx_tsne = np.unique(idx_tsne)
    idx_constraints_row_tsne = idx_tsne.searchsorted(i_cl_row)  # idx_tsne is already sorted so it returns the indices
                                                                # of i_cl_row in idx_tsne
    idx_constraints_col_tsne = idx_tsne.searchsorted(i_cl_col)
    u_tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300).fit_transform(
    [encoded_data[i] for i in idx_tsne])

    # Plot figure with subplots of different sizes - large 10*10 for tsne + two columns of 10 images for the pairs
    plt.figure(1, clear=True)
    # set up subplot grid
    gridspec.GridSpec(10, 12)

    # large subplot - tsne:
    plt.subplot2grid((10, 12), (0, 0), colspan=10, rowspan=10)
    plt.scatter(u_tsne[:, 0], u_tsne[:, 1], marker='o', c=[y[i] for i in idx_tsne],
                cmap=mymap, s=10, edgecolors='k', linewidths=0.3)
    plt.colorbar(CB, ticks=range(num_labels))
    plt.clim(-0.5, float(num_labels) - 0.5)
    norm = colors.Normalize(0,num_labels)
    for i in range(len(idx_constraints_col_tsne)):
        color = cmap(norm(y[i_cl_col[i]]))
        plt.plot([u_tsne[idx_constraints_row_tsne[i], 0], u_tsne[idx_constraints_col_tsne[i], 0]],
                 [u_tsne[idx_constraints_row_tsne[i], 1], u_tsne[idx_constraints_col_tsne[i], 1]],
                 color='k', marker='o', markersize=3,
                 markerfacecolor=color, markeredgewidth=0.3)
    plt.savefig( './epoch_{}.png'.format(epoch))

def plot_pca_constraints(dataset, encoded_data, i_cl_row,i_cl_col,idx_tsne, epoch):
    """
    plots pca of data points with connections between cannot-link constraints
    :param dataset:
    :param encoded_data:
    :param i_cl_row:
    :param i_cl_col:
    :param idx_tsne:
    :param epoch:
    :return:
    """
    if isinstance(dataset.train_labels, np.ndarray):
        y = dataset.train_labels
    else:
        y = dataset.train_labels.numpy()
    num_labels = int(max(y) + 1)
    cmap = plt.get_cmap('jet')
    mymap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=0, b=1),
        cmap(np.linspace(0, 1, num_labels)))
    Z = [[0, 0], [0, 0]]
    levels = range(0, num_labels + 1, 1)
    CB = plt.contourf(Z, levels, cmap=mymap)
    idx_tsne = np.concatenate((idx_tsne, i_cl_row, i_cl_col))
    idx_tsne = np.unique(idx_tsne)
    idx_constraints_row_tsne = idx_tsne.searchsorted(i_cl_row)
    idx_constraints_col_tsne = idx_tsne.searchsorted(i_cl_col)
    encoded_data = [encoded_data[i] for i in idx_tsne]
    pca = decomposition.PCA(n_components=2)
    pca.fit(encoded_data)
    u_pca = pca.transform(encoded_data)
    y_pca = [y[i] for i in idx_tsne]
    # save output:
    with open('./pca_epoch_{}.png'.format(epoch), 'w') as f:  # Python 3: open(..., 'wb')
        pickle.dump([u_pca, idx_constraints_row_tsne, idx_constraints_col_tsne, y, idx_tsne], f)

    # Plot figure with subplots of different sizes - large 10*10 for tsne + two columns of 10 images for the pairs
    plt.figure(1, clear=True)
    # set up subplot grid
    gridspec.GridSpec(10, 12)

    # large subplot - tsne:
    plt.subplot2grid((10, 12), (0, 0), colspan=10, rowspan=10)
    plt.scatter(u_pca[:, 0], u_pca[:, 1], marker='o', c=y_pca,
                cmap=mymap, s=1)
    plt.colorbar(CB, ticks=range(num_labels))
    plt.clim(-0.5, float(num_labels) - 0.5)

    norm = colors.Normalize(0, num_labels)
    for i in range(len(idx_constraints_col_tsne)):
        color = cmap(norm(y[idx_constraints_row_tsne[i]]))
        plt.plot([u_pca[idx_constraints_row_tsne[i], 0], u_pca[idx_constraints_col_tsne[i], 0]],
                 [u_pca[idx_constraints_row_tsne[i], 1], u_pca[idx_constraints_col_tsne[i], 1]],
                 color='k', marker='o', markersize=0.3,
                 markerfacecolor=color, markeredgewidth=0.1)
    plt.savefig('./pca_epoch_{}.png'.format(epoch))


def plot_connections(y,encoded_data,idx_tsne, epoch, E):
    u_pca = plot_pca(y, encoded_data, idx_tsne, epoch)
    idx = random.sample(range(0, len(E.row)), min(len(E.row), len(E.row)))
    for i in idx:
        if y[E.row[i]]==y[E.col[i]]:
            color = 'b'
        else:
            color='r'
        plt.plot([u_pca[E.row[i], 0], u_pca[E.col[i], 0]],
                 [u_pca[E.row[i], 1], u_pca[E.col[i], 1]],
                 color=color, marker='o', markersize=0.001,
                 markerfacecolor='k', markeredgewidth=0.001)
        plt.ylim(-50, 50)
        plt.xlim(-50, 50)
    plt.savefig('./connections_epoch_{}.png'.format(epoch))

def plot_results(acc, ami, nmi, clust_loss, ml_loss,
                 reconst_loss, rep_loss, rep_mult, total_loss, plot_name):
    """
    plots the result of the clustering run
    :param acc:
    :param ami:
    :param nmi:
    :param total_loss:
    :param delta_label:
    :param plot_name:
    :return:
    """
    epochs = range(0, len(acc), 1)
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    tot_loss_plot = ax1.plot(epochs[0:-1], total_loss, 'b--', label='total loss')
    clust_loss_plot = ax1.plot(epochs[0:-1], clust_loss, 'y--', label='clust_loss')
    ml_loss_plot = ax1.plot(epochs[0:-1], ml_loss, 'c--', label='ml_loss')
    rec_loss_plot = ax1.plot(epochs[0:-1], ml_loss, 'm--', label='ml_loss')
    rep_loss_plot = ax1.plot(epochs[0:-1], rep_loss, 'g--', label='rep_loss')
    rep_mult_plot = ax1.plot(epochs[0:-1], rep_mult, 'r--', label='rep_mult')
    ax1.set_ylabel('losses')
    ax1.set_ylim(0,100)
    # plt.legend(bbox_to_anchor=(1., 1.),bbox_transform=plt.gcf().transFigure, loc=2, fontsize='small')
    plt.legend(loc=2, fontsize='small')

    diff_loss = np.abs(np.asarray(total_loss[1:len(total_loss)])-np.asarray(total_loss[0:-1]))
    diff_loss_plot = ax1.plot(epochs[0:-2], diff_loss, 'k-', label='diff loss')

    ax2 = ax1.twinx()
    acc_plot = ax2.plot(epochs, acc, 'r-', label='acc')
    ax2.set_ylabel('acc, ami, nmi, delta_label', color='r')
    for tl in ax2.get_yticklabels():
        tl.set_color('r')
    ami_plot = ax2.plot(epochs, ami, 'g-', label='ami')
    nmi_plot = ax2.plot(epochs, nmi, 'm-', label='nmi')

    plt.legend(loc=2, fontsize='small')
    plt.savefig(plot_name+'.png')


def plot_cl_pairs_images(image_pairs, cl_places):
    """
    plots pairs of images with high clustering loss. Cannot link pairs are marked in red.
    :param image_pairs: pairs of images
    :param cl_places: places of cannot link pairs
    :return:
    """
    color_cl = [1,0,0,1] #red
    color_default = [0, 0, 0, 1] # black
    # set up subplot grid - first column is low si image, the next 5 are the closest 5 cluster representatives
    heights = [a[0].shape[0] for a in image_pairs[0,:,:,:]]
    widths = [a.shape[1] for a in image_pairs[:,0,:,:]]
    plt.figure(1, clear=True)
    fig_width = 8.  # inches
    fig_height = fig_width * sum(heights) / sum(widths)
    f, axarr = plt.subplots(2, len(cl_places), figsize=(fig_width, fig_height),
                            gridspec_kw={'height_ratios': heights})
    # plot low_si images:
    for i in range(len(cl_places)):
        if cl_places[i]:
            img1 = frame_image(image_pairs[i,0]/255.0,1,color_cl)
            img2 = frame_image(image_pairs[i,1]/255.0,1,color_cl)
        else:
            img1 = frame_image(image_pairs[i,0]/255.0,1,color_default)
            img2 = frame_image(image_pairs[i,1]/255.0,1,color_default)
        axarr[0, i].imshow(img1)
        axarr[0, i].axis('off')
        axarr[1, i].imshow(img2)
        axarr[1, i].axis('off')
    plt.subplots_adjust(wspace=0.1, hspace=0.01, left=0, right=1, bottom=0, top=1)
    plt.savefig( './high_loss_pairs.png')

