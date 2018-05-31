from CPAC import CPAC
from create_datasets import *
from torchvision import transforms
import matplotlib as mpl
mpl.use('Agg')
import argparse
from lw_ae import lw_ae
from pretrained_model import *

# setting parameters
parser = argparse.ArgumentParser(description='CPAC Clustering')
parser.add_argument("--plot-reconst", help="plot reconstructed images while running, default=False",
                    action="store_true")
parser.add_argument("--plot-tsne", help="plot tsne while running the clustering stage, default=False",
                    action="store_true")

# dataset parameters
parser.add_argument('--input', type=str, metavar='<file path>',
                   help='dataset name should be the name of the h5 file without the 4torch prefix. If not specified USPS is used', default='usps', required=False)
parser.add_argument('--not-image', help="input data is not images (changes the preprocessing)", action="store_true")
parser.add_argument("--rgb", help="Specify to use rgb images instead of grayscale images (default is false)",
                    action="store_true")
parser.add_argument('--input-length', type=int, metavar='<dataset_length>',
                   help='number of samples taken from the dataset, use -1 to take the entire dataset', default=-1, required=False)
parser.add_argument("--use-vgg19", help="use output of vgg as input to autoencoder (default is false)",
                    action="store_true")

# AE parameters
parser.add_argument('--ae-dims', metavar='<ae dims>',
                    help='dimensions of linear layers in AE (after conv layers)', type=int, nargs='+', default=[500, 500, 2000, 10], required=False)
parser.add_argument('--dropout', type=float, metavar='<dropout>',
                    help='autoencoder dropout during layerwise training, default is set to 0.2',
                    default=0.2, required=False)


# AE learning parameters
parser.add_argument('--encoder-weights-input-path', type=str, metavar='<file path>',
                    help='Path to a file that contains the autoencoder weights', default=None, required=False)
parser.add_argument('--encoder-weights-output-path', type=str, metavar='<file path>',
                    help='The path of the file in which the autoencoder weights will be saved', default=None, required=False)
parser.add_argument('--ae-lw-epochs', type=int, metavar='<ae_lw_epochs>',
                   help='number of learning epochs for autoencoder layerwise training. default=30', default=50, required=False)
parser.add_argument('--ae-finetune-epochs', type=int, metavar='<ae_finetune_epochs>',
                   help='number of learning epochs for autoencoder finetuning training. default=30', default=50, required=False)
parser.add_argument('--batch-size-ae', type=int, metavar='<batch_size_ae>',
                   help='batch size for training ae', default=128, required=False)
parser.add_argument('--learning-rate-ae', type=float, metavar='<learning_rate_ae>',
                   help='learning rate for ae training', default=0.0001, required=False)

# constraints parameters
parser.add_argument('--num-constraints', type=float, metavar='<num_constraints>',
                    help='number of labeled pairs. Default is 0 (no constraints)', default=0, required=False)

# clustering parameters
parser.add_argument('--n-neighbors', type=int,metavar='<n_neighbors>',
                    help='number of nn used in th mknn to find pairs of points', default=10, required=False)
parser.add_argument("--not-calculate-conn-mat", help="Don't calculate the connection matrix of the encoded data,"
                                                     " take it from the conn-mat-path file, default=False",
                    action="store_true")
parser.add_argument('--conn-mat-path', type=str, metavar='<conn_mat_path>',
                    help='Path to a file that contains connection matrix', default=None, required=False)
parser.add_argument('--final-conn-threshold', type=float, metavar='<final_conn_threshold>',
                   help='threshold multiplier for connections when calculating the final clustering', default=1, required=False)
parser.add_argument('--dist-meas', type=str, metavar='<dist_meas>',
                    help='distance measurement for MKNN calculation ', default='cosine', required=False)
parser.add_argument('--conn-mat-var', type=str, metavar='<conn_mat_var>',
                    help='variable used for connectivity matrix calculation, default is X', default='X', required=False)

# clustering learning parameters
parser.add_argument('--batch-size-c', type=int, metavar='<batch_size_c>',
                   help='batch size for training clustering', default=128, required=False)
parser.add_argument('--learning-rate-c', type=float, metavar='<learning_rate_c>',
                   help='learning rate for clustering training', default=0.0001, required=False)
parser.add_argument('--learning-rate-u', type=float, metavar='<learning_rate_u>',
                   help='learning rate for clustering training for u', default=None, required=False)
parser.add_argument('--mu-epoch-update', type=int, metavar='<mu_epoch_update>',
                   help='epochs between mu update, default is 30', default=None, required=False)
parser.add_argument('--max-cluster-epochs', type=int, metavar='<max_cluster_epochs>',
                    help='The number of training epochs for clustering', required=False,
                    default=int(1000))
parser.add_argument('--output', type=str, metavar='<file path>',
                    help='Path to output file, which contains the clustering results', required=False)
parser.add_argument('--clust-method', type=str, metavar='<clust_method>',
                    help='method for final clustering: MKNN, kmeans, spectral, dbscan', required=False,
                    default='MKNN')
parser.add_argument("--save-net", help="save final network, default=False",
                    action="store_true")
parser.add_argument("--ADMM", help="use ADMM framework for training, default=False",
                    action="store_true")
parser.add_argument('--epochs-u', type=int, metavar='<epochs_u>',
                   help='number of epochs training on u in ADMM framework', default=1, required=False)
parser.add_argument('--epochs-z', type=int, metavar='<epochs_z>',
                   help='number of epochs training on z in ADMM framework', default=1, required=False)
parser.add_argument('--lr-lm', type=float, metavar='<lr_lm>',
                   help='learning rate lagrange multiplier', default=0.01, required=False)


# get all parameters
args = parser.parse_args()

# setting parameters
plot_reconst = args.plot_reconst
plot_tsne = args.plot_tsne

# dataset parameters
input = args.input
rgb = args.rgb
length = args.input_length
use_vgg19 = args.use_vgg19
not_image = args.not_image

# AE parameters
ae_dims = args.ae_dims
dropout = args.dropout

# AE learning parameters
encoder_weights_path = args.encoder_weights_input_path
encoder_weights_output_path = args.encoder_weights_output_path
ae_lw_epochs = args.ae_lw_epochs
ae_finetune_epochs = args.ae_finetune_epochs
batch_size_ae = args.batch_size_ae
learning_rate_ae = args.learning_rate_ae

# constraints parameters
num_constraints = args.num_constraints

# clustering parameters
n_neighbors = args.n_neighbors
not_calculate_conn_mat = args.not_calculate_conn_mat
conn_mat_path = args.conn_mat_path
final_conn_threshold = args.final_conn_threshold
conn_mat_var=args.conn_mat_var
dist_meas = args.dist_meas
save_net = args.save_net

# clustering learning  parameters
batch_size_c = args.batch_size_c
learning_rate_c = args.learning_rate_c
learning_rate_u = args.learning_rate_u
mu_epoch_update = args.mu_epoch_update
max_cluster_epochs = args.max_cluster_epochs
output_file_path = args.output
clust_method = args.clust_method
ADMM = args.ADMM
epochs_u = args.epochs_u
epochs_z = args.epochs_z
lr_lm = args.lr_lm

if encoder_weights_path is not None and encoder_weights_output_path is not None:
    parser.error("Cannot provider both input and output path for autoencoder weights")

#prepare data
torch.manual_seed(2)

img_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
if input=='MNIST' or input=='usps':
    img_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

if not_image:
    dataset = New_Dataset(name=input, rgb=False, use_vgg19=False, transform=None)
else:
    dataset = New_Dataset(name=input, rgb=rgb, use_vgg19=use_vgg19, transform=img_transform)

# take part of the dataset
if length>0:
    new_data = dataset.train_data[0:length]
    new_labels = dataset.train_labels[0:length]
    dataset.train_labels = new_labels
    dataset.train_data = new_data

# a = dataset.train_labels<10
# dataset.train_labels = dataset.train_labels[a]
# dataset.train_data = dataset.train_data[a, :, :, :]


# create autoencoder
data_size = dataset.train_data.shape
if use_vgg19:
    ae_dims = [dataset.train_data.shape[1]]+ae_dims
    data_size = dataset.train_data.shape
    y_true = dataset.train_labels
else:
    if len(dataset.train_data.shape)<3:
        ae_dims = [dataset.train_data.shape[1]] + ae_dims
    else:
        ae_dims = [dataset.train_data.shape[1]*dataset.train_data.shape[2]*dataset.train_data.shape[3]]+ae_dims

ae = lw_ae(data_size,ae_dims, dropout)

cluster_net = CPAC(autoencoder=ae,
                   learning_rate_ae=learning_rate_ae,
                   pretrained_weights_path=encoder_weights_path,
                   batch_size_ae=batch_size_ae,
                   batch_size_c=batch_size_c,
                   )


# train autoencoder
cluster_net.initialize(dataset=dataset,
                       autoencoder_weights_save_path=encoder_weights_output_path,
                       layerwise_epochs=ae_lw_epochs,
                       finetune_epochs=ae_finetune_epochs,
                       plot=plot_reconst,
                       rgb=rgb)

# train clustering
y_pred = cluster_net.cluster(dataset=dataset,
                             use_vgg=use_vgg19,
                             epochs_max=max_cluster_epochs,
                             plot=plot_reconst,
                             plot_tsne_bool=plot_tsne,
                             not_calculate_conn_mat=not_calculate_conn_mat,
                             conn_mat_path=conn_mat_path,
                             dist_meas=dist_meas,
                             n_neighbors=n_neighbors,
                             num_constraints=num_constraints,
                             final_conn_threshold=final_conn_threshold,
                             clust_method=clust_method,
                             save_result=True,
                             lr_u = learning_rate_u,
                             mu_epoch_update=mu_epoch_update,
                             learning_rate_c=learning_rate_c,
                             conn_mat_var=conn_mat_var,
                             save_net=save_net,
                             ADMM=ADMM,
                             epochs_u=epochs_u,
                             epochs_z=epochs_z,
                             lr_lm=lr_lm
                             )
