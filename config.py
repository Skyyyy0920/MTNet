import torch
import argparse

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_args():
    parser = argparse.ArgumentParser(description="MTNet's args")
    # Operation environment
    parser.add_argument('--seed',
                        type=int,
                        default=20020310,
                        help='Random seed')
    parser.add_argument('--device',
                        type=str,
                        default=device,
                        help='Running on which device')
    # Data
    parser.add_argument('--dataset',
                        type=str,
                        default='NYC',
                        # default='TKY',
                        # default='Gowalla-CA',
                        help='Dataset name')

    # Model hyper-parameters
    parser.add_argument('--n_time_slot',
                        type=int,
                        default=4,
                        help='number of time slot (period node)')
    parser.add_argument('--user_embed_dim',
                        type=int,
                        default=128,
                        help='User embedding dimensions')
    parser.add_argument('--POI_embed_dim',
                        type=int,
                        default=128,
                        help='POI embedding dimensions')
    parser.add_argument('--cat_embed_dim',
                        type=int,
                        default=32,
                        help='Category embedding dimensions')
    parser.add_argument('--K_cluster',
                        type=int,
                        default=60,
                        help='number of cluster for geographic info')
    parser.add_argument('--coo_embed_dim',
                        type=int,
                        default=32,
                        help='Coordinate embedding dimensions')
    parser.add_argument('--embed_dropout',
                        type=float,
                        default=0.4,
                        help='Dropout rate for embedding')
    parser.add_argument('--model_dropout',
                        type=float,
                        default=0.6,
                        help='Dropout rate for TreeLSTM')
    parser.add_argument('--h_size',
                        type=int,
                        default=512,
                        help='Hidden size for TreeLSTM')  # 512

    # Training hyper-parameters
    parser.add_argument('--batch_size',
                        type=int,
                        default=1024,
                        help='Batch size')  # 1024
    parser.add_argument('--accumulation_steps',
                        type=int,
                        default=32,
                        help='Gradient accumulation to solve the GPU memory problem')
    parser.add_argument('--epochs',
                        type=int,
                        default=50,
                        help='Number of epochs to train')
    parser.add_argument('--lr',
                        type=float,
                        default=1e-3,
                        help='Initial learning rate')
    parser.add_argument('--lr_step_size',
                        type=int,
                        default=6,
                        help='Learning rate scheduler factor')
    parser.add_argument('--lr_gamma',
                        type=float,
                        default=0.9,
                        help='Learning rate scheduler factor')
    parser.add_argument('--weight_decay',
                        type=float,
                        default=1e-4,
                        help='Weight decay (L2 loss on parameters)')
    parser.add_argument('--patience',
                        type=int,
                        default=4,
                        help='the patience for early stopping')

    # Experiment configuration
    parser.add_argument('--plot_tree',
                        type=bool,
                        default=False,
                        help='Whether to plot the tree')
    parser.add_argument('--workers',
                        type=int,
                        default=0,
                        help='Num of workers for dataloader.')
    parser.add_argument('--port',
                        type=int,
                        default=19923,
                        help='Python console use only')
    parser.add_argument('--save_path',
                        type=str,
                        default='./checkpoints/',
                        help='Checkpoints saving path')
    parser.add_argument('--load_path',
                        type=str,
                        default='',
                        help='Loading model path')
    parser.add_argument('--save_model',
                        type=bool,
                        default=False,
                        help='Whether to save model or not')
    parser.add_argument('--save_data',
                        type=bool,
                        default=False,
                        help='Whether to save data or not')

    args = parser.parse_args()
    return args
