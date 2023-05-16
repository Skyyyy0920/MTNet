import yaml
import time
import random
import pickle
import zipfile
import logging
import collections

from model import *
from utils import *
from config import *
from dataset import *

SSTBatch = collections.namedtuple(
    "SSTBatch", ["graph", "features", "label"]
)


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


if __name__ == '__main__':
    # ==================================================================================================
    # 1. Get experiment args and seed
    # ==================================================================================================
    print('\n' + '=' * 36 + ' Get experiment args ' + '=' * 36)
    args = get_args()
    setup_seed(args.seed)  # make the experiment repeatable

    # ==================================================================================================
    # 2. Setup logger
    # ==================================================================================================
    print('\n' + '=' * 36 + ' Setup logger ' + '=' * 36)
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging_time = time.strftime('%m-%d_%H-%M', time.localtime())
    save_dir = os.path.join(args.save_path, args.dataset + '_' + str(args.nary) + '-ary_' + logging_time)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    print(f"Saving path: {save_dir}")
    logging.basicConfig(level=logging.INFO,
                        format='[%(asctime)s %(levelname)s]%(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        filename=os.path.join(save_dir, 'running.log'))
    console = logging.StreamHandler()  # Simultaneously output to console
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter(fmt='[%(asctime)s %(levelname)s]%(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
    logging.getLogger('').addHandler(console)
    logging.getLogger('matplotlib.font_manager').disabled = True

    # ==================================================================================================
    # 3. Save codes and settings
    # ==================================================================================================
    print('\n' + '=' * 36 + ' Save codes and settings ' + '=' * 36)
    zipf = zipfile.ZipFile(file=os.path.join(save_dir, 'codes.zip'), mode='a', compression=zipfile.ZIP_DEFLATED)
    zipdir(Path().absolute(), zipf, include_format=['.py'])
    zipf.close()
    with open(os.path.join(save_dir, 'args.yml'), 'a') as f:
        yaml.dump(vars(args), f, sort_keys=False)

    # ==================================================================================================
    # 4. Prepare data
    # ==================================================================================================
    print('\n' + '=' * 36 + ' Prepare data ' + '=' * 36)
    # Load data
    train_df = pd.read_csv(f'dataset/{args.dataset}/{args.dataset}_train.csv')
    val_df = pd.read_csv(f'dataset/{args.dataset}/{args.dataset}_val.csv')
    test_df = pd.read_csv(f'dataset/{args.dataset}/{args.dataset}_test.csv')

    if args.dataset == 'Gowalla-CA':
        train_df = process_for_GowallaCA(train_df)
        val_df = process_for_GowallaCA(val_df)
        test_df = process_for_GowallaCA(test_df)
        train_df['POI_catid'] = train_df.apply(lambda x: eval(x['POI_catname'].replace(";", ","))[0]['url'], axis=1)
        val_df['POI_catid'] = val_df.apply(lambda x: eval(x['POI_catname'].replace(";", ","))[0]['url'], axis=1)
        test_df['POI_catid'] = test_df.apply(lambda x: eval(x['POI_catname'].replace(";", ","))[0]['url'], axis=1)

    # User id to index
    uid_list = [str(uid) for uid in list(set(train_df['user_id'].to_list()))]
    user_id2idx_dict = dict(zip(uid_list, range(len(uid_list))))
    # POI id to index
    POI_list = list(set(train_df['POI_id'].tolist()))
    POI_id2idx_dict = dict(zip(POI_list, range(len(POI_list))))
    # Cat id to index
    cat_ids = list(set(train_df['POI_catid'].tolist()))
    cat_id2idx_dict = dict(zip(cat_ids, range(len(cat_ids))))
    # POI index to cat index
    POI_idx2cat_idx_dict = {}
    for i, row in train_df.iterrows():
        POI_idx2cat_idx_dict[POI_id2idx_dict[row['POI_id']]] = cat_id2idx_dict[row['POI_catid']]
    map_set = (user_id2idx_dict, POI_id2idx_dict, POI_idx2cat_idx_dict)

    n_clusters = args.lon_parts * args.lat_parts
    max_lon, min_lon = train_df.loc[:, "longitude"].max() + 1, train_df.loc[:, "longitude"].min() - 1
    max_lat, min_lat = train_df.loc[:, "latitude"].max() + 1, train_df.loc[:, "latitude"].min() - 1
    column = (max_lon - min_lon) / args.lon_parts
    row = (max_lat - min_lat) / args.lat_parts


    def gen_coo_ID(lon, lat):
        if lon <= min_lon or lon >= max_lon or lat <= min_lat or lat >= max_lat:
            return -1
        return int((lon - min_lon) / column) + 1 + int((lat - min_lat) / row) * args.lon_parts


    train_df['coo_label'] = train_df.apply(lambda x: gen_coo_ID(x['longitude'], x['latitude']), axis=1)
    val_df['coo_label'] = val_df.apply(lambda x: gen_coo_ID(x['longitude'], x['latitude']), axis=1)
    test_df['coo_label'] = test_df.apply(lambda x: gen_coo_ID(x['longitude'], x['latitude']), axis=1)

    # Build dataset
    train_dataset = TrajectoryTrainDataset(train_df, map_set)
    val_dataset = TrajectoryValDataset(val_df, map_set)
    test_dataset = TrajectoryTestDataset(test_df, map_set)
    train_batch_size = int(args.batch_size / args.accumulation_steps)
    train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, drop_last=False,
                                  pin_memory=True, num_workers=args.workers, collate_fn=lambda x: x)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False,
                                pin_memory=True, num_workers=args.workers, collate_fn=lambda x: x)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False,
                                 pin_memory=True, num_workers=args.workers, collate_fn=lambda x: x)

    # ==================================================================================================
    # 5. Build models, define overall loss and optimizer
    # ==================================================================================================
    print('\n' + '=' * 36 + ' Build models ' + '=' * 36)
    num_users = len(user_id2idx_dict)
    num_POIs = len(POI_id2idx_dict)
    num_cats = len(cat_id2idx_dict)
    TreeLSTM_model = TreeLSTM(h_size=args.h_size,
                              embed_dropout=args.embed_dropout, model_dropout=args.model_dropout,
                              num_users=num_users, user_embed_dim=args.user_embed_dim,
                              num_POIs=num_POIs, POI_embed_dim=args.POI_embed_dim,
                              num_cats=num_cats, cat_embed_dim=args.cat_embed_dim,
                              num_coos=n_clusters, coo_embed_dim=args.coo_embed_dim,
                              time_embed_dim=args.time_embed_dim,
                              cell_type=args.cell_type, nary=args.nary,
                              head_num=args.transformer_head_num, hid_dim=args.transformer_hid_dim,
                              layer_num=args.transformer_layer_num, t_dropout=args.transformer_dropout,
                              device=args.device).to(device=args.device)

    criterion_POI = nn.CrossEntropyLoss(ignore_index=-1)  # -1 is ignored
    criterion_cat = nn.CrossEntropyLoss(ignore_index=-1)
    criterion_coo = nn.CrossEntropyLoss(ignore_index=-1)
    optimizer = torch.optim.Adam(params=list(TreeLSTM_model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)

    # ==================================================================================================
    # 6. Load pre-trained model
    # ==================================================================================================
    if args.load_path:
        print('\nLoad pre-trained model...')
        checkpoint = torch.load(os.path.join(args.load_path, f"checkpoint_50.pth"))
        TreeLSTM_model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state'])

    logging.info(f"\n{TreeLSTM_model}")
    logging.info(f"\n{optimizer}")

    # ==================================================================================================
    # 7. Training and validation
    # ==================================================================================================
    print('\n' + '=' * 36 + ' Start training ' + '=' * 36)
    # Training loop
    for epoch in range(args.epochs):
        TreeLSTM_model.train()
        TreeLSTM_model.in_cell.train()
        TreeLSTM_model.out_cell.train()

        y_pred_POI_list, y_label_POI_list, loss_list = [], [], []

        for b_idx, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc="Training"):
            batcher = []
            re_batcher = []
            for trajectory in batch:
                traj_tree = construct_dgl_tree(trajectory, args.cell_type, args.nary, args.need_plot_tree, 'in')
                batcher.append(traj_tree.to(device=args.device))
                re_traj_tree = construct_dgl_tree(trajectory, args.cell_type, args.nary, args.need_plot_tree, 'out')
                re_batcher.append(re_traj_tree.to(device=args.device))

            batch_trees = dgl.batch(batcher).to(device=args.device)
            batch_input = SSTBatch(graph=batch_trees,
                                   features=batch_trees.ndata["x"].to(device=args.device),
                                   label=batch_trees.ndata["y"].to(device=args.device))
            re_batch_trees = dgl.batch(re_batcher).to(device=args.device)
            re_batch_input = SSTBatch(graph=re_batch_trees,
                                      features=re_batch_trees.ndata["x"].to(device=args.device),
                                      label=re_batch_trees.ndata["y"].to(device=args.device))

            g, re_g = batch_input.graph.to(device=args.device), re_batch_input.graph.to(device=args.device)
            n, re_n = g.num_nodes(), re_g.num_nodes()
            h = torch.zeros((n, args.h_size)).to(device=args.device)
            c = torch.zeros((n, args.h_size)).to(device=args.device)
            re_h = torch.zeros((re_n, args.h_size)).to(device=args.device)
            re_c = torch.zeros((re_n, args.h_size)).to(device=args.device)
            h_child = torch.zeros((n, args.nary, args.h_size)).to(device=args.device)
            c_child = torch.zeros((n, args.nary, args.h_size)).to(device=args.device)
            re_h_child = torch.zeros((re_n, args.nary, args.h_size)).to(device=args.device)
            re_c_child = torch.zeros((re_n, args.nary, args.h_size)).to(device=args.device)

            y_pred_POI_in, y_pred_cat_in, y_pred_coo_in, y_pred_POI_out, y_pred_cat_out, y_pred_coo_out = \
                TreeLSTM_model(batch_input, g, h, c, h_child, c_child,
                               re_batch_input, re_g, re_h, re_c, re_h_child, re_c_child)

            y_POI_in, y_cat_in, y_coo_in = \
                batch_input.label[:, 0], batch_input.label[:, 1], batch_input.label[:, 2]
            y_POI_out, y_cat_out, y_coo_out = \
                re_batch_input.label[:, 0], re_batch_input.label[:, 1], re_batch_input.label[:, 2]

            y_pred_POI = y_pred_POI_in + y_pred_POI_out
            y_pred_POI_list.append(y_pred_POI.detach().cpu().numpy())
            y_label_POI_list.append(y_POI_in.detach().cpu().numpy())

            loss_POI = criterion_POI(y_pred_POI_in, y_POI_in.long()) + criterion_POI(y_pred_POI_out, y_POI_out.long())
            loss_cat = criterion_cat(y_pred_cat_in, y_cat_in.long()) + criterion_cat(y_pred_cat_out, y_cat_out.long())
            loss_coo = criterion_coo(y_pred_coo_in, y_coo_in.long()) + criterion_coo(y_pred_coo_out, y_coo_out.long())
            loss = loss_POI + loss_cat + loss_coo
            loss_list.append(loss.item())
            loss.backward()

            # Gradient accumulation to solve the GPU memory problem
            if (b_idx + 1) % args.accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

        lr_scheduler.step()  # update learning rate

        # Measurement
        y_pred_numpy = np.concatenate(y_pred_POI_list, axis=0)
        y_label_numpy = np.concatenate(y_label_POI_list, axis=0)
        acc1 = top_k_acc(y_label_numpy, y_pred_numpy, k=1)
        acc5 = top_k_acc(y_label_numpy, y_pred_numpy, k=5)
        acc10 = top_k_acc(y_label_numpy, y_pred_numpy, k=10)
        acc20 = top_k_acc(y_label_numpy, y_pred_numpy, k=20)
        # mrr = MRR_metric(batch_labels_numpy, y_pred_numpy)
        mrr = 0
        mAP = mAP_metric(y_label_numpy, y_pred_numpy, k=20)

        # Logging
        logging.info(f"************************  Training epoch: {epoch + 1}/{args.epochs}  ************************")
        logging.info(f"Current epoch's mean loss: {np.mean(loss_list)}\t\tlr: {optimizer.param_groups[0]['lr']}")
        logging.info(f"acc@1: {acc1}\tacc@5: {acc5}\tacc@10: {acc10}\tacc@20: {acc20}")
        logging.info(f"mrr: {mrr}\tmAP: {mAP}\n")

        # ==================================================================================================
        # 8. Evaluation
        # ==================================================================================================

        # Save model
        if (epoch + 1) % 20 == 0:
            checkpoint = {
                'model_state': TreeLSTM_model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'lr_scheduler_state': lr_scheduler.state_dict()
            }
            torch.save(checkpoint, os.path.join(save_dir, f"checkpoint_{epoch + 1}.pth"))

        # ==================================================================================================
        # 9. Testing
        # ==================================================================================================
        TreeLSTM_model.eval()
        TreeLSTM_model.in_cell.eval()
        TreeLSTM_model.out_cell.eval()

        with torch.no_grad():
            y_pred_POI_list, y_label_POI_list = [], []
            y_pred_POI_list_in, y_pred_POI_list_out = [], []
            # Start testing
            for batch in test_dataloader:
                batcher = []
                re_batcher = []
                for trajectory in batch:
                    traj_tree = construct_dgl_tree(trajectory, args.cell_type, args.nary, args.need_plot_tree, 'in')
                    batcher.append(traj_tree.to(device=args.device))
                    re_traj_tree = construct_dgl_tree(trajectory, args.cell_type, args.nary, args.need_plot_tree, 'out')
                    re_batcher.append(re_traj_tree.to(device=args.device))

                batch_trees = dgl.batch(batcher).to(device=args.device)
                batch_input = SSTBatch(graph=batch_trees,
                                       features=batch_trees.ndata["x"].to(device=args.device),
                                       label=batch_trees.ndata["y"].to(device=args.device))
                re_batch_trees = dgl.batch(re_batcher).to(device=args.device)
                re_batch_input = SSTBatch(graph=re_batch_trees,
                                          features=re_batch_trees.ndata["x"].to(device=args.device),
                                          label=re_batch_trees.ndata["y"].to(device=args.device))

                g, re_g = batch_input.graph.to(device=args.device), re_batch_input.graph.to(device=args.device)
                n, re_n = g.num_nodes(), re_g.num_nodes()
                h = torch.zeros((n, args.h_size)).to(device=args.device)
                c = torch.zeros((n, args.h_size)).to(device=args.device)
                re_h = torch.zeros((re_n, args.h_size)).to(device=args.device)
                re_c = torch.zeros((re_n, args.h_size)).to(device=args.device)
                h_child = torch.zeros((n, args.nary, args.h_size)).to(device=args.device)
                c_child = torch.zeros((n, args.nary, args.h_size)).to(device=args.device)
                re_h_child = torch.zeros((re_n, args.nary, args.h_size)).to(device=args.device)
                re_c_child = torch.zeros((re_n, args.nary, args.h_size)).to(device=args.device)

                y_pred_POI_in, y_pred_cat_in, y_pred_coo_in, y_pred_POI_out, y_pred_cat_out, y_pred_coo_out = \
                    TreeLSTM_model(batch_input, g, h, c, h_child, c_child,
                                   re_batch_input, re_g, re_h, re_c, re_h_child, re_c_child)
                y_pred_POI = y_pred_POI_in + y_pred_POI_out * args.out_tree_weight
                y_POI_in = batch_input.label[:, 0]

                y_pred_POI_list.append(y_pred_POI.detach().cpu().numpy())
                y_label_POI_list.append(y_POI_in.detach().cpu().numpy())
                y_pred_POI_list_in.append(y_pred_POI_in.detach().cpu().numpy())
                y_pred_POI_list_out.append(y_pred_POI_out.detach().cpu().numpy())

            # Measurement
            y_pred_numpy = np.concatenate(y_pred_POI_list, axis=0)
            y_label_numpy = np.concatenate(y_label_POI_list, axis=0)
            acc1 = top_k_acc(y_label_numpy, y_pred_numpy, k=1)
            acc5 = top_k_acc(y_label_numpy, y_pred_numpy, k=5)
            acc10 = top_k_acc(y_label_numpy, y_pred_numpy, k=10)
            acc20 = top_k_acc(y_label_numpy, y_pred_numpy, k=20)
            # mrr = MRR_metric(batch_labels_numpy, y_pred_numpy)
            mrr = 0
            mAP = mAP_metric(y_label_numpy, y_pred_numpy, k=20)

            if (epoch + 1) % 20 == 0:
                y_pred_in = np.concatenate(y_pred_POI_list_in, axis=0)
                y_pred_out = np.concatenate(y_pred_POI_list_out, axis=0)
                pickle.dump(y_pred_in, open(os.path.join(save_dir, f"recommendation_list_in_tree_{epoch + 1}"), 'wb'))
                pickle.dump(y_pred_out, open(os.path.join(save_dir, f"recommendation_list_out_tree_{epoch + 1}"), 'wb'))
                pickle.dump(y_label_numpy, open(os.path.join(save_dir, f"ground_truth_{epoch + 1}"), 'wb'))

            # Logging
            logging.info(f"================================ Testing ================================")
            logging.info(f"acc@1: {acc1}\tacc@5: {acc5}\tacc@10: {acc10}\tacc@20: {acc20}")
            logging.info(f"mrr: {mrr}\tmAP: {mAP}\n")
