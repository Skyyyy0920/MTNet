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
    "SSTBatch", ["graph", "features", "label", "mask"]
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
    print(f"Using device: {args.device}")
    setup_seed(args.seed)  # make the experiment repeatable

    # ==================================================================================================
    # 2. Setup logger
    # ==================================================================================================
    print('\n' + '=' * 36 + ' Setup logger ' + '=' * 36)
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging_time = time.strftime('%m-%d_%H-%M', time.localtime())
    save_dir = os.path.join(args.save_path, f"{args.dataset}_{args.nary}-ary_{logging_time}")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    print(f"Saving path: {save_dir}")
    logging.basicConfig(level=logging.INFO,
                        format='[%(asctime)s %(levelname)s]%(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        filename=os.path.join(save_dir, f'{args.dataset}_{args.nary}.log'))
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

    # with open('dataset/NYC/NYC_train.csv', 'r') as f:
    #     checkin = f.readlines()
    # checkin = checkin[1:]
    # # poi_data = {}
    # traj_data = []
    # for i in checkin:
    #     data = i.replace('\n', '').split(',')
    #     traj = str(data[12])
    #     traj_data.append(traj.split('_'))
    # uesr = {}
    # for i in range(len(traj_data)):
    #     if not (int(traj_data[i][0]) in uesr):
    #         uesr[int(traj_data[i][0])] = int(traj_data[i][1])
    #     elif uesr[int(traj_data[i][0])] < int(traj_data[i][1]):
    #         uesr[int(traj_data[i][0])] = int(traj_data[i][1])
    # count = 0
    # user_ids = []
    # for k in uesr.keys():
    #     if uesr[k] < 13:
    #         count += 1
    #         user_ids.append(k)
    # print(count)
    # print(len(user_ids))
    # print(user_ids)
    # test_df = test_df[test_df['user_id'].isin(user_ids)]

    # grouped_train_df = train_df.groupby('user_id')
    # user_ids = []
    # for user_id, group in grouped_train_df:
    #     # if 4 <= len(group.groupby('trajectory_id')) <= 20:
    #     # if len(group.groupby('trajectory_id')) > 25:
    #     # if len(group.groupby('trajectory_id')) < 6:
    #     # if len(group.groupby('trajectory_id')) > 41:
    #     # if 13 <= len(group.groupby('trajectory_id')) <= 41:
    #     if len(group.groupby('trajectory_id')) < 13:
    #     # if len(group) > 41:
    #         user_ids.append(user_id)
    # print(len(user_ids))
    # print(user_ids)
    # test_df = test_df[test_df['user_id'].isin(user_ids)]

    if args.dataset == 'Gowalla-CA':
        train_df = process_for_GowallaCA(train_df)
        val_df = process_for_GowallaCA(val_df)
        test_df = process_for_GowallaCA(test_df)

    # User id to index
    uid_list = [str(uid) for uid in list(set(train_df['user_id'].to_list()))]
    user_id2idx_dict = dict(zip(uid_list, range(len(uid_list))))
    # POI id to index
    POI_list = list(set(train_df['POI_id'].tolist()))
    POI_id2idx_dict = dict(zip(POI_list, range(len(POI_list))))
    # Cat id to index
    cat_ids = list(set(train_df['POI_catid'].tolist()))
    cat_id2idx_dict = dict(zip(cat_ids, range(len(cat_ids))))

    num_users = len(user_id2idx_dict)
    num_POIs = len(POI_id2idx_dict)
    num_cats = len(cat_id2idx_dict)
    print(f"users: {num_users}, POIs: {num_POIs}, cats: {num_cats}")

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
    map_set = (user_id2idx_dict, POI_id2idx_dict, cat_id2idx_dict)
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

    TreeLSTM_model = TreeLSTM(h_size=args.h_size,
                              embed_dropout=args.embed_dropout, model_dropout=args.model_dropout,
                              num_users=num_users, user_embed_dim=args.user_embed_dim,
                              num_POIs=num_POIs, POI_embed_dim=args.POI_embed_dim,
                              num_cats=num_cats, cat_embed_dim=args.cat_embed_dim,
                              time_embed_dim=args.time_embed_dim,
                              num_coos=n_clusters, coo_embed_dim=args.coo_embed_dim,
                              nary=args.nary, device=args.device).to(device=args.device)

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
        TreeLSTM_model.cell.train()
        TreeLSTM_model.cell_o.train()

        loss_list = []

        for b_idx, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc="Training"):
            in_tree_batcher, out_tree_batcher = [], []
            for trajectory in batch:
                traj_in_tree = construct_dgl_tree(trajectory, args.nary, args.plot_tree, 'in')
                in_tree_batcher.append(traj_in_tree.to(args.device))
                traj_out_tree = construct_dgl_tree(trajectory, args.nary, args.plot_tree, 'out')
                out_tree_batcher.append(traj_out_tree.to(args.device))

            in_tree_batch = dgl.batch(in_tree_batcher).to(args.device)
            in_trees = SSTBatch(graph=in_tree_batch,
                                features=in_tree_batch.ndata["x"].to(args.device),
                                label=in_tree_batch.ndata["y"].to(args.device),
                                mask=in_tree_batch.ndata["mask"].to(args.device))
            out_tree_batch = dgl.batch(out_tree_batcher).to(args.device)
            out_trees = SSTBatch(graph=out_tree_batch,
                                 features=out_tree_batch.ndata["x"].to(args.device),
                                 label=out_tree_batch.ndata["y"].to(args.device),
                                 mask=out_tree_batch.ndata["mask"].to(args.device))

            y_pred_POI, y_pred_cat, y_pred_coo, y_pred_POI_o, y_pred_cat_o, y_pred_coo_o = \
                TreeLSTM_model(in_trees, out_trees)

            y_POI, y_cat, y_tim, y_coo = \
                in_trees.label[:, 0], in_trees.label[:, 1], in_trees.label[:, 2], in_trees.label[:, 3]
            y_POI_o, y_cat_o, y_tim_o, y_coo_o = \
                out_trees.label[:, 0], out_trees.label[:, 1], out_trees.label[:, 2], out_trees.label[:, 3]

            loss_POI = criterion_POI(y_pred_POI, y_POI.long()) + criterion_POI(y_pred_POI_o, y_POI_o.long())
            loss_cat = criterion_cat(y_pred_cat, y_cat.long()) + criterion_cat(y_pred_cat_o, y_cat_o.long())
            loss_coo = criterion_coo(y_pred_coo, y_coo.long()) + criterion_coo(y_pred_coo_o, y_coo_o.long())
            loss = loss_POI + loss_cat + loss_coo
            loss_list.append(loss.item())
            loss.backward()

            # Gradient accumulation to solve the GPU memory problem
            if (b_idx + 1) % args.accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

        lr_scheduler.step()  # update learning rate

        # Logging
        logging.info(f"************************  Training epoch: {epoch + 1}/{args.epochs}  ************************")
        logging.info(f"Current epoch's mean loss: {np.mean(loss_list)}\t\tlr: {optimizer.param_groups[0]['lr']}")

        # Save model
        if (epoch + 1) % 60 == 0:
            checkpoint = {
                'model_state': TreeLSTM_model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'lr_scheduler_state': lr_scheduler.state_dict()
            }
            torch.save(checkpoint, os.path.join(save_dir, f"checkpoint_{epoch + 1}.pth"))

        # ==================================================================================================
        # 8. Testing
        # ==================================================================================================
        TreeLSTM_model.eval()
        TreeLSTM_model.cell.eval()
        TreeLSTM_model.cell_o.eval()

        with torch.no_grad():
            y_pred_POI_list, y_label_POI_list = [], []
            y_pred_cat_list, y_label_cat_list = [], []
            y_pred_coo_list, y_label_coo_list = [], []
            # Start testing
            for batch in test_dataloader:
                in_tree_batcher, out_tree_batcher = [], []
                for trajectory in batch:
                    traj_in_tree = construct_dgl_tree(trajectory, args.nary, args.plot_tree, 'in')
                    in_tree_batcher.append(traj_in_tree.to(args.device))
                    traj_out_tree = construct_dgl_tree(trajectory, args.nary, args.plot_tree, 'out')
                    out_tree_batcher.append(traj_out_tree.to(args.device))

                in_tree_batch = dgl.batch(in_tree_batcher).to(args.device)
                in_trees = SSTBatch(graph=in_tree_batch,
                                    features=in_tree_batch.ndata["x"].to(args.device),
                                    label=in_tree_batch.ndata["y"].to(args.device),
                                    mask=in_tree_batch.ndata["mask"].to(args.device))
                out_tree_batch = dgl.batch(out_tree_batcher).to(args.device)
                out_trees = SSTBatch(graph=out_tree_batch,
                                     features=out_tree_batch.ndata["x"].to(args.device),
                                     label=out_tree_batch.ndata["y"].to(args.device),
                                     mask=out_tree_batch.ndata["mask"].to(args.device))

                y_pred_POI, y_pred_cat, y_pred_coo, y_pred_POI_o, y_pred_cat_o, y_pred_coo_o = \
                    TreeLSTM_model(in_trees, out_trees)

                y_POI, y_cat, y_tim, y_coo = \
                    in_trees.label[:, 0], in_trees.label[:, 1], in_trees.label[:, 2], in_trees.label[:, 3]

                y_pred_POI_all = y_pred_POI + y_pred_POI_o
                y_pred_cat_all = y_pred_cat + y_pred_cat_o
                y_pred_coo_all = y_pred_coo + y_pred_coo_o
                y_pred_POI_list.append(y_pred_POI_all.detach().cpu().numpy())
                y_label_POI_list.append(y_POI.detach().cpu().numpy())
                y_pred_cat_list.append(y_pred_cat_all.detach().cpu().numpy())
                y_label_cat_list.append(y_cat.detach().cpu().numpy())
                y_pred_coo_list.append(y_pred_coo_all.detach().cpu().numpy())
                y_label_coo_list.append(y_coo.detach().cpu().numpy())

            y_label_POI_numpy, y_pred_POI_numpy = get_pred_label(y_label_POI_list, y_pred_POI_list)
            y_label_cat_numpy, y_pred_cat_numpy = get_pred_label(y_label_cat_list, y_pred_cat_list)
            y_label_coo_numpy, y_pred_coo_numpy = get_pred_label(y_label_coo_list, y_pred_coo_list)

            if (epoch + 1) % 60 == 0:
                pickle.dump(y_pred_POI_numpy, open(os.path.join(save_dir, f"recommendation_list_{epoch + 1}"), 'wb'))
                pickle.dump(y_label_POI_numpy, open(os.path.join(save_dir, f"ground_truth_{epoch + 1}"), 'wb'))

            # Logging
            logging.info(f"================================ Testing ================================")
            acc1, acc5, acc10, acc20, mrr = get_performance(y_label_POI_numpy, y_pred_POI_numpy)
            logging.info(f" <POI> acc@1: {acc1}\tacc@5: {acc5}\tacc@10: {acc10}\tacc@20: {acc20}\tmrr: {mrr}")
            acc1, acc5, acc10, acc20, mrr = get_performance(y_label_cat_numpy, y_pred_cat_numpy)
            logging.info(f" <cat> acc@1: {acc1}\tacc@5: {acc5}\tacc@10: {acc10}\tacc@20: {acc20}\tmrr: {mrr}")
            acc1, acc5, acc10, acc20, mrr = get_performance(y_label_coo_numpy, y_pred_coo_numpy)
            logging.info(f" <coo> acc@1: {acc1}\tacc@5: {acc5}\tacc@10: {acc10}\tacc@20: {acc20}\tmrr: {mrr}")
