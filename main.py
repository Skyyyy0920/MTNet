import yaml
import time
import pickle
import zipfile
import logging
from sklearn.cluster import KMeans
from model import *
from utils import *
from config import *
from dataset import *

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
    # val_df = pd.read_csv(f'dataset/{args.dataset}/{args.dataset}_val.csv')
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
        # val_df = process_for_GowallaCA(val_df)
        test_df = process_for_GowallaCA(test_df)

    # User id to index
    uid_list = [str(uid) for uid in list(set(train_df['user_id'].to_list()))]
    user_id2idx_dict = dict(zip(uid_list, range(len(uid_list))))
    # POI id to index
    POI_list = list(set(train_df['POI_id'].tolist()))
    POI_list.sort()
    POI_id2idx_dict = dict(zip(POI_list, range(len(POI_list))))
    fuse_len = len(POI_id2idx_dict)
    # Cat id to index
    cat_list = list(set(train_df['POI_catid'].tolist()))
    cat_list.sort()
    cat_id2idx_dict = dict(zip(cat_list, range(fuse_len, fuse_len + len(cat_list))))
    fuse_len = fuse_len + len(cat_id2idx_dict)

    data_train = np.column_stack((train_df['longitude'], train_df['latitude']))
    kmeans_train = KMeans(n_clusters=args.K_cluster)
    kmeans_train.fit(data_train)
    train_df['coo_label'] = kmeans_train.labels_ + fuse_len
    data_test = np.column_stack((test_df['longitude'], test_df['latitude']))
    test_df['coo_label'] = kmeans_train.predict(data_test) + fuse_len

    num_users = len(user_id2idx_dict)
    num_POIs = len(POI_id2idx_dict)
    num_cats = len(cat_id2idx_dict)
    print(f"users: {num_users}, POIs: {num_POIs}, cats: {num_cats}, coos: {args.K_cluster}")

    # Build dataset
    map_set = (user_id2idx_dict, POI_id2idx_dict, cat_id2idx_dict)
    train_dataset = TrajectoryTrainDataset(train_df, map_set)
    # val_dataset = TrajectoryValDataset(val_df, map_set)
    test_dataset = TrajectoryTestDataset(test_df, map_set)
    train_batch_size = int(args.batch_size / args.accumulation_steps)
    train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, drop_last=False,
                                  pin_memory=True, num_workers=args.workers, collate_fn=lambda x: x)
    # val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False,
    #                             pin_memory=True, num_workers=args.workers, collate_fn=lambda x: x)
    test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=False, drop_last=False,
                                 pin_memory=True, num_workers=args.workers, collate_fn=lambda x: x)

    # ==================================================================================================
    # 5. Build models, define overall loss and optimizer
    # ==================================================================================================
    print('\n' + '=' * 36 + ' Build models ' + '=' * 36)

    TreeLSTM_model = TreeLSTM(h_size=args.h_size,
                              embed_dropout=args.embed_dropout, model_dropout=args.model_dropout,
                              num_users=num_users, num_POIs=num_POIs, num_cats=num_cats, num_coos=args.K_cluster,
                              user_embed_dim=args.user_embed_dim, fuse_embed_dim=args.fuse_embed_dim,
                              nary=args.nary + 3, device=args.device).to(device=args.device)
    multi_task_loss = MultiTaskLoss(3).to(device=args.device)

    criterion_POI = nn.CrossEntropyLoss(ignore_index=-1)  # -1 is ignored
    criterion_cat = nn.CrossEntropyLoss(ignore_index=-1)
    criterion_coo = nn.CrossEntropyLoss(ignore_index=-1)
    optimizer = torch.optim.Adam(params=list(TreeLSTM_model.parameters())
                                        + list(multi_task_loss.parameters()),
                                 lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)

    # ==================================================================================================
    # 6. Load pre-trained model
    # ==================================================================================================
    if args.load_path:
        print('\nLoad pre-trained model...')
        checkpoint = torch.load(os.path.join(args.load_path, f"checkpoint_50.pth"))
        TreeLSTM_model.load_state_dict(checkpoint['model_state'])
        multi_task_loss.load_state_dict(checkpoint['multi_task_loss_state'])
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
        multi_task_loss.train()

        loss_list = []

        for b_idx, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc="Training"):
            in_tree_batcher, out_tree_batcher = [], []
            for trajectory in batch:
                traj_in_tree = construct_MobilityTree(trajectory, args.nary, args.plot_tree, 'in',
                                                      fuse_len + args.K_cluster)
                in_tree_batcher.append(traj_in_tree.to(args.device))
                traj_out_tree = construct_MobilityTree(trajectory, args.nary, args.plot_tree, 'out',
                                                       fuse_len + args.K_cluster)
                out_tree_batcher.append(traj_out_tree.to(args.device))

            in_tree_batch = dgl.batch(in_tree_batcher).to(args.device)
            in_trees = SSTBatch(graph=in_tree_batch,
                                user=in_tree_batch.ndata["u"].to(args.device),
                                features=in_tree_batch.ndata["x"].to(args.device),
                                time=in_tree_batch.ndata["time"].to(args.device),
                                label=in_tree_batch.ndata["y"].to(args.device),
                                mask=in_tree_batch.ndata["mask"].to(args.device),
                                type=in_tree_batch.ndata["type"].to(args.device))
            out_tree_batch = dgl.batch(out_tree_batcher).to(args.device)
            out_trees = SSTBatch(graph=out_tree_batch,
                                 user=out_tree_batch.ndata["u"].to(args.device),
                                 features=out_tree_batch.ndata["x"].to(args.device),
                                 time=out_tree_batch.ndata["time"].to(args.device),
                                 label=out_tree_batch.ndata["y"].to(args.device),
                                 mask=out_tree_batch.ndata["mask"].to(args.device),
                                 type=out_tree_batch.ndata["type"].to(args.device))

            y_pred_POI, y_pred_cat, y_pred_coo, y_pred_POI_o, y_pred_cat_o, y_pred_coo_o, h, h_ = \
                TreeLSTM_model(in_trees, out_trees)

            indices = torch.any(h_ != 0, dim=1)
            h, h_ = h[indices], h_[indices]

            y_POI, y_cat, y_coo = in_trees.label[:, 0], in_trees.label[:, 1], in_trees.label[:, 2]
            y_POI_o, y_cat_o, y_coo_o = out_trees.label[:, 0], out_trees.label[:, 1], out_trees.label[:, 2]

            loss_POI = criterion_POI(y_pred_POI, y_POI.long()) + criterion_POI(y_pred_POI_o, y_POI_o.long())
            loss_cat = criterion_cat(y_pred_cat, y_cat.long()) + criterion_cat(y_pred_cat_o, y_cat_o.long())
            loss_coo = criterion_coo(y_pred_coo, y_coo.long()) + criterion_coo(y_pred_coo_o, y_coo_o.long())
            loss = multi_task_loss(loss_POI, loss_cat, loss_coo) + SSL(h, h_) * 0.2
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
        if (epoch + 1) % 5 == 0 and epoch >= 80:
            checkpoint = {
                'model_state': TreeLSTM_model.state_dict(),
                'multi_task_loss_state': multi_task_loss.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'lr_scheduler_state': lr_scheduler.state_dict()
            }
            torch.save(checkpoint, os.path.join(save_dir, f"checkpoint_{epoch + 1}.pth"))

        # if (epoch + 1) % 10 == 0:
        # ==================================================================================================
        # 8. Testing
        # ==================================================================================================
        TreeLSTM_model.eval()
        TreeLSTM_model.cell.eval()
        TreeLSTM_model.cell_o.eval()
        multi_task_loss.eval()

        with torch.no_grad():
            y_pred_POI_list, y_label_POI_list = [], []
            y_pred_cat_list, y_label_cat_list = [], []
            y_pred_coo_list, y_label_coo_list = [], []
            y_pred_POI_list_0, y_label_POI_list_0 = [], []
            y_pred_POI_list_1, y_label_POI_list_1 = [], []
            y_pred_POI_list_2, y_label_POI_list_2 = [], []
            y_pred_POI_list_3, y_label_POI_list_3 = [], []
            # Start testing
            for batch in test_dataloader:
                in_tree_batcher, out_tree_batcher = [], []
                for trajectory in batch:
                    traj_in_tree = construct_MobilityTree(trajectory, args.nary, args.plot_tree, 'in',
                                                          fuse_len + args.K_cluster)
                    in_tree_batcher.append(traj_in_tree.to(args.device))
                    traj_out_tree = construct_MobilityTree(trajectory, args.nary, args.plot_tree, 'out',
                                                           fuse_len + args.K_cluster)
                    out_tree_batcher.append(traj_out_tree.to(args.device))

                in_tree_batch = dgl.batch(in_tree_batcher).to(args.device)
                in_trees = SSTBatch(graph=in_tree_batch,
                                    user=in_tree_batch.ndata["u"].to(args.device),
                                    features=in_tree_batch.ndata["x"].to(args.device),
                                    time=in_tree_batch.ndata["time"].to(args.device),
                                    label=in_tree_batch.ndata["y"].to(args.device),
                                    mask=in_tree_batch.ndata["mask"].to(args.device),
                                    type=in_tree_batch.ndata["type"].to(args.device))
                out_tree_batch = dgl.batch(out_tree_batcher).to(args.device)
                out_trees = SSTBatch(graph=out_tree_batch,
                                     user=out_tree_batch.ndata["u"].to(args.device),
                                     features=out_tree_batch.ndata["x"].to(args.device),
                                     time=out_tree_batch.ndata["time"].to(args.device),
                                     label=out_tree_batch.ndata["y"].to(args.device),
                                     mask=out_tree_batch.ndata["mask"].to(args.device),
                                     type=out_tree_batch.ndata["type"].to(args.device))

                y_pred_POI, y_pred_cat, y_pred_coo, y_pred_POI_o, y_pred_cat_o, y_pred_coo_o, h, h_ = \
                    TreeLSTM_model(in_trees, out_trees)

                y_POI, y_cat, y_coo = in_trees.label[:, 0], in_trees.label[:, 1], in_trees.label[:, 2]

                # alpha = 0.5
                # y_pred_POI_all = alpha * y_pred_POI + (1 - alpha) * y_pred_POI_o
                y_pred_POI_all = y_pred_POI + y_pred_POI_o
                y_pred_cat_all = y_pred_cat + y_pred_cat_o
                y_pred_coo_all = y_pred_coo + y_pred_coo_o

                indices_0 = torch.where(in_trees.type == 0)[0]
                indices_1 = torch.where(in_trees.type == 1)[0]
                indices_2 = torch.where(in_trees.type == 2)[0]
                indices_3 = torch.where(in_trees.type == 3)[0]
                # indices_5 = torch.where(in_trees.type == 5)[0]
                y_pred_POI_0, y_POI_0 = y_pred_POI_all[indices_0], y_POI[indices_0]
                y_pred_POI_1, y_POI_1 = y_pred_POI_all[indices_1], y_POI[indices_1]
                y_pred_POI_2, y_POI_2 = y_pred_POI_all[indices_2], y_POI[indices_2]
                y_pred_POI_3, y_POI_3 = y_pred_POI_all[indices_3], y_POI[indices_3]
                # y_pred_POI_5, y_POI_5 = y_pred_POI_all[indices_5], y_POI[indices_5]

                y_pred_POI_list_0.append(y_pred_POI_0.detach().cpu().numpy())
                y_label_POI_list_0.append(y_POI_0.detach().cpu().numpy())
                y_pred_POI_list_1.append(y_pred_POI_1.detach().cpu().numpy())
                y_label_POI_list_1.append(y_POI_1.detach().cpu().numpy())
                y_pred_POI_list_2.append(y_pred_POI_2.detach().cpu().numpy())
                y_label_POI_list_2.append(y_POI_2.detach().cpu().numpy())
                y_pred_POI_list_3.append(y_pred_POI_3.detach().cpu().numpy())
                y_label_POI_list_3.append(y_POI_3.detach().cpu().numpy())

                y_pred_POI_list.append(y_pred_POI_all.detach().cpu().numpy())
                y_label_POI_list.append(y_POI.detach().cpu().numpy())
                y_pred_cat_list.append(y_pred_cat_all.detach().cpu().numpy())
                y_label_cat_list.append(y_cat.detach().cpu().numpy())
                y_pred_coo_list.append(y_pred_coo_all.detach().cpu().numpy())
                y_label_coo_list.append(y_coo.detach().cpu().numpy())

            y_label_POI_numpy_0, y_pred_POI_numpy_0 = get_pred_label(y_label_POI_list_0, y_pred_POI_list_0)
            y_label_POI_numpy_1, y_pred_POI_numpy_1 = get_pred_label(y_label_POI_list_1, y_pred_POI_list_1)
            y_label_POI_numpy_2, y_pred_POI_numpy_2 = get_pred_label(y_label_POI_list_2, y_pred_POI_list_2)
            y_label_POI_numpy_3, y_pred_POI_numpy_3 = get_pred_label(y_label_POI_list_3, y_pred_POI_list_3)

            y_label_POI_numpy, y_pred_POI_numpy = get_pred_label(y_label_POI_list_0, y_pred_POI_list_0)
            y_label_cat_numpy, y_pred_cat_numpy = get_pred_label(y_label_cat_list, y_pred_cat_list)
            y_label_coo_numpy, y_pred_coo_numpy = get_pred_label(y_label_coo_list, y_pred_coo_list)

            if epoch >= 80:
                pickle.dump(y_pred_POI_numpy_0, open(os.path.join(save_dir, f"recommend_list_POI_{epoch + 1}"), 'wb'))
                pickle.dump(y_pred_POI_numpy_1, open(os.path.join(save_dir, f"recommend_list_cat_{epoch + 1}"), 'wb'))
                pickle.dump(y_pred_POI_numpy_2, open(os.path.join(save_dir, f"recommend_list_coo_{epoch + 1}"), 'wb'))
                pickle.dump(y_label_POI_numpy_0, open(os.path.join(save_dir, f"ground_truth_POI_{epoch + 1}"), 'wb'))
                pickle.dump(y_label_POI_numpy_1, open(os.path.join(save_dir, f"ground_truth_cat_{epoch + 1}"), 'wb'))
                pickle.dump(y_label_POI_numpy_2, open(os.path.join(save_dir, f"ground_truth_coo_{epoch + 1}"), 'wb'))
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
