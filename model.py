import dgl
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


# class PositionalEncoding(nn.Module):
#     def __init__(self, d_model, max_len=500):
#         super(PositionalEncoding, self).__init__()
#
#         pe = torch.zeros(max_len, d_model)
#         position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         pe = pe.unsqueeze(0).transpose(0, 1)
#         self.register_buffer('pe', pe)
#
#     def forward(self, x):
#         x = x + self.pe[:x.size(0), :]
#         return x


class PositionalEncoding(nn.Embedding):
    def __init__(self, d_model, max_len=5000):
        super().__init__(max_len, d_model)

    def forward(self, x):
        weight = self.weight.data.unsqueeze(1)
        x = x + weight[:x.size(0), :]
        return x


class Cell(nn.Module):
    def __init__(self, embedding_dim, h_size, nary):
        super(Cell, self).__init__()
        self.nary = nary
        self.W_f = nn.Linear(embedding_dim, h_size, bias=False)  # W_f -> [embedding_dim, h_size]
        self.U_f = nn.Linear(nary * h_size, nary * h_size, bias=False)
        self.b_f = nn.Parameter(torch.zeros(1, h_size))
        self.W_iou = nn.Linear(embedding_dim, 3 * h_size, bias=False)  # [W_i, W_u, W_o] -> [embedding_dim, 3 * h_size]
        self.U_iou = nn.Linear(nary * h_size, 3 * h_size, bias=False)
        self.b_iou = nn.Parameter(torch.zeros(1, 3 * h_size))
        # Transformer encoder
        encoder_layers = nn.TransformerEncoderLayer(h_size, 2, h_size, 0.4)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, 2)

    def apply_node_func(self, nodes):
        iou = nodes.data["iou"]
        i, o, u = torch.chunk(iou, 3, 1)
        i, o, u = torch.sigmoid(i), torch.sigmoid(o), torch.tanh(u)
        c = i * u + nodes.data["c"]  # [batch, h_size]
        h = o * torch.tanh(c)
        return {"h": h, "c": c, "rel": nodes.data["rel"], "h_pre": nodes.data["h_pre"]}

    def message_func(self, edges):
        return {"h_child": edges.src["h"], "c_child": edges.src["c"],
                "type": edges.src["type"], "x_ori": edges.src["x_ori"]}

    def reduce_func(self, nodes):
        Wx = torch.cat([self.W_f(nodes.data["x"]) for _ in range(self.nary)], dim=1)
        b_f = torch.cat([self.b_f for _ in range(self.nary)], dim=1)
        h_children = nodes.mailbox["h_child"]  # [batch, nary, h_size]
        h_children = self.transformer_encoder(h_children)
        h_children = h_children.view(h_children.size(0), -1)  # [batch, nary * h_size]
        f = torch.sigmoid(Wx + h_children + b_f)
        iou = self.W_iou(nodes.data["x"]) + self.U_iou(h_children) + self.b_iou  # [batch, 3 * h_size]
        c = torch.sum(f.view(nodes.mailbox["c_child"].size()) * nodes.mailbox["c_child"], 1)

        relation = nodes.mailbox["x_ori"][:, 1:]
        h_pre = nodes.mailbox["h_child"][:, 0, :].squeeze(1)
        # 这里取第0列是因为POI在子节点中处于第0列的位置, 详情可以看nodes.mailbox["type"]

        return {"c": c.view(c.size(0), -1), "iou": iou, "rel": relation, "h_pre": h_pre}


class TreeLSTM(nn.Module):
    def __init__(self,
                 h_size=512,
                 embed_dropout=0.2, model_dropout=0.4,
                 num_users=3000, user_embed_dim=128,
                 num_POIs=5000, POI_embed_dim=128,
                 num_cats=300, cat_embed_dim=32,
                 time_embed_dim=32,
                 num_coos=1024, coo_embed_dim=64,
                 nary=3, device='cuda'):
        super(TreeLSTM, self).__init__()
        self.device = device
        self.h_size = h_size
        self.nary = nary
        # embedding
        self.embedding_dim = user_embed_dim + POI_embed_dim
        self.fuse_len = num_POIs + num_cats + num_coos + 24
        self.user_embedding = nn.Embedding(num_embeddings=num_users, embedding_dim=user_embed_dim)
        self.fuse_embedding = nn.Embedding(num_embeddings=self.fuse_len, embedding_dim=POI_embed_dim)
        self.user_embedding_o = nn.Embedding(num_embeddings=num_users, embedding_dim=user_embed_dim)
        self.fuse_embedding_o = nn.Embedding(num_embeddings=self.fuse_len, embedding_dim=POI_embed_dim)
        # positional encoding
        self.time_pos_encoder = nn.Embedding(num_embeddings=600, embedding_dim=self.embedding_dim)
        self.time_pos_encoder_o = nn.Embedding(num_embeddings=600, embedding_dim=self.embedding_dim)
        # dropout
        self.embed_dropout = nn.Dropout(embed_dropout)
        self.model_dropout = nn.Dropout(model_dropout)
        # cell
        self.cell = Cell(self.embedding_dim, h_size, nary)
        self.cell_o = Cell(self.embedding_dim, h_size, nary)
        # decoder
        self.decoder_POI = nn.Linear(h_size, num_POIs)
        self.decoder_cat = nn.Linear(h_size, num_cats)
        self.decoder_coo = nn.Linear(h_size, num_coos)
        self.decoder_POI_o = nn.Linear(h_size, num_POIs)
        self.decoder_cat_o = nn.Linear(h_size, num_cats)
        self.decoder_coo_o = nn.Linear(h_size, num_coos)

    def forward(self, in_trees, out_trees):
        user_embedding = self.user_embedding(in_trees.user.long() * in_trees.mask)  # 1694 128
        fuse_embedding = self.fuse_embedding(in_trees.features.long() * in_trees.mask)  # 1694 128
        pe = self.time_pos_encoder(in_trees.time.long() * in_trees.mask)  # 256
        concat_embedding = torch.cat((user_embedding, fuse_embedding), dim=1)  # 256
        concat_embedding = concat_embedding + pe * 0.5

        user_embedding_o = self.user_embedding_o(out_trees.user.long() * out_trees.mask)
        fuse_embedding_o = self.fuse_embedding_o(out_trees.features.long() * out_trees.mask)
        pe_o = self.time_pos_encoder_o(out_trees.time.long() * in_trees.mask)
        concat_embedding_o = torch.cat((user_embedding_o, fuse_embedding_o), dim=1)
        concat_embedding_o = concat_embedding_o + pe_o * 0.5

        g = in_trees.graph.to(self.device)
        n = g.num_nodes()
        g.ndata["iou"] = self.cell.W_iou(self.embed_dropout(concat_embedding)) * in_trees.mask.float().unsqueeze(-1)
        g.ndata["x"] = self.embed_dropout(concat_embedding) * in_trees.mask.float().unsqueeze(-1)
        g.ndata["h"] = torch.zeros((n, self.h_size)).to(self.device)
        g.ndata["c"] = torch.zeros((n, self.h_size)).to(self.device)
        g.ndata["h_child"] = torch.zeros((n, self.nary, self.h_size)).to(self.device)
        g.ndata["c_child"] = torch.zeros((n, self.nary, self.h_size)).to(self.device)
        g.ndata["x_ori"] = in_trees.features.long() * in_trees.mask
        g.ndata["rel"] = torch.zeros((n, 2), dtype=torch.int64).to(self.device)
        g.ndata["h_pre"] = torch.zeros((n, self.h_size)).to(self.device)

        dgl.prop_nodes_topo(graph=g,
                            message_func=self.cell.message_func,
                            reduce_func=self.cell.reduce_func,
                            apply_node_func=self.cell.apply_node_func)

        h = self.model_dropout(g.ndata.pop("h"))  # [batch_size, h_size]
        h_pre = g.ndata.pop("h_pre")
        rel = g.ndata.pop("rel")

        y_pred_POI = self.decoder_POI(h)
        y_pred_cat = self.decoder_cat(h)
        y_pred_coo = self.decoder_coo(h)

        g_o = out_trees.graph.to(self.device)
        n = g_o.num_nodes()
        g_o.ndata["iou"] = self.cell_o.W_iou(self.embed_dropout(concat_embedding_o)) \
                           * out_trees.mask.float().unsqueeze(-1)
        g_o.ndata["x"] = self.embed_dropout(concat_embedding_o) * out_trees.mask.float().unsqueeze(-1)
        g_o.ndata["h"] = torch.zeros((n, self.h_size)).to(self.device)
        g_o.ndata["c"] = torch.zeros((n, self.h_size)).to(self.device)
        g_o.ndata["h_child"] = torch.zeros((n, self.nary, self.h_size)).to(self.device)
        g_o.ndata["c_child"] = torch.zeros((n, self.nary, self.h_size)).to(self.device)
        g_o.ndata["x_ori"] = out_trees.features.long() * out_trees.mask
        g_o.ndata["rel"] = torch.zeros((n, 2), dtype=torch.int64).to(self.device)
        g_o.ndata["h_pre"] = torch.zeros((n, self.h_size)).to(self.device)

        dgl.prop_nodes_topo(graph=g_o,
                            message_func=self.cell_o.message_func,
                            reduce_func=self.cell_o.reduce_func,
                            apply_node_func=self.cell_o.apply_node_func)

        h_o = self.model_dropout(g_o.ndata.pop("h"))  # [batch_size, h_size]
        h_pre_o = g_o.ndata.pop("h_pre")
        rel_o = g_o.ndata.pop("rel")

        y_pred_POI_o = self.decoder_POI_o(h_o)
        y_pred_cat_o = self.decoder_cat_o(h_o)
        y_pred_coo_o = self.decoder_coo_o(h_o)

        return y_pred_POI, y_pred_cat, y_pred_coo, y_pred_POI_o, y_pred_cat_o, y_pred_coo_o, \
               h, h_pre, rel, h_o, h_pre_o, rel_o

    def predict(self, in_trees, out_trees):
        y_pred_POI, y_pred_cat, y_pred_coo, y_pred_POI_o, y_pred_cat_o, y_pred_coo_o = self.forward(in_trees, out_trees)
        return y_pred_POI, y_pred_cat, y_pred_coo, y_pred_POI_o, y_pred_cat_o, y_pred_coo_o


class KnowledgeGraph(nn.Module):
    def __init__(self, h_size, rel_num, rel_dim):
        super(KnowledgeGraph, self).__init__()
        # self.head_embedding = nn.Embedding(1, 1)
        # self.tail_embedding = nn.Embedding(1, 1)
        self.head_embedding = nn.Linear(h_size, rel_dim)
        self.tail_embedding = nn.Linear(h_size, rel_dim)
        self.relation_embedding = nn.Embedding(rel_num, rel_dim)
        self.W_h = nn.Linear(rel_dim, rel_dim)
        self.W_t = nn.Linear(rel_dim, rel_dim)
        self.W_r = nn.Linear(rel_dim * 2, rel_dim)

    def forward(self, head, tail, relation):
        head = F.normalize(self.head_embedding(head), 2, -1)
        tail = F.normalize(self.tail_embedding(tail), 2, -1)
        cat, coo = relation[:, 0], relation[:, 1]
        cat = F.normalize(self.relation_embedding(cat), 2, -1)
        coo = F.normalize(self.relation_embedding(coo), 2, -1)

        h_v = torch.tanh(self.W_h(cat))
        h_t = torch.tanh(self.W_t(coo))
        z = torch.sigmoid(self.W_r(torch.cat((h_v, h_t), dim=-1)))
        relation = z * h_v + (1 - z) * h_t

        score = head + relation - tail
        return score


class MarginLoss(nn.Module):
    def __init__(self, adv_temperature=None, margin=6.0):
        super(MarginLoss, self).__init__()
        self.margin = nn.Parameter(torch.Tensor([margin]))
        if adv_temperature is not None:
            self.adv_temperature = nn.Parameter(torch.Tensor([adv_temperature]))
            self.adv_temperature.requires_grad = False
            self.adv_flag = True
        else:
            self.adv_flag = False

    def get_weights(self, n_score):
        return F.softmax(-n_score * self.adv_temperature, dim=-1).detach()

    # def forward(self, p_score, n_score):
    #     if self.adv_flag:
    #         return (self.get_weights(n_score) * torch.max(p_score - n_score, -self.margin)).sum(
    #             dim=-1).mean() + self.margin
    #     else:
    #         return (torch.max(p_score - n_score, -self.margin)).mean() + self.margin
    def forward(self, p_score):
        return (torch.max(p_score, -self.margin)).mean() + self.margin

    def predict(self, p_score, n_score):
        # score = self.forward(p_score, n_score)
        score = self.forward(p_score)
        return score.cpu().data.numpy()


class GeoPrediction(nn.Module):
    def __init__(self):
        super(GeoPrediction, self).__init__()
        self.cia = CheckInActivity(config.train_checkins_filename, config.test_filename, config.tune_filename)
        self.a = Variable(torch.zeros(1).type(torch.FloatTensor), requires_grad=True).to(device)
        self.W1 = Variable(torch.zeros(config.dim_m, config.dim_m).type(torch.FloatTensor), requires_grad=True).to(
            device)
        self.W2 = Variable(torch.zeros(config.dim_m, config.dim_m).type(torch.FloatTensor), requires_grad=True).to(
            device)
        self.P = Variable(torch.zeros(config.history_num, config.dim_m).type(torch.FloatTensor), requires_grad=True).to(
            device)
        self.bias = Variable(torch.zeros(config.history_num, config.dim_m).type(torch.FloatTensor),
                             requires_grad=True).to(device)
        # self.beta = Variable(torch.zeros(1).type(torch.FloatTensor), requires_grad=True).to(device)

        self.a = nn.init.normal_(self.a)
        self.W1 = nn.init.xavier_uniform_(self.W1)
        self.W2 = nn.init.xavier_uniform_(self.W2)
        self.P = nn.init.normal_(self.P)
        self.bias = nn.init.xavier_uniform_(self.bias)

        self.user_embeddings = nn.Embedding(self.cia.u_tot, config.dim_d, max_norm=1)
        self.location_embeddings = nn.Embedding(20 + config.time_slot, config.dim_m, max_norm=1)
        self.venue_embeddings = nn.Embedding(self.cia.v_tot, config.dim_d, max_norm=1)

        nn.init.normal_(self.user_embeddings.weight.data)
        nn.init.normal_(self.location_embeddings.weight.data)
        nn.init.normal_(self.venue_embeddings.weight.data)

        self.location_matrix = nn.Embedding(20 + config.time_slot, config.dim_m * config.dim_d, max_norm=1)
        if not config.rand_init:
            identity = torch.zeros(config.dim_m, config.dim_d)
            for i in range(min(config.dim_d, config.dim_m)):
                identity[i][i] = 1
            identity = identity.view(config.dim_m * config.dim_d)
            for i in range(20 + config.time_slot):
                self.location_matrix.weight.data[i] = identity
        else:
            self.location_matrix = nn.init.xavier_uniform_(self.location_matrix.weight.data)

        if config.margin is not None:
            self.margin = nn.Parameter(torch.Tensor([config.margin]))
            self.margin.requires_grad = False
            self.margin_flag = True
        else:
            self.margin_flag = False

    @staticmethod
    def _calc(u, l, v):
        u = F.normalize(u, 2, -1)
        l = F.normalize(l, 2, -1)
        v = F.normalize(v, 2, -1)

        score = u + l - v
        score = torch.norm(score, config.p_norm, -1).flatten()
        return score

    @staticmethod
    def _transfer(emb, l_transfer):
        l_transfer = l_transfer.view(-1, config.dim_d, config.dim_m)
        if emb.shape[0] != l_transfer.shape[0]:
            emb = emb.view(-1, l_transfer.shape[0], config.dim_d).permute(1, 0, 2)
            emb = emb.matmul(emb, l_transfer).permute(1, 0, 2)
        else:
            emb = emb.view(-1, 1, config.dim_d)
            emb = torch.matmul(emb, l_transfer)
        return emb.view(-1, config.dim_m)

    def forward(self, data):
        batch_u = data['batch_u']
        batch_v = data['batch_v']
        batch_olc = data['batch_olc']
        batch_cluster_olc = data['batch_cluster_olc']
        batch_history = data['batch_history']

        u = self.user_embeddings(batch_u)
        v = self.venue_embeddings(batch_v)

        # window
        history_emb = self.venue_embeddings(batch_history)

        # Attention network
        h = torch.tanh(torch.matmul(history_emb, self.W1).to(device)
                       + torch.matmul(self.P, self.W2).repeat(len(history_emb), 1, 1).to(device)
                       + self.bias.repeat(len(history_emb), 1, 1)).to(device)
        alpha = torch.softmax(h, dim=1).to(device)
        l = alpha * history_emb
        l = torch.sum(l, 1) + self.a

        score = self._calc(u, l, v)
        if self.margin_flag:
            return self.margin - score
        else:
            return score

    def predict(self, data):
        score = self.forward_test(data)
        if self.margin_flag:
            score = self.margin - score
            return score.cpu().data.numpy()
        else:
            return score.cpu().data.numpy()

    def forward_test(self, data):
        batch_u = data['batch_u']
        batch_v = data['batch_v']
        batch_olc = data['batch_olc']
        batch_cluster_olc = data['batch_cluster_olc']
        batch_history = data['batch_history']

        u = self.user_embeddings(batch_u)
        v = self.venue_embeddings(batch_v)

        # window
        history_emb = self.venue_embeddings(batch_history)

        score_sum = Variable(torch.zeros(int(history_emb.size()[0])).type(torch.FloatTensor), requires_grad=True).to(
            device)
        for i in range(int(history_emb[0].size()[0])):
            if int(history_emb[0].size()[0]) - i >= config.history_num:
                history_sub = history_emb[:, i:config.history_num + i, :]
                h = torch.tanh(torch.matmul(history_sub, self.W1).to(device)
                               + torch.matmul(self.P, self.W2).repeat(len(history_sub), 1, 1).to(device)
                               + self.bias.repeat(len(history_sub), 1, 1)).to(device)
                alpha = torch.softmax(h, dim=1).to(device)
                l = alpha * history_sub
                l = torch.sum(l, 1) + self.a
                score = self._calc(u, l, v)
                if self.margin_flag:
                    score = self.margin - score
                score_sum += score

        return score_sum
