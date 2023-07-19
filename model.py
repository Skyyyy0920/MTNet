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
        return {"h": h, "c": c}

    def message_func(self, edges):
        return {"h_child": edges.src["h"], "c_child": edges.src["c"]}

    def reduce_func(self, nodes):
        Wx = torch.cat([self.W_f(nodes.data["x"]) for _ in range(self.nary)], dim=1)
        b_f = torch.cat([self.b_f for _ in range(self.nary)], dim=1)
        h_children = nodes.mailbox["h_child"]  # [batch, nary, h_size]
        h_children = self.transformer_encoder(h_children)
        h_children = h_children.view(h_children.size(0), -1)  # [batch, nary * h_size]
        f = torch.sigmoid(Wx + h_children + b_f)
        iou = self.W_iou(nodes.data["x"]) + self.U_iou(h_children) + self.b_iou  # [batch, 3 * h_size]
        c = torch.sum(f.view(nodes.mailbox["c_child"].size()) * nodes.mailbox["c_child"], 1)
        return {"c": c.view(c.size(0), -1), "iou": iou}


class TreeLSTM(nn.Module):
    def __init__(self,
                 h_size=512,
                 embed_dropout=0.2, model_dropout=0.4,
                 num_users=3000, user_embed_dim=128,
                 num_POIs=5000, fuse_embed_dim=128,
                 num_cats=300, cat_embed_dim=32,
                 num_coos=1024, coo_embed_dim=64,
                 nary=3, device='cuda'):
        super(TreeLSTM, self).__init__()
        self.device = device
        self.h_size = h_size
        self.nary = nary
        # embedding
        self.embedding_dim = user_embed_dim + fuse_embed_dim
        self.fuse_len = num_POIs + num_cats + num_coos + 24
        self.user_embedding = nn.Embedding(num_embeddings=num_users, embedding_dim=user_embed_dim)
        self.fuse_embedding = nn.Embedding(num_embeddings=self.fuse_len, embedding_dim=fuse_embed_dim)
        self.user_embedding_o = nn.Embedding(num_embeddings=num_users, embedding_dim=user_embed_dim)
        self.fuse_embedding_o = nn.Embedding(num_embeddings=self.fuse_len, embedding_dim=fuse_embed_dim)
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

        dgl.prop_nodes_topo(graph=g,
                            message_func=self.cell.message_func,
                            reduce_func=self.cell.reduce_func,
                            apply_node_func=self.cell.apply_node_func)

        h = self.model_dropout(g.ndata.pop("h"))  # [batch_size, h_size]

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

        dgl.prop_nodes_topo(graph=g_o,
                            message_func=self.cell_o.message_func,
                            reduce_func=self.cell_o.reduce_func,
                            apply_node_func=self.cell_o.apply_node_func)

        h_o = self.model_dropout(g_o.ndata.pop("h"))  # [batch_size, h_size]

        y_pred_POI_o = self.decoder_POI_o(h_o)
        y_pred_cat_o = self.decoder_cat_o(h_o)
        y_pred_coo_o = self.decoder_coo_o(h_o)

        return y_pred_POI, y_pred_cat, y_pred_coo, y_pred_POI_o, y_pred_cat_o, y_pred_coo_o, h, h_o

    def get_embedding(self, y_POI, y_cat, y_coo, user):
        y_POI_emb = torch.cat((self.fuse_embedding(y_POI), self.user_embedding(user)), dim=1)
        y_cat_emb = torch.cat((self.fuse_embedding(y_cat), self.user_embedding(user)), dim=1)
        y_coo_emb = torch.cat((self.fuse_embedding(y_coo), self.user_embedding(user)), dim=1)
        return y_POI_emb, y_cat_emb, y_coo_emb

    def get_embedding_test(self, y_POI, y_cat, y_coo, user):
        return self.fuse_embedding(y_POI), self.fuse_embedding(y_cat), self.fuse_embedding(y_coo), self.user_embedding(
            user)


class KnowledgeGraph(nn.Module):
    def __init__(self, h_size, dim, num_POIs):
        super(KnowledgeGraph, self).__init__()
        self.head_embedding = nn.Linear(h_size, dim)
        self.W_h = nn.Linear(dim, dim)
        self.W_t = nn.Linear(dim, dim)
        self.W_r = nn.Linear(dim * 2, dim)
        self.num_POIs = num_POIs

    def forward(self, head, tail, relation):
        head = F.normalize(self.head_embedding(head), 2, -1)
        tail = F.normalize(tail, 2, -1)
        cat, coo = relation
        cat = F.normalize(cat, 2, -1)
        coo = F.normalize(coo, 2, -1)

        h_v = torch.tanh(self.W_h(cat))
        h_t = torch.tanh(self.W_t(coo))
        z = torch.sigmoid(self.W_r(torch.cat((h_v, h_t), dim=-1)))
        relation = z * h_v + (1 - z) * h_t

        score = head + relation - tail
        score = torch.norm(score, 1, -1).flatten()
        return score

    def predict(self, head, tail, relation, user):
        """
        tail and relation are fixed, their shapes are [num_POI, embedding_dim]
        """
        recommendation_list = []
        for i in range(len(head)):
            he = head[i].expand(tail.size(0), -1)
            u = user[i].expand(tail.size(0), -1)
            ta = torch.cat((tail, u), dim=1)
            cat, coo = relation
            cat = torch.cat((cat, u), dim=1)
            coo = torch.cat((coo, u), dim=1)
            rel = (cat, coo)
            score_matrix = self.forward(he, ta, rel)  # [1, num_POI]
            recommendation_list.append(score_matrix.unsqueeze(0))
        recommendation_list = torch.cat(recommendation_list, dim=0)
        return recommendation_list


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

    def forward(self, p_score, n_score):
        if self.adv_flag:
            return (self.get_weights(n_score) * torch.max(p_score - n_score, -self.margin)).sum(
                dim=-1).mean() + self.margin
        else:
            return (torch.max(p_score - n_score, -self.margin)).mean() + self.margin
