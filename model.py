import dgl
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


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
                 num_users=2000, num_POIs=5000, num_cats=300, num_coos=50,
                 user_embed_dim=128, fuse_embed_dim=128,
                 nary=3, device='cuda'):
        super(TreeLSTM, self).__init__()
        self.device = device
        self.h_size = h_size
        self.nary = nary
        # embedding
        self.embedding_dim = user_embed_dim + fuse_embed_dim
        self.fuse_len = num_POIs + num_cats + num_coos
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
        # transition matrix
        self.cat2POI = nn.Linear(num_cats, num_POIs)
        self.coo2POI = nn.Linear(num_coos, num_POIs)
        self.cat2POI_o = nn.Linear(num_cats, num_POIs)
        self.coo2POI_o = nn.Linear(num_coos, num_POIs)
        # MultiTask
        self.AutomaticWeighted = MultiTaskLoss(3)
        self.AutomaticWeighted_o = MultiTaskLoss(3)

    def forward(self, in_trees, out_trees):
        user_embedding = self.user_embedding(in_trees.user.long() * in_trees.mask)  # 1694 128
        fuse_embedding = self.fuse_embedding(in_trees.features.long() * in_trees.mask)  # 1694 128
        pe = self.time_pos_encoder(in_trees.time.long() * in_trees.mask)  # 256
        concat_embedding = torch.cat((user_embedding, fuse_embedding), dim=1)  # 256
        concat_embedding = concat_embedding + pe * 0.5

        user_embedding_o = self.user_embedding_o(out_trees.user.long() * out_trees.mask)
        fuse_embedding_o = self.fuse_embedding_o(out_trees.features.long() * out_trees.mask)
        pe_o = self.time_pos_encoder_o(out_trees.time.long() * out_trees.mask)
        concat_embedding_o = torch.cat((user_embedding_o, fuse_embedding_o), dim=1)
        concat_embedding_o = concat_embedding_o + pe_o * 0.5

        g = in_trees.graph.to(self.device)
        n = g.num_nodes()
        g.ndata["iou"] = self.cell.W_iou(self.embed_dropout(concat_embedding)) * in_trees.mask.float().unsqueeze(-1)
        g.ndata["x"] = self.embed_dropout(concat_embedding) * in_trees.mask.float().unsqueeze(-1)
        g.ndata["h"] = nn.init.xavier_uniform_(torch.zeros((n, self.h_size)).to(self.device))
        g.ndata["c"] = nn.init.xavier_uniform_(torch.zeros((n, self.h_size)).to(self.device))
        g.ndata["h_child"] = nn.init.xavier_uniform_(torch.zeros((n, self.nary, self.h_size)).to(self.device))
        g.ndata["c_child"] = nn.init.xavier_uniform_(torch.zeros((n, self.nary, self.h_size)).to(self.device))

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
        g_o.ndata["h"] = nn.init.xavier_uniform_(torch.zeros((n, self.h_size)).to(self.device))
        g_o.ndata["c"] = nn.init.xavier_uniform_(torch.zeros((n, self.h_size)).to(self.device))
        g_o.ndata["h_child"] = nn.init.xavier_uniform_(torch.zeros((n, self.nary, self.h_size)).to(self.device))
        g_o.ndata["c_child"] = nn.init.xavier_uniform_(torch.zeros((n, self.nary, self.h_size)).to(self.device))

        dgl.prop_nodes_topo(graph=g_o,
                            message_func=self.cell_o.message_func,
                            reduce_func=self.cell_o.reduce_func,
                            apply_node_func=self.cell_o.apply_node_func)

        h_o = self.model_dropout(g_o.ndata.pop("h"))  # [batch_size, h_size]

        y_pred_POI_o = self.decoder_POI_o(h_o)
        y_pred_cat_o = self.decoder_cat_o(h_o)
        y_pred_coo_o = self.decoder_coo_o(h_o)

        y_pred_POI = self.AutomaticWeighted(y_pred_POI, self.cat2POI(y_pred_cat), self.coo2POI(y_pred_coo))
        y_pred_POI_o = self.AutomaticWeighted_o(y_pred_POI_o, self.cat2POI_o(y_pred_cat_o),
                                                self.coo2POI_o(y_pred_coo_o))

        return y_pred_POI, y_pred_cat, y_pred_coo, y_pred_POI_o, y_pred_cat_o, y_pred_coo_o


class MultiTaskLoss(nn.Module):
    def __init__(self, num=3):
        super(MultiTaskLoss, self).__init__()
        # params = nn.init.xavier_uniform_(torch.ones(num, requires_grad=True))
        params = torch.ones(num, requires_grad=True)
        self.params = nn.Parameter(params)

    def forward(self, *losses):
        loss_sum = 0
        for i, loss in enumerate(losses):
            # loss_sum += 0.5 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i] ** 2)
            loss_sum += 0.5 * torch.exp(-self.params[i]) * loss + self.params[i]
        return loss_sum
