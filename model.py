import dgl
import torch
import torch.nn as nn


class ChildSumTreeLSTMCell(nn.Module):
    def __init__(self, embedding_dim, h_size):
        super(ChildSumTreeLSTMCell, self).__init__()
        self.W_f = nn.Linear(embedding_dim, h_size, bias=False)  # W_f -> [embedding_dim, h_size]
        self.U_f = nn.Linear(h_size, h_size, bias=False)
        self.b_f = nn.Parameter(torch.zeros(1, h_size))
        self.W_iou = nn.Linear(embedding_dim, 3 * h_size, bias=False)
        self.U_iou = nn.Linear(h_size, 3 * h_size, bias=False)
        self.b_iou = nn.Parameter(torch.zeros(1, 3 * h_size))

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
        h_sum = torch.sum(nodes.mailbox["h_child"], 1)
        f = torch.sigmoid(self.W_f(nodes.data["x"]) + self.U_f(h_sum) + self.b_f)
        iou = self.W_iou(nodes.data["x"]) + self.U_iou(h_sum) + self.b_iou  # [batch, 3 * h_size]
        c = torch.sum(f * nodes.mailbox["c_child"], 1)
        return {"c": c, "iou": iou}


class NaryTreeLSTMCell(nn.Module):
    def __init__(self, embedding_dim, h_size, nary, head_num, hid_dim, layer_num, dropout):
        super(NaryTreeLSTMCell, self).__init__()
        self.nary = nary
        self.W_f = nn.Linear(embedding_dim, h_size, bias=False)  # W_f -> [embedding_dim, h_size]
        self.U_f = nn.Linear(nary * h_size, nary * h_size, bias=False)
        self.b_f = nn.Parameter(torch.zeros(1, h_size))
        self.W_iou = nn.Linear(embedding_dim, 3 * h_size, bias=False)  # [W_i, W_u, W_o] -> [embedding_dim, 3 * h_size]
        self.U_iou = nn.Linear(nary * h_size, 3 * h_size, bias=False)
        self.b_iou = nn.Parameter(torch.zeros(1, 3 * h_size))
        # Transformer encoder
        encoder_layers = nn.TransformerEncoderLayer(h_size, head_num, hid_dim, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, layer_num)

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
        h_cat = nodes.mailbox["h_child"]  # [batch, nary, h_size]
        h_cat = h_cat.view(h_cat.size(0), -1)  # [batch, nary * h_size]
        f = torch.sigmoid(Wx + self.U_f(h_cat) + b_f)
        h_cat_att = self.transformer_encoder(nodes.mailbox["h_child"])
        h_cat_att = h_cat_att.view(h_cat_att.size(0), -1)
        iou = self.W_iou(nodes.data["x"]) + self.U_iou(h_cat_att) + self.b_iou  # [batch, 3 * h_size]
        c = torch.sum(f.view(nodes.mailbox["c_child"].size()) * nodes.mailbox["c_child"], 1)
        return {"c": c.view(c.size(0), -1), "iou": iou}


class TreeLSTM(nn.Module):
    def __init__(self,
                 h_size=128,
                 embed_dropout=0.3, model_dropout=0.5,
                 num_users=3000, user_embed_dim=128,
                 num_POIs=5000, POI_embed_dim=128,
                 num_cats=300, cat_embed_dim=32,
                 time_embed_dim=32,
                 num_coos=1024, coo_embed_dim=64,
                 cell_type='N-ary', nary=3,
                 head_num=4, hid_dim=1024, layer_num=2, t_dropout=0.3,
                 device='cuda'):
        super(TreeLSTM, self).__init__()
        self.device = device
        self.h_size = h_size
        self.nary = nary
        # embedding
        self.embedding_dim = user_embed_dim + POI_embed_dim + cat_embed_dim + time_embed_dim + coo_embed_dim
        self.user_embedding = nn.Embedding(num_users, user_embed_dim)
        self.in_POI_embedding = nn.Embedding(num_POIs, POI_embed_dim)
        self.out_POI_embedding = nn.Embedding(num_POIs, POI_embed_dim)
        self.cat_embedding = nn.Embedding(num_cats, cat_embed_dim)
        self.time_embedding = nn.Embedding(24, time_embed_dim)
        self.coo_embedding = nn.Embedding(num_coos, coo_embed_dim)
        # fuse embedding
        self.user_POI_fusion = nn.Linear(user_embed_dim + POI_embed_dim, user_embed_dim + POI_embed_dim)
        self.cat_time_fusion = nn.Linear(cat_embed_dim + time_embed_dim, cat_embed_dim + time_embed_dim)
        self.coo_time_fusion = nn.Linear(coo_embed_dim + time_embed_dim, coo_embed_dim + time_embed_dim)
        # dropout
        self.embed_dropout = nn.Dropout(embed_dropout)
        self.model_dropout = nn.Dropout(model_dropout)
        # cell
        if cell_type == 'N-ary':
            self.POI_cell = NaryTreeLSTMCell(user_embed_dim + POI_embed_dim, h_size, nary, head_num, hid_dim, layer_num,
                                             t_dropout)
            self.cat_cell = NaryTreeLSTMCell(cat_embed_dim + time_embed_dim, h_size, nary, head_num, hid_dim, layer_num,
                                             t_dropout)
            self.coo_cell = NaryTreeLSTMCell(coo_embed_dim + time_embed_dim, h_size, nary, head_num, hid_dim, layer_num,
                                             t_dropout)
        # decoder
        self.decoder_POI = nn.Linear(h_size + cat_embed_dim + coo_embed_dim, num_POIs)
        self.decoder_cat = nn.Linear(h_size, num_cats)
        self.decoder_coo = nn.Linear(h_size, num_coos)
        # transfer
        self.cat_trans = nn.Linear(h_size, cat_embed_dim)
        self.coo_trans = nn.Linear(h_size, coo_embed_dim)

    def forward(self, in_trees, out_trees=None):
        user_embedding = self.user_embedding(in_trees.features[:, 0].long())
        POI_embedding = self.in_POI_embedding(in_trees.features[:, 1].long())
        cat_embedding = self.cat_embedding(in_trees.features[:, 2].long())
        time_embedding = self.time_embedding(in_trees.features[:, 3].long())
        coo_embedding = self.coo_embedding(in_trees.features[:, 4].long())
        # fusion
        user_POI_fusion = self.user_POI_fusion(torch.cat((user_embedding, POI_embedding), dim=-1))
        cat_time_fusion = self.cat_time_fusion(torch.cat((cat_embedding, time_embedding), dim=-1))
        coo_time_fusion = self.coo_time_fusion(torch.cat((coo_embedding, time_embedding), dim=-1))

        g = in_trees.graph.to(self.device)
        n = g.num_nodes()
        g.ndata["iou"] = self.POI_cell.W_iou(self.embed_dropout(user_POI_fusion))  # [batch_size, nary * h_size]
        g.ndata["x"] = self.embed_dropout(user_POI_fusion)  # [batch_size, embedding_dim]
        g.ndata["h"] = torch.zeros((n, self.h_size)).to(self.device)
        g.ndata["c"] = torch.zeros((n, self.h_size)).to(self.device)
        g.ndata["h_child"] = torch.zeros((n, self.nary, self.h_size)).to(self.device)
        g.ndata["c_child"] = torch.zeros((n, self.nary, self.h_size)).to(self.device)

        dgl.prop_nodes_topo(graph=g,
                            message_func=self.POI_cell.message_func,
                            reduce_func=self.POI_cell.reduce_func,
                            apply_node_func=self.POI_cell.apply_node_func)

        h_POI = self.model_dropout(g.ndata.pop("h"))  # [batch_size, h_size]

        g.ndata["iou"] = self.cat_cell.W_iou(self.embed_dropout(cat_time_fusion))  # [batch_size, nary * h_size]
        g.ndata["x"] = self.embed_dropout(cat_time_fusion)  # [batch_size, embedding_dim]
        g.ndata["h"] = torch.zeros((n, self.h_size)).to(self.device)
        g.ndata["c"] = torch.zeros((n, self.h_size)).to(self.device)
        g.ndata["h_child"] = torch.zeros((n, self.nary, self.h_size)).to(self.device)
        g.ndata["c_child"] = torch.zeros((n, self.nary, self.h_size)).to(self.device)

        dgl.prop_nodes_topo(graph=g,
                            message_func=self.cat_cell.message_func,
                            reduce_func=self.cat_cell.reduce_func,
                            apply_node_func=self.cat_cell.apply_node_func)

        h_cat = self.model_dropout(g.ndata.pop("h"))  # [batch_size, h_size]

        g.ndata["iou"] = self.coo_cell.W_iou(self.embed_dropout(coo_time_fusion))  # [batch_size, nary * h_size]
        g.ndata["x"] = self.embed_dropout(coo_time_fusion)  # [batch_size, embedding_dim]
        g.ndata["h"] = torch.zeros((n, self.h_size)).to(self.device)
        g.ndata["c"] = torch.zeros((n, self.h_size)).to(self.device)
        g.ndata["h_child"] = torch.zeros((n, self.nary, self.h_size)).to(self.device)
        g.ndata["c_child"] = torch.zeros((n, self.nary, self.h_size)).to(self.device)

        dgl.prop_nodes_topo(graph=g,
                            message_func=self.coo_cell.message_func,
                            reduce_func=self.coo_cell.reduce_func,
                            apply_node_func=self.coo_cell.apply_node_func)

        h_coo = self.model_dropout(g.ndata.pop("h"))

        y_pred_cat = self.decoder_cat(h_cat)
        y_pred_coo = self.decoder_coo(h_coo)

        cat_trans = self.cat_trans(h_cat)
        coo_trans = self.coo_trans(h_coo)
        y_pred_POI = self.decoder_POI(torch.concat((h_POI, cat_trans, coo_trans), dim=-1))

        return y_pred_POI, y_pred_cat, y_pred_coo
