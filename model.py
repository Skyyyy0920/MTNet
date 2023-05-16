import dgl
import numpy as np
import torch
import torch.nn as nn


def t2v(tau, f, out_features, w, b, w0, b0, arg=None):
    if arg:
        v1 = f(torch.matmul(tau, w) + b, arg)
    else:
        v1 = f(torch.matmul(tau, w) + b)
    v2 = torch.matmul(tau, w0) + b0
    return torch.cat([v1, v2], 1)


class SineActivation(nn.Module):
    def __init__(self, in_features, out_features):
        super(SineActivation, self).__init__()
        self.out_features = out_features
        self.w0 = nn.parameter.Parameter(torch.randn(in_features, 1))
        self.b0 = nn.parameter.Parameter(torch.randn(in_features, 1))
        self.w = nn.parameter.Parameter(torch.randn(in_features, out_features - 1))
        self.b = nn.parameter.Parameter(torch.randn(in_features, out_features - 1))
        self.f = torch.sin

    def forward(self, tau):
        return t2v(tau, self.f, self.out_features, self.w, self.b, self.w0, self.b0)


class CosineActivation(nn.Module):
    def __init__(self, in_features, out_features):
        super(CosineActivation, self).__init__()
        self.out_features = out_features
        self.w0 = nn.parameter.Parameter(torch.randn(in_features, 1))
        self.b0 = nn.parameter.Parameter(torch.randn(in_features, 1))
        self.w = nn.parameter.Parameter(torch.randn(in_features, out_features - 1))
        self.b = nn.parameter.Parameter(torch.randn(in_features, out_features - 1))
        self.f = torch.cos

    def forward(self, tau):
        return t2v(tau, self.f, self.out_features, self.w, self.b, self.w0, self.b0)


class Time2Vec(nn.Module):
    def __init__(self, activation='sin', out_dim=32):
        super(Time2Vec, self).__init__()
        if activation == 'sin':
            self.l1 = SineActivation(1, out_dim)
        elif activation == 'cos':
            self.l1 = CosineActivation(1, out_dim)

    def forward(self, x):
        x = self.l1(x)
        return x


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
                 num_coos=1024, coo_embed_dim=32,
                 time_embed_dim=32,
                 cell_type='N-ary', nary=3,
                 head_num=4, hid_dim=1024, layer_num=2, t_dropout=0.3,
                 device='cuda'):
        super(TreeLSTM, self).__init__()
        self.device = device
        # embedding
        self.embedding_dim = user_embed_dim + POI_embed_dim + cat_embed_dim + time_embed_dim + coo_embed_dim
        self.user_embedding = nn.Embedding(num_embeddings=num_users, embedding_dim=user_embed_dim)
        self.in_POI_embedding = nn.Embedding(num_embeddings=num_POIs, embedding_dim=POI_embed_dim)
        self.out_POI_embedding = nn.Embedding(num_embeddings=num_POIs, embedding_dim=POI_embed_dim)
        self.cat_embedding = nn.Embedding(num_embeddings=num_cats, embedding_dim=cat_embed_dim)
        self.coo_embedding = nn.Embedding(num_embeddings=num_coos, embedding_dim=coo_embed_dim)
        self.l1 = SineActivation(in_features=1, out_features=time_embed_dim)
        # dropout
        self.embed_dropout = nn.Dropout(embed_dropout)
        self.model_dropout = nn.Dropout(model_dropout)
        # cell
        if cell_type == 'N-ary':
            self.in_cell = NaryTreeLSTMCell(self.embedding_dim, h_size, nary, head_num, hid_dim, layer_num, t_dropout)
            self.out_cell = NaryTreeLSTMCell(self.embedding_dim, h_size, nary, head_num, hid_dim, layer_num, t_dropout)
        else:
            self.in_cell = ChildSumTreeLSTMCell(self.embedding_dim, h_size)
            self.out_cell = ChildSumTreeLSTMCell(self.embedding_dim, h_size)
        # decoder
        self.decoder_POI_in = nn.Linear(h_size, num_POIs)
        self.decoder_cat_in = nn.Linear(h_size, num_cats)
        self.decoder_coo_in = nn.Linear(h_size, num_coos)
        self.decoder_POI_out = nn.Linear(h_size, num_POIs)
        self.decoder_cat_out = nn.Linear(h_size, num_cats)
        self.decoder_coo_out = nn.Linear(h_size, num_coos)

    def forward(self, batch, g, h, c, h_child, c_child,
                re_batch, re_g, re_h, re_c, re_h_child, re_c_child):
        user_embedding = self.user_embedding(batch.features[:, 0].long())
        POI_embedding = self.in_POI_embedding(batch.features[:, 1].long())
        cat_embedding = self.cat_embedding(batch.features[:, 2].long())
        time_embedding = []
        for time in batch.features[:, 3]:
            t_input = torch.tensor([time], dtype=torch.float).to(device=self.device)
            time_embedding.append(torch.squeeze(self.l1(t_input)))
        time_embedding = np.array([item.cpu().detach().numpy() for item in time_embedding])
        time_embedding = torch.tensor(time_embedding).to(device=self.device)
        coo_embedding = self.coo_embedding(batch.features[:, 4].long())
        concat_embedding = torch.cat(
            (user_embedding, time_embedding, POI_embedding, cat_embedding, coo_embedding),
            dim=1)  # concat -> [batch_size, embedding_dim]

        re_user_embedding = self.user_embedding(re_batch.features[:, 0].long())
        re_POI_embedding = self.out_POI_embedding(re_batch.features[:, 1].long())
        re_cat_embedding = self.cat_embedding(re_batch.features[:, 2].long())
        re_time_embedding = []
        for time in re_batch.features[:, 3]:
            t_input = torch.tensor([time], dtype=torch.float).to(device=self.device)
            re_time_embedding.append(torch.squeeze(self.l1(t_input)))
        re_time_embedding = np.array([item.cpu().detach().numpy() for item in re_time_embedding])
        re_time_embedding = torch.tensor(re_time_embedding).to(device=self.device)
        re_coo_embedding = self.coo_embedding(re_batch.features[:, 4].long())
        re_concat_embedding = torch.cat(
            (re_user_embedding, re_time_embedding, re_POI_embedding, re_cat_embedding, re_coo_embedding),
            dim=1)  # concat -> [batch_size, embedding_dim]

        g.ndata["iou"] = self.in_cell.W_iou(self.embed_dropout(concat_embedding))  # [batch_size, nary * h_size]
        g.ndata["x"] = self.embed_dropout(concat_embedding)  # [batch_size, embedding_dim]
        g.ndata["h"] = h
        g.ndata["c"] = c
        g.ndata["h_child"] = h_child
        g.ndata["c_child"] = c_child
        re_g.ndata["iou"] = self.out_cell.W_iou(self.embed_dropout(re_concat_embedding))  # [batch_size, nary * h_size]
        re_g.ndata["x"] = self.embed_dropout(re_concat_embedding)  # [batch_size, embedding_dim]
        re_g.ndata["h"] = re_h
        re_g.ndata["c"] = re_c
        re_g.ndata["h_child"] = re_h_child
        re_g.ndata["c_child"] = re_c_child

        dgl.prop_nodes_topo(graph=g,
                            message_func=self.in_cell.message_func,
                            reduce_func=self.in_cell.reduce_func,
                            apply_node_func=self.in_cell.apply_node_func)
        dgl.prop_nodes_topo(graph=re_g,
                            message_func=self.out_cell.message_func,
                            reduce_func=self.out_cell.reduce_func,
                            apply_node_func=self.out_cell.apply_node_func)

        h = self.model_dropout(g.ndata.pop("h"))  # [batch_size, h_size]
        re_h = self.model_dropout(re_g.ndata.pop("h"))
        y_pred_POI_in = self.decoder_POI_in(h)
        y_pred_cat_in = self.decoder_cat_in(h)
        y_pred_coo_in = self.decoder_coo_in(h)
        y_pred_POI_out = self.decoder_POI_out(re_h)
        y_pred_cat_out = self.decoder_cat_out(re_h)
        y_pred_coo_out = self.decoder_coo_out(re_h)
        return y_pred_POI_in, y_pred_cat_in, y_pred_coo_in, y_pred_POI_out, y_pred_cat_out, y_pred_coo_out
