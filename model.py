import dgl
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
        return {"h_child": edges.src["h"], "c_child": edges.src["c"], "type": edges.src["type"]}

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
                 h_size=512, nary=3,
                 embed_dropout=0.2, model_dropout=0.4,
                 num_users=3000, user_embed_dim=128,
                 num_POIs=5000, POI_embed_dim=128,
                 num_cats=300, cat_embed_dim=32,
                 num_coos=1024, coo_embed_dim=64,
                 device='cuda'):
        super(TreeLSTM, self).__init__()
        self.device = device
        self.h_size = h_size
        self.nary = nary
        # embedding
        self.embedding_dim = user_embed_dim + POI_embed_dim + cat_embed_dim + coo_embed_dim
        self.user_embedding = nn.Embedding(num_embeddings=num_users, embedding_dim=user_embed_dim)
        self.POI_embedding = nn.Embedding(num_embeddings=num_POIs, embedding_dim=POI_embed_dim)
        self.cat_embedding = nn.Embedding(num_embeddings=num_cats, embedding_dim=cat_embed_dim)
        self.coo_embedding = nn.Embedding(num_embeddings=num_coos, embedding_dim=coo_embed_dim)
        # self.user_embedding_o = nn.Embedding(num_embeddings=num_users, embedding_dim=user_embed_dim)
        # self.POI_embedding_o = nn.Embedding(num_embeddings=num_POIs, embedding_dim=POI_embed_dim)
        # self.cat_embedding_o = nn.Embedding(num_embeddings=num_cats, embedding_dim=cat_embed_dim)
        # self.coo_embedding_o = nn.Embedding(num_embeddings=num_coos, embedding_dim=coo_embed_dim)
        # positional encoding
        self.time_pos_encoder = nn.Embedding(num_embeddings=96, embedding_dim=self.embedding_dim)
        # self.time_pos_encoder_o = nn.Embedding(num_embeddings=96, embedding_dim=self.embedding_dim)
        # dropout
        self.embed_dropout = nn.Dropout(embed_dropout)
        self.model_dropout = nn.Dropout(model_dropout)
        # cell
        self.cell = Cell(self.embedding_dim, h_size, nary)
        # self.cell_o = Cell(self.embedding_dim, h_size, nary)
        # decoder
        self.decoder_POI = nn.Linear(h_size, num_POIs)
        self.decoder_cat = nn.Linear(h_size, num_cats)
        self.decoder_coo = nn.Linear(h_size, num_coos)
        # self.decoder_POI_o = nn.Linear(h_size, num_POIs)
        # self.decoder_cat_o = nn.Linear(h_size, num_cats)
        # self.decoder_coo_o = nn.Linear(h_size, num_coos)

    # def forward(self, in_trees, out_trees):
    def forward(self, in_trees):
        user_embedding = self.user_embedding(in_trees.features[:, 0].long() * in_trees.mask)
        POI_embedding = self.POI_embedding(in_trees.features[:, 1].long() * in_trees.mask)
        cat_embedding = self.cat_embedding(in_trees.features[:, 2].long() * in_trees.mask)
        coo_embedding = self.coo_embedding(in_trees.features[:, 3].long() * in_trees.mask)
        pe = self.time_pos_encoder(in_trees.time.long() * in_trees.mask)
        concat_embedding = torch.cat((user_embedding, POI_embedding, cat_embedding, coo_embedding), dim=1)
        concat_embedding = concat_embedding + pe * 0.5

        # user_embedding_o = self.user_embedding_o(out_trees.features[:, 0].long() * out_trees.mask)
        # POI_embedding_o = self.POI_embedding_o(out_trees.features[:, 1].long() * out_trees.mask)
        # cat_embedding_o = self.cat_embedding_o(out_trees.features[:, 2].long() * out_trees.mask)
        # coo_embedding_o = self.coo_embedding_o(out_trees.features[:, 3].long() * out_trees.mask)
        # pe_o = self.time_pos_encoder_o(out_trees.time.long() * out_trees.mask)
        # concat_embedding_o = torch.cat((user_embedding_o, POI_embedding_o, cat_embedding_o, coo_embedding_o), dim=1)
        # concat_embedding_o = concat_embedding_o + pe_o * 0.5

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

        # g_o = out_trees.graph.to(self.device)
        # n = g_o.num_nodes()
        # g_o.ndata["iou"] = self.cell_o.W_iou(self.embed_dropout(concat_embedding_o)) \
        #                    * out_trees.mask.float().unsqueeze(-1)
        # g_o.ndata["x"] = self.embed_dropout(concat_embedding_o) * out_trees.mask.float().unsqueeze(-1)
        # g_o.ndata["h"] = torch.zeros((n, self.h_size)).to(self.device)
        # g_o.ndata["c"] = torch.zeros((n, self.h_size)).to(self.device)
        # g_o.ndata["h_child"] = torch.zeros((n, self.nary, self.h_size)).to(self.device)
        # g_o.ndata["c_child"] = torch.zeros((n, self.nary, self.h_size)).to(self.device)
        #
        # dgl.prop_nodes_topo(graph=g_o,
        #                     message_func=self.cell_o.message_func,
        #                     reduce_func=self.cell_o.reduce_func,
        #                     apply_node_func=self.cell_o.apply_node_func)
        #
        # h_o = self.model_dropout(g_o.ndata.pop("h"))  # [batch_size, h_size]
        #
        # y_pred_POI_o = self.decoder_POI_o(h_o)
        # y_pred_cat_o = self.decoder_cat_o(h_o)
        # y_pred_coo_o = self.decoder_coo_o(h_o)

        # return y_pred_POI, y_pred_cat, y_pred_coo, y_pred_POI_o, y_pred_cat_o, y_pred_coo_o
        return y_pred_POI, y_pred_cat, y_pred_coo


class MultiTaskLoss(nn.Module):
    def __init__(self, num=3):
        super(MultiTaskLoss, self).__init__()
        params = torch.ones(num, requires_grad=True)
        self.params = nn.Parameter(params)

    def forward(self, *losses):
        loss_sum = 0
        for i, loss in enumerate(losses):
            # loss_sum += 0.5 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i] ** 2)
            loss_sum += 0.5 * torch.exp(-self.params[i]) * loss + self.params[i]
        return loss_sum