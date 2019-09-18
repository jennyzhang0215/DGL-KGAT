import torch as th
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
from dgl.nn.pytorch.softmax import edge_softmax


def _cal_score(pos_score, neg_score):
    ### L = -1. * ln(sigmpid(neg_score, pos_score))
    return (-1.) * th.mean(th.log(F.sigmoid(neg_score - pos_score)))
def _L2_norm(x):
    ### sum(t ** 2) / 2
    return th.sum(th.pow(x, 2), dim=1, keepdims=False) / 2.
def _L2_norm_mean(x):
    ### ### mean( sum(t ** 2) / 2)
    return th.mean(th.sum(th.pow(x, 2), dim=1, keepdims=False)/2.)

# class KGEModel(nn.Module):
#     def __init__(self, n_entities, n_relations, entity_dim, relation_dim, reg_lambda):
#         super(KGEModel, self).__init__()
#         self._reg_lambda = reg_lambda
#         self.entity_embed = nn.Embedding(n_entities, entity_dim)
#         self.relation_embed = nn.Embedding(n_relations, relation_dim)
#         self.W_entity = nn.Linear(entity_dim, relation_dim, bias=False)
#
#     def forward(self, h, r, pos_t, neg_t):
#         h_embed = th.norm(self.W_entity(self.entity_embed(h)), p="fro", dim=1) ### Shape(batch_size, dim)
#         r_embed = th.norm(self.relation_embed(r), p="fro", dim=1)
#         pos_t_embed = th.norm(self.W_entity(self.entity_embed(pos_t)), p="fro", dim=1)
#         neg_t_embed = th.norm(self.W_entity(self.entity_embed(neg_t)), p="fro", dim=1)
#
#         pos_kg_score = _L2_norm(h_embed + r_embed - pos_t_embed) ### Shape(batch_size,)
#         neg_kg_score = _L2_norm(h_embed + r_embed - neg_t_embed)
#         kg_loss = _cal_score(pos_kg_score, neg_kg_score)
#         kg_reg_loss = _L2_norm_mean(h_embed) + _L2_norm_mean(r_embed) + \
#                       _L2_norm_mean(pos_t_embed) + _L2_norm_mean(neg_t_embed)
#
#         loss = kg_loss + self._reg_lambda * kg_reg_loss
#         return loss


class KGATConv(nn.Module):
    def __init__(self, in_feats, out_feats, n_relations, feat_drop, res_type="Bi"):
        super(KGATConv, self).__init__()
        self._in_feats = in_feats
        self.relation_weight = nn.Parameter(th.Tensor(n_relations, in_feats, out_feats))  ### W_r
        nn.init.xavier_uniform_(self.relation_weight, gain=nn.init.calculate_gain('relu'))
        self.feat_drop = nn.Dropout(feat_drop)
        self._res_type = res_type
        if res_type == "Bi":
            self.res_fc = nn.Linear(out_feats, out_feats, bias=False)
            self.res_fc_2 = nn.Linear(out_feats, out_feats, bias=False)
        else:
            raise NotImplementedError

    def forward(self, graph, nfeat, efeat):
        print(g)
        graph = graph.local_var()
        node_embed = self.feat_drop(nfeat)
        graph.ndata.update({'h': node_embed})
        print("start compute edge weight ...")
        edge_W = self.relation_weight.index_select(0, graph.edata["type"])
        print(edge_W)
        edge_embed = self.feat_drop(efeat)
        print(edge_embed)
        graph.edata.update({'R_W':edge_W, 'e': edge_embed})
        print("update node features")
        graph.apply_edges(fn.u_mul_e('h', 'R_W', 't_r'))
        graph.apply_edges(fn.v_mul_e('h', 'R_W', 'h_r'))
        print("apply edges h R_W")
        graph.edata.update({'att_w': th.dot(graph.edata["t_r"],
                                            F.tanh(graph.edata["h_r"] + self.relation_embed(graph.edata['e'])))})
        graph.edata['a'] = edge_softmax(graph, graph.edata.pop('att_w'))
        graph.update_all(fn.u_mul_e('h', 'a', 'm'),
                         fn.sum('m', 'h_neighbor'))
        if self._res_type == "Bi":
            graph.ndata.update({'h': F.leaky_relu(self.res_fc(graph.ndata['h']+graph.ndata['h_neighbor']))+
                                     F.leaky_relu(self.res_fc_2(th.mul(graph.ndata['h'],graph.ndata['h_neighbor'])))})
        return graph.ndata['h']


class CFModel(nn.Module):
    def __init__(self, n_entities, n_relations, entity_dim, num_gnn_layers, n_hidden, dropout, reg_lambda):
        super(CFModel, self).__init__()
        self._reg_lambda = reg_lambda
        self.relation_embed = nn.Embedding(n_relations, n_hidden)  ### e_r
        self.entity_embed = nn.Embedding(n_entities, entity_dim)
        self.layers = nn.ModuleList()
        for i in range(num_gnn_layers):
            if i == 0:
                # in_feats, out_feats, n_relations, feat_drop,
                kgatConv = KGATConv(entity_dim, n_hidden, n_relations, dropout)
            else:
                kgatConv = KGATConv(n_hidden, n_hidden, n_relations, dropout)
            self.layers.append(kgatConv)

    def forward(self, g, src_ids, pos_dst_ids, neg_dst_ids):
        h = self.entity_embed(th.arange(g.number_of_nodes()))
        efeat = self.relation_embed(g.edata['type'])
        print("embedding finished")
        print("h:", h.shape, "\n", h)
        print("efeat:",efeat.shape, "\n", efeat)
        node_embed_cache = [h]
        for i, layer in enumerate(self.layers):
            print(i)
            h = layer(g, h, efeat)
            node_embed_cache.append(h)
        final_h = th.cat(node_embed_cache, 1)
        src_vec = final_h[src_ids]
        pos_dst_vec = final_h[pos_dst_ids]
        neg_dst_vec = final_h[neg_dst_ids]
        pos_score = th.sum(th.mul(src_vec, pos_dst_vec), dim=1, keepdims=False)
        neg_score = th.sum(th.mul(src_vec, neg_dst_vec), dim=1, keepdims=False)
        cf_reg_loss = _L2_norm_mean(src_vec) + _L2_norm_mean(pos_dst_vec) + _L2_norm_mean(neg_dst_vec)
        cf_loss = _cal_score(pos_score, neg_score)
        loss = cf_loss + self._reg_lambda * cf_reg_loss

        return loss















