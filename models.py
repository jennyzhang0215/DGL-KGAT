import torch as th
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
from dgl.nn.pytorch.softmax import edge_softmax


def _cal_score(pos_score, neg_score):
    ### L = -1. * ln(sigmpid(neg_score, pos_score))
    s = th.log(th.sigmoid(neg_score - pos_score))
    return (-1.) * th.mean(s.double())
def _L2_norm(x):
    ### sum(t ** 2) / 2
    return th.sum(th.pow(x, 2), dim=1, keepdim=False) / 2.
def _L2_norm_mean(x):
    ### ### mean( sum(t ** 2) / 2)
    s = _L2_norm(x)
    return th.mean(s.double())

def bmm_maybe_select(A, B, index):
    """Slice submatrices of B by the given index and perform bmm.
    B is a 3D tensor of shape (N, D1, D2), which can be viewed as a stack of
    N matrices of shape (D1, D2). The input index is an integer vector of length M.
    A could be either:
    (1) a dense tensor of shape (M, D1),
    (2) an integer vector of length M.
    The result C is a 2D matrix of shape (M, D2)
    For case (1), C is computed by bmm:
    ::
        C[i, :] = matmul(A[i, :], B[index[i], :, :])
    For case (2), C is computed by index select:
    ::
        C[i, :] = B[index[i], A[i], :]
    Parameters
    ----------
    A : torch.Tensor
        lhs tensor
    B : torch.Tensor
        rhs tensor
    index : torch.Tensor
        index tensor
    Returns
    -------
    C : torch.Tensor
        return tensor
    """
    if A.dtype == th.int64 and len(A.shape) == 1:
        # following is a faster version of B[index, A, :]
        B = B.view(-1, B.shape[2])
        flatidx = index * B.shape[1] + A
        return B.index_select(0, flatidx)
    else:
        BB = B.index_select(0, index)
        return th.bmm(A.unsqueeze(1), BB).squeeze()

class KGEModel(nn.Module):
    def __init__(self, n_entities, n_relations, entity_dim, relation_dim, reg_lambda):
        super(KGEModel, self).__init__()
        self._reg_lambda = reg_lambda
        self.entity_embed = nn.Embedding(n_entities, entity_dim)
        self.relation_embed = nn.Embedding(n_relations, relation_dim)
        self.W_entity = nn.Linear(entity_dim, relation_dim, bias=False)

    def forward(self, h, r, pos_t, neg_t):
        h_embed = th.norm(self.W_entity(self.entity_embed(h)), p="fro", dim=1) ### Shape(batch_size, dim)
        r_embed = th.norm(self.relation_embed(r), p="fro", dim=1)
        pos_t_embed = th.norm(self.W_entity(self.entity_embed(pos_t)), p="fro", dim=1)
        neg_t_embed = th.norm(self.W_entity(self.entity_embed(neg_t)), p="fro", dim=1)

        pos_kg_score = _L2_norm(h_embed + r_embed - pos_t_embed) ### Shape(batch_size,)
        neg_kg_score = _L2_norm(h_embed + r_embed - neg_t_embed)
        kg_loss = _cal_score(pos_kg_score, neg_kg_score)
        kg_reg_loss = _L2_norm_mean(h_embed) + _L2_norm_mean(r_embed) + \
                      _L2_norm_mean(pos_t_embed) + _L2_norm_mean(neg_t_embed)

        loss = kg_loss + self._reg_lambda * kg_reg_loss
        return loss


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
    def att_score(self, edges):
        """
        att_score = (W_r h_t)^T tanh(W_r h_r + e_r)
        Parameters
        ----------
        edges

        Returns
        -------

        """
        t_r = bmm_maybe_select(edges.src['h'], self.relation_weight, edges.data['type']) ### (edge_num, hidden_dim)
        h_r = bmm_maybe_select(edges.dst['h'], self.relation_weight, edges.data['type']) ### (edge_num, hidden_dim)
        att_w = th.bmm(t_r.unsqueeze(1), th.tanh(h_r + edges.data['e']).unsqueeze(2)).squeeze(-1)
        return {'att_w': att_w}

    def forward(self, graph, nfeat, efeat):
        graph = graph.local_var()
        node_embed = self.feat_drop(nfeat)
        graph.ndata.update({'h': node_embed})
        edge_embed = self.feat_drop(efeat)
        graph.edata.update({'e': edge_embed})

        ### compute attention weight using edge_softmax
        graph.apply_edges(self.att_score)
        att_w = edge_softmax(graph, graph.edata.pop('att_w'))
        graph.edata['a'] = att_w
        graph.update_all(fn.u_mul_e('h', 'a', 'm'), fn.sum('m', 'h_neighbor'))
        if self._res_type == "Bi":
            h = F.leaky_relu(self.res_fc(graph.ndata['h']+graph.ndata['h_neighbor']))+\
                F.leaky_relu(self.res_fc_2(th.mul(graph.ndata['h'], graph.ndata['h_neighbor'])))
        else:
            raise NotImplementedError
        return h

class CFModel(nn.Module):
    def __init__(self, n_entities, n_relations, entity_dim, num_gnn_layers, n_hidden, dropout, reg_lambda):
        super(CFModel, self).__init__()
        self._reg_lambda = reg_lambda
        self.relation_embed = nn.Embedding(n_relations, n_hidden)  ### e_r
        self.entity_embed = nn.Embedding(n_entities, n_hidden)
        self.layers = nn.ModuleList()
        for i in range(num_gnn_layers):
            if i == 0:
                # in_feats, out_feats, n_relations, feat_drop,
                kgatConv = KGATConv(n_hidden, n_hidden, n_relations, dropout)
            else:
                kgatConv = KGATConv(n_hidden, n_hidden, n_relations, dropout)
            self.layers.append(kgatConv)

    def forward(self, g, src_ids, pos_dst_ids, neg_dst_ids):
        h = self.entity_embed(g.ndata['id'])
        efeat = self.relation_embed(g.edata['type'])
        node_embed_cache = [h]
        for i, layer in enumerate(self.layers):
            h = layer(g, h, efeat)
            node_embed_cache.append(h)
        final_h = th.cat(node_embed_cache, 1)
        #print("final_h", final_h.shape, final_h)
        src_vec = final_h[src_ids]
        pos_dst_vec = final_h[pos_dst_ids]
        neg_dst_vec = final_h[neg_dst_ids]
        #print("src_vec", src_vec.shape)
        #print("pos_dst_vec", pos_dst_vec.shape)
        #print("neg_dst_vec", neg_dst_vec.shape)
        pos_score = th.bmm(src_vec.unsqueeze(1), pos_dst_vec.unsqueeze(2)).squeeze() ### (batch_size, )
        neg_score = th.bmm(src_vec.unsqueeze(1), neg_dst_vec.unsqueeze(2)).squeeze() ### (batch_size, )
        cf_reg_loss = _L2_norm_mean(src_vec) + _L2_norm_mean(pos_dst_vec) + _L2_norm_mean(neg_dst_vec)
        cf_loss = _cal_score(pos_score, neg_score)
        loss = cf_loss + self._reg_lambda * cf_reg_loss

        return loss















