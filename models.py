import torch as th
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
from dgl.nn.pytorch.softmax import edge_softmax


def _L2_norm(x):
    ### sum(t ** 2) / 2
    ### th.pow(th.norm(x, dim=1), 2) / 2.
    return th.sum(th.pow(x, 2), dim=1, keepdim=False) / 2.
def _L2_norm_mean(x):
    ### ### mean( sum(t ** 2) / 2)
    return th.mean(_L2_norm(x))

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
        self.relation_weight = nn.Parameter(th.Tensor(n_relations, entity_dim, relation_dim))  ### W_r
        nn.init.xavier_uniform_(self.relation_weight, gain=nn.init.calculate_gain('relu'))

    def forward(self, h, r, pos_t, neg_t):
        h_embed = self.entity_embed(h)  ### Shape(batch_size, dim)
        r_embed = self.relation_embed(r)
        pos_t_embed = self.entity_embed(pos_t)
        neg_t_embed = self.entity_embed(neg_t)

        h_vec = bmm_maybe_select(h_embed, self.relation_weight, r)
        pos_t_vec = bmm_maybe_select(pos_t_embed, self.relation_weight, r)
        neg_t_vec = bmm_maybe_select(neg_t_embed, self.relation_weight, r)
        # print("h_vec:", h_vec.shape)
        # print("r_vec", r_embed.shape)
        # print("pos_t_vec", pos_t_vec.shape)
        # print("neg_t_vec", neg_t_vec.shape)
        pos_score = _L2_norm(h_vec + r_embed - pos_t_vec)
        neg_score = _L2_norm(h_vec + r_embed - neg_t_vec)
        kg_loss = F.logsigmoid(pos_score - neg_score) * (-1.0)
        print(kg_loss)
        kg_loss = th.mean(kg_loss)
        kg_reg_loss = _L2_norm_mean(h_embed) + _L2_norm_mean(r_embed) + \
                      _L2_norm_mean(pos_t_embed) + _L2_norm_mean(neg_t_embed)
        loss = kg_loss + self._reg_lambda * kg_reg_loss
        return loss


class KGATConv(nn.Module):
    def __init__(self, entity_in_feats, out_feats, relation_weight, feat_drop, res_type="Bi"):
        super(KGATConv, self).__init__()
        self.feat_drop = nn.Dropout(feat_drop)
        self._res_type = res_type
        self.relation_W = relation_weight
        if res_type == "Bi":
            self.res_fc = nn.Linear(entity_in_feats, out_feats, bias=False)
            self.res_fc_2 = nn.Linear(entity_in_feats, out_feats, bias=False)
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
        #print(self.relation_W)
        t_r = bmm_maybe_select(edges.src['h'], self.relation_W, edges.data['type']) ### (edge_num, hidden_dim)
        h_r = bmm_maybe_select(edges.dst['h'], self.relation_W, edges.data['type']) ### (edge_num, hidden_dim)
        # print("t_r", t_r)
        # print("h_r", h_r)
        att_w = th.bmm(t_r.unsqueeze(1), th.tanh(h_r + edges.data['e']).unsqueeze(2)).squeeze(-1)
        return {'att_w': att_w}

    def forward(self, graph, nfeat, efeat):
        graph = graph.local_var()
        # node_embed = self.feat_drop(nfeat)
        graph.ndata.update({'h': nfeat})
        graph.edata.update({'e': efeat})
        #print("relation_W", self.relation_W.shape,  self.relation_W)
        ### compute attention weight using edge_softmax
        graph.apply_edges(self.att_score)
        #print("attention_score:", graph.edata['att_w'])
        att_w = edge_softmax(graph, graph.edata.pop('att_w'))
        #print("att_w", att_w)
        graph.edata['a'] = att_w
        graph.update_all(fn.u_mul_e('h', 'a', 'm'), fn.sum('m', 'h_neighbor'))
        h_neighbor = graph.ndata['h_neighbor']
        if self._res_type == "Bi":
            out = F.leaky_relu(self.res_fc(graph.ndata['h']+h_neighbor))+\
                  F.leaky_relu(self.res_fc_2(th.mul(graph.ndata['h'], h_neighbor)))
        else:
            raise NotImplementedError
        return h_neighbor, out

class CFModel(nn.Module):
    def __init__(self, n_entities, n_relations, entity_dim, relation_dim, num_gnn_layers, n_hidden,
                 dropout, reg_lambda, res_type="Bi"):
        super(CFModel, self).__init__()
        self._reg_lambda = reg_lambda
        self.entity_embed = nn.Embedding(n_entities, entity_dim) ### e_h, e_t
        self.relation_embed = nn.Embedding(n_relations, relation_dim)  ### e_r
        self.relation_weight = nn.Parameter(th.Tensor(n_relations, entity_dim, relation_dim))  ### W_r
        nn.init.xavier_uniform_(self.relation_weight, gain=nn.init.calculate_gain('relu'))
        self.layers = nn.ModuleList()
        for i in range(num_gnn_layers):
            self.layers.append(KGATConv(entity_dim, n_hidden, self.relation_weight, dropout))

    def forward(self, g, node_ids, relation_ids):
        #print("node_ids", node_ids.shape, node_ids)
        #print("relation_ids", relation_ids.shape, relation_ids)
        g.edata['type'] = relation_ids
        h = self.entity_embed(node_ids)
        efeat = self.relation_embed(relation_ids)
        #print("h", h.shape, h)
        #print("efeat", efeat.shape, efeat)
        node_embed_cache = [h]
        for i, layer in enumerate(self.layers):
            h, out = layer(g, h, efeat)
            #print(i, "h", h.shape, h)
            node_embed_cache.append(out)
        final_h = th.cat(node_embed_cache, 1)
        #print("final_h", final_h)
        return final_h

    def get_loss(self, embedding, src_ids, pos_dst_ids, neg_dst_ids):
        src_vec = embedding[src_ids]
        pos_dst_vec = embedding[pos_dst_ids]
        neg_dst_vec = embedding[neg_dst_ids]
        pos_score = th.bmm(src_vec.unsqueeze(1), pos_dst_vec.unsqueeze(2)).squeeze()  ### (batch_size, )
        neg_score = th.bmm(src_vec.unsqueeze(1), neg_dst_vec.unsqueeze(2)).squeeze()  ### (batch_size, )
        #print("pos_score", pos_score)
        #print("neg_score", neg_score)
        self.cf_loss = th.mean(F.logsigmoid(pos_score - neg_score) ) * (-1.0)
        self.reg_loss = _L2_norm_mean(self.relation_embed.weight) + _L2_norm_mean(self.entity_embed.weight) +\
                        _L2_norm_mean(self.relation_weight)
        print("\tcf_loss:{}, reg_loss:{}".format(self.cf_loss.item(), self.reg_loss.item()))
        return self.cf_loss + self._reg_lambda * self.reg_loss



















