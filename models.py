import torch as th
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
from dgl.nn.pytorch.softmax import edge_softmax
from dgl.nn.pytorch.conv import SAGEConv
import math

def _L2_loss_mean(x):
    ### ### mean( sum(t ** 2) / 2)
    return th.mean(th.sum(th.pow(x, 2), dim=1, keepdim=False) / 2.)

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

class KGATConv(nn.Module):
    def __init__(self, entity_in_feats, out_feats, dropout, res_type="Bi"):
        super(KGATConv, self).__init__()
        self.mess_drop = nn.Dropout(dropout)
        self._res_type = res_type
        if res_type == "Bi":
            #self.res_fc = nn.Linear(entity_in_feats, out_feats, bias=False)
            self.res_fc_2 = nn.Linear(entity_in_feats, out_feats, bias=False)
        else:
            raise NotImplementedError

    def forward(self, g, nfeat):
        g = g.local_var()
        g.ndata['h'] = nfeat
        g.update_all(fn.u_mul_e('h', 'w', 'm'), fn.sum('m', 'h_neighbor'))
        h_neighbor = g.ndata['h_neighbor']
        if self._res_type == "Bi":
            out = F.leaky_relu(self.res_fc_2(th.mul(g.ndata['h'], h_neighbor)))
        else:
            raise NotImplementedError

        return self.mess_drop(out)

class Model(nn.Module):
    def __init__(self, use_KG, input_node_dim, gnn_model, num_gnn_layers, n_hidden, dropout, use_attention=True,
                 n_entities=None, n_relations=None, relation_dim=None,
                 reg_lambda_kg=0.01, reg_lambda_gnn=0.01, res_type="Bi"):
        super(Model, self).__init__()
        self._use_KG = use_KG
        self._n_entities = n_entities
        self._n_relations = n_relations
        self._gnn_model = gnn_model
        self._use_attention = use_attention
        self._reg_lambda_kg = reg_lambda_kg
        self._reg_lambda_gnn = reg_lambda_gnn

        ### for input node embedding
        self.entity_embed = nn.Embedding(n_entities, input_node_dim) ### e_h, e_t
        self.relation_embed = nn.Embedding(n_relations, relation_dim)  ### e_r
        self.W_R = nn.Parameter(th.Tensor(n_relations, input_node_dim, relation_dim))  ### W_r
        nn.init.xavier_uniform_(self.W_R, gain=nn.init.calculate_gain('relu'))

        self.layers = nn.ModuleList()
        for i in range(num_gnn_layers):
            r = int(math.pow(2, i))
            act = None if i+1 == num_gnn_layers else F.relu
            if i==0:
                if gnn_model == "kgat":
                    self.layers.append(KGATConv(input_node_dim, n_hidden // r, dropout))
                elif gnn_model == "graphsage":
                    self.layers.append(SAGEConv(input_node_dim, n_hidden // r, aggregator_type="mean",
                                                feat_drop=dropout, activation=act))
                else:
                    raise NotImplementedError
            else:
                r2 = int(math.pow(2, i - 1))
                if gnn_model == "kgat":
                    self.layers.append(KGATConv(n_hidden // r2, n_hidden // r, dropout))
                elif gnn_model == "graphsage":
                    self.layers.append(SAGEConv(n_hidden // r2, n_hidden // r, aggregator_type="mean",
                                                feat_drop=dropout, activation=act))
                else:
                    raise NotImplementedError


    def transR(self, h, r, pos_t, neg_t):
        h_embed = self.entity_embed(h)  ### Shape(batch_size, dim)
        r_embed = self.relation_embed(r)
        pos_t_embed = self.entity_embed(pos_t)
        neg_t_embed = self.entity_embed(neg_t)

        h_vec = F.normalize(bmm_maybe_select(h_embed, self.W_R, r), p=2, dim=1)
        r_vec = F.normalize(r_embed, p=2, dim=1)
        pos_t_vec = F.normalize(bmm_maybe_select(pos_t_embed, self.W_R, r), p=2, dim=1)
        neg_t_vec = F.normalize(bmm_maybe_select(neg_t_embed, self.W_R, r), p=2, dim=1)

        pos_score = th.sum(th.pow(h_vec + r_vec - pos_t_vec, 2), dim=1, keepdim=True)
        neg_score = th.sum(th.pow(h_vec + r_vec - neg_t_vec, 2), dim=1, keepdim=True)
        ### pairwise ranking loss
        l = (-1.0) * F.logsigmoid(neg_score-pos_score)
        l = th.mean(l)
        reg_loss = _L2_loss_mean(h_vec) + _L2_loss_mean(r_vec) + \
                   _L2_loss_mean(pos_t_vec) + _L2_loss_mean(neg_t_vec)
        loss = l + self._reg_lambda_kg * reg_loss
        return loss

    def _att_score(self, edges):
        """
        att_score = (W_r h_t)^T tanh(W_r h_r + e_r)

        """
        t_r = th.matmul(self.entity_embed(edges.src['id']), self.W_r) ### (edge_num, hidden_dim)
        h_r = th.matmul(self.entity_embed(edges.dst['id']), self.W_r) ### (edge_num, hidden_dim)
        att_w = th.bmm(t_r.unsqueeze(1),
                       th.tanh(h_r + self.relation_embed(edges.data['type'])).unsqueeze(2)).squeeze(-1)
        return {'att_w': att_w}

    def compute_attention(self, g):
        ## compute attention weight and store it on edges
        g = g.local_var()
        for i in range(self._n_relations):
            e_idxs = g.filter_edges(lambda edges: edges.data['type'] == i)
            self.W_r = self.W_R[i]
            g.apply_edges(self._att_score, e_idxs)
        w = edge_softmax(g, g.edata.pop('att_w'))
        return w

    def gnn(self, g, x):
        g = g.local_var()
        if self._use_KG:
            h = self.entity_embed(g.ndata['id'])
        else:
            h = th.cat((self.item_proj(x[0]), self.user_proj(x[1])), dim=0)
        node_embed_cache = [h]
        for i, layer in enumerate(self.layers):
            h = layer(g, h)
            out = F.normalize(h, p=2, dim=1)
            node_embed_cache.append(out)
        final_h = th.cat(node_embed_cache, 1)
        return final_h

    def get_loss(self, embedding, src_ids, pos_dst_ids, neg_dst_ids):
        src_vec = embedding[src_ids]
        pos_dst_vec = embedding[pos_dst_ids]
        neg_dst_vec = embedding[neg_dst_ids]
        pos_score = th.bmm(src_vec.unsqueeze(1), pos_dst_vec.unsqueeze(2)).squeeze()  ### (batch_size, )
        neg_score = th.bmm(src_vec.unsqueeze(1), neg_dst_vec.unsqueeze(2)).squeeze()  ### (batch_size, )
        cf_loss = th.mean(F.logsigmoid(pos_score - neg_score) ) * (-1.0)
        reg_loss = _L2_loss_mean(src_vec) + _L2_loss_mean(pos_dst_vec) + _L2_loss_mean(neg_dst_vec)
        return cf_loss + self._reg_lambda_gnn * reg_loss



















