import torch as th
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
from dgl.nn.pytorch.softmax import edge_softmax
import math

def _L2_loss_mean(x):
    ### ### mean( sum(t ** 2) / 2)
    return th.mean(th.sum(th.pow(x, 2), dim=1, keepdim=False) / 2.)

def _L2_loss_sum(x):
    ### ### sum( sum(t ** 2) / 2)
    return th.sum(th.sum(th.pow(x, 2), dim=1, keepdim=False) / 2.)

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
        self.feat_drop = nn.Dropout(dropout)
        self._res_type = res_type
        if res_type == "Bi":
            self.res_fc = nn.Linear(entity_in_feats, out_feats, bias=False)
            self.res_fc_2 = nn.Linear(entity_in_feats, out_feats, bias=False)
        else:
            raise NotImplementedError

    def forward(self, g, nfeat):
        g = g.local_var()
        # node_embed = self.feat_drop(nfeat)
        g.ndata['h'] = self.feat_drop(nfeat)
        #g.ndata['h'] = th.matmul(nfeat, self.W_r).squeeze() ### (#node, #rel, entity_dim)
        #print("relation_W", self.relation_W.shape,  self.relation_W)
        ### compute attention weight using edge_softmax
        #print("attention_score:", graph.edata['att_w'])
        #print("att_w", att_w)

        g.update_all(fn.u_mul_e('h', 'w', 'm'), fn.sum('m', 'h_neighbor'))
        h_neighbor = g.ndata['h_neighbor']
        if self._res_type == "Bi":
            out = F.leaky_relu(self.res_fc(g.ndata['h']+h_neighbor))+\
                  F.leaky_relu(self.res_fc_2(th.mul(g.ndata['h'], h_neighbor)))
        else:
            raise NotImplementedError
        return h_neighbor, out

class Model(nn.Module):
    def __init__(self, n_entities, n_relations, entity_dim, relation_dim, num_gnn_layers, n_hidden,
                 dropout, reg_lambda_kg=0.01, reg_lambda_gnn=0.01, res_type="Bi"):
        super(Model, self).__init__()
        self._n_entities = n_entities
        self._n_relations = n_relations
        self._reg_lambda_kg = reg_lambda_kg
        self._reg_lambda_gnn = reg_lambda_gnn
        self.entity_embed = nn.Embedding(n_entities, entity_dim) ### e_h, e_t
        self.relation_embed = nn.Embedding(n_relations, relation_dim)  ### e_r
        self.W_R = nn.Parameter(th.Tensor(n_relations, entity_dim, relation_dim))  ### W_r
        nn.init.xavier_uniform_(self.W_R, gain=nn.init.calculate_gain('relu'))
        self.layers = nn.ModuleList()
        for i in range(num_gnn_layers):
            r = int(math.pow(2, i))
            self.layers.append(KGATConv(entity_dim, n_hidden//r, dropout))

    def transR(self, h, r, pos_t, neg_t):
        h_embed = self.entity_embed(h)  ### Shape(batch_size, dim)
        r_embed = self.relation_embed(r)
        pos_t_embed = self.entity_embed(pos_t)
        neg_t_embed = self.entity_embed(neg_t)
        # print("h_embed", h_embed)
        h_vec = F.normalize(bmm_maybe_select(h_embed, self.W_R, r), p=2, dim=1)
        # print("h_vec", h_vec)
        r_vec = F.normalize(r_embed, p=2, dim=1)
        pos_t_vec = F.normalize(bmm_maybe_select(pos_t_embed, self.W_R, r), p=2, dim=1)
        neg_t_vec = F.normalize(bmm_maybe_select(neg_t_embed, self.W_R, r), p=2, dim=1)

        pos_score = th.sum(th.pow(h_vec + r_vec - pos_t_vec, 2), dim=1, keepdim=True)
        neg_score = th.sum(th.pow(h_vec + r_vec - neg_t_vec, 2), dim=1, keepdim=True)
        l = F.logsigmoid(pos_score - neg_score) * (-1.0)
        l = th.mean(l)
        ## tf.reduce_sum(tf.square((h_e + r_e - t_e)), 1, keepdims=True)
        ###
        reg_loss =_L2_loss_mean(self.relation_embed.weight) + _L2_loss_mean(self.entity_embed.weight) + \
                  _L2_loss_mean(self.W_R)
        #print("\tkg loss:", l.items(), "reg loss:", reg_loss.items())
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

    def gnn(self, g, node_ids, rel_ids):
        #print("node_ids", node_ids.shape, node_ids)
        #print("relation_ids", relation_ids.shape, relation_ids)
        h = self.entity_embed(node_ids)
        self.g = g
        g.ndata['id'] = node_ids
        g.edata['type'] = rel_ids
        ## compute attention weight and store it on edges
        for i in range(self._n_relations):
            e_idxs = self.g.filter_edges(lambda edges: edges.data['type'] == i)
            self.W_r = self.W_R[i]
            g.apply_edges(self._att_score, e_idxs)
        g.edata['w'] = edge_softmax(g, g.edata.pop('att_w'))
        node_embed_cache = [h]
        for i, layer in enumerate(self.layers):
            h, out = layer(g, h)
            #print(i, "h", h.shape, h)
            out = F.normalize(out, p=2, dim=1)
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
        self.reg_loss = _L2_loss_mean(self.relation_embed.weight) + _L2_loss_mean(self.entity_embed.weight) +\
                        _L2_loss_mean(self.W_R)
        #print("\tcf_loss:{}, reg_loss:{}".format(self.cf_loss.item(), self.reg_loss.item()))
        return self.cf_loss + self._reg_lambda_gnn * self.reg_loss



















