import numpy as np
import scipy.sparse as sp
from scipy.sparse import coo_matrix

def _si_norm_lap(adj):
    rowsum = np.array(adj.sum(1))
    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat_inv = sp.diags(d_inv)
    norm_adj = d_mat_inv.dot(adj)
    return norm_adj.tocoo()

A = coo_matrix([[1.0, 2.0, 0.0], [0.0, 0.0, 3.0], [4.0, 0.0, 5.0]])
lap_A = _si_norm_lap(A)




def _bi_norm_lap(adj):
    print("adj", adj.toarray())
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    bi_lap = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
    print(bi_lap)
    return bi_lap.tocoo()