import numpy as np
import torch as th
import heapq

def one_recall_at_k(r, k, all_pos_num):
    r = np.asfarray(r)[:k]
    return np.sum(r) / all_pos_num
def one_dcg_at_k(r, k, method=1):
    """Score is discounted cumulative gain (dcg)
    Relevance is positive real values.  Can use binary
    as the previous methods.
    Returns:
        Discounted cumulative gain
    """
    r = np.asfarray(r)[:k]
    if r.size:
        if method == 0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        else:
            raise ValueError('method must be 0 or 1.')
    return 0.
def one_ndcg_at_k(r, k, method=1):
    """Score is normalized discounted cumulative gain (ndcg)
    Relevance is positive real values.  Can use binary
    as the previous methods.
    Returns:
        Normalized discounted cumulative gain
    """
    dcg_max = one_dcg_at_k(sorted(r, reverse=True), k, method)
    if not dcg_max:
        return 0.
    return one_dcg_at_k(r, k, method) / dcg_max

def calc_recall_ndcg(embedding, train_user_dict, test_user_dict, all_item_id_range, K, use_cuda=False):
    with th.no_grad():
        # perturb subject
        #print("Test size: {}".format(len(dataset.test_user_dict)))
        recall_all = 0.0
        ndcg_all = 0.0
        all_pos_item_num = 0

        for u_id, pos_item_l in test_user_dict.items():
            all_pos_item_num += len(pos_item_l)
            emb_u = embedding[u_id]
            emb_all = embedding[all_item_id_range].transpose(0, 1)
            score = th.matmul(emb_u, emb_all)
            ### mask scores of the training items as 0
            score[train_user_dict[u_id]] = 0.0
            _, rank_indices = th.sort(score, descending=True)
            if use_cuda:
                rank_indices = rank_indices.cpu().numpy()
            else:
                rank_indices = rank_indices.numpy()
            binary_rank_K = np.zeros(K, dtype=np.float32)
            for i in range(K):
                if rank_indices[i] in pos_item_l:
                    binary_rank_K[i] = 1.0
            if len(pos_item_l) > 0:
                recall_all += one_recall_at_k(binary_rank_K.tolist(), K, len(pos_item_l))
            else:
                print("Nan user dict:", u_id, test_user_dict[u_id])
            ndcg_all += one_ndcg_at_k(binary_rank_K.tolist(), K)

    recall = recall_all / len(test_user_dict)
    ndcg = ndcg_all / len(test_user_dict)

    return recall, ndcg