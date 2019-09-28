import numpy as np
import torch as th
import heapq


def recall_at_k(r, k, all_pos_num):
    r = np.asfarray(r)[:k]
    return np.sum(r) / all_pos_num
def dcg_at_k(r, k):
    """Score is discounted cumulative gain (dcg)
    Relevance is positive real values.  Can use binary
    as the previous methods.
    Returns:
        Discounted cumulative gain
    """
    r = np.asfarray(r)[:k]
    assert r.size > 0
    return np.sum(r / np.log2(np.arange(2, k + 2)), axis=1)
def ndcg_at_k(r, k):
    """Score is normalized discounted cumulative gain (ndcg)
    Relevance is positive real values.  Can use binary
    as the previous methods.
    Returns:
        Normalized discounted cumulative gain
    """
    dcg_max = dcg_at_k(np.flip(np.sort(r), axis=1), k)
    dcg = dcg_at_k(r, k)
    dcg_max[dcg_max==0] = np.inf
    return np.mean(dcg / dcg_max)


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


def calc_recall_ndcg(embedding, dataset, all_item_id_range, K, use_cuda):
    with th.no_grad():
        # perturb subject
        #print("Test size: {}".format(len(dataset.test_user_dict)))
        ranks = []
        pos_item_num = []
        recall_all = 0.0
        ndcg_all = 0.0
        for u_id, pos_item_l in dataset.test_user_dict.items():
            pos_item_num.append(len(pos_item_l))
            emb_u = embedding[u_id]
            #print("emb_u", emb_u.shape, emb_u)
            emb_all = embedding[all_item_id_range].transpose(0, 1)
            #print("emb_all", emb_all.shape, emb_all)
            score = th.matmul(emb_u, emb_all)
            #print("score", score.shape, score)
            ### mask scores of the training items as 0
            score[dataset.train_user_dict[u_id]] = 0.0
            _, rank_indices = th.sort(score, descending=True)
            if use_cuda:
                rank_indices = rank_indices.cpu().numpy()
            else:
                rank_indices = rank_indices.numpy()
            binary_rank_K = np.zeros(K)
            for i in range(K):
                if rank_indices[i] in pos_item_l:
                    binary_rank_K[i] = 1
            ranks.append(binary_rank_K)
            recall_all += one_recall_at_k(binary_rank_K.tolist(), K, len(pos_item_l))
            ndcg_all += one_ndcg_at_k(binary_rank_K.tolist(), K)


    ranks = np.vstack(ranks)
    ### the output is the sum
    recall = recall_at_k(ranks, K, dataset.num_test)
    one_by_one_recall = recall_all / len(dataset.test_user_dict)
    ndcg = ndcg_at_k(ranks, K)
    one_by_one_ndcg = ndcg_all / len(dataset.test_user_dict)
    print("Comparision: recall (all v.s. one by one) {} v.s. {}".format(recall_all, one_by_one_recall))
    print("\tndcg (all v.s. one by one) {} v.s. {}".format(ndcg_all, one_by_one_ndcg))

    return recall, ndcg


#
#
# def recall_at_k(r, k, all_pos_num):
#     r = np.asfarray(r)[:k]
#     return np.sum(r) / all_pos_num
#
# def dcg_at_k(r, k, method=1):
#     """Score is discounted cumulative gain (dcg)
#     Relevance is positive real values.  Can use binary
#     as the previous methods.
#     Returns:
#         Discounted cumulative gain
#     """
#     r = np.asfarray(r)[:k]
#     if r.size:
#         if method == 0:
#             return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
#         elif method == 1:
#             return np.sum(r / np.log2(np.arange(2, r.size + 2)))
#         else:
#             raise ValueError('method must be 0 or 1.')
#     return 0.
#
# def ndcg_at_k(r, k, method=1):
#     """Score is normalized discounted cumulative gain (ndcg)
#     Relevance is positive real values.  Can use binary
#     as the previous methods.
#     Returns:
#         Normalized discounted cumulative gain
#     """
#     dcg_max = dcg_at_k(sorted(r, reverse=True), k, method)
#     if not dcg_max:
#         return 0.
#     return dcg_at_k(r, k, method) / dcg_max