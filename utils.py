import numpy as np
import torch as th

def sort_and_rank(score, target):
    _, indices = th.sort(score, dim=1, descending=True)
    indices = th.nonzero(indices == target.view(-1, 1))
    indices = indices[:, 1].view(-1)
    return indices

def perturb_and_get_rank(embedding, user, item, all_item_id_range, batch_size=100):
    """ Perturb one element in the triplets
    """
    test_size = user.nelement()
    n_batch = (test_size + batch_size - 1) // batch_size
    ranks = []
    for idx in range(n_batch):
        print("batch {} / {}".format(idx, n_batch))
        batch_start = idx * batch_size
        batch_end = min(test_size, (idx + 1) * batch_size)
        batch_u = user[batch_start: batch_end]
        emb_u = embedding[batch_u]
        emb_u = emb_u.transpose(0, 1).unsqueeze(2) # size: D x E x 1
        emb_all = embedding[all_item_id_range].transpose(0, 1).unsqueeze(1) # size: D x 1 x V
        # out-prod and reduce sum
        out_prod = th.bmm(emb_u, emb_all) # size D x E x V
        score = th.sum(out_prod, dim=0) # size E x V
        score = th.sigmoid(score)
        target = item[batch_start: batch_end]
        ranks.append(sort_and_rank(score, target).detach())
    return th.cat(ranks)

# return Hits @ (K)
def calc_hit(embedding, test_pairs, all_item_id_range, K, eval_bz=100):
    with th.no_grad():
        # perturb subject
        ranks = perturb_and_get_rank(embedding=embedding, user=test_pairs[0], item=test_pairs[1],
                                     all_item_id_range=all_item_id_range, batch_size=eval_bz)
        ranks += 1 # change to 1-indexed

        avg_count = th.mean((ranks <= K).float())
        return avg_count.item()


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