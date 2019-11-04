import argparse
from dataset import DataLoader
import torch as th
import metric
from utils import creat_log_id, logging_config
import numpy as np
import os
import logging

def parse_args():
    parser = argparse.ArgumentParser(description="Reproduce KGAT using DGL")
    parser.add_argument('--gpu', type=int, default=0, help='use GPU')
    parser.add_argument('--data_name', nargs='?', default='last-fm',  help='Choose a dataset from {yelp2018, last-fm, amazon-book}')
    parser.add_argument('--use_pretrain', type=bool, default=True, help='whether to use pretrain embeddings or not')
    parser.add_argument('--seed', type=int, default=1234, help='the random seed')
    args = parser.parse_args()

    save_dir = "{}_bprmf_test".format(args.data_name)
    args.save_dir = os.path.join('log', save_dir)
    if not os.path.isdir('log'):
        os.makedirs('log')
    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)
    args.save_id = creat_log_id(args.save_dir)
    return args

def eval(all_embedding, train_user_dict, eval_user_dict, item_id_range, use_cuda):
    with th.no_grad():
        recall, ndcg = metric.calc_recall_ndcg(all_embedding, train_user_dict, eval_user_dict,
                                               item_id_range, K=20, use_cuda=use_cuda)
    return recall, ndcg

if __name__ == '__main__':
    args = parse_args()
    logging_config(folder=args.save_dir, name='log{:d}'.format(args.save_id), no_console=False)
    ### check context
    use_cuda = args.gpu >= 0 and th.cuda.is_available()
    if use_cuda:
        th.cuda.set_device(args.gpu)

    dataset = DataLoader(args.data_name, use_KG=False, use_pretrain=args.use_pretrain, seed=args.seed)
    assert dataset.num_items + dataset.num_users == dataset.num_all_nodes

    item_id_range = th.arange(dataset.num_items, dtype=th.long)
    embeds = th.tensor(np.vstack((dataset.item_pre_embed, dataset.user_pre_embed)))
    if use_cuda:
        item_id_range, embeds = item_id_range.cuda(), embeds.cuda()
    print("Start validation ...")
    val_recall, val_ndcg = eval(embeds, dataset.train_user_dict, dataset.valid_user_dict, item_id_range, use_cuda)
    print("Start testing ...")
    test_recall, test_ndcg = eval(embeds, dataset.train_user_dict, dataset.test_user_dict, item_id_range, use_cuda)

    logging.info("Test recall: {:.5f}, ndcg:{:.5f}\t\tvalid recall: {:.5f}, ndcg:{:.5f}".format(
        test_recall, test_ndcg, val_recall, val_ndcg))