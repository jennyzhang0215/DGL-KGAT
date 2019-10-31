import argparse
from dataset import DataLoader
from models import Model
import torch as th
import torch.optim as optim
import metric
from utils import creat_log_id, logging_config, MetricLogger
import numpy as np
from time import time
import os
import logging

def parse_args():
    parser = argparse.ArgumentParser(description="Reproduce KGAT using DGL")
    parser.add_argument('--gpu', type=int, default=0, help='use GPU')
    parser.add_argument('--seed', type=int, default=1234, help='the random seed')
    parser.add_argument('--data_name', nargs='?', default='last-fm',  help='Choose a dataset from {yelp2018, last-fm, amazon-book}')
    ### Model parameters
    parser.add_argument('--node_dim', type=int, default=64, help='the input node dimension')
    parser.add_argument('--gnn_num_layer', type=int, default=3, help='the number of layers')
    parser.add_argument('--gnn_hidden_size', type=int, default=64, help='Output sizes of every layer')
    parser.add_argument('--dropout_rate', type=float, default=0.1, help='Keep probability w.r.t. node dropout (i.e., 1-dropout_ratio) for each deep layer. 1: no dropout.')
    parser.add_argument('--regs', type=float, default=0.0001, help='Regularization for user and item embeddings.')

    ### Training parameters
    parser.add_argument('--max_epoch', type=int, default=1000, help='train xx iterations')
    parser.add_argument("--grad_norm", type=float, default=1.0, help="norm to clip gradient to")
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate.')
    parser.add_argument('--batch_size', type=int, default=1024, help='CF batch size.')
    parser.add_argument('--batch_size_kg', type=int, default=2048, help='KG batch size.')
    parser.add_argument('--joint_train', type=bool, default=True, help='Whether to jointly-train the mode or '
                                                                        'alternative train the model ')
    parser.add_argument('--evaluate_every', type=int, default=1, help='the evaluation duration')
    parser.add_argument('--print_every', type=int, default=1000, help='the print duration')
    #parser.add_argument("--eval_batch_size", type=int, default=-1, help="batch size when evaluating")
    args = parser.parse_args()
    save_dir = "{}_kg{}_graphsage_d{}_l{}_h{}_dp{}_lr{}_bz{}_seed{}".format(args.data_name, 0,
                args.node_dim, args.gnn_num_layer, args.gnn_hidden_size,
                args.dropout_rate, args.lr, args.batch_size, args.seed)
    args.save_dir = os.path.join('log', save_dir)
    if not os.path.isdir('log'):
        os.makedirs('log')
    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)
    args.save_id = creat_log_id(args.save_dir)
    logging.info(args)
    return args


def eval(model, g, x_input, train_user_dict, eval_user_dict, item_id_range, use_cuda):
    with th.no_grad():
        all_embedding = model.gnn(g, x_input)
        recall, ndcg = metric.calc_recall_ndcg(all_embedding, train_user_dict, eval_user_dict,
                                               item_id_range, K=20, use_cuda=use_cuda)
    return recall, ndcg

def train(args):
    logging_config(folder=args.save_dir, name='log{:d}'.format(args.save_id), no_console=False)
    ### check context
    use_cuda = args.gpu >= 0 and th.cuda.is_available()
    if use_cuda:
        th.cuda.set_device(args.gpu)

    ### load data
    dataset = DataLoader(args.data_name, use_KG=False, seed=args.seed)

    ### model
    model = Model(use_KG=False, input_node_dim=args.node_dim, gnn_model="graphsage",
                  num_gnn_layers=args.gnn_num_layer, n_hidden=args.gnn_hidden_size, dropout=args.dropout_rate,
                  input_item_dim=dataset.item_dim, input_user_dim=dataset.user_dim,
                  item_num=dataset.num_items, user_num=dataset.num_users,
                  reg_lambda_gnn=args.regs)
    if use_cuda:
        model.cuda()
    logging.info(model)
    ### optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    valid_metric_logger = MetricLogger(['epoch', 'recall', 'ndcg', 'is_best'],
                                       ['%d', '%.5f', '%.5f', '%d'],
                                       os.path.join(args.save_dir, 'valid{:d}.csv'.format(args.save_id)))
    test_metric_logger = MetricLogger(['epoch', 'recall', 'ndcg'],
                                       ['%d', '%.5f', '%.5f'],
                                       os.path.join(args.save_dir, 'test{:d}.csv'.format(args.save_id)))
    best_epoch = -1
    best_recall = 0.0
    best_ndcg = 0.0
    model_state_file = 'model_state.pth'

    train_g = dataset.train_g
    if dataset.item_dim:
        item_fea = th.Tensor(dataset.item_fea)
        print("item_fea", item_fea)
    else:
        item_fea = th.arange(dataset.num_items, dtype=th.long)
    if dataset.user_dim:
        user_fea = th.Tensor(dataset.user_fea)
    else:
        user_fea = th.arange(dataset.num_users, dtype=th.long)
    if use_cuda:
        item_fea, user_fea = item_fea.cuda(), user_fea.cuda()
    x_input = [item_fea, user_fea]

    test_g = dataset.test_g
    if use_cuda:
        item_id_range = th.arange(dataset.num_items, dtype=th.long).cuda()
    else:
        item_id_range = th.arange(dataset.num_items, dtype=th.long)
    """ For initializing the edge weights """
    # A_w = th.tensor(dataset.w).view(-1, 1)
    # if use_cuda:
    #     A_w = A_w.cuda()
    # print(A_w)
    # g.edata['w'] = A_w

    for epoch in range(1, args.max_epoch+1):
        cf_sampler = dataset.CF_pair_uniform_sampler(batch_size=args.batch_size)
        total_loss = 0.0
        time1 = time()
        total_iter = dataset.num_train // args.batch_size
        print("Total iter:", total_iter)
        for iter in range(total_iter):
            user_ids, item_pos_ids, item_neg_ids =next(cf_sampler)
            model.train()
            user_ids_th = th.LongTensor(user_ids)
            item_pos_ids_th = th.LongTensor(item_pos_ids)
            item_neg_ids_th = th.LongTensor(item_neg_ids)
            if use_cuda:
                user_ids_th, item_pos_ids_th, item_neg_ids_th = \
                    user_ids_th.cuda(), item_pos_ids_th.cuda(), item_neg_ids_th.cuda()
            embedding = model.gnn(train_g, x_input)
            loss = model.get_loss(embedding, user_ids_th, item_pos_ids_th, item_neg_ids_th)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()
            if (iter % args.print_every) == 0:
                logging.info("Epoch {:04d} Iter {:04d} | Loss {:.4f} ".format(epoch, iter+1, total_loss / (iter+1)))
        logging.info('Time: {:.1f}s, loss {:.4f}'.format(time() - time1, total_loss / total_iter))

        if epoch % args.evaluate_every == 0:
            time1 = time()
            val_recall, val_ndcg = eval(model, train_g, x_input, dataset.train_user_dict, dataset.valid_user_dict,
                                        item_id_range, use_cuda)
            info = "Epoch: {}, [{:.1f}s] val recall:{:.5f}, val ndcg:{:.5f}".format(
                epoch, time() - time1, val_recall, val_ndcg)
            # save best model
            if val_recall > best_recall:
                valid_metric_logger.log(epoch=epoch, recall=val_recall, ndcg=val_ndcg, is_best=1)
                best_recall = val_recall
                #best_ndcg = val_ndcg
                best_epoch = epoch
                time1 = time()
                test_recall, test_ndcg = eval(model, test_g, x_input, dataset.train_user_dict, dataset.test_user_dict,
                                              item_id_range, use_cuda)
                test_metric_logger.log(epoch=epoch, recall=test_recall, ndcg=test_ndcg)

                info += "\t[{:.1f}s] test recall:{:.5f}, test ndcg:{:.5f}".format(time() - time1, test_recall, test_ndcg)
                #th.save({'state_dict': model.state_dict(), 'epoch': epoch}, model_state_file)
            else:
                valid_metric_logger.log(epoch=epoch, recall=val_recall, ndcg=val_ndcg, is_best=0)
            logging.info(info)

    logging.info("Final test recall:{:.5f}, test ndcg:{:.5f}, best epoch:{}".format(test_recall, test_ndcg, best_epoch))

if __name__ == '__main__':
    args = parse_args()
    train(args)