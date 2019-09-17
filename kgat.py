import argparse

from dataset import DataLoader
from models import CFModel

import torch as th
import torch.optim as optim

def parse_args():
    parser = argparse.ArgumentParser(description="Reproduce KGAT using DGL")
    parser.add_argument('--gpu', type=int, default=-1, help='use GPU')
    ### Data parameters
    parser.add_argument('--data_name', nargs='?', default='yelp2018',  help='Choose a dataset from {yelp2018, last-fm, amazon-book}')
    parser.add_argument('--pretrain', type=int, default=0, help='0: No pretrain, -1: Pretrain with the learned embeddings, 1:Pretrain with stored models.')
    parser.add_argument('--adj_type', nargs='?', default='si', help='Specify the type of the adjacency (laplacian) matrix from {bi, si}.')
    parser.add_argument('--pbg', action='store_true', help='use pbg negative sampling.')

    ### Model parameters
    parser.add_argument('--model_type', nargs='?', default='kgat', help='Specify a loss type from {kgat, bprmf, fm, nfm, cke, cfkg}.')
    parser.add_argument('--use_kge', type=bool, default=True, help='whether using knowledge graph embedding')
    parser.add_argument('--kge_size', type=int, default=64, help='KG Embedding size.')
    parser.add_argument('--embed_size', type=int, default=64, help='CF Embedding size.')
    parser.add_argument('--layer_size', nargs='?', default='[64]', help='Output sizes of every layer')
    parser.add_argument('--Ks', nargs='?', default='[20, 40, 60, 80, 100]', help='Output sizes of every layer')
    parser.add_argument('--node_dropout', nargs='?', default='[0.1]', help='Keep probability w.r.t. node dropout (i.e., 1-dropout_ratio) for each deep layer. 1: no dropout.')
    parser.add_argument('--mess_dropout', nargs='?', default='[0.1]', help='Keep probability w.r.t. message dropout (i.e., 1-dropout_ratio) for each deep layer. 1: no dropout.')
    parser.add_argument('--alg_type', nargs='?', default='ngcf', help='Specify the type of the graph convolutional layer from {bi, gcn, graphsage}.')
    parser.add_argument('--use_att', type=bool, default=True, help='whether using attention mechanism')
    parser.add_argument('--regs', nargs='?', default='[1e-5,1e-5,1e-2]', help='Regularization for user and item embeddings.')
    parser.add_argument('--adj_uni_type', nargs='?', default='sum', help='Specify a loss type (uni, sum).')

    ### Training parameters
    parser.add_argument('--max_iter', type=int, default=80000, help='train xx iterations')
    parser.add_argument('--batch_size', type=int, default=1024, help='CF batch size.')
    parser.add_argument('--batch_size_kg', type=int, default=2048, help='KG batch size.')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate.')
    args = parser.parse_args()

    return args


def train(args):
    ### check context
    use_cuda = args.gpu >= 0 and th.cuda.is_available()
    if use_cuda:
        th.cuda.set_device(args.gpu)

    ### load data
    dataset = DataLoader(args.data_name)
    graph = dataset.g
    th_e_type = th.from_numpy(dataset.etype)
    if use_cuda:
        th_e_type = th_e_type.cuda()
    graph.edata['type'] = th_e_type
    cf_sampler = dataset.CF_sampler(segment='train')

    ### model
    cf_model =CFModel(n_entities=dataset.num_all_entities, n_relations=dataset.num_all_relations,
                      entity_dim=args.embed_size, num_gnn_layers=2, n_hidden=64, dropout=0.1, reg_lambda=0.01)
    if use_cuda:
        cf_model = cf_model.cuda()

    ### optimizer
    optimizer = optim.Adam(cf_model.parameters(), lr=args.lr)

    for iter in range(1, args.max_iter+1):
        cf_model.train()
        user_ids, item_pos_ids, item_neg_ids = next(cf_sampler)
        user_ids_th = th.from_numpy(user_ids)
        item_pos_ids_th = th.from_numpy(item_pos_ids)
        item_neg_ids_th = th.from_numpy(item_neg_ids)
        if use_cuda:
            user_ids_th, item_pos_ids_th, item_neg_ids_th = \
                user_ids_th.cuda(), item_pos_ids_th.cuda(), item_neg_ids_th.cuda()

        loss = cf_model(graph, user_ids_th, item_pos_ids_th, item_neg_ids_th)
        loss.backward()
        optimizer.step()
        print("Iter {:04d} | Loss {:.4f} ".format(iter, loss.item()))
        optimizer.zero_grad()


if __name__ == '__main__':
    args = parse_args()
    train(args)