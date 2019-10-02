import argparse
from dataset import DataLoader
from models import Model
import torch as th
import torch.optim as optim
import utils
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description="Reproduce KGAT using DGL")
    parser.add_argument('--gpu', type=int, default=0, help='use GPU')
    ### Data parameters
    parser.add_argument('--data_name', nargs='?', default='last-fm',  help='Choose a dataset from {yelp2018, last-fm, amazon-book}')
    parser.add_argument('--adj_type', nargs='?', default='si', help='Specify the type of the adjacency (laplacian) matrix from {bi, si}.')

    ### Model parameters
    parser.add_argument('--entity_embed_dim', type=int, default=64, help='CF Embedding size.')
    parser.add_argument('--relation_embed_dim', type=int, default=64, help='CF Embedding size.')
    parser.add_argument('--gnn_num_layer', type=int, default=3, help='the number of layers')
    parser.add_argument('--gnn_hidden_size', type=int, default=64, help='Output sizes of every layer')
    parser.add_argument('--dropout_rate', type=float, default=0.1, help='Keep probability w.r.t. node dropout (i.e., 1-dropout_ratio) for each deep layer. 1: no dropout.')
    parser.add_argument('--regs', type=float, default=0.00001, help='Regularization for user and item embeddings.')

    ### Training parameters
    parser.add_argument('--max_epoch', type=int, default=100, help='train xx iterations')
    parser.add_argument("--grad_norm", type=float, default=1.0, help="norm to clip gradient to")
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate.')
    parser.add_argument('--batch_size', type=int, default=1024, help='CF batch size.')
    parser.add_argument('--batch_size_kg', type=int, default=2048, help='KG batch size.')
    parser.add_argument('--evaluate_every', type=int, default=1, help='the evaluation duration')
    parser.add_argument("--eval_batch_size", type=int, default=-1, help="batch size when evaluating")
    args = parser.parse_args()
    print(args)
    return args


def train(args):
    ### check context
    use_cuda = args.gpu >= 0 and th.cuda.is_available()
    if use_cuda:
        th.cuda.set_device(args.gpu)

    ### load data
    dataset = DataLoader(args.data_name, num_neighbor_hop=args.gnn_num_layer)
    print("Dataset prepared ...")
    ### model
    # n_entities, n_relations, entity_dim, relation_dim, num_gnn_layers, n_hidden,
    # dropout, reg_lambda_kg=0.01, reg_lambda_gnn=0.01,
    model = Model(n_entities=dataset.num_all_entities, n_relations=dataset.num_all_relations,
                  entity_dim=args.entity_embed_dim, relation_dim=args.relation_embed_dim,
                  num_gnn_layers=args.gnn_num_layer, n_hidden=args.gnn_hidden_size, dropout=args.dropout_rate,
                  reg_lambda_kg=args.regs, reg_lambda_gnn=args.regs)
    if use_cuda:
        model = model.cuda()
    print(model)
    ### optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    print("Start training ...")
    best_recall = 0.0
    model_state_file = 'model_state.pth'

    for epoch in range(1, args.max_epoch+1):
        """
        ### train kg first
        kg_sampler = dataset.KG_sampler(batch_size=args.batch_size_kg)
        iter = 0
        for h, r, pos_t, neg_t in kg_sampler:
            iter += 1
            model.train()
            h_th = th.LongTensor(h)
            r_th = th.LongTensor(r)
            pos_t_th = th.LongTensor(pos_t)
            neg_t_th = th.LongTensor(neg_t)
            if use_cuda:
                h_th, r_th, pos_t_th, neg_t_th = h_th.cuda(), r_th.cuda(), pos_t_th.cuda(), neg_t_th.cuda()
            loss = model.transR(h_th, r_th, pos_t_th, neg_t_th)
            # print("loss", loss)
            loss.backward()
            # print("start computing gradient ...")
            optimizer.step()
            optimizer.zero_grad()
            if (iter % 100) == 0:
               print("Epoch {:04d}, Iter {:04d} | Loss {:.4f} ".format(epoch, iter, loss.item()))
        """
        ### Then train GNN
        """
        g, all_etype = dataset.generate_whole_g()
        nid_th = th.arange(dataset.num_all_entities)
        etype_th = th.LongTensor(all_etype)
        if use_cuda:
            nid_th, etype_th, = nid_th.cuda(), etype_th.cuda()
        model.train()
        embedding = model.gnn(g, nid_th, etype_th)
        print("\t\tembedding", embedding.shape)
        cf_sampler = dataset.CF_sampler(batch_size=args.batch_size, segment='train', sequential=True)
        loss_sum = None
        train_pairs = 0
        for user_ids, item_pos_ids, item_neg_ids, batch_size in cf_sampler:
            user_ids_th = th.LongTensor(user_ids)
            item_pos_ids_th = th.LongTensor(item_pos_ids)
            item_neg_ids_th = th.LongTensor(item_neg_ids)
            if use_cuda:
                user_ids_th, item_pos_ids_th, item_neg_ids_th = \
                    user_ids_th.cuda(), item_pos_ids_th.cuda(), item_neg_ids_th.cuda()

            loss = model.get_loss(embedding, user_ids_th, item_pos_ids_th, item_neg_ids_th)
            train_pairs += 1
            if loss_sum is None:
                loss_sum = loss
            else:
                loss_sum += loss
        loss = loss_sum / train_pairs
        """
        """
        cf_sampler = dataset.CF_batchwise_sampler(batch_size=args.batch_size)
        iter = 0
        for user_ids, item_pos_ids, item_neg_ids, g, uniq_v, etype in cf_sampler:
            iter += 1
            user_ids_th = th.LongTensor(user_ids)
            item_pos_ids_th = th.LongTensor(item_pos_ids)
            item_neg_ids_th = th.LongTensor(item_neg_ids)
            nid_th = th.LongTensor(uniq_v)
            etype_th = th.LongTensor(etype)
            if use_cuda:
                user_ids_th, item_pos_ids_th, item_neg_ids_th, nid_th, etype_th = \
                    user_ids_th.cuda(), item_pos_ids_th.cuda(), item_neg_ids_th.cuda(), nid_th.cuda(), etype_th.cuda()
            model.train()
            embedding = model.gnn(g, nid_th, etype_th)
            loss = model.get_loss(embedding, user_ids_th, item_pos_ids_th, item_neg_ids_th)
            loss.backward()
            #th.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)  # clip gradients
            #print("start computing gradient ...")
            optimizer.step()
            optimizer.zero_grad()
            if (iter % 10) == 0:
                print("Epoch {:04d}  Iter: {:04d} Loss {:.4f} ".format(epoch, iter, loss.item()))
        """
        model.train()
        g, all_etype = dataset.generate_whole_g()
        nid_th = th.arange(dataset.num_all_entities)
        etype_th = th.LongTensor(all_etype)
        cf_sampler = dataset.CF_pair_sampler(batch_size=args.batch_size)
        iter = 0
        if use_cuda:
            nid_th, etype_th = nid_th.cuda(), etype_th.cuda()
        g.ndata['id'] = nid_th
        g.edata['type'] = etype_th
        with th.no_grad():
            att_w = model.compute_attention(g)
        g.edata['w'] = att_w
        print(g)
        for user_ids, item_pos_ids, item_neg_ids in cf_sampler:
            iter += 1
            user_ids_th = th.LongTensor(user_ids)
            item_pos_ids_th = th.LongTensor(item_pos_ids)
            item_neg_ids_th = th.LongTensor(item_neg_ids)
            if use_cuda:
                user_ids_th, item_pos_ids_th, item_neg_ids_th = \
                    user_ids_th.cuda(), item_pos_ids_th.cuda(), item_neg_ids_th.cuda()
            embedding = model.gnn(g)
            loss = model.get_loss(embedding, user_ids_th, item_pos_ids_th, item_neg_ids_th)
            print("embedding", embedding.shape, embedding)
            print("Before backward ...", g)
            print(model.parameters())
            loss.backward()
            # th.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)  # clip gradients
            # print("start computing gradient ...")
            optimizer.step()
            optimizer.zero_grad()
            if (iter % 10) == 0:
               print("Epoch {:04d}  Iter: {:04d} Loss {:.4f} ".format(epoch, iter, loss.item()))

        if epoch % args.evaluate_every == 0:
            with th.no_grad():
                model.eval()
                g, all_etype = dataset.generate_whole_g()
                nid_th = th.arange(dataset.num_all_entities)
                etype_th = th.LongTensor(all_etype)
                if use_cuda:
                    nid_th, etype_th, = nid_th.cuda(), etype_th.cuda()

                all_embedding = model.gnn(g, nid_th, etype_th)
                if use_cuda:
                    item_id_range = th.arange(dataset.num_items).cuda()
                else:
                    item_id_range = th.arange(dataset.num_items)
                recall, ndcg = utils.calc_recall_ndcg(all_embedding, dataset, item_id_range, K=20, use_cuda=use_cuda)
                print("Epoch: {}, Test recall:{:.6%}, ndcg:{:.6%}".format(epoch, recall, ndcg))
            # save best model
            # if recall > best_recall:
            #     best_recall = recall
            #     th.save({'state_dict': model.state_dict(), 'epoch': epoch}, model_state_file)
            if use_cuda:
                model.cuda()

if __name__ == '__main__':
    args = parse_args()
    train(args)