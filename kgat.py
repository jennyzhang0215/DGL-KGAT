import argparse

from dataset import DataLoader
from models import KGEModel, CFModel
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
    parser.add_argument('--use_kge', type=bool, default=True, help='whether using knowledge graph embedding')
    parser.add_argument('--kge_size', type=int, default=5, help='KG Embedding size.')
    parser.add_argument('--embed_size', type=int, default=4, help='CF Embedding size.')
    parser.add_argument('--gnn_num_layer', type=int, default=2, help='the number of layers')
    parser.add_argument('--gnn_hidden_size', type=int, default=8, help='Output sizes of every layer')
    parser.add_argument('--dropout_rate', type=float, default=0.1, help='Keep probability w.r.t. node dropout (i.e., 1-dropout_ratio) for each deep layer. 1: no dropout.')
    parser.add_argument('--regs', nargs='?', default='[1e-5,1e-5,1e-2]', help='Regularization for user and item embeddings.')

    ### Training parameters
    parser.add_argument('--train_kge', type=bool, default=False, help='Just for testing. Train KGE model')
    parser.add_argument('--max_epoch', type=int, default=10, help='train xx iterations')
    parser.add_argument("--grad_norm", type=float, default=1.0, help="norm to clip gradient to")
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate.')
    parser.add_argument('--batch_size', type=int, default=1024, help='CF batch size.')
    parser.add_argument('--batch_size_kg', type=int, default=4096, help='KG batch size.')
    parser.add_argument('--evaluate_every', type=int, default=1, help='the evaluation duration')
    parser.add_argument("--eval-batch-size", type=int, default=1000, help="batch size when evaluating")
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
    th_e_type = th.LongTensor(dataset.etype)
    th_n_id = th.arange(graph.number_of_nodes(), dtype=th.int32).long()
    if use_cuda:
        th_e_type = th_e_type.cuda()
        th_n_id = th_n_id.cuda()
    print("Dataset prepared ...")
    ### model
    if args.train_kge:
        model = KGEModel(n_entities=dataset.num_KG_entities, n_relations=dataset.num_KG_relations,
                             entity_dim=args.embed_size, relation_dim=args.embed_size, reg_lambda=0.01)
    else:
        model = CFModel(n_entities=dataset.num_all_entities, n_relations=dataset.num_all_relations,
                           entity_dim=args.embed_size, relation_dim=args.embed_size, num_gnn_layers=args.gnn_num_layer,
                           n_hidden=args.gnn_hidden_size, dropout=args.dropout_rate, reg_lambda=0.01)
    if use_cuda:
        model = model.cuda()
    ### optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    print("Start training ...")
    best_hit_score = 0.0
    model_state_file = 'model_state.pth'

    for epoch in range(1, args.max_epoch+1):
        if args.train_kge:
            kg_sampler = dataset.KG_sampler(batch_size=args.batch_size_kg, sequential=True, segment='train')
            iter = 0
            for h, r, pos_t, neg_t in kg_sampler:
                iter +=1
                model.train()
                h_th = th.LongTensor(h)
                r_th = th.LongTensor(r)
                pos_t_th = th.LongTensor(pos_t)
                neg_t_th = th.LongTensor(neg_t)
                if use_cuda:
                    h_th, r_th, pos_t_th, neg_t_th = h_th.cuda(), r_th.cuda(), pos_t_th.cuda(), neg_t_th.cuda()
                loss = model(h_th, r_th, pos_t_th, neg_t_th)
                # print("loss", loss)
                loss.backward()
                # print("start computing gradient ...")
                optimizer.step()
                print("Epoch {:04d}, Iter {:04d} | Loss {:.4f} ".format(epoch, iter, loss.item()))
                optimizer.zero_grad()
        else:
            model.train()
            ### sample graph and sample user-item pairs
            cf_sampler = dataset.CF_sampler(segment='train')
            user_ids, item_pos_ids, item_neg_ids = next(cf_sampler)
            user_ids_th = th.LongTensor(user_ids)
            item_pos_ids_th = th.LongTensor(item_pos_ids)
            item_neg_ids_th = th.LongTensor(item_neg_ids)
            if use_cuda:
                user_ids_th, item_pos_ids_th, item_neg_ids_th = \
                    user_ids_th.cuda(), item_pos_ids_th.cuda(), item_neg_ids_th.cuda()

            embedding = model(graph, th_n_id, th_e_type)
            print("embedding", embedding)
            loss = model.get_loss(embedding, user_ids_th, item_pos_ids_th, item_neg_ids_th)
            #print("loss", loss)
            loss.backward()
            th.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)  # clip gradients

            #print("start computing gradient ...")
            optimizer.step()
            print("Epoch {:04d} | Loss {:.4f} ".format(epoch, loss.item()))
            optimizer.zero_grad()


        if epoch % args.evaluate_every == 0:
            if use_cuda:
                model.cpu()
                th_n_id = th_n_id.cpu()
                th_e_type = th_e_type.cpu()
            model.eval()
            cf_sampler = dataset.CF_sampler(segment='test')
            test_hit_l = []
            for test_user_ids, test_item_ids, _ in cf_sampler:
                test_user_ids_th = th.LongTensor(test_user_ids)
                test_item_ids_th = th.LongTensor(test_item_ids)
                item_id_range = th.arange(dataset.num_items)
                # if use_cuda:
                #     test_user_ids_th, test_item_ids_th, item_id_range = \
                #         test_user_ids_th.cuda(), test_item_ids_th.cuda(), item_id_range.cuda()
                embedding = model(graph, th_n_id, th_e_type)
                hit_rate = utils.calc_hit(embedding, (test_user_ids_th, test_item_ids_th), item_id_range,
                                      K=20, eval_bz=args.eval_batch_size)
                print("hit_rate", hit_rate)
                test_hit_l.append(hit_rate)
            hit_score = th.mean(th.cat(test_hit_l).float()).items()
            # save best model
            if hit_score > best_hit_score:
                best_hit_score = hit_score
                th.save({'state_dict': model.state_dict(), 'epoch': epoch}, model_state_file)
            if use_cuda:
                model.cuda()
                th_n_id = th_n_id.cuda()
                th_e_type = th_e_type.cuda()

if __name__ == '__main__':
    args = parse_args()
    train(args)