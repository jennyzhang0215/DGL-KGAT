import numpy as np
import os
import dgl
import pandas as pd
import collections
import random as rd
from time import time
import scipy.sparse as sp
import json
import torch as th

class L_DataLoader(object):
    def __init__(self, data_name, num_neighbor_hop=3, adj_type='si', seed=1234):
        self._adj_type = adj_type
        data_dir =  os.path.realpath(os.path.join(os.path.abspath(__file__), '..', "datasets", data_name))
        train_file = os.path.join(data_dir, "train.txt")
        test_file = os.path.join(data_dir, "test.txt")
        kg_file = os.path.join(data_dir, "kg_final.txt")

        train_data, train_user_dict = self._load_ratings(train_file)
        test_data, test_user_dict = self._load_ratings(test_file)
        self._n_users = max(max(train_data[:, 0]), max(test_data[:, 0])) + 1
        assert self._n_users == np.unique(np.concatenate((train_data[:, 0], test_data[:, 0]))).size
        self._n_items = max(max(train_data[:, 1]), max(test_data[:, 1])) + 1
        assert self._n_items ==  np.unique(np.concatenate((train_data[:, 1], test_data[:, 1]))).size
        self._n_train = len(train_data)
        self._n_test = len(test_data)

        kg_np, kg_dict, relation_dict = self._load_kg(kg_file)
        self._n_KG_relations = max(kg_np[:, 1]) + 1
        self._n_KG_entities = max(max(kg_np[:, 0]), max(kg_np[:, 2])) + 1
        self._n_KG_triples = len(kg_np)

        ## remapping user ids to new ids which is after entity ids
        self.user_mapping = {i: i+self.num_KG_entities for i in range(self.num_users)}
        train_data[:, 0] = train_data[:, 0] + self.num_KG_entities
        test_data[:, 0] = test_data[:, 0] + self.num_KG_entities

        self.train_user_dict = {}
        for k,v in train_user_dict.items():
            self.train_user_dict[k+self.num_KG_entities] = np.unique(v)
        self.test_user_dict = {}
        for k,v in test_user_dict.items():
            self.test_user_dict[k+self.num_KG_entities] = np.unique(v)

        ## merge KG and UI-pairs
        adj_list, adj_r_list = self._get_relational_adj_list(train_data, relation_dict)
        lap_list = self._get_relational_lap_list(adj_list)
        all_h_list, all_r_list, all_t_list, all_v_list = self._get_all_kg_data(lap_list, adj_r_list)

        self.all_triplet_np = np.zeros((len(all_h_list), 3), dtype=np.int32)
        self.all_triplet_np[:, 0] = all_h_list
        self.all_triplet_np[:, 1] = all_r_list
        self.all_triplet_np[:, 2] = all_t_list
        self.all_triplet_np = self.all_triplet_np.astype(np.int32)
        self.w = all_v_list

        self.all_kg_dict = self._get_all_kg_dict()
        print("The whole graph: #entities {}, #relations {}, #triplets {}".format(
            self.num_all_entities, self.num_all_relations, self.num_all_triplets))
        print("The KG: #entities {}, #relations {}, #triplets {}".format(
            self.num_KG_entities, self.num_KG_relations, self.num_KG_triples))
        print("The user-item pairs: #users {}, #items {}, #train pairs {}, #test pairs {}".format(
            self.num_users, self.num_items, self.num_train, self.num_test))

    @property
    def num_all_entities(self):
        return self._n_KG_entities + self._n_users
    @property
    def num_all_relations(self):
        return (self._n_KG_relations+1) * 2
    @property
    def num_all_triplets(self):
        return self.all_triplet_np.shape[0]
    @property
    def num_KG_entities(self):
        return self._n_KG_entities
    @property
    def num_KG_relations(self):
        return self._n_KG_relations
    @property
    def num_KG_triples(self):
        return self._n_KG_triples
    @property
    def num_train(self):
        return self._n_train
    @property
    def num_test(self):
        return self._n_test
    @property
    def num_users(self):
        return self._n_users
    @property
    def num_items(self):
        return self._n_items


    def generate_whole_g(self):
        g = dgl.DGLGraph()
        g.add_nodes(self.num_all_entities)
        g.add_edges(self.all_triplet_np[:, 2], self.all_triplet_np[:, 0])
        all_etype = self.all_triplet_np[:, 1]
        return g, all_etype

    def _get_all_kg_dict(self):
        all_kg_dict = collections.defaultdict(list)
        for h, r, t in self.all_triplet_np:
            if h in all_kg_dict.keys():
                all_kg_dict[h].append((t, r))
            else:
                all_kg_dict[h] = [(t, r)]
        return all_kg_dict


    ### KG sampler
    def _sample_pos_triples_for_h(self, h):
        t, r = rd.choice(self.all_kg_dict[h])
        return r, t

    def _sample_neg_triples_for_h(self, h, r):
        while True:
            t = np.random.randint(low=0, high=self.num_all_entities, size=1)[0]
            if (t, r) not in self.all_kg_dict[h]:
                return t

    def KG_sampler(self, batch_size):
        exist_heads = list(self.all_kg_dict.keys())
        n_batch = self.num_all_triplets // batch_size + 1
        # print("#num_all_triplets", self.num_all_triplets, "batch_size", batch_size, "n_batch", n_batch,
        #      "#exist_heads", len(exist_heads))
        i = 0
        while i < n_batch:
            i += 1
            if batch_size <= len(exist_heads):
                heads = rd.sample(exist_heads, batch_size)
            else:
                heads = [rd.choice(exist_heads) for _ in range(batch_size)]
            pos_r_batch, pos_t_batch, neg_t_batch = [], [], []
            for h in heads:
                pos_rs, pos_ts = self._sample_pos_triples_for_h(h)
                pos_r_batch.append(pos_rs)
                pos_t_batch.append(pos_ts)
                neg_ts = self._sample_neg_triples_for_h(h, pos_rs)
                neg_t_batch.append(neg_ts)

            yield heads, pos_r_batch, pos_t_batch, neg_t_batch

    ### for GNN sampler
    def _sample_pos_items_for_u(self, u):
        return rd.choice(self.train_user_dict[u])

    def _sample_neg_items_for_u(self, u):
        while True:
            neg_i_id = np.random.randint(low=0, high=self.num_items, size=1)[0]
            if neg_i_id not in self.train_user_dict[u]:
                return neg_i_id

    def CF_pair_sampler(self, batch_size):
        exist_users = list(self.train_user_dict.keys())
        if batch_size < 0:
            batch_size = self.num_train
            n_batch = 1
        elif batch_size > self.num_train:
            batch_size = min(batch_size, self.num_train)
            n_batch = 1
        else:
            n_batch = self.num_train // batch_size + 1
        # print("#train_pair", self.num_train, "batch_size", batch_size, "n_batch", n_batch, "#exist_user", len(exist_users))

        i = 0
        while i < n_batch:
            i += 1
            if batch_size <= self.num_users:
                ## test1
                users = rd.sample(exist_users, batch_size)
            else:
                users = [rd.choice(exist_users) for _ in range(batch_size)]

            pos_items, neg_items = [], []
            for u in users:
                pos_items.append(self._sample_pos_items_for_u(u))
                neg_items.append(self._sample_neg_items_for_u(u))
            yield users, pos_items, neg_items

        # reading train & test interaction data.
    def _load_ratings(self, file_name):
        user_dict = dict()
        inter_mat = list()

        lines = open(file_name, 'r').readlines()
        for l in lines:
            tmps = l.strip()
            inters = [int(i) for i in tmps.split(' ')]

            u_id, pos_ids = inters[0], inters[1:]
            pos_ids = list(set(pos_ids))

            for i_id in pos_ids:
                inter_mat.append([u_id, i_id])

            if len(pos_ids) > 0:
                user_dict[u_id] = pos_ids
        return np.array(inter_mat), user_dict

    def _load_kg(self, file_name):
        def _construct_kg(kg_np):
            kg = collections.defaultdict(list)
            rd = collections.defaultdict(list)

            for head, relation, tail in kg_np:
                kg[head].append((tail, relation))
                rd[relation].append((head, tail))
            return kg, rd

        kg_np = np.loadtxt(file_name, dtype=np.int32)
        kg_np = np.unique(kg_np, axis=0)

        kg_dict, relation_dict = _construct_kg(kg_np)

        return kg_np, kg_dict, relation_dict

    def _get_relational_adj_list(self, train_data, relation_dict):
        t1 = time()
        adj_mat_list = []
        adj_r_list = []

        def _np_mat2sp_adj(np_mat, row_pre, col_pre):
            n_all = self.num_users + self.num_KG_entities
            # single-direction
            a_rows = np_mat[:, 0] + row_pre
            a_cols = np_mat[:, 1] + col_pre
            a_vals = [1.] * len(a_rows)

            b_rows = a_cols
            b_cols = a_rows
            b_vals = [1.] * len(b_rows)

            a_adj = sp.coo_matrix((a_vals, (a_rows, a_cols)), shape=(n_all, n_all))
            b_adj = sp.coo_matrix((b_vals, (b_rows, b_cols)), shape=(n_all, n_all))

            return a_adj, b_adj
        R, R_inv = _np_mat2sp_adj(train_data, row_pre=0, col_pre=0)
        adj_mat_list.append(R)
        adj_r_list.append(0)

        adj_mat_list.append(R_inv)
        adj_r_list.append(self.num_KG_relations + 1)
        print('\tconvert ratings into adj mat done.')

        for r_id in relation_dict.keys():
            K, K_inv = _np_mat2sp_adj(np.array(relation_dict[r_id]), row_pre=0, col_pre=0)
            adj_mat_list.append(K)
            adj_r_list.append(r_id + 1)

            adj_mat_list.append(K_inv)
            adj_r_list.append(r_id + 2 + self.num_KG_relations)
        print('\tconvert %d relational triples into adj mat done. @%.4fs' %(len(adj_mat_list), time()-t1))

        print('\tadj relation list is', adj_r_list, np.unique(adj_r_list))
        return adj_mat_list, adj_r_list


    def _get_relational_lap_list(self, adj_list):
        ### TODO have some problems here
        def _bi_norm_lap(adj):
            #print("adj", adj.toarray())
            rowsum = np.array(adj.sum(1))

            d_inv_sqrt = np.power(rowsum, -0.5).flatten()
            d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
            d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

            bi_lap = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
            return bi_lap.tocoo()

        def _si_norm_lap(adj):
            rowsum = np.array(adj.sum(1))

            d_inv = np.power(rowsum, -1).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)

            norm_adj = d_mat_inv.dot(adj)
            return norm_adj.tocoo()

        if self._adj_type == 'bi':
            lap_list = [_bi_norm_lap(adj) for adj in adj_list]
            print('\tgenerate bi-normalized adjacency matrix.')
        else:
            lap_list = [_si_norm_lap(adj) for adj in adj_list]
            print('\tgenerate si-normalized adjacency matrix.')
        return lap_list

    def _get_all_kg_data(self, lap_list, adj_r_list):
        all_h_list, all_t_list, all_r_list = [], [], []
        all_v_list = []

        for l_id, lap in enumerate(lap_list):
            all_h_list += list(lap.row)
            all_t_list += list(lap.col)
            all_v_list += list(lap.data)
            all_r_list += [adj_r_list[l_id]] * len(lap.row)
        return all_h_list, all_r_list, all_t_list, all_v_list

class DataLoader(object):
    def __init__(self, data_name, use_KG, seed=1234):
        print("\n{}->".format(data_name))
        self._data_name = data_name
        self._use_KG = use_KG
        self._rng = np.random.RandomState(seed=seed)
        data_dir = os.path.realpath(os.path.join(os.path.abspath(__file__), '..', "datasets", data_name))

        train_file = os.path.join(data_dir, "train1.txt")
        train_pairs, train_user_dict = self.load_train_interaction(train_file)
        self.train_user_ids = np.unique(train_pairs[0])  ### requires to remap later
        self.item_ids = np.unique(train_pairs[1])
        self.num_users = self.train_user_ids.size
        self.num_items = self.item_ids.size
        self.num_train = train_pairs[0].size
        valid_file = os.path.join(data_dir, "valid1.txt")
        valid_pairs, valid_user_dict = self.load_test_interaction(valid_file)
        self.num_valid = valid_pairs[0].size
        test_file = os.path.join(data_dir, "test.txt")
        test_pairs, test_user_dict = self.load_test_interaction(test_file)
        self.num_test = test_pairs[0].size
        print("Train data: #user:{}, #item:{}, #pairs:{}".format(self.num_users, self.num_items, self.num_train))
        print("Valid data: #pairs:{}".format(self.num_valid))
        print("Test data: #pairs:{}".format(self.num_test))

        kg_file = os.path.join(data_dir, "kg_final.txt")
        kg_pd = self.load_kg(kg_file)
        if use_KG:
            kg_triples_pd = self.plus_inverse_kg(kg_pd)
            self.num_KG_relations = kg_triples_pd["r"].nunique()
            self.num_KG_entities = pd.concat([kg_pd['h'], kg_pd['t']]).nunique()
            self.num_KG_triples = kg_triples_pd.shape[0]
            kg_triples_np = kg_triples_pd.values
            print("After adding inverse KG triplets ...")
            print("#KG entities:{}, relations:{}, triplet:{}, #head:{}, #tail:{}".format(
                self.num_KG_entities, self.num_KG_relations, self.num_KG_triples,
                kg_triples_pd['h'].nunique(), kg_triples_pd['t'].nunique()))
            ## Keep item and entity ids unchanged and remap user ids
            ## stack user ids after entities
            self.user_mapping = {i: i + self.num_KG_entities for i in range(self.num_users)}
        else:
            ### convert KG to feature vectors
            ## Only leave attributes pointed to items
            item_fea_file = "item_fea.npz"
            if os.path.exists(os.path.join(data_dir, item_fea_file)):
                item_fea = np.load(os.path.join(data_dir, item_fea_file))
            else:
                item_fea = self.construct_item_fea(kg_pd, data_dir)
                np.save(os.path.join(data_dir, item_fea_file), item_fea)
            self.item_dim = item_fea.shape[1]
            self.user_dim = None
            ## stack user ids after items
            self.user_mapping = {i: i + self.num_items for i in range(self.num_users)}
            self.item_fea = item_fea

        self.train_pairs = (np.array(list(map(self.user_mapping.get, train_pairs[0]))).astype(np.int32),
                            train_pairs[1].astype(np.int32))
        self.train_user_dict= {self.user_mapping[k]: np.unique(v).astype(np.int32) for k,v in train_user_dict.items()}
        self.valid_pairs = (np.array(list(map(self.user_mapping.get, valid_pairs[0]))).astype(np.int32),
                           valid_pairs[1].astype(np.int32))
        self.valid_user_dict = {self.user_mapping[k]: np.unique(v).astype(np.int32) for k, v in valid_user_dict.items()}
        self.test_pairs = (np.array(list(map(self.user_mapping.get, test_pairs[0]))).astype(np.int32),
                           test_pairs[1].astype(np.int32))
        self.test_user_dict= {self.user_mapping[k]: np.unique(v).astype(np.int32) for k,v in test_user_dict.items()}

        print("The user-item pairs: #users {}, #items {}, #train pairs {}, #valid pairs {}, #test pairs {}".format(
            self.num_users, self.num_items, self.num_train, self.num_valid, self.num_test))

        ### the relation id for user->item == 0 and item->user == 1
        train_user_item_triplet = np.zeros((self.num_train*2, 3), dtype=np.int32)
        train_user_item_triplet[:, 0] = np.concatenate((self.train_pairs[0], self.train_pairs[1]))
        train_user_item_triplet[:, 1] = np.concatenate((np.zeros(self.num_train, dtype=np.int32),
                                                        np.ones(self.num_train, dtype=np.int32)))
        train_user_item_triplet[:, 2] = np.concatenate((self.train_pairs[1], self.train_pairs[0]))
        train_user_item_triplet = train_user_item_triplet.astype(np.int32)
        valid_user_item_triplet = np.zeros((self.num_valid*2, 3), dtype=np.int32)
        valid_user_item_triplet[:, 0] = np.concatenate((self.valid_pairs[0], self.valid_pairs[1]))
        valid_user_item_triplet[:, 1] = np.concatenate((np.zeros(self.num_valid, dtype=np.int32),
                                                        np.ones(self.num_valid, dtype=np.int32)))
        valid_user_item_triplet[:, 2] = np.concatenate((self.valid_pairs[1], self.valid_pairs[0]))
        valid_user_item_triplet = valid_user_item_triplet.astype(np.int32)

        if use_KG:
            ###              |<item>  <att entity> | <user>
            ### <item>       |=====================|=======
            ### <att entity> |=====================|+++++++
            ### <user>       |=======|+++++++++++++++++++++

            ### the first two relation ids are for user->item and item->user
            print(kg_triples_np[:, 1])
            kg_triples_np[:, 1] = kg_triples_np[:, 1] + 2
            print(kg_triples_np[:, 1])

            all_train_triplet = np.vstack((kg_triples_np, train_user_item_triplet))
            all_test_triplet = np.vstack((kg_triples_np, train_user_item_triplet, valid_user_item_triplet))
            self.all_train_triplet_np = all_train_triplet
            self.all_test_triplet_np = all_test_triplet

            self.num_all_entities = self.num_KG_entities + self.num_users
            assert np.max(all_train_triplet) + 1 == self.num_all_entities
            self.num_all_relations = self.num_KG_relations + 2
            self.num_all_train_triplets = self.all_train_triplet_np.shape[0]
            self.num_all_test_triplets = self.all_test_triplet_np.shape[0]
            self.num_all_nodes = self.num_all_entities
            print("The KG: #entities {}, #relations {}, #triplets {}".format(
                self.num_KG_entities, self.num_KG_relations, self.num_KG_triples))
            print("The train graph: #nodes {}, #relations {}, #edges {}".format(
                self.num_all_nodes, self.num_all_relations, self.num_all_train_triplets))
            print("The test graph: #nodes {}, #relations {}, #triplets {}".format(
                self.num_all_nodes, self.num_all_relations, self.num_all_test_triplets))
            self.all_train_kg_dict = self._get_all_train_kg_dict()

        else:
            self.num_all_nodes = self.num_users + self.num_items
            self.all_train_triplet_np = train_user_item_triplet
            self.all_test_triplet_np = np.vstack((train_user_item_triplet, valid_user_item_triplet))

    def construct_item_fea(self, kg_pd, data_dir):
        item_kg_pg = kg_pd[kg_pd['h'] < self.num_items]
        r_ecount = item_kg_pg.groupby("r")["t"].nunique()
        r_e_freq = item_kg_pg.groupby(['r'])['t'].value_counts()
        r_ecount.to_csv(os.path.join(data_dir, "_rel_ecount"), sep=" ", index_label=['r'], header=["count"])
        r_e_freq.to_csv(os.path.join(data_dir, "_rel_entity_freq"), sep=" ", index_label=['r', 't'], header=["freq"])

        rel_entity_dict = {}  ## {r_id: [entity_id]}
        for (r_id, entity_id), freq in r_e_freq.items():
            # print("r_id", r_id,"entity_id", entity_id, "freq", freq)
            if r_id in rel_entity_dict:
                rel_entity_dict[r_id].append(entity_id)
            else:
                rel_entity_dict[r_id] = []
                rel_entity_dict[r_id].append(entity_id)

        pos_file = open(os.path.join(data_dir, "_itemFea_Record"), "w")
        pos_file.write("fea_idx rel_id entity_id\n")
        fea_pos_dict = {}
        fea_idx = 0
        for r_id, e_l in rel_entity_dict.items():
            fea_pos_dict[r_id] = {}
            for e in e_l:
                fea_pos_dict[r_id][e] = fea_idx
                pos_file.write("{} {} {}\n".format(fea_idx, r_id, e))
                fea_idx += 1
        # print(fea_pos_dict)
        pos_file.close()

        total_fea_dim = fea_idx
        print("total_fea_dim", total_fea_dim)
        fea_np = np.zeros((self.num_items, total_fea_dim))
        for _, row in item_kg_pg.iterrows():
            h, r, t = row['h'], row['r'], row['t']
            fea_np[h][fea_pos_dict[r][t]] = 1.0

        sparsity = fea_np.sum() / (self.num_items * total_fea_dim)
        print("feature sparsity: {:.2f}%".format(sparsity*100))

        return fea_np
    @property
    def train_g(self):
        g = dgl.DGLGraph()
        g.add_nodes(self.num_all_nodes)
        g.add_edges(self.all_train_triplet_np[:, 2], self.all_train_triplet_np[:, 0])
        g.readonly()
        g.ndata['id'] = th.arange(self.num_all_nodes, dtype=th.long)
        g.edata['type'] = th.LongTensor(self.all_train_triplet_np[:, 1])
        return g

    @property
    def test_g(self):
        g = dgl.DGLGraph()
        g.add_nodes(self.num_all_nodes)
        g.add_edges(self.all_test_triplet_np[:, 2], self.all_test_triplet_np[:, 0])
        g.readonly()
        g.ndata['id'] = th.arange(self.num_all_nodes, dtype=th.long)
        g.edata['type'] = th.LongTensor(self.all_test_triplet_np[:, 1])
        return g

    def load_kg(self, file_name):
        kg_pd = pd.read_csv(file_name, sep=" ", names=['h', "r", "t"], engine='python')
        kg_pd = kg_pd.drop_duplicates()
        kg_pd = kg_pd.sort_values(by=['h'])
        unique_rel = kg_pd['r'].nunique()
        entity_ids = pd.unique(pd.concat([kg_pd['h'], kg_pd['t']]))
        if kg_pd["r"].nunique() != kg_pd["r"].max()+1:
            relation_mapping = {old_id: idx for idx, old_id in enumerate(pd.unique(kg_pd["r"]))}
            kg_pd['r'] = list(map(relation_mapping.get, kg_pd['r'].values))

        print("#KG entities:{}, relations:{}, triplet:{}, #head:{}, #tail:{}".format(
            entity_ids.size, unique_rel, kg_pd.shape[0], kg_pd['h'].nunique(), kg_pd['t'].nunique()))

        return kg_pd

    def plus_inverse_kg(self, kg_pd):
        unique_rel = kg_pd['r'].nunique()
        rev = kg_pd.copy()
        rev = rev.rename({'h':'t', 't':'h'})
        rev['r'] += unique_rel
        new_kg_pd = pd.concat([kg_pd, rev], ignore_index=True)

        return new_kg_pd

    def _get_all_train_kg_dict(self):
        ### generate all_kg_dict for sampling
        all_kg_dict = collections.defaultdict(list)
        for h, r, t in self.all_train_triplet_np:
            if h in all_kg_dict.keys():
                all_kg_dict[h].append((t, r))
            else:
                all_kg_dict[h] = [(t, r)]
        return all_kg_dict

    # def _sample_pos_triples_for_h(self, h, num):
    #     pos_triples = self.all_kg_dict[h]
    #     n_pos_triples = len(pos_triples)
    #     pos_rs, pos_ts = [], []
    #     while True:
    #         if len(pos_rs) == num: break
    #         pos_id = np.random.randint(low=0, high=n_pos_triples, size=1)[0]
    #         t = pos_triples[pos_id][0]
    #         r = pos_triples[pos_id][1]
    #         if r not in pos_rs and t not in pos_ts:
    #             pos_rs.append(r)
    #             pos_ts.append(t)
    #     return pos_rs, pos_ts
    #
    #
    # def _sample_neg_triples_for_h(self, h, r, num):
    #     neg_ts = []
    #     while True:
    #         if len(neg_ts) == num: break
    #         t = np.random.randint(low=0, high=self.num_all_entities, size=1)[0]
    #         if (t, r) not in self.all_kg_dict[h] and t not in neg_ts:
    #             neg_ts.append(t)
    #     return neg_ts
    #
    # def KG_sampler(self, batch_size):
    #     ### generate negative triplets
    #     self._get_all_kg_dict()
    #     exist_heads = list(self.all_kg_dict.keys())
    #     print("len(exist_heads)", len(exist_heads))
    #     n_batch = self.num_all_triplets // batch_size + 1
    #     print("num_all_triplets", self.num_all_triplets, "batch_size", batch_size, "n_batch", n_batch)
    #     i = 0
    #     #print("Batch_size:{}, #batches:{}".format(batch_size, n_batch))
    #     while i < n_batch:
    #         i += 1
    #         if batch_size <= len(exist_heads):
    #             heads = rd.sample(exist_heads, batch_size)
    #         else:
    #             heads = [rd.choice(exist_heads) for _ in range(batch_size)]
    #         pos_r_batch, pos_t_batch, neg_t_batch = [], [], []
    #         for h in heads:
    #             pos_rs, pos_ts = self._sample_pos_triples_for_h(h, 1)
    #             pos_r_batch += pos_rs
    #             pos_t_batch += pos_ts
    #             neg_ts = self._sample_neg_triples_for_h(h, pos_rs[0], 1)
    #             neg_t_batch += neg_ts
    #         yield heads, pos_r_batch, pos_t_batch, neg_t_batch

    def _sample_pos_triples_for_h(self, h):
        t, r = rd.choice(self.all_train_kg_dict[h])
        return r, t
    def _sample_neg_triples_for_h(self, h, r):
        while True:
            t = rd.choice(range(self.num_all_entities))
            if (t, r) not in self.all_train_kg_dict[h]:
                return t
    def KG_sampler(self, batch_size):
        assert self._use_KG
        exist_heads = list(self.all_train_kg_dict.keys())
        n_batch = self.num_all_train_triplets // batch_size + 1
        #print("#num_all_triplets", self.num_all_triplets, "batch_size", batch_size, "n_batch", n_batch,
        #      "#exist_heads", len(exist_heads))
        i = 0
        while i < n_batch:
            i += 1
            if batch_size <= len(exist_heads):
                heads = rd.sample(exist_heads, batch_size)
            else:
                heads = [rd.choice(exist_heads) for _ in range(batch_size)]
            pos_r_batch, pos_t_batch, neg_t_batch = [], [], []
            for h in heads:
                pos_rs, pos_ts = self._sample_pos_triples_for_h(h)
                pos_r_batch.append(pos_rs)
                pos_t_batch.append(pos_ts)
                neg_ts = self._sample_neg_triples_for_h(h, pos_rs)
                neg_t_batch.append(neg_ts)
            yield heads, pos_r_batch, pos_t_batch, neg_t_batch
    def KG_sampler_uniform(self, batch_size):
        assert self._use_KG
        # pos_pool = []
        # for i in range(self.num_all_train_triplets):
        #     pos_pool.append(self.all_train_triplet_np[i, :].tolist())
        if batch_size < 0:
            batch_size = self.num_all_train_triplets
            n_batch = 1
        elif batch_size > self.num_all_train_triplets:
            batch_size = min(batch_size, self.num_all_train_triplets)
            n_batch = 1
        else:
            n_batch = self.num_all_train_triplets // batch_size + 1
        i = 0
        while i < n_batch:
            i += 1
            sel = rd.sample(range(self.num_all_train_triplets), k=batch_size)
            h = self.all_train_triplet_np[sel][:, 0]
            r = self.all_train_triplet_np[sel][:, 1]
            pos_t = self.all_train_triplet_np[sel][:, 2]
            neg_t = rd.choices(range(self.num_all_entities), k=batch_size)
            ### check whether negative triplets are true negative TODO too slow
            # neg_l = [[neg_t[j], r[j], h[j]] for j in range(batch_size)]
            # true_neg = list(map(lambda x: x not in pos_pool, neg_l))
            # h, r, pos_t, neg_t = h[true_neg], r[true_neg], pos_t[true_neg], neg_t[true_neg]
            # print(h.shape, r.shape, pos_t.shape, neg_t.shape)
            yield h, r, pos_t, neg_t
    def create_Edge_sampler(self, batch_size, num_workers=8, shuffle=True, exclude_positive=False):
        EdgeSampler = getattr(dgl.contrib.sampling, 'EdgeSampler')
        return EdgeSampler(self.train_g,
                           batch_size=batch_size,
                           neg_sample_size=0,
                           negative_mode="tail",
                           num_workers=num_workers,
                           shuffle=shuffle,
                           exclude_positive=exclude_positive,
                           return_false_neg=False)
    def KG_sampler_DGL(self, batch_size):
        if batch_size < 0:
            batch_size = self.num_all_train_triplets
            n_batch = 1
        elif batch_size > self.num_all_train_triplets:
            batch_size = min(batch_size, self.num_all_train_triplets)
            n_batch = 1
        else:
            n_batch = self.num_train // batch_size + 1
        i = 0
        while i < n_batch:
            i += 1
            for pos_g, _ in self.create_Edge_sampler(batch_size):
                pos_g.copy_from_parent()
                #neg_g.copy_from_parent()
                h_idx, t_idx = pos_g.all_edges(order='eid')
                #neg_h_idx, neg_t_idx = neg_g.all_edges(order='eid')
                # print("Positive Graph ...")
                # print("(h_idx, t_idx)", h_idx, t_idx)
                # print("ndata", pos_g.ndata["id"])
                # print("(h_id, t_id)", pos_g.ndata['id'][h_idx], pos_g.ndata['id'][t_idx])
                #
                # print("Negative Graph ...")
                # print("(h_idx, t_idx)", neg_h_idx, neg_t_idx)
                # print("ndata", neg_g.ndata['id'])
                # print("(h_id, t_id)", neg_g.ndata['id'][neg_h_idx], neg_g.ndata['id'][neg_t_idx])
                h = pos_g.ndata['id'][h_idx]
                r = pos_g.edata['type']
                pos_t = pos_g.ndata['id'][t_idx]
                #neg_t = neg_g.ndata['id'][neg_t_idx]
                neg_t = th.LongTensor(rd.choices(range(self.num_all_entities), k=batch_size))
                #print(h, r, pos_t, neg_t)
                yield h, r, pos_t, neg_t

    # reading train & test interaction data
    def _load_interaction(self, file_name):
        src = []
        dst = []
        user_dict = {}
        lines = open(file_name, 'r').readlines()
        for l in lines:
            tmps = l.strip()
            inters = [int(i) for i in tmps.split(' ')]
            if len(inters) > 1:
                user_id, item_ids = inters[0], inters[1:]
                item_ids = list(set(item_ids))
                for i_id in item_ids:
                    src.append(user_id)
                    dst.append(i_id)
                user_dict[user_id] = item_ids
        return np.array(src, dtype=np.int32), np.array(dst, dtype=np.int32), user_dict

    def load_train_interaction(self, file_name):
        src, dst, train_user_dict = self._load_interaction(file_name)
        ### check whether the user id / item id are continuous and starting from 0
        assert np.unique(src).size == max(src) + 1
        assert np.unique(dst).size == max(dst) + 1

        return (src, dst), train_user_dict

    def load_test_interaction(self, file_name):
        src, dst, test_user_dict = self._load_interaction(file_name)
        unique_users = np.unique(src)
        unique_items = np.unique(dst)
        for ele in unique_users:
            assert ele in self.train_user_ids
        for ele in unique_items:
            assert ele in self.item_ids

        return (src, dst), test_user_dict

    # def _sample_pos_items_for_u(self, u, num):
    #     pos_items = self.train_user_dict[u]
    #     n_pos_items = len(pos_items)
    #     pos_batch = []
    #     while True:
    #         if len(pos_batch) == num: break
    #         pos_id = np.random.randint(low=0, high=n_pos_items, size=1)[0]
    #         pos_i_id = pos_items[pos_id]
    #
    #         if pos_i_id not in pos_batch:
    #             pos_batch.append(pos_i_id)
    #     return pos_batch
    # def _sample_neg_items_for_u(self, u, num):
    #     neg_items = []
    #     while True:
    #         if len(neg_items) == num: break
    #         neg_i_id = np.random.randint(low=0, high=self.num_items, size=1)[0]
    #         if neg_i_id not in self.train_user_dict[u] and neg_i_id not in neg_items:
    #             neg_items.append(neg_i_id)
    #     return neg_items
    # def _generate_user_pos_neg_items(self, batch_size):
    #     if batch_size <= self.num_users:
    #         ## test1
    #         users = rd.sample(self.exist_users, batch_size)
    #     else:
    #         users = [rd.choice(self.exist_users) for _ in range(batch_size)]
    #
    #     pos_items, neg_items = [], []
    #     for u in users:
    #         pos_items += self._sample_pos_items_for_u(u, 1)
    #         neg_items += self._sample_neg_items_for_u(u, 1)
    #
    #     return users, pos_items, neg_items

    def _sample_pos_items_for_u(self, u):
        return rd.choice(self.train_user_dict[u])
    def _sample_neg_items_for_u(self, u):
        while True:
            neg_i_id = rd.choice(range(self.num_items))
            if neg_i_id not in self.train_user_dict[u]:
                return neg_i_id
    def CF_pair_sampler(self, batch_size):
        exist_users = list(self.train_user_dict.keys())
        if batch_size < 0:
            batch_size = self.num_train
            n_batch = 1
        elif batch_size > self.num_train:
            batch_size = min(batch_size, self.num_train)
            n_batch = 1
        else:
            n_batch = self.num_train // batch_size + 1
        #print("#train_pair", self.num_train, "batch_size", batch_size, "n_batch", n_batch, "#exist_user", len(exist_users))
        i = 0
        while i < n_batch:
            i += 1
            if batch_size <= self.num_users:
                ## test1
                users = rd.sample(exist_users, batch_size)
            else:
                users = [rd.choice(exist_users) for _ in range(batch_size)]

            pos_items, neg_items = [], []
            for u in users:
                pos_items.append(self._sample_pos_items_for_u(u))
                neg_items.append(self._sample_neg_items_for_u(u))
            yield users, pos_items, neg_items

    def CF_pair_uniform_sampler(self, batch_size):
        if batch_size < 0:
            batch_size = self.num_train
            n_batch = 1
        elif batch_size > self.num_train:
            batch_size = min(batch_size, self.num_train)
            n_batch = 1
        else:
            n_batch = self.num_train // batch_size + 1
        #print("#train_pair", self.num_train, "batch_size", batch_size, "n_batch", n_batch, "#exist_user", len(exist_users))
        i = 0
        while i < n_batch:
            i += 1
            sel = rd.sample(range(self.num_train), k = batch_size)
            users = self.train_pairs[0][sel]
            pos_items = self.train_pairs[1][sel]
            neg_items = rd.choices(range(self.num_items), k = batch_size)
            yield users, pos_items, neg_items

    # def CF_batchwise_sampler(self, batch_size):
    #     self.exist_users = self.train_user_dict.keys()
    #     if batch_size < 0:
    #         batch_size = self.num_train-1
    #         n_batch = 1
    #     else:
    #         batch_size = min(batch_size, self.num_train)
    #         n_batch = self.num_train // batch_size + 1
    #
    #     i = 0
    #     #print("Batch_size:{}, #batches:{}".format(batch_size, n_batch))
    #     while i < n_batch:
    #         i += 1
    #         user_ids, item_ids, neg_item_ids = self._generate_user_pos_neg_items(batch_size)
    #         new_entity_ids, new_pd = self._filter_neighbor(np.concatenate((user_ids, item_ids, neg_item_ids)),
    #                                                        self.all_triplet_dp)
    #         etype = new_pd['r'].values
    #         ### relabel nodes to have consecutive node ids
    #         uniq_v, edges = np.unique((new_pd['h'].values, new_pd['t'].values), return_inverse=True)
    #         src, dst = np.reshape(edges, (2, -1))
    #         g = dgl.DGLGraph()
    #         g.add_nodes(uniq_v.size)
    #         g.add_edges(dst, src)
    #         ### map user_ids and items_ids into indicies in the graph
    #         node_map = {ele: idx for idx, ele in enumerate(uniq_v)}
    #         user_ids = np.array(list(map(node_map.get, user_ids)), dtype=np.int32)
    #         item_ids = np.array(list(map(node_map.get, item_ids)), dtype=np.int32)
    #         neg_item_ids = np.array(list(map(node_map.get, neg_item_ids)), dtype=np.int32)
    #
    #         yield user_ids, item_ids, neg_item_ids, g, uniq_v, etype

if __name__ == '__main__':
    """
    batch_size = 1024
    d_loader = DataLoader("yelp2018", use_KG=True)
    
    ## convert positive triplets to sets
    pos_pool = []

    for i in range(d_loader.num_all_train_triplets):
        pos_pool.append(d_loader.all_train_triplet_np[i, :].tolist())
    #print(pos_pool)
    kg_sampler = d_loader.KG_sampler_DGL(batch_size)
    for h, r, pos_t, neg_t in kg_sampler:
        print("\n\n")
        count = 0
        for j in range(batch_size):
            pos_l = [pos_t[j].item(), r[j].item(), h[j].item()]
            neg_l = [neg_t[j].item(), r[j].item(), h[j].item()]
            try:
                assert pos_l in pos_pool
            except:
                print("pos_l", pos_l, "not in the true positive triplets!!!!~~~~")

            try:
                assert neg_l not in pos_pool
            except:
                count += 1
                print("neg_l", neg_l, "not in the true negative triplets!")
        print(count)
        #print(h, r, pos_t, neg_t)
    """
    DataLoader("yelp2018", use_KG=True)
    DataLoader("yelp2018", use_KG=False)
    DataLoader("last-fm", use_KG=True)
    DataLoader("last-fm", use_KG=False)
    DataLoader("amazon-book", use_KG=True)
    DataLoader("amazon-book", use_KG=False)


