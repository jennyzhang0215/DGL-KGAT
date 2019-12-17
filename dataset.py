import numpy as np
import os
import dgl
import pandas as pd
import collections
import random as rd
import torch as th

class DataLoader(object):
    def __init__(self, data_name, use_pretrain, seed=1234):
        print("\n{}->".format(data_name))
        self._data_name = data_name
        self._use_pretrain = use_pretrain
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


        self.train_pairs = (np.array(list(map(self.user_mapping.get, train_pairs[0]))).astype(np.int32),
                            train_pairs[1].astype(np.int32))
        self.train_user_dict = {self.user_mapping[k]: np.unique(v).astype(np.int32) for k,v in train_user_dict.items()}
        self.valid_pairs = (np.array(list(map(self.user_mapping.get, valid_pairs[0]))).astype(np.int32),
                           valid_pairs[1].astype(np.int32))
        self.valid_user_dict = {self.user_mapping[k]: np.unique(v).astype(np.int32) for k, v in valid_user_dict.items()}
        train_val_user_dict = self.train_user_dict.copy()
        for k, v in self.valid_user_dict.items():
            train_val_user_dict[k] = np.concatenate((v, train_val_user_dict[k]))
        self.train_val_user_dict = train_val_user_dict
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

        ###              |<item>  <att entity> | <user>
        ### <item>       |=====================|=======
        ### <att entity> |=====================|+++++++
        ### <user>       |=======|+++++++++++++++++++++

        ### the first two relation ids are for user->item and item->user
        kg_triples_np[:, 1] = kg_triples_np[:, 1] + 2

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

        if use_pretrain:
            pre_model = 'mf'
            file_name = os.path.realpath(os.path.join(os.path.abspath(__file__), '..', "datasets",
                                                      "pretrain", data_name, "{}.npz".format(pre_model)))
            pretrain_data = np.load(file_name)
            self.user_pre_embed = pretrain_data['user_embed']
            self.item_pre_embed = pretrain_data['item_embed']
            assert self.user_pre_embed.shape[0] == self.num_users
            assert self.item_pre_embed.shape[0] == self.num_items


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
                all_kg_dict[h].append((r, t))
            else:
                all_kg_dict[h] = [(r, t)]
        return all_kg_dict
    def create_Edge_sampler(self, graph, batch_size, neg_sample_size=1, negative_mode="tail", num_workers=8, shuffle=True,
                            exclude_positive=False, return_false_neg=True):
        EdgeSampler = getattr(dgl.contrib.sampling, 'EdgeSampler')
        return EdgeSampler(graph,
                           batch_size=batch_size,
                           neg_sample_size=neg_sample_size,
                           negative_mode=negative_mode,
                           num_workers=num_workers,
                           shuffle=shuffle,
                           exclude_positive=exclude_positive,
                           return_false_neg=return_false_neg)
    def KG_sampler(self, batch_size, pos_mode="uniform", neg_mode="tail",
                   num_workers=8, exclude_positive=False):
        if batch_size < 0:
            batch_size = self.num_all_train_triplets
            n_batch = 1
        elif batch_size > self.num_all_train_triplets:
            batch_size = min(batch_size, self.num_all_train_triplets)
            n_batch = 1
        else:
            n_batch = self.num_all_train_triplets // batch_size + 1

        print("n_batch", n_batch)
        ### start sampling
        if pos_mode == "unique" and neg_mode == "tail":
            train_kg_h_dict = self._get_all_train_kg_dict()
            exist_heads = list(train_kg_h_dict.keys())
            i = 0
            while i < n_batch:
                i += 1
                ### this sampler is for KGAT
                if batch_size <= len(exist_heads):
                    hs = rd.sample(exist_heads, batch_size)
                else:
                    hs = [rd.choice(exist_heads) for _ in range(batch_size)]
                rs, pos_ts, neg_ts = [], [], []
                for h in hs:
                    pos_r, pos_t = rd.choice(train_kg_h_dict[h])
                    while True:
                        neg_t = rd.choice(range(self.num_all_entities))
                        if (pos_r, neg_t) not in train_kg_h_dict[h]: break
                    rs.append(pos_r)
                    pos_ts.append(pos_t)
                    neg_ts.append(neg_t)
                yield hs, rs, pos_ts, neg_ts, None
        elif pos_mode == "uniform" and neg_mode == "tail":
            ### python code for uniform sampling
            # sel = rd.sample(range(self.n_train_KG_triplet), k=batch_size)
            # h = self.train_KG_triplet[sel][:, 0]
            # r = self.train_KG_triplet[sel][:, 1]
            # pos_t = self.train_KG_triplet[sel][:, 2]
            # neg_t = rd.choices(range(self.n_KG_entity), k=batch_size)
            for pos_g, neg_g in self.create_Edge_sampler(self.train_g, batch_size, neg_sample_size=1,
                                                         num_workers=num_workers, shuffle=True,
                                                         exclude_positive=exclude_positive, return_false_neg=True):
                false_neg = neg_g.edata["false_neg"]
                pos_g.copy_from_parent()
                neg_g.copy_from_parent()
                h_idx, t_idx = pos_g.all_edges(order='eid')
                neg_h_idx, neg_t_idx = neg_g.all_edges(order='eid')
                hs = pos_g.ndata['id'][h_idx]
                rs = pos_g.edata['type']
                pos_ts = pos_g.ndata['id'][t_idx]
                neg_ts = neg_g.ndata['id'][neg_t_idx]
                yield hs, rs, pos_ts, neg_ts, false_neg
        else:
            raise NotImplementedError


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
