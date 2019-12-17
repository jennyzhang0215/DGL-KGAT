import numpy as np
import os
import dgl
import pandas as pd
import collections
import random as rd
from time import time
import scipy.sparse as sp
import torch as th
import pickle
import time
from scipy import sparse as sp

_UV_FILES = ["uv_train.pd", "uv_val.pd", "uv_test.pd"]
### in KGAT it assums that user has no kg data, thus we set the file name to be None
_KG_FILES = [None, "kg_item.pd"]

_SEP = "\t"

class DataLoader(object):
    def __init__(self, data_name, use_pretrain=False, symmetric=True, as_continuous_ids=True, add_uv2KG=True, seed=1234):
        self._data_name = data_name
        rd.seed(seed)
        _DATA_DIR = os.path.realpath(os.path.join(os.path.abspath(__file__), '..', '..', "datasets", data_name))

        train_pd = self._load_pd(os.path.join(_DATA_DIR, _UV_FILES[0])).sort_values(by=["u"])
        valid_pd = self._load_pd(os.path.join(_DATA_DIR, _UV_FILES[1])).sort_values(by=["u"])
        test_pd = self._load_pd(os.path.join(_DATA_DIR, _UV_FILES[2])).sort_values(by=["u"])
        self.n_users = train_pd["u"].nunique()
        self.n_items = train_pd["v"].nunique()
        self.n_train = train_pd.shape[0]
        self.n_valid = valid_pd.shape[0]
        self.n_test = test_pd.shape[0]
        ### test_user_ids, test_item_ids, test_uv_sp, pruned_train_uv_pairs,
        ###TODO assum all the items and users are seen in the training set
        self.eval_val_user_ids = np.unique(valid_pd["u"].unique())
        self.eval_val_uv_pair = valid_pd.copy()
        self.eval_train_uv_pairv4val = train_pd[train_pd["u"].isin(self.eval_val_user_ids)]

        self.eval_test_user_ids = np.unique(test_pd["u"].unique())
        self.eval_test_uv_pair = test_pd.copy()
        train_val_pd = train_pd.append(valid_pd, ignore_index=True).sort_values(by=["u"])
        self.eval_train_uv_pairv4test = train_val_pd[train_val_pd["u"].isin(self.eval_test_user_ids)]

        if len(train_pd.columns) == 2:
            ### just use implicit feedback
            self.train_user_dict = self._convert_uv_pair2dict(train_pd)
            # self.valid_user_dict = self._convert_uv_pair2dict(valid_pd)
            # self.test_user_dict = self._convert_uv_pair2dict(test_pd)
            # self.train_valid_user_dict = self._convert_uv_pair2dict(train_pd.append(valid_pd, ignore_index=True))
            n_uv_rel = 1
        elif len(train_pd.columns) > 2:
            n_uv_rel = train_pd["r"].nunique()
        else:
            raise NotImplementedError

        item_kg_pd = self._load_pd(os.path.join(_DATA_DIR, _KG_FILES[1]))

        ### ids will be remapping
        if as_continuous_ids:
            ## process the entities ids to be consecutive
            ## by stacking item/entity after user ids
            ## <user> | <item>  <att entity>
            item_entity_offset = self.n_users
            self.item_id_range = np.arange(item_entity_offset, item_entity_offset + self.n_items)
            ## process the uv data first
            train_pd["v"] = train_pd["v"] + item_entity_offset
            valid_pd["v"] = valid_pd["v"] + item_entity_offset
            test_pd["v"] = test_pd["v"] + item_entity_offset
            ## then process the item kg
            if item_kg_pd is not None:
                item_kg_pd["h"] = item_kg_pd["h"] + item_entity_offset
                item_kg_pd["t"] = item_kg_pd["t"] + item_entity_offset
        self.eval_item_ids = np.unique(train_pd["v"].unique())

        kg_pd = item_kg_pd
        n_KG_relations = kg_pd["r"].nunique()
        train_uv_triplet = self._convert_uv2triplet_np(train_pd, offset_rel=n_KG_relations,
                                                       n_uv_rel=n_uv_rel, symmetric=symmetric)
        train_valid_uv_triplet = self._convert_uv2triplet_np(train_pd.append(valid_pd, ignore_index=True),
                                                             offset_rel=n_KG_relations,
                                                             n_uv_rel=n_uv_rel, symmetric=symmetric)

        if add_uv2KG:
            ###               <user> | <item>  <att entity>
            ### <user>       |+++++++|========|++++++++++++
            ### <item>       |=======|++++++++|============
            ### <att entity> |+++++++|========|============
            self.train_KG_triplet = np.vstack((kg_pd.values, train_uv_triplet))
            self.test_KG_triplet = np.vstack((kg_pd.values, train_valid_uv_triplet))
        else:
            self.train_KG_triplet = kg_pd.values
            self.test_KG_triplet = kg_pd.values

        self.n_KG_relation = np.unique(self.train_KG_triplet[:, 1]).size
        self.n_KG_entity = np.unique(np.concatenate((self.train_KG_triplet[:, 0], self.train_KG_triplet[:, 2]))).size
        self.n_train_KG_triplet = self.train_KG_triplet.shape[0]
        self.n_test_KG_triplet = self.test_KG_triplet.shape[0]

        self.train_KG = self._generate_KG(self.n_KG_entity, self.train_KG_triplet, add_etype=True)
        #self.test_KG = self._generate_KG(self.n_KG_entity, self.test_KG_triplet, add_etype=True)

        self.train_uv_graph = self._generate_KG(n_entity=self.n_users + self.n_items,
                                                KG_triplet=train_uv_triplet,  add_etype=False)
        #self.train_val_uv_graph = self._generate_KG(n_entity=n_user_entities+self.n_items,
        #                                            KG_triplet=train_valid_uv_triplet, add_etype=False)

        self.train_pairs = train_pd.values ##[[u_id, v_id], ...]
        self.valid_pairs = valid_pd.values
        self.test_pairs = test_pd.values

        if use_pretrain:
            pre_model = 'mf'
            file_name = os.path.realpath(os.path.join(os.path.abspath(__file__), '..', "datasets",
                                                      "pretrain", data_name, "{}.npz".format(pre_model)))
            pretrain_data = np.load(file_name)
            self.user_pre_embed = pretrain_data['user_embed']
            self.item_pre_embed = pretrain_data['item_embed']

    @property
    def train_g(self):
        g = dgl.DGLGraph()
        g.add_nodes(self.n_KG_entity)
        g.add_edges(self.train_KG_triplet[:, 2], self.train_KG_triplet[:, 0])
        g.ndata['id'] = th.arange(self.n_KG_entity, dtype=th.long)
        g.edata['type'] = th.LongTensor(self.train_KG_triplet[:, 1])
        return g

    @property
    def test_g(self):
        g = dgl.DGLGraph()
        g.add_nodes(self.n_KG_entity)
        g.add_edges(self.test_KG_triplet[:, 2], self.test_KG_triplet[:, 0])
        g.ndata['id'] = th.arange(self.n_KG_entity, dtype=th.long)
        g.edata['type'] = th.LongTensor(self.test_KG_triplet[:, 1])
        return g


    def __repr__(self):
        info = "{}:\n".format(self._data_name)
        info += "\t#user {}, #item {}\n".format(self.n_users, self.n_items)
        info += "\t#train {}, #valid {}, #test {}\n".format(self.n_train, self.n_valid, self.n_test)

        info += "In the knowledge graph: {}\n"
        info += "\tKG: #entity {}, #relation {}, #train triplet {}, #test triplet {}\n".format(
                self.n_KG_entity, self.n_KG_relation, self.n_train_KG_triplet, self.n_test_KG_triplet)
        return info


    def _load_np(self, file_name):
        if os.path.isfile(file_name):
            return np.load(file_name)
        else:
            print("{} does not exit.".format(file_name))
            return None
    def _load_pd(self, file_name):
        if os.path.isfile(file_name):
            return pd.read_csv(file_name, sep=_SEP, header=0, engine="python", dtype=np.int32)
        else:
            print("{} does not exit.".format(file_name))
            return None
    def _load_pkl(self, file_name):
        if os.path.isfile(file_name):
            f = open(file_name, "rb")
            data_dict = pickle.load(f)
            f.close()
            return data_dict
        else:
            print("{} does not exit.".format(file_name))
            return None
    def _convert_uv2triplet_np(self, uv_pd, offset_rel, n_uv_rel=1, symmetric=True):
        num_pair = uv_pd.shape[0]
        uv_triplet = np.zeros((num_pair, 3), dtype=np.int32)
        uv_triplet[:, 0] = uv_pd["u"].values
        uv_triplet[:, 2] = uv_pd["v"].values
        if len(uv_pd.columns) == 2:
            uv_triplet[:, 1] = np.ones(num_pair, dtype=np.int32) * offset_rel
        else:
            uv_triplet[:, 1] = uv_pd["r"].values + offset_rel
        if symmetric:
            vu_triplet = np.zeros((num_pair, 3), dtype=np.int32)
            vu_triplet[:, 0] = uv_pd["v"].values
            vu_triplet[:, 2] = uv_pd["u"].values
            if len(uv_pd.columns) == 2:
                vu_triplet[:, 1] = np.ones(num_pair, dtype=np.int32) * (offset_rel + n_uv_rel)
            else:
                vu_triplet[:, 1] = uv_pd["r"].values + (offset_rel + n_uv_rel)

            return np.vstack((uv_triplet, vu_triplet))
        else:
            return uv_triplet

    def _convert_uv_pair2dict(self, uv_pd):
        start = time.time()
        #uv_pd = uv_pd.sort_values(by=['u'])
        u_v_dict = {}
        groups = uv_pd.groupby("u")
        for _, g in groups:
            assert g["u"].nunique() == 1
            u_v_dict[g["u"].iloc[0]] = g["v"].values
        print("{:.1f}s for convert_uv_pair2dict() ...".format(time.time() - start))
        return u_v_dict
    def _symmetrize_kg(self, kg_pd):
        unique_rel = kg_pd['r'].nunique()
        rev_kg = kg_pd.copy()
        rev_kg = rev_kg.rename({'h':'t', 't':'h'})
        rev_kg['r'] += unique_rel
        return pd.concat([kg_pd, rev_kg], ignore_index=True)
    def _generate_KG(self, n_entity, KG_triplet, add_etype=True):
        g = dgl.DGLGraph()
        g.add_nodes(n_entity)
        g.add_edges(KG_triplet[:, 2], KG_triplet[:, 0])
        g.readonly()
        g.ndata['id'] = th.arange(n_entity, dtype=th.long)
        if add_etype:
            g.edata['type'] = th.LongTensor(KG_triplet[:, 1])
        return g


    def _get_all_train_kg_dict(self):
        ### generate all_kg_dict for sampling
        all_kg_dict = collections.defaultdict(list)
        for h, r, t in self.train_KG_triplet:
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
            batch_size = self.n_train_KG_triplet
            n_batch = 1
        elif batch_size > self.n_train_KG_triplet:
            batch_size = min(batch_size, self.n_train_KG_triplet)
            n_batch = 1
        else:
            n_batch = self.n_train_KG_triplet // batch_size + 1

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
                        neg_t = rd.choice(range(self.n_KG_entity))
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
            for pos_g, neg_g in self.create_Edge_sampler(self.train_KG, batch_size, neg_sample_size=1,
                                                         num_workers=num_workers, shuffle=True,
                                                         exclude_positive=exclude_positive, return_false_neg=True):
                false_neg = neg_g.edata["false_neg"]
                pos_g.copy_from_parent()
                neg_g.copy_from_parent()
                h_idx, t_idx = pos_g.all_edges(order='eid')
                neg_h_idx, neg_t_idx = neg_g.all_edges(order='eid')
                # print("Positive Graph ...")
                # print("h_id", pos_g.ndata['id'][h_idx])
                # print("r_id", pos_g.edata['type'])
                # print("t_id", pos_g.ndata['id'][t_idx])
                # print("Negative Graph ...")
                # print("h_id", neg_g.ndata['id'][neg_h_idx])
                # print("r_id", neg_g.edata['type'])
                # print("t_id", neg_g.ndata['id'][neg_t_idx])
                hs = pos_g.ndata['id'][h_idx]
                rs = pos_g.edata['type']
                pos_ts = pos_g.ndata['id'][t_idx]
                neg_ts = neg_g.ndata['id'][neg_t_idx]
                yield hs, rs, pos_ts, neg_ts, false_neg
        else:
            raise NotImplementedError
    def CF_pair_sampler(self, batch_size, pos_mode="uniform", neg_mode="random",
                        num_workers=8, exclude_positive=False):
        if batch_size < 0:
            batch_size = self.n_train
            n_batch = 1
        elif batch_size > self.n_train:
            batch_size = min(batch_size, self.n_train)
            n_batch = 1
        else:
            n_batch = self.n_train // batch_size + 1

        if pos_mode == "unique" and neg_mode == "exclude_pos":
            exist_users = list(self.train_user_dict.keys())
            i = 0
            while i < n_batch:
                i += 1
                if batch_size <= self.n_users:
                    users = rd.sample(exist_users, batch_size)
                else:
                    users = [rd.choice(exist_users) for _ in range(batch_size)]
                pos_items, neg_items = [], []
                for u in users:
                    pos_items.append(rd.choice(self.train_user_dict[u]))
                    while True:
                        neg_i_id = rd.choice(range(self.n_items))
                        if neg_i_id not in self.train_user_dict[u]: break
                    neg_items.append(neg_i_id)
                yield users, pos_items, neg_items, None
        elif pos_mode == "uniform" and neg_mode == "random":
            for pos_g, neg_g in self.create_Edge_sampler(self.train_uv_graph, batch_size, neg_sample_size=1,
                                                         num_workers=num_workers, shuffle=True, negative_mode="head",
                                                         exclude_positive=exclude_positive, return_false_neg=True):
                false_neg = neg_g.edata["false_neg"]
                pos_g.copy_from_parent()
                neg_g.copy_from_parent()
                i_idx, u_idx = pos_g.all_edges(order='eid')
                neg_i_idx, neg_u_idx = neg_g.all_edges(order='eid')
                users = pos_g.ndata['id'][u_idx]
                pos_items = pos_g.ndata['id'][i_idx]
                neg_items = neg_g.ndata['id'][neg_i_idx]
                yield users, pos_items, neg_items, false_neg
        else:
            raise NotImplementedError
