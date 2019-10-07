import numpy as np
import os
import dgl
import pandas as pd
import collections
import random as rd
from time import time
import scipy.sparse as sp
import multiprocessing

class L_DataLoader(object):
    def __init__(self, data_name, batch_size=1024, adj_type='bi', num_neighbor_hop=3, seed=1234):
        self._batch_size = batch_size
        self._adj_type = adj_type
        data_dir =  os.path.realpath(os.path.join(os.path.abspath(__file__), '..', "datasets", data_name))
        train_file = os.path.join(data_dir, "train.txt")
        test_file = os.path.join(data_dir, "test.txt")
        kg_file = os.path.join(data_dir, "kg_final.txt")
        self.n_train, self.n_test = 0, 0
        self.n_users, self.n_items = 0, 0

        self.train_data, self.train_user_dict = self._load_ratings(train_file)
        self.test_data, self.test_user_dict = self._load_ratings(test_file)
        self.exist_users = self.train_user_dict.keys()
        self.n_users = max(max(self.train_data[:, 0]), max(self.test_data[:, 0])) + 1
        self.n_items = max(max(self.train_data[:, 1]), max(self.test_data[:, 1])) + 1
        self.n_train = len(self.train_data)
        self.n_test = len(self.test_data)

        self.n_relations, self.n_entities, self.n_triples = 0, 0, 0
        self.kg_data, self.kg_dict, self.relation_dict = self._load_kg(kg_file)

        self.batch_size_kg = self.n_triples // (self.n_train // self._batch_size)
        print('[n_users, n_items]=[%d, %d]' % (self.n_users, self.n_items))
        print('[n_train, n_test]=[%d, %d]' % (self.n_train, self.n_test))
        print('[n_entities, n_relations, n_triples]=[%d, %d, %d]' % (self.n_entities, self.n_relations, self.n_triples))
        print('[batch_size, batch_size_kg]=[%d, %d]' % (self._batch_size, self.batch_size_kg))

        self.adj_list, self.adj_r_list = self._get_relational_adj_list()
        self.lap_list = self._get_relational_lap_list()
        self.all_kg_dict = self._get_all_kg_dict()
        self.all_h_list, self.all_r_list, self.all_t_list, self.all_v_list = self._get_all_kg_data()



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

        self.n_relations = max(kg_np[:, 1]) + 1
        self.n_entities = max(max(kg_np[:, 0]), max(kg_np[:, 2])) + 1
        self.n_triples = len(kg_np)

        kg_dict, relation_dict = _construct_kg(kg_np)

        return kg_np, kg_dict, relation_dict

    def _get_relational_adj_list(self):
        t1 = time()
        adj_mat_list = []
        adj_r_list = []

        def _np_mat2sp_adj(np_mat, row_pre, col_pre):
            n_all = self.n_users + self.n_entities
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

        R, R_inv = _np_mat2sp_adj(self.train_data, row_pre=0, col_pre=self.n_users)
        adj_mat_list.append(R)
        adj_r_list.append(0)

        adj_mat_list.append(R_inv)
        adj_r_list.append(self.n_relations + 1)
        print('\tconvert ratings into adj mat done.')

        for r_id in self.relation_dict.keys():
            K, K_inv = _np_mat2sp_adj(np.array(self.relation_dict[r_id]), row_pre=self.n_users, col_pre=self.n_users)
            adj_mat_list.append(K)
            adj_r_list.append(r_id + 1)

            adj_mat_list.append(K_inv)
            adj_r_list.append(r_id + 2 + self.n_relations)
        print('\tconvert %d relational triples into adj mat done. @%.4fs' %(len(adj_mat_list), time()-t1))

        self.n_relations = len(adj_r_list)
        # print('\tadj relation list is', adj_r_list)

        return adj_mat_list, adj_r_list

    def _get_relational_lap_list(self):
        def _bi_norm_lap(adj):
            print("adj", adj.toarray())
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
            lap_list = [_bi_norm_lap(adj) for adj in self.adj_list]
            print('\tgenerate bi-normalized adjacency matrix.')
        else:
            lap_list = [_si_norm_lap(adj) for adj in self.adj_list]
            print('\tgenerate si-normalized adjacency matrix.')
        print("#lap_list: {}".format(len(lap_list)))
        return lap_list

    def _get_all_kg_dict(self):
        all_kg_dict = collections.defaultdict(list)
        for l_id, lap in enumerate(self.lap_list):

            rows = lap.row
            cols = lap.col

            for i_id in range(len(rows)):
                head = rows[i_id]
                tail = cols[i_id]
                relation = self.adj_r_list[l_id]

                all_kg_dict[head].append((tail, relation))
        return all_kg_dict

    def _get_all_kg_data(self):
        def _reorder_list(org_list, order):
            new_list = np.array(org_list)
            new_list = new_list[order]
            return new_list

        all_h_list, all_t_list, all_r_list = [], [], []
        all_v_list = []

        for l_id, lap in enumerate(self.lap_list):
            all_h_list += list(lap.row)
            all_t_list += list(lap.col)
            all_v_list += list(lap.data)
            all_r_list += [self.adj_r_list[l_id]] * len(lap.row)

        assert len(all_h_list) == sum([len(lap.data) for lap in self.lap_list])

        # resort the all_h/t/r/v_list,
        # ... since tensorflow.sparse.softmax requires indices sorted in the canonical lexicographic order
        print('\treordering indices...')
        org_h_dict = dict()

        for idx, h in enumerate(all_h_list):
            if h not in org_h_dict.keys():
                org_h_dict[h] = [[],[],[]]

            org_h_dict[h][0].append(all_t_list[idx])
            org_h_dict[h][1].append(all_r_list[idx])
            org_h_dict[h][2].append(all_v_list[idx])
        print('\treorganize all kg data done.')

        sorted_h_dict = dict()
        for h in org_h_dict.keys():
            org_t_list, org_r_list, org_v_list = org_h_dict[h]
            sort_t_list = np.array(org_t_list)
            sort_order = np.argsort(sort_t_list)

            sort_t_list = _reorder_list(org_t_list, sort_order)
            sort_r_list = _reorder_list(org_r_list, sort_order)
            sort_v_list = _reorder_list(org_v_list, sort_order)

            sorted_h_dict[h] = [sort_t_list, sort_r_list, sort_v_list]
        print('\tsort meta-data done.')

        od = collections.OrderedDict(sorted(sorted_h_dict.items()))
        new_h_list, new_t_list, new_r_list, new_v_list = [], [], [], []

        for h, vals in od.items():
            new_h_list += [h] * len(vals[0])
            new_t_list += list(vals[0])
            new_r_list += list(vals[1])
            new_v_list += list(vals[2])


        assert sum(new_h_list) == sum(all_h_list)
        assert sum(new_t_list) == sum(all_t_list)
        assert sum(new_r_list) == sum(all_r_list)
        # try:
        #     assert sum(new_v_list) == sum(all_v_list)
        # except Exception:
        #     print(sum(new_v_list), '\n')
        #     print(sum(all_v_list), '\n')
        print('\tsort all data done.')


        return new_h_list, new_r_list, new_t_list, new_v_list


class DataLoader(object):
    def __init__(self, data_name, num_neighbor_hop=3, seed=1234):
        print("\n{}->".format(data_name))
        self._data_name = data_name
        self._num_neighbor_hop = num_neighbor_hop
        self._rng = np.random.RandomState(seed=seed)
        data_dir =  os.path.realpath(os.path.join(os.path.abspath(__file__), '..', "datasets", data_name))
        train_file = os.path.join(data_dir, "train.txt")
        train_pairs, train_user_dict = self.load_train_interaction(train_file)
        test_file = os.path.join(data_dir, "test.txt")
        test_pairs, test_user_dict = self.load_test_interaction(test_file)
        kg_file = os.path.join(data_dir, "kg_final.txt")
        kg_triples_np = self.load_kg_plus_inverse(kg_file)

        ## stack user ids after entities
        self.user_mapping = {i: i+self.num_KG_entities for i in range(self.num_users)}

        self.train_pairs = ((train_pairs[0] + self.num_KG_entities).astype(np.int32),
                            train_pairs[1].astype(np.int32))
        self.train_user_dict= {self.user_mapping[k]: np.unique(v).astype(np.int32)
                               for k,v in train_user_dict.items()}
        self.test_pairs = ((test_pairs[0] + self.num_KG_entities).astype(np.int32),
                           test_pairs[1].astype(np.int32))
        self.test_user_dict= {self.user_mapping[k]: np.unique(v).astype(np.int32)
                              for k,v in test_user_dict.items()}
        user_item_triplet = np.zeros((self.num_train*2, 3), dtype=np.int32)
        user_item_triplet[:, 0] = np.concatenate((self.train_pairs[0], self.train_pairs[1]))
        user_item_triplet[:, 1] = np.concatenate(((np.ones(self.num_train)*self.num_KG_relations).astype(np.int32),
                                                  (np.ones(self.num_train)*(self.num_KG_relations+1)).astype(np.int32)))
        user_item_triplet[:, 2] = np.concatenate((self.train_pairs[1], self.train_pairs[0]))

        ###              |<item>  <att entity> | <user>
        ### <item>       |=====================|=======
        ### <att entity> |=====================|+++++++
        ### <user>       |=======|+++++++++++++++++++++
        all_triplet = np.vstack((kg_triples_np,  user_item_triplet)).astype(np.int32)
        self.all_triplet_np = all_triplet
        self.all_triplet_dp = pd.DataFrame(all_triplet, columns=['h', 'r', 't'], dtype=np.int32)
        assert np.max(all_triplet) + 1 == self.num_all_entities
        print("The whole graph: {} entities, {} relations, {} triplets".format(
            self.num_all_entities, self.num_all_relations, self.num_all_triplets))

    def generate_whole_g(self):
        g = dgl.DGLGraph()
        g.add_nodes(self.num_all_entities)
        ### TODO when adding edges, remember to reverse the direction, e.g., t-->h
        g.add_edges(self.all_triplet_np[:, 2], self.all_triplet_np[:, 0])
        #print(g)
        all_etype = self.all_triplet_np[:, 1]
        return g, all_etype

    @property
    def num_all_entities(self):
        return self._n_KG_entities + self._n_users
    @property
    def num_all_relations(self):
        return self._n_KG_relations + 2
    @property
    def num_all_triplets(self):
        return self.all_triplet_np.shape[0]

    def _filter_neighbor(self, item_ids, kg_pd):
        new_pd = None
        item_ids = np.unique(item_ids)
        #print("original:\t#triplets:{}, #entities:{}".format(kg_pd.shape[0], item_ids.size))
        for i in range(self._num_neighbor_hop):
            new_pd = kg_pd[kg_pd.h.isin(item_ids)]
            item_ids = np.unique(np.concatenate((new_pd['h'].values, new_pd['t'].values)))
            #print("{}-> new #triplets:{}, #entities:{}".format(i+1, new_pd.shape[0], item_ids.size))
        new_entity_ids = item_ids
        #print("original:\t#h:{}, #r:{}, #t:{}".format(kg_pd['h'].nunique(), kg_pd['r'].nunique(), kg_pd['t'].nunique()))
        #print("filtered:\t#h:{}, #r:{}, #t:{}".format(new_pd["h"].nunique(), new_pd["r"].nunique(),
        #                                              new_pd["t"].nunique()))
        #print("Filtered G:\t#entities:{}, #triplets:{}".format(new_entity_ids.size, new_pd.shape[0]))
        return new_entity_ids, new_pd

    def load_kg_plus_inverse(self, file_name):
        kg_pd = pd.read_csv(file_name, sep=" ", names=['h', "r", "t"], engine='python')
        kg_pd = kg_pd.sort_values(by=['h'])
        unique_rel = kg_pd['r'].nunique()
        entity_ids = np.unique(np.concatenate((kg_pd['h'].values, kg_pd['t'].values)))

        if kg_pd["r"].nunique() != kg_pd["r"].max()+1:
            relation_mapping = {old_id: idx for idx, old_id in enumerate(np.unique(kg_pd["r"].values))}
            kg_pd['r'] = list(map(relation_mapping.get, kg_pd['r'].values))

        print("#KG entities:{}, relations:{}, triplet:{}, #head:{}, #tail:{}".format(
            entity_ids.size, unique_rel, kg_pd.shape[0], kg_pd['h'].nunique(), kg_pd['t'].nunique()))

        ### TODO filter neighbors are not implemented here
        # entity_ids, new_kg_pd = self._filter_neighbor(self.item_ids, kg_pd)
        # ## construct kg_np by relabelling node ids
        # kg_np = np.zeros((new_kg_pd.shape[0], 3))
        # self.entity_mapping = {old_id: idx for idx, old_id in enumerate(entity_ids)}
        # kg_np[:, 0] = list(map(self.entity_mapping.get, new_kg_pd['h'].values))
        # kg_np[:, 2] = list(map(self.entity_mapping.get, new_kg_pd['t'].values))
        # if new_kg_pd["r"].nunique() != kg_pd['r'].nunique():
        #     print("Relation mapping..")
        #     relation_mapping = {old_id: idx for idx, old_id in enumerate(np.unique(new_kg_pd["r"].values))}
        #     kg_np[:, 1] = list(map(relation_mapping.get, new_kg_pd['r'].values))
        # else:
        #     kg_np[:, 1] = new_kg_pd["r"].values

        rev = kg_pd.copy()
        rev = rev.rename({'h':'t', 't':'h'})
        rev['r'] += unique_rel
        new_kg_pd = pd.concat([kg_pd, rev], ignore_index=True)

        self._n_KG_relations = new_kg_pd["r"].nunique()
        self._n_KG_entities = entity_ids.size
        self._n_KG_triples = new_kg_pd.shape[0]
        print("#KG entities:{}, relations:{}, triplet:{}, #head:{}, #tail:{}".format(
            self.num_KG_entities, self.num_KG_relations, self.num_KG_triples,
            new_kg_pd['h'].nunique(), new_kg_pd['t'].nunique()))

        return new_kg_pd.values

    def _get_all_kg_dict(self):
        all_kg_dict = collections.defaultdict(list)
        for h, r, t in self.all_triplet_np:
            if h in all_kg_dict.keys():
                all_kg_dict[h].append((t, r))
            else:
                all_kg_dict[h] = [(t, r)]
        self.all_kg_dict = all_kg_dict

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
        pos_triples = self.all_kg_dict[h]
        n_pos_triples = len(pos_triples)
        pos_id = np.random.randint(low=0, high=n_pos_triples)
        t = pos_triples[pos_id][0]
        r = pos_triples[pos_id][1]
        return r, t
    def _sample_neg_triples_for_h(self, h, r):
        while True:
            t = np.random.randint(low=0, high=self.num_all_entities)
            if (t, r) in self.all_kg_dict[h]:
                continue
            else:
                return t

    def KG_sampler(self, batch_size):
        ### generate negative triplets
        print("#Core", multiprocessing.cpu_count() // 2)
        self._get_all_kg_dict()
        exist_heads = list(self.all_kg_dict.keys())
        n_batch = self.num_all_triplets // batch_size + 1
        i = 0
        #print("Batch_size:{}, #batches:{}".format(batch_size, n_batch))
        while i < n_batch:
            pool = multiprocessing.Pool(multiprocessing.cpu_count() // 2)
            i += 1
            if batch_size <= len(exist_heads):
                heads = rd.sample(exist_heads, batch_size)
            else:
                heads = [rd.choice(exist_heads) for _ in range(batch_size)]
            pos_r_batch, pos_t_batch, neg_t_batch = [], [], []
            ### generate positive samples
            print("Heads", heads)

            print("Pool mapping .......")
            pos_r_batch, pos_t_batch = pool.map(self._sample_pos_triples_for_h, heads)
            print("pos_r_batch", pos_r_batch)
            print("pos_t_batch", pos_t_batch)

            for h in heads:
                pos_rs, pos_ts = self._sample_pos_triples_for_h(h)
                pos_r_batch.append(pos_rs)
                pos_t_batch.append(pos_ts)
                neg_ts = self._sample_neg_triples_for_h(h, pos_rs)
                neg_t_batch.append(neg_ts)
            yield heads, pos_r_batch, pos_t_batch, neg_t_batch

    def KG_sampler2(self, batch_size, sequential=True):
        if batch_size < 0:
            batch_size = self.num_all_triplets
        else:
            batch_size = min(self.num_all_triplets, batch_size)
        if sequential:
            shuffled_idx = self._rng.permutation(self.num_all_triplets)
            all_triplet_np = self.all_triplet_np[shuffled_idx, :]
            for start in range(0, self.num_all_triplets, batch_size):
                end = min(start+batch_size, self.num_all_triplets)
                h = all_triplet_np[start: end][:, 0]
                r = all_triplet_np[start: end][:, 1]
                pos_t = all_triplet_np[start: end][:, 2]
                neg_t = self._rng.choice(self.num_all_entities, end-start, replace=True).astype(np.int32)
                yield h, r, pos_t, neg_t
        else:
            while True:
                sel = self._rng.choice(self.num_all_triplets, batch_size, replace=False)
                h = self.all_triplet_np[sel][:, 0]
                r = self.all_triplet_np[sel][:, 1]
                pos_t = self.all_triplet_np[sel][:, 2]
                neg_t = self._rng.choice(self.num_all_entities, batch_size, replace=True).astype(np.int32)
                yield h, r, pos_t, neg_t

    @property
    def num_KG_entities(self):
        return self._n_KG_entities

    @property
    def num_KG_relations(self):
        return self._n_KG_relations

    @property
    def num_KG_triples(self):
        return self._n_KG_triples

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
        self.train_user_ids = np.unique(src) ### requires to remap later
        self.item_ids = np.unique(dst)

        self._n_users = self.train_user_ids.size
        self._n_items = self.item_ids.size
        self._n_train = src.size
        print("Train data: #user:{}, #item:{}, #pairs:{}".format(
            self.num_users, self.num_items, self.num_train))
        return (src, dst), train_user_dict

    def load_test_interaction(self, file_name):
        src, dst, test_user_dict = self._load_interaction(file_name)
        unique_users = np.unique(src)
        unique_items = np.unique(dst)
        for ele in unique_users:
            assert ele in self.train_user_ids
        for ele in unique_items:
            assert ele in self.item_ids
        self._n_test = src.size
        print("Test data: #user:{}, #item:{}, #pairs:{}".format(
            unique_users.size, unique_items.size, self.num_test))
        return (src, dst), test_user_dict

    def _sample_pos_items_for_u(self, u, num):
        pos_items = self.train_user_dict[u]
        n_pos_items = len(pos_items)
        pos_batch = []
        while True:
            if len(pos_batch) == num: break
            pos_id = np.random.randint(low=0, high=n_pos_items)
            pos_i_id = pos_items[pos_id]

            if pos_i_id not in pos_batch:
                pos_batch.append(pos_i_id)
        return pos_batch
    def _sample_neg_items_for_u(self, u, num):
        neg_items = []
        while True:
            if len(neg_items) == num: break
            neg_i_id = np.random.randint(low=0, high=self.num_items)
            if neg_i_id not in self.train_user_dict[u] and neg_i_id not in neg_items:
                neg_items.append(neg_i_id)
        return neg_items
    def _generate_user_pos_neg_items(self, batch_size):
        if batch_size <= self.num_users:
            ## test1
            users = rd.sample(self.exist_users, batch_size)
        else:
            users = [rd.choice(self.exist_users) for _ in range(batch_size)]

        pos_items, neg_items = [], []
        for u in users:
            pos_items += self._sample_pos_items_for_u(u, 1)
            neg_items += self._sample_neg_items_for_u(u, 1)

        return users, pos_items, neg_items

    def CF_batchwise_sampler(self, batch_size):
        self.exist_users = self.train_user_dict.keys()
        if batch_size < 0:
            batch_size = self.num_train-1
            n_batch = 1
        else:
            batch_size = min(batch_size, self.num_train)
            n_batch = self.num_train // batch_size + 1

        i = 0
        #print("Batch_size:{}, #batches:{}".format(batch_size, n_batch))
        while i < n_batch:
            i += 1
            user_ids, item_ids, neg_item_ids = self._generate_user_pos_neg_items(batch_size)
            new_entity_ids, new_pd = self._filter_neighbor(np.concatenate((user_ids, item_ids, neg_item_ids)),
                                                           self.all_triplet_dp)
            etype = new_pd['r'].values
            ### relabel nodes to have consecutive node ids
            uniq_v, edges = np.unique((new_pd['h'].values, new_pd['t'].values), return_inverse=True)
            src, dst = np.reshape(edges, (2, -1))
            g = dgl.DGLGraph()
            g.add_nodes(uniq_v.size)
            g.add_edges(dst, src)
            ### map user_ids and items_ids into indicies in the graph
            node_map = {ele: idx for idx, ele in enumerate(uniq_v)}
            user_ids = np.array(list(map(node_map.get, user_ids)), dtype=np.int32)
            item_ids = np.array(list(map(node_map.get, item_ids)), dtype=np.int32)
            neg_item_ids = np.array(list(map(node_map.get, neg_item_ids)), dtype=np.int32)

            yield user_ids, item_ids, neg_item_ids, g, uniq_v, etype

    def CF_pair_sampler(self, batch_size):
        self.exist_users = list(self.train_user_dict.keys())
        print("exist_user", len(self.exist_users))
        if batch_size < 0:
            batch_size = self.num_train
            n_batch = 1
        elif batch_size > self.num_train:
            batch_size = min(batch_size, self.num_train)
            n_batch = 1
        else:
            n_batch = self.num_train // batch_size + 1
        print("num_train", self.num_train, "batch_size", batch_size, "n_batch", n_batch)
        i = 0
        #print("Batch_size:{}, #batches:{}".format(batch_size, n_batch))
        while i < n_batch:
            i += 1
            user_ids, item_ids, neg_item_ids = self._generate_user_pos_neg_items(batch_size)
            yield user_ids, item_ids, neg_item_ids

    def old_CF_all_sampler(self, batch_size, segment='train', sequential=True):
        if segment == 'train':
            node_pairs = self.train_pairs
            all_num = self.num_train
        else:
            raise NotImplementedError
        if batch_size < 0:
            batch_size = all_num
        else:
            batch_size = min(batch_size, all_num)
        if batch_size == all_num:
            neg_item_ids = self._rng.choice(self.item_ids, batch_size, replace=True).astype(np.int32)
            uniq_v = np.arange(self.num_all_entities)
            g, etype = self.generate_whole_g()
            yield node_pairs[0], node_pairs[1], neg_item_ids, g, uniq_v, etype
        if sequential:
            shuffled_idx = self._rng.permutation(self.num_train)
            all_user_ids = node_pairs[0][shuffled_idx]
            all_item_idx = node_pairs[1][shuffled_idx]
            for start in range(0, all_num, batch_size):
                ## choose user item pairs
                end = min(start+batch_size, all_num)
                user_ids = all_user_ids[start: end]
                item_ids = all_item_idx[start: end]
                ## obtain k-hop neighbors
                new_entity_ids, new_pd = self._filter_neighbor(np.concatenate((user_ids, item_ids)),
                                                               self.all_triplet_dp)
                etype = new_pd['r'].values
                ### relabel nodes to have consecutive node ids
                uniq_v, edges = np.unique((new_pd['h'].values, new_pd['t'].values), return_inverse=True)
                src, dst = np.reshape(edges, (2, -1))
                g = dgl.DGLGraph()
                g.add_nodes(uniq_v.size)
                g.add_edges(dst, src)
                ### map user_ids and items_ids into indicies in the graph
                node_map = {ele: idx for idx, ele in enumerate(uniq_v)}
                user_ids = np.array(list(map(node_map.get, user_ids)), dtype=np.int32)
                item_ids = np.array(list(map(node_map.get, item_ids)), dtype=np.int32)
                neg_item_ids = self._rng.choice(item_ids, end-start, replace=True).astype(np.int32)

                yield user_ids, item_ids, neg_item_ids, g, uniq_v, etype
        else:
            while True:
                sel = self._rng.choice(all_num, batch_size, replace=False)
                user_ids = node_pairs[0][sel]
                item_ids = node_pairs[1][sel]
                new_entity_ids, new_pd = self._filter_neighbor(np.concatenate((user_ids, item_ids)),
                                                               self.all_triplet_dp)
                etype = new_pd['r'].values
                ### relabel nodes to have consecutive node ids
                uniq_v, edges = np.unique((new_pd['h'].values, new_pd['t'].values), return_inverse=True)
                src, dst = np.reshape(edges, (2, -1))
                g = dgl.DGLGraph()
                g.add_nodes(uniq_v.size)
                g.add_edges(dst, src)
                ### map user_ids and items_ids into indicies in the graph
                node_map = {ele: idx for idx, ele in enumerate(uniq_v)}
                user_ids = np.array(list(map(node_map.get, user_ids)), dtype=np.int32)
                item_ids = np.array(list(map(node_map.get, item_ids)), dtype=np.int32)
                neg_item_ids = self._rng.choice(item_ids, batch_size, replace=True).astype(np.int32)
                yield user_ids, item_ids, neg_item_ids, g, uniq_v, etype

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


if __name__ == '__main__':
    DataLoader("yelp2018")
    DataLoader("last-fm")
    DataLoader("amazon-book")

