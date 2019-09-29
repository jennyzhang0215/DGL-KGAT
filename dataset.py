import numpy as np
import os
import dgl
import pandas as pd
import collections
import random as rd

class DataLoader(object):
    def __init__(self, data_name, full_batch=True, num_neighbor_hop=2, seed=1234):
        print("\n{}->".format(data_name))
        self._data_name = data_name
        self._full_batch = full_batch
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
        ### reverse the (head, relation, tail) direction, because we need tail --> head
        all_triplet = np.vstack((kg_triples_np,  user_item_triplet)).astype(np.int32)
        self.all_triplet_np = all_triplet
        self.all_triplet_dp = pd.DataFrame(all_triplet, columns=['h', 'r', 't'], dtype=np.int32)
        print("The whole graph: {} entities, {} relations, {} triplets".format(
            self.num_all_entities, self.num_all_relations, self.num_all_triplets))
        assert np.max(all_triplet) + 1 == self.num_all_entities
        ###              |<item>  <att entity> | <user>
        ### <item>       |=====================|=======
        ### <att entity> |=====================|+++++++
        ### <user>       |=======|+++++++++++++++++++++
        print("Overall KG #entities:{}, #triplets:{}".format(self.num_all_entities, self.all_triplet_np.shape[0]) )

    def generate_whole_g(self):
        g = dgl.DGLGraph()
        g.add_nodes(self.num_all_entities)
        ### TODO when adding edges, remember to reverse the direction, e.g., t-->h
        g.add_edges(self.all_triplet_np[:, 2], self.all_triplet_np[:, 0])
        #print(g)
        all_etype = self.all_triplet_np[:, 1]

        ### compute support
        def _calc_norm(x):
            x = x.asnumpy().astype('float32')
            x[x == 0.] = np.inf
            x = mx.array(1. / np.sqrt(x))
            return x.as_in_context(self._ctx).expand_dims(1)
        user_ci = []
        user_cj = []
        movie_ci = []
        movie_cj = []
        for r in self.possible_rating_values:
            r = str(r)
            user_ci.append(graph['rev-%s' % r].in_degrees())
            movie_ci.append(graph[r].in_degrees())
            if self._symm:
                user_cj.append(graph[r].out_degrees())
                movie_cj.append(graph['rev-%s' % r].out_degrees())
            else:
                user_cj.append(mx.nd.zeros((self.num_user,)))
                movie_cj.append(mx.nd.zeros((self.num_movie,)))
        user_ci = _calc_norm(mx.nd.add_n(*user_ci))
        movie_ci = _calc_norm(mx.nd.add_n(*movie_ci))
        if self._symm:
            user_cj = _calc_norm(mx.nd.add_n(*user_cj))
            movie_cj = _calc_norm(mx.nd.add_n(*movie_cj))
        else:
            user_cj = mx.nd.ones((self.num_user,), ctx=self._ctx)
            movie_cj = mx.nd.ones((self.num_movie,), ctx=self._ctx)
        graph.nodes['user'].data.update({'ci': user_ci, 'cj': user_cj})
        graph.nodes['movie'].data.update({'ci': movie_ci, 'cj': movie_cj})
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
        unique_rel = np.unique(kg_pd['h'].values).size
        entity_ids = np.unique(np.concatenate((kg_pd['h'].values, kg_pd['t'].values)))

        if kg_pd["r"].nunique() != kg_pd["r"].max()+1:
            relation_mapping = {old_id: idx for idx, old_id in enumerate(np.unique(kg_pd["r"].values))}
            kg_pd['r'] = list(map(relation_mapping.get, kg_pd['r'].values))

        print("#KG entities:{}, relations:{}, triplet:{}, #head:{}, #tail:{}".format(
            entity_ids.size, kg_pd['r'].nunique(), kg_pd.shape[0], kg_pd['h'].nunique(), kg_pd['t'].nunique()))

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

    def _sample_pos_triples_for_h(self, h, num):
        pos_triples = self.all_kg_dict[h]
        n_pos_triples = len(pos_triples)
        pos_rs, pos_ts = [], []
        while True:
            if len(pos_rs) == num: break
            pos_id = np.random.randint(low=0, high=n_pos_triples, size=1)[0]
            t = pos_triples[pos_id][0]
            r = pos_triples[pos_id][1]
            if r not in pos_rs and t not in pos_ts:
                pos_rs.append(r)
                pos_ts.append(t)
        return pos_rs, pos_ts

    def _sample_neg_triples_for_h(self, h, r, num):
        neg_ts = []
        while True:
            if len(neg_ts) == num: break
            t = np.random.randint(low=0, high=self.num_all_entities, size=1)[0]
            if (t, r) not in self.all_kg_dict[h] and t not in neg_ts:
                neg_ts.append(t)
        return neg_ts

    def KG_sampler(self, batch_size, sequential=True):
        ### generate negative triplets
        self._get_all_kg_dict()
        exist_heads = self.all_kg_dict.keys()
        n_batch = self.num_all_triplets // batch_size + 1
        i = 0
        while i < n_batch:
            i += 1
            if batch_size <= len(exist_heads):
                heads = rd.sample(exist_heads, batch_size)
            else:
                heads = [rd.choice(exist_heads) for _ in range(batch_size)]
            pos_r_batch, pos_t_batch, neg_t_batch = [], [], []
            for h in heads:
                pos_rs, pos_ts = self._sample_pos_triples_for_h(h, 1)
                pos_r_batch += pos_rs
                pos_t_batch += pos_ts
                neg_ts = self._sample_neg_triples_for_h(h, pos_rs[0], 1)
                neg_t_batch += neg_ts
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
            pos_id = np.random.randint(low=0, high=n_pos_items, size=1)[0]
            pos_i_id = pos_items[pos_id]

            if pos_i_id not in pos_batch:
                pos_batch.append(pos_i_id)
        return pos_batch
    def _sample_neg_items_for_u(self, u, num):
        neg_items = []
        while True:
            if len(neg_items) == num: break
            neg_i_id = np.random.randint(low=0, high=self.num_items, size=1)[0]
            if neg_i_id not in self.train_user_dict[u] and neg_i_id not in neg_items:
                neg_items.append(neg_i_id)
        return neg_items
    def _generate_user_pos_neg_items(self, batch_size):
        if batch_size <= self.num_users:
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
        if batch_size < 0:
            batch_size = self.num_train
            n_batch = 1
        else:
            batch_size = min(batch_size, self.num_train)
            n_batch = self.num_train // batch_size + 1
        i = 0
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

