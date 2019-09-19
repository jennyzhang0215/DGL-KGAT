import numpy as np
import os
import dgl
import pandas as pd

class DataLoader(object):
    def __init__(self, data_name, full_batch=True, num_neighbor_hop=2, seed=1234):
        self._data_name = data_name
        self._full_batch = full_batch
        self._num_neighbor_hop = num_neighbor_hop
        self._rng = np.random.RandomState(seed=seed)
        data_dir =  os.path.realpath(os.path.join(os.path.abspath(__file__), '..', "datasets", data_name))

        train_file = os.path.join(data_dir, "train.txt")
        train_pairs = self.load_train_interaction(train_file)
        test_file = os.path.join(data_dir, "test.txt")
        test_pairs = self.load_test_interaction(test_file)

        kg_file = os.path.join(data_dir, "kg_final.txt")
        self.kg_triples_np = self.load_kg_filter_neighbor(kg_file)

        ## stack user ids after entities
        self.user_mapping = {i: i+self.num_KG_entities for i in range(self.num_users)}
        self.train_pairs = ((train_pairs[0] + self.num_KG_entities).astype(np.int32),
                            train_pairs[1].astype(np.int32))
        self.test_pairs = ((test_pairs[0] + self.num_KG_entities).astype(np.int32),
                           test_pairs[1].astype(np.int32))
        user_item_triplet = np.zeros((self.num_train*2, 3), dtype=np.int32)
        user_item_triplet[:, 0] = np.concatenate((self.train_pairs[0], self.train_pairs[1]))
        user_item_triplet[:, 1] = np.concatenate(((np.ones(self.num_train)*self.num_KG_relations).astype(np.int32),
                                               (np.ones(self.num_train) * (self.num_KG_relations+1)).astype(np.int32)))
        user_item_triplet[:, 2] = np.concatenate((self.train_pairs[1], self.train_pairs[0]))
        ### reverse the (head, relation, tail) direction, because we need tail --> head
        all_triplet = np.vstack((self.kg_triples_np[:, [2,1,0]],  user_item_triplet)).astype(np.int32)

        ###              |<item>  <att entity> | <user>
        ### <item>       |=====================|=======
        ### <att entity> |=====================|+++++++
        ### <user>       |=======|+++++++++++++++++++++
        g = dgl.DGLGraph()
        g.add_nodes(self.num_all_entities)
        g.add_edges(all_triplet[:, 0], all_triplet[:, 2])

        self.g = g
        self.etype = all_triplet[:, 1]

    @property
    def num_all_entities(self):
        return self._n_KG_entities + self._n_users
    @property
    def num_all_relations(self):
        return self._n_KG_relations + 2

    def load_kg_filter_neighbor(self, file_name):
        kg_pd = pd.read_csv(file_name, sep=" ", names=['h', "r", "t"], engine='python')
        kg_pd = kg_pd.sort_values(by=['h'])

        item_ids = self.item_ids
        print("original_items", item_ids.size)
        for i in range(self._num_neighbor_hop):
            new_pd = kg_pd[kg_pd.h.isin(item_ids)]
            item_ids = np.unique(np.concatenate((new_pd['h'].values, new_pd['t'].values)))
            print("{}-> new #triplets:{}, new #items:{}".format(i+1, new_pd.shape[0], item_ids.size))

        print("original: #h:{}, #r:{}, #t:{}".format(kg_pd['h'].nunique(), kg_pd['r'].nunique(), kg_pd['t'].nunique()))
        print("#h:{}, #r:{}, #t:{}".format(new_pd["h"].nunique(), new_pd["r"].nunique(), new_pd["t"].nunique()))

        kg_np = np.zeros((new_pd.shape[0], 3))
        self.entity_mapping = {old_id: idx for idx, old_id in enumerate(item_ids)}
        kg_np[:, 0] = list(map(self.entity_mapping.get, new_pd['h'].values))
        kg_np[:, 2] = list(map(self.entity_mapping.get, new_pd['t'].values))
        if new_pd["r"].nunique() != kg_pd['r'].nunique():
            relation_mapping = {old_id: idx for idx, old_id in enumerate(np.unique(new_pd["r"].values))}
            print("Relation mapping..")
            kg_np[:, 1] = list(map(relation_mapping.get, new_pd['r'].values))
        else:
            kg_np[:, 1] = new_pd["r"].values
        self._n_KG_relations = new_pd["r"].nunique()
        self._n_KG_entities = item_ids.size
        self._n_KG_triples = new_pd.shape[0]
        print("{}: #KG entities:{}, #KG relations:{}, #KG triplet:{}, #head:{}, #tail:{}".format(
            self._data_name, self.num_KG_entities, self.num_KG_relations, self.num_KG_triples,
            new_pd['h'].nunique(), new_pd['t'].nunique()))
        return kg_np


    def _load_kg2triplet(self, file_name):
        """ Load the KG txt file into the 2-d numpy array

        Parameters
        ----------
        file_name: str

        Returns
        -------
        np.array Shape:(num_triples, 3)
        """
        kg_triples_np = np.loadtxt(file_name, dtype=np.int32)
        ### check whether entity ids are continuous and start from 0
        assert np.unique(kg_triples_np[:, 1]).size == max(kg_triples_np[:, 1]) + 1
        assert np.unique(np.concatenate((kg_triples_np[:, 0], kg_triples_np[:, 2]))).size == \
               max(max(kg_triples_np[:, 0]), max(kg_triples_np[:, 2])) + 1
        self._n_KG_relations = np.unique(kg_triples_np[:, 1]).size
        self._n_KG_entities = np.unique(np.concatenate((kg_triples_np[:, 0], kg_triples_np[:, 2]))).size
        self._n_KG_triples = kg_triples_np.shape[0]
        print("{}: #KG entities:{}, #KG relations:{}, #KG triplet:{}, #head:{}, #tail:{}".format(
            self._data_name, self.num_KG_entities, self.num_KG_relations, self.num_KG_triples,
            np.unique(kg_triples_np[:, 0]).size, np.unique(kg_triples_np[:, 2]).size))


        return kg_triples_np







    def KG_sampler(self, batch_size, sequential=True, segment='train'):
        if batch_size < 0:
            batch_size = self.num_KG_triples
        else:
            batch_size = min(self.num_KG_triples, batch_size)
        if sequential:
            for start in range(0, self.num_KG_triples, batch_size):
                end = min(start+batch_size, self.num_KG_triples)
                h = self.kg_triples_np[start: end][:, 0]
                r = self.kg_triples_np[start: end][:, 1]
                pos_t = self.kg_triples_np[start: end][:, 2]
                neg_t = self._rng.choice(self.num_KG_entities, end-start, replace=True).astype(np.int32)
                yield h, r, pos_t, neg_t
        else:
            while True:
                sel = self._rng.choice(self.num_KG_triples, batch_size, replace=False)
                h = self.kg_triples_np[sel][:, 0]
                r = self.kg_triples_np[sel][:, 1]
                pos_t = self.kg_triples_np[sel][:, 2]
                neg_t = self._rng.choice(self.num_KG_entities, batch_size, replace=True).astype(np.int32)
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
        lines = open(file_name, 'r').readlines()
        for l in lines:
            tmps = l.strip()
            inters = [int(i) for i in tmps.split(' ')]
            user_id, item_ids = inters[0], inters[1:]
            item_ids = list(set(item_ids))
            for i_id in item_ids:
                src.append(user_id)
                dst.append(i_id)
        print("{}: #user:{}, #item:{}, #pair:{}".format(
            self._data_name, np.unique(src).size, np.unique(dst).size, len(src)))
        return np.array(src, dtype=np.int32), np.array(dst, dtype=np.int32)

    def load_train_interaction(self, file_name):
        src, dst = self._load_interaction(file_name)
        ### check whether the user id / item id are continuous and starting from 0
        assert np.unique(src).size == max(src) + 1
        assert np.unique(dst).size == max(dst) + 1
        self.train_user_ids = np.unique(src) ### requires to remap later
        self.item_ids = np.unique(dst)
        print("item_ids", self.item_ids)
        self._n_users = self.train_user_ids.size
        self._n_items = self.item_ids.size
        self._n_train = src.size
        print("Train data: #user:{}, #item:{}, #pairs:{}".format(
            self.num_users, self.num_items, self.num_train))
        return (src, dst)
    def load_test_interaction(self, file_name):
        src, dst = self._load_interaction(file_name)
        unique_users = np.unique(src)
        unique_items = np.unique(dst)
        for ele in unique_users:
            assert ele in self.train_user_ids
        for ele in unique_items:
            assert ele in self.item_ids
        self._n_test = src.size
        print("Test data: #user:{}, #item:{}, #pairs:{}".format(
            self.num_users, self.num_items, self.num_test))
        return (src, dst)


    def CF_sampler(self, segment='train'):
        if segment == 'train':
            node_pairs = self.train_pairs
        elif segment == 'test':
            node_pairs = self.test_pairs
        else:
            raise NotImplementedError
        while True:
            ### full batch sampling
            neg_item_ids = self._rng.choice(self.num_items, self.num_train, replace=True).astype(np.int32)
            yield node_pairs[0], node_pairs[1], neg_item_ids

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

