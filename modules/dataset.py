import numpy as np
import random as rd
import os
import scipy as sp
import dgl
import torch as th

class DataLoader(object):
    def __init__(self, data_name):
        self._data_name = data_name
        data_dir =  os.path.realpath(os.path.join(os.path.abspath(__file__), '..', "..", "datasets", data_name))

        # ----------get number of entities and relations & then load kg data from kg_file into the DGLGraph------------.
        kg_file = os.path.join(data_dir, "kg_final.txt")
        self.kg_triples_np = self.load_kg2triplet(kg_file)

        train_file = os.path.join(data_dir, "train.txt")
        train_user_id_np, train_item_id_np = self.load_rating(train_file)
        ## remapping user ids after entities
        self._user_mapping = {i: i+self.num_KG_entities for i in range(self.num_users)}
        recsys_triplet = np.zeros((self.num_train*2, 3))
        for i in range(self.num_train):
            recsys_triplet[i, 0] = self._user_mapping[train_user_id_np[i]]
            ### add the user2item interact relation
            recsys_triplet[i, 1] = self.num_KG_relations
            recsys_triplet[i, 2] = train_item_id_np[i]
        for i in range(0, self.num_train):
            recsys_triplet[i+self.num_train, 0] = train_item_id_np[i]
            ### add the item2user interacted relation
            recsys_triplet[i+self.num_train, 1] = self.num_KG_relations + 1
            recsys_triplet[i+self.num_train, 2] = self._user_mapping[train_user_id_np[i]]
        ### reverse the (head, relation, tail) direction, because we need tail --> head
        all_triplet = np.vstack((self.kg_triples_np[:, [2,1,0]],  recsys_triplet))
        g = dgl.DGLGraph()
        g.add_nodes(self.num_all_entity)
        g.add_edges(all_triplet[:, 0], all_triplet[:, 2])
        self.g = g
        self.rel_np = all_triplet[:, 1]

    @property
    def num_all_entities(self):
        return self._n_KG_entities + self._n_users
    @property
    def num_all_relations(self):
        return self._n_KG_relations + 2
    @property
    def num_KG_entities(self):
        return self._n_KG_entities
    @property
    def num_KG_relations(self):
        return self._n_KG_relations
    @property
    def num_KG_triples(self):
        return self._n_KG_triples


    def load_kg2triplet(self, file_name):
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
        print("{}: #KG entities:{}, #KG relations:{}, #KG triplet:{}".format(
            self._data_name, self.num_KG_entities, self.num_KG_relations, self.num_KG_triples))
        return kg_triples_np

    def create_KG_sampler(self, batch_size, neg_sample_size=1, mode=None, num_workers=5,
                          shuffle=True, exclude_positive=False):
        raise NotImplementedError

    # reading train & test interaction data
    def _load_rating(self, file_name):
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
        return np.array(src, dtype=np.int32), np.array(dst, dtype=np.int32)

    def load_rating(self, file_name):
        src, dst = self._load_rating(file_name)
        ### check whether the user id / item id are continuous and starting from 0
        assert np.unique(src).size == max(src) + 1
        assert np.unique(dst).size == max(dst) + 1
        self._n_users = np.unique(src).size
        self._n_items = np.unique(dst).size
        self._n_train = src.size
        print("{}: #user:{}, #items:{}, #train:{}".format(
            self._data_name, self.num_users, self.num_items, self.num_train))
        return src, dst

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

