import numpy as np
import random as rd
import os
import scipy as sp
import dgl
import torch as th

class DataLoader(object):
    def __init__(self, data_name):
        data_dir =  os.path.realpath(os.path.join(os.path.abspath(__file__), '..', "..", "datasets", data_name))

        # ----------get number of entities and relations & then load kg data from kg_file into the DGLGraph------------.
        kg_file = os.path.join(data_dir, "kg_final.txt")
        self.kg = self.load_kg2graph(kg_file)
        train_file = os.path.join(data_dir, "train.txt")
        test_file = os.path.join(data_dir, "test.txt")
        self.rating_g, = self.load_rating2graph(train_file)
        #self.test_data,  = self._load_rating2graph(test_file)
        self.all_g = dgl.hetero_from_relations([self.kg, self.rating_g])
        print("Data Statistic:\n\t#user:{}, #items:{}, #entities:{}, #relations:{}".format(
            self.num_users, self.num_items, self.num_entities, self.num_relations))

    @property
    def num_entities(self):
        return self._n_entities
    @property
    def num_relations(self):
        return self._n_relations
    @property
    def num_triples(self):
        return self._n_triples

    def load_kg2graph(self, file_name):
        """ Load the KG txt file into the 2-d numpy array

        Parameters
        ----------
        file_name: str

        Returns
        -------
        np.array Shape:(num_triples, 3)
        """
        kg_triples_np = np.loadtxt(file_name, dtype=np.int32)

        self._n_relations = np.unique(kg_triples_np[:, 1]).size
        assert self._n_relations == kg_triples_np[:, 1].max() + 1
        self._n_entities = np.unique(np.concatenate((kg_triples_np[:, 0], kg_triples_np[:, 2]))).size
        assert self._n_entities == max(max(kg_triples_np[:, 0]), max(kg_triples_np[:, 2])) + 1
        self._n_triples = kg_triples_np.shape[0]

        src = kg_triples_np[:, 0]
        etype_id = kg_triples_np[:, 1]
        dst = kg_triples_np[:, 2]
        coo = sp.sparse.coo_matrix((np.ones(self.num_triples), (src, dst)),
                                   shape=[self.num_entities, self.num_entities])
        g = dgl.graph(coo, ntype='entity', etype='relation')
        g.ndata['id'] = th.arange(g.number_of_nodes())
        g.edata['id'] = th.LongTensor(etype_id)
        return g

    def create_KG_sampler(self, batch_size, neg_sample_size=1, mode=None, num_workers=5,
                          shuffle=True, exclude_positive=False):
        raise NotImplementedError

    # reading train & test interaction data
    def _load_rating(self, file_name):
        src = []
        dst = []
        lines = open(file_name, 'r').readlines()
        user_l = []
        item_l = []
        for l in lines:
            tmps = l.strip()
            inters = [int(i) for i in tmps.split(' ')]
            user_id, item_ids = inters[0], inters[1:]
            item_ids = list(set(item_ids))
            user_l.append(user_id)
            item_l += item_ids
            for i_id in item_ids:
                src.append(user_id)
                dst.append(i_id)
        return src, dst, user_l, item_l

    def load_rating2graph(self, file_name):
        src, dst, user_l, item_l = self._load_rating(file_name)
        assert np.unique(user_l).size == len(user_l)
        assert len(user_l) == max(user_l) + 1
        self._n_users = np.unique(user_l).size
        self._n_items = np.unique(item_l).size
        assert self.num_items == max(item_l) + 1
        self._n_train = len(src)
        g = dgl.bipartite((np.array(src, dtype=np.int32), np.array(dst, dtype=np.int32)),
                          'user', 'interact', 'items')
        return g

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
    dataset = DataLoader("yelp2018")

