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
        self.kg, relation_ids = self.load_kg2graph(kg_file)
        train_file = os.path.join(data_dir, "train.txt")
        self.rating_g = self.load_rating2graph(train_file)
        #test_file = os.path.join(data_dir, "test.txt")
        #self.test_data,  = self._load_rating2graph(test_file)
        print(len(self.kg), len(self.rating_g), len(self.kg + self.rating_g))
        self.all_g = dgl.hetero_from_relations(self.kg + self.rating_g)
        self.all_g.edges["relation"].data['type'] = relation_ids
        print("Data Statistic:\n\t#user:{}, #items:{}, #interactions:{}, #entities:{}, #relations:{}, #triplets:{}".format(
            self.num_users, self.num_items, self.num_train, self.num_entities, self.num_relations, self.num_triples))
        print("#users:", self.all_g.number_of_nodes('user'))
        print("#entities:", self.all_g.number_of_nodes('entity'))
        print("#interactions:", self.all_g.number_of_edges(('user', 'interact', 'entity')))
        print("self.all_g['relation'].number_of_edges()", self.all_g['relation'].number_of_edges())
        print("g.ntypes", self.all_g.ntypes)
        print("g.etypes", self.all_g.etypes)
        print("self.all_g['relation'].edata", self.all_g['relation'].edata)
        print("metagraph", self.all_g.metagraph.edges())

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
        self.kg_triples_np = kg_triples_np
        self._n_relations = np.unique(kg_triples_np[:, 1]).size
        assert self._n_relations == kg_triples_np[:, 1].max() + 1
        self._n_entities = np.unique(np.concatenate((kg_triples_np[:, 0], kg_triples_np[:, 2]))).size
        assert self._n_entities == max(max(kg_triples_np[:, 0]), max(kg_triples_np[:, 2])) + 1
        self._n_triples = kg_triples_np.shape[0]

        src = kg_triples_np[:, 0]
        etype_id = kg_triples_np[:, 1]
        dst = kg_triples_np[:, 2]
        coo = sp.sparse.coo_matrix((np.ones(self.num_triples), (dst, src)),
                                   shape=[self.num_entities, self.num_entities])
        g = dgl.graph(coo, ntype='entity', etype='relation')
        return [g], th.LongTensor(etype_id)

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
        g = dgl.bipartite((src, dst), 'user', 'interact', 'entity',
                          card=(self.num_users, self.num_entities))
        rev_g = dgl.bipartite((dst, src), 'entity', 'interacted_by', 'user',
                              card=(self.num_entities, self.num_users))
        return [g, rev_g]

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

