import numpy as np
import os
import dgl

class DataLoader(object):
    def __init__(self, data_name, full_batch=True, seed=1234):
        self._data_name = data_name
        self._full_batch = full_batch
        self._rng = np.random.RandomState(seed=seed)
        data_dir =  os.path.realpath(os.path.join(os.path.abspath(__file__), '..', "..", "datasets", data_name))

        kg_file = os.path.join(data_dir, "kg_final.txt")
        self.kg_triples_np = self.load_kg2triplet(kg_file)

        train_file = os.path.join(data_dir, "train.txt")
        train_user_id_np, train_item_id_np = self.load_train_interaction(train_file)
        ## remapping user ids after entities
        self._user_mapping = {i: i+self.num_KG_entities for i in range(self.num_users)}
        self.train_pairs = ((train_user_id_np + self.num_KG_entities).astype(np.int32),
                            train_item_id_np.astype(np.int32))
        recsys_triplet = np.zeros((self.num_train*2, 3), dtype=np.int32)
        recsys_triplet[:, 0] = np.concatenate((self.train_pairs[0], self.train_pairs[1]))
        recsys_triplet[:, 1] = np.concatenate(((np.ones(self.num_train)*self.num_KG_relations).astype(np.int32),
                                               (np.ones(self.num_train) * (self.num_KG_relations+1)).astype(np.int32)))
        recsys_triplet[:, 2] = np.concatenate((self.train_pairs[1], self.train_pairs[0]))
        ### reverse the (head, relation, tail) direction, because we need tail --> head
        all_triplet = np.vstack((self.kg_triples_np[:, [2,1,0]],  recsys_triplet)).astype(np.int32)
        g = dgl.DGLGraph()
        g.add_nodes(self.num_all_entities)
        g.add_edges(all_triplet[:, 0], all_triplet[:, 2])
        self.g = g
        self.rel_np = all_triplet[:, 1]

        ### generate testing pairs
        test_file = os.path.join(data_dir, "test.txt")
        test_user_id_np, test_item_id_np = self._load_interaction(test_file)
        self.test_pairs = ((test_user_id_np + self.num_KG_entities).astype(np.int32),
                            test_item_id_np.astype(np.int32))


    @property
    def num_all_entities(self):
        return self._n_KG_entities + self._n_users
    @property
    def num_all_relations(self):
        return self._n_KG_relations + 2

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
        print("{}: #KG entities:{}, #KG relations:{}, #KG triplet:{}, #head:{}, #tail:{}".format(
            self._data_name, self.num_KG_entities, self.num_KG_relations, self.num_KG_triples,
            np.unique(kg_triples_np[:, 0]).size, np.unique(kg_triples_np[:, 2]).size))
        return kg_triples_np

    def create_KG_sampler(self, batch_size, neg_sample_size=1, mode=None, num_workers=5,
                          shuffle=True, exclude_positive=False):
        raise NotImplementedError

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
        self._n_users = np.unique(src).size
        self._n_items = np.unique(dst).size
        self._n_train = src.size
        print("{}: #user:{}, #item:{}, #train pair:{}".format(
            self._data_name, self.num_users, self.num_items, self.num_train))
        return src, dst

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

