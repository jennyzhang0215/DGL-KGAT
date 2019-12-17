import os
import argparse
import numpy as np
import pandas as pd
import time
import pickle


def download_data(data_name, path, url):
    if not os.path.exists(os.path.join(path, data_name)):
        assert url is not None
        print('{} not found. Downloading from {}'.format(data_name, url))
        from urllib import request
        import zipfile, subprocess

        fn = os.path.join(path, "{}.zip".format(data_name))
        while True:
            try:
                with zipfile.ZipFile(fn) as zf:
                    zf.extractall(path)
                print('Unzip finished.')
                ### change the data dir name
                #subprocess.call(['mv', '{}/ml-100k'.format(path), '{}/movielens/raw/ml-100k'.format(path)])
                break
            except Exception:
                data_file = request.urlopen(url)
                with open(fn, 'wb') as output:
                    output.write(data_file.read())
                print('Download finished. Unzipping the file...')

class Dataset(object):
    def __init__(self, dir, data_name, uv_files, user_file, item_file, val_ratio, url=None):
        super(Dataset, self).__init__()
        self._data_name = data_name
        download_data(data_name, dir, url)
        data_dir = os.path.join(dir, data_name)
        self._new_dir = os.path.join(data_dir,  'data')
        if not os.path.isdir(self._new_dir):
            os.makedirs(self._new_dir)

        if data_name in ["amazon-book", "yelp2018"]:
            ### read uv_files
            sep = ' '
            if len(uv_files) == 1:
                raise NotImplementedError
            elif len(uv_files) == 2:
                assert val_ratio is not None
                train_val_uv_pd, user_map, item_map = \
                    self.convert_u_v_dict2pair(self.read2u_v_dict(os.path.join(data_dir, uv_files[0]), sep=sep),
                                              re_mapping=True, map_dict=None)
                train_uv_pd, val_uv_pd= self.split_val(train_val_uv_pd, val_ratio, mode="seen")
                test_uv_pd, _, _ = self.convert_u_v_dict2pair(self.read2u_v_dict(os.path.join(data_dir, uv_files[1]), sep=sep),
                                                             re_mapping=True, map_dict=[user_map, item_map])
            else:
                assert len(uv_files) == 3
                train_uv_pd, user_map, item_map = \
                    self.convert_u_v_dict2pair(self.read2u_v_dict(os.path.join(data_dir, uv_files[0]), sep=sep),
                                              re_mapping=True, map_dict=None)
                val_uv_pd, _, _ = self.convert_u_v_dict2pair(self.read2u_v_dict(os.path.join(data_dir, uv_files[1]), sep=sep),
                                                            re_mapping=True, map_dict=[user_map, item_map])
                train_val_uv_pd = pd.concat([train_uv_pd, val_uv_pd], ignore_index=True)
                test_uv_pd, _, _ = self.convert_u_v_dict2pair(self.read2u_v_dict(os.path.join(data_dir, uv_files[2]), sep=sep),
                                                             re_mapping=True, map_dict=[user_map, item_map])
            self.user_map, self.item_map = user_map, item_map
            self.train_uv_pd, self.val_uv_pd, self.train_val_uv_pd, self.test_uv_pd =\
                train_uv_pd, val_uv_pd, train_val_uv_pd, test_uv_pd

            if user_file is not None:
                self.user_kg_pd, user_entity_map, user_rel_map = \
                    self.read_kg2pd(os.path.join(data_dir, user_file), sep=sep, _map=user_map)
            else:
                self.user_kg_pd = None

            if item_file is not None:
                self.item_kg_pd, item_entity_map, item_rel_map = \
                    self.read_kg2pd(os.path.join(data_dir, item_file), sep=sep, _map=user_map)
            else:
                self.item_kg_pd = None

            self.save_all()
            self.load_all()

        else:
            raise NotImplementedError
    def _save_info(self, log_file, info):
        f = open(log_file, "a")
        f.write("--------------------------------------------------\n")
        f.write(info)
        print(info)
        f.close()

    def save_all(self):
        uv_data = [self.train_uv_pd, self.val_uv_pd, self.test_uv_pd]
        uv_file_name = ["uv_train.pd", "uv_val.pd", "uv_test.pd"]
        for data, file_name in zip(uv_data, uv_file_name):
            if data is not None:
                file_name = os.path.join(self._new_dir, file_name)
                data.to_csv(file_name, sep="\t", index=False)

        kg_data = [self.user_kg_pd, self.item_kg_pd]
        kg_file_name = ["kg_user.pd", "kg_item.pd"]
        for data, file_name in zip(kg_data, kg_file_name):
            if data is not None:
                file_name = os.path.join(self._new_dir, file_name)
                data.to_csv(file_name, sep="\t", index=False)

    def load_all(self):
        file_names = ["uv_train.pd", "uv_val.pd", "uv_test.pd"] + ["kg_user.pd", "kg_item.pd"]
        for file_name in file_names:
            file_name = os.path.join(self._new_dir, file_name)
            if os.path.isfile(file_name):
                data = pd.read_csv(file_name, sep="\t", header=0)
                print(file_name)
                print(data, "\n\n")
        np_file_names = ["item_fea_id.npy", "item_fea_emb.npy"]
        for file_name in np_file_names:
            file_name = os.path.join(self._new_dir, file_name)
            if os.path.isfile(file_name):
                data = np.load(file_name)
                print(file_name)
                print(data.shape, data, "\n\n")

    def _save_pickle(self, data_dict, save_file):
        f = open(save_file, "wb")
        pickle.dump(data_dict, f)
        f.close()
    def _load_pickle(self, file_name):
        f = open(file_name, "rb")
        data_dict = pickle.load(f)
        f.close()
        return data_dict
    def _save_map(self, data_dict, save_file):
        ## the input dictionary is {old_id: new_id}
        ## convert to {new_id: old_id}
        kv_dic = {v:k for k,v in data_dict.items()}
        d_pd = pd.DataFrame.from_dict(kv_dic, orient="index", columns=["old_id"])
        d_pd.to_csv(save_file, sep="\t", index_label="new_id")

    def read2u_v_dict(self, file_name, sep):
        user_dict = {}
        lines = open(file_name, 'r').readlines()
        for l in lines:
            tmps = l.strip()
            inters = [i for i in tmps.split(sep)]
            if len(inters) > 1:
                user_id, item_ids = inters[0], inters[1:]
                item_ids = list(set(item_ids)) ## remove duplicated pairs
                user_dict[user_id] = item_ids
        return user_dict

    def convert_u_v_dict2pair(self, dict, re_mapping=False, map_dict=None, is_digit=True):
        start = time.time()
        pairs = []
        for k,v_l in dict.items():
            for v in v_l:
                if is_digit:
                    pairs.append((int(k), int(v)))
                else:
                    pairs.append((k, v))
        pair_pd = pd.DataFrame(data=pairs, index=None, columns=['u', 'v'], copy=False)
        if re_mapping and map_dict is None:
            user_ids = pair_pd['u'].unique()
            item_ids = pair_pd['v'].unique()
            user_ids.sort()
            item_ids.sort()
            user_map = {id:idx for idx, id in enumerate(user_ids)}
            item_map = {id:idx for idx, id in enumerate(item_ids)}
            map_dict = [user_map, item_map]
            self._save_map(user_map, os.path.join(self._new_dir, "_map_use_id"))
            self._save_map(item_map, os.path.join(self._new_dir, "_map_item_id"))
        elif re_mapping and map_dict is not None:
            assert len(map_dict) == 2
        if re_mapping:
            pair_pd['u'] = list(map(map_dict[0].get, pair_pd['u']))
            pair_pd['v'] = list(map(map_dict[1].get, pair_pd['v']))
        print("{:.1f}s for convert_u_v_dict2pair() ...".format(time.time() - start))
        return pair_pd, map_dict[0], map_dict[1]

    def split_val(self, uv_pd, val_ratio, mode="seen"):
        start = time.time()
        num_train = uv_pd.shape[0]
        all_u = uv_pd['u'].unique()
        all_v = uv_pd['v'].unique()
        num_val = int(num_train * val_ratio)
        if mode in ["random", "seen"]:
            idx_all = np.random.permutation(num_train)
            val_uv_pd = uv_pd.iloc[idx_all[:num_val]]
            train_uv_pd = uv_pd.iloc[idx_all[num_val:]]
            if mode == "seen":
                ### move the users/items those are not in the current training data
                ### from validation set to training set
                sampled_train_users = train_uv_pd['u'].unique()
                for u in all_u:
                    if u not in sampled_train_users:
                        filter_valid_index = val_uv_pd[val_uv_pd['u'] == u].index[0]
                        v_id = val_uv_pd.loc[filter_valid_index]['v']
                        add_row = pd.DataFrame({'u': [u], "v": [v_id]})
                        train_uv_pd = train_uv_pd.append(add_row, ignore_index=True)
                        val_uv_pd = val_uv_pd.drop(index = filter_valid_index)
                sampled_train_items = train_uv_pd['v'].unique()
                for v in all_v:
                    if v not in sampled_train_items:
                        filter_valid_index = val_uv_pd[val_uv_pd['v'] == v].index[0]
                        u_id = val_uv_pd.loc[filter_valid_index]['u']
                        add_row = pd.DataFrame({'u': [u_id], "v": [v]})
                        train_uv_pd = train_uv_pd.append(add_row, ignore_index=True)
                        val_uv_pd = val_uv_pd.drop(index = filter_valid_index)
        else:
            raise NotImplementedError
        print("{:.1f}s for split_val() ...".format(time.time() - start))
        return train_uv_pd, val_uv_pd

    def read_kg2pd(self, file_name, sep, _map=None):
        start = time.time()
        kg_pd = pd.read_csv(file_name, sep=sep, names=["h", "r", "t"])
        kg_pd = kg_pd.drop_duplicates() ### remove duplicate items
        unique_ent = kg_pd["h"].append(kg_pd["t"]).unique()
        unique_rel = kg_pd["r"].unique()
        unique_ent.sort()
        unique_rel.sort()
        if _map is not None:
            ## remap the entity ids, items rank first and then non-item ids
            entity_map = {}
            curr_id = len(_map)
            for ele in unique_ent:
                if ele not in _map:
                    entity_map[ele] = curr_id
                    curr_id += 1
                else:
                    entity_map[ele] = _map[ele]
            kg_pd['h'] = list(map(entity_map.get, kg_pd["h"]))
            kg_pd['t'] = list(map(entity_map.get, kg_pd["t"]))
            self._save_map(entity_map, os.path.join(self._new_dir, "_map_ent"))
            rel_map = {rel: idx for idx, rel in enumerate(unique_rel)}
            kg_pd["r"] = list(map(rel_map.get, kg_pd["r"]))
            self._save_map(rel_map, os.path.join(self._new_dir, "_map_rel"))
        else:
            ### the entities are already mapped with consecutive ids
            entity_map, rel_map = None, None
            assert max(unique_ent)+1 == unique_ent.size
            assert max(unique_rel)+1 == unique_rel.size
        print("{:.1f}s for read_kg2pd() ...".format(time.time() - start))
        return kg_pd, entity_map, rel_map

    def __repr__(self):
        info = "{}:\n".format(self._data_name)
        info += "{} users, {} items, {} train, {} valid, {} test pairs\n".format(
            len(self.user_map), len(self.item_map),
            self.train_uv_pd.shape[0], self.val_uv_pd.shape[0], self.test_uv_pd.shape[0])
        if self.user_kg_pd is not None:
            info += "User KG: {} entities, {} relations, {} triplets\n".format(
                self.user_kg_pd["h"].append(self.user_kg_pd["t"]).nunique(), self.user_kg_pd["r"].nunique(),
                self.user_kg_pd.shape[0])
        if self.item_kg_pd is not None:
            info += "Item KG: {} entities, {} relations, {} triplets\n".format(
                self.item_kg_pd["h"].append(self.item_kg_pd["t"]).nunique(), self.item_kg_pd["r"].nunique(),
                self.item_kg_pd.shape[0])
        return info




DATA_PATH = os.path.realpath(os.path.join(os.path.abspath(__file__), '..'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Reproduce KGAT using DGL")
    parser.add_argument('--data_name', nargs='?', default='yelp2018',
                        help='Choose a dataset from {"amazon-book", "yelp2018"}')
    args = parser.parse_args()

    URL = 'https://s3.us-east-2.amazonaws.com/dgl.ai/dataset/{}.zip'.format(args.data_name)

    Dataset(dir=DATA_PATH, data_name=args.data_name,
            uv_files=["train.txt", "test.txt"],
            user_file=None, item_file=["kg_final.txt"],
            val_ratio=0.1, url=URL)
