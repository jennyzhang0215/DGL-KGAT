import numpy as np
import pandas as pd
import os
import argparse

def read_kg(file_name, data_name, UNIQUE_ITEMs):
    kg_pd = pd.read_csv(file_name, sep=" ", names=["h", "r", "t"])
    print("All relations: {}".format(kg_pd['r'].nunique()))
    try:
        assert kg_pd['t'].min() > UNIQUE_ITEMs-1
    except:
        tail_kg_pd = kg_pd[kg_pd['t'] < UNIQUE_ITEMs]
        print(tail_kg_pd)
        print("relation", tail_kg_pd['r'].nunique(), tail_kg_pd['r'].unique())
    item_kg_pg = kg_pd[kg_pd['h'] < UNIQUE_ITEMs]
    r_ecount = item_kg_pg.groupby("r")["t"].nunique()
    r_ecount.to_csv(os.path.join(data_name, "_rel_ecount"), sep=" ", index_label=['r'], header=["count"])

    other_kg_pg = kg_pd[kg_pd['h'] > UNIQUE_ITEMs-1]
    assert kg_pd.shape[0] == item_kg_pg.shape[0] + other_kg_pg.shape[0]

    r_e_freq = item_kg_pg.groupby(['r'])['t'].value_counts()
    r_e_freq.to_csv(os.path.join(data_name, "_rel_entity_freq"), sep=" ", index_label=['r', 't'], header=["freq"])

    ## {r_id: [entity_id]}
    rel_entity_dict = {}
    for (r_id, entity_id), freq in r_e_freq.items():
        # print("r_id", r_id,"entity_id", entity_id, "freq", freq)
        if r_id in rel_entity_dict:
            rel_entity_dict[r_id].append(entity_id)
        else:
            rel_entity_dict[r_id] = []
            rel_entity_dict[r_id].append(entity_id)
    #print(rel_entity_dict)

    pos_file = open(os.path.join(data_name, "_fea_bit_record"), "w")
    pos_file.write("_fea_idx rel_id entity_id\n")
    fea_pos_dict = {}
    fea_idx = 0
    for r_id, e_l in rel_entity_dict.items():
        fea_pos_dict[r_id] = {}
        for e in e_l:
            fea_pos_dict[r_id][e] = fea_idx
            pos_file.write("{} {} {}\n".format(fea_idx, r_id, e))
            fea_idx += 1
    #print(fea_pos_dict)
    pos_file.close()

    total_fea_bit = fea_idx
    print("total_fea_bit", total_fea_bit)
    fea_np = np.zeros((UNIQUE_ITEMs, total_fea_bit))
    for _, row in item_kg_pg.iterrows():
        h,r,t = row['h'], row['r'], row['t']
        fea_np[h][fea_pos_dict[r][t]] = 1.0
    print(fea_np)

    sparsity = fea_np.sum() / (UNIQUE_ITEMs * total_fea_bit)
    print("sparsity: {:.2f}%".format(sparsity*100))
    np.savez(os.path.join(data_name, "fea.npz"),
             item_fea = fea_np)

def load_interaction(file_name):
    src = []
    dst = []
    user_dict = {}
    lines = open(file_name, 'r').readlines()
    for l in lines:
        tmps = l.strip()
        inters = [i for i in tmps.split(' ')]
        if len(inters) > 1:
            user_id, item_ids = inters[0], inters[1:]
            item_ids = list(set(item_ids))
            for i_id in item_ids:
                src.append(user_id)
                dst.append(i_id)
            user_dict[user_id] = item_ids
    data_pd = pd.DataFrame(np.stack((src, dst)).T, columns=["src", "dst"])
    return data_pd, user_dict

def split_train_interaction(file_name, UNIQUE_ITEMs, val_ratio = 0.1):
    train_src = []
    train_dst = []
    train_user_dict = {}
    valid_src = []
    valid_dst = []
    valid_user_dict = {}

    lines = open(file_name, 'r').readlines()
    for l in lines:
        tmps = l.strip()
        inters = [i for i in tmps.split(' ')]
        if len(inters) > 1:
            user_id, item_ids = inters[0], inters[1:]
            item_ids = np.array(list(set(item_ids)))
            all_num = len(item_ids)
            val_num = int(np.ceil(all_num * val_ratio)) if all_num > 1 else 0
            idx = np.random.permutation(all_num)
            train_items = item_ids[idx[val_num : ]]
            valid_items = item_ids[idx[: val_num]]
            for i_id in train_items:
                train_src.append(user_id)
                train_dst.append(i_id)
            for i_id in valid_items:
                valid_src.append(user_id)
                valid_dst.append(i_id)
            train_user_dict[user_id] = train_items.tolist()
            valid_user_dict[user_id] = valid_items.tolist()

    train_data_pd = pd.DataFrame(np.stack((train_src, train_dst)).T, columns=["src", "dst"])
    valid_data_pd = pd.DataFrame(np.stack((valid_src, valid_dst)).T, columns=["src", "dst"])

    ### move the items those are not in the current training data from validation set to training set
    sampled_train_items = train_data_pd['dst'].unique()
    for i in range(UNIQUE_ITEMs):
        if str(i) not in sampled_train_items:
            filter_valid_pd = valid_data_pd[valid_data_pd['dst'] == str(i)]
            user_id_str = filter_valid_pd.iloc[0]['src']
            print("\n", str(i))
            print(filter_valid_pd)
            print("Previous: Train {}, Valid {}".format(len(train_user_dict[user_id_str]),
                                                          len(valid_user_dict[user_id_str])))
            train_user_dict[user_id_str].append(str(i))
            valid_user_dict[user_id_str].remove(str(i))
            print("Updated: Train {}, Valid {}".format(len(train_user_dict[user_id_str]),
                                                       len(valid_user_dict[user_id_str])))
            if len(valid_user_dict[user_id_str]) == 0:
                del valid_user_dict[user_id_str]
                assert user_id_str not in valid_user_dict


    return train_user_dict, valid_user_dict

def convert_dict2pd(user_dict):
    src = []
    dst = []
    for user_str, item_str_l in user_dict.items():
        user_id = int(user_str)
        if len(item_str_l) > 0:
            item_ids = [int(i) for i in item_str_l]
            for i_id in item_ids:
                src.append(user_id)
                dst.append(i_id)
    data_pd = pd.DataFrame(np.stack((src, dst)).T, columns=["src", "dst"])
    return data_pd


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="parameters")
    parser.add_argument('--data_name', type=str, default="yelp2018", help='use GPU')
    args = parser.parse_args()

    if args.data_name == "last-fm":
        UNIQUE_ITEMs = 48123
    elif args.data_name == "amazon-book":
        UNIQUE_ITEMs = 24915
    elif args.data_name == "yelp2018":
        UNIQUE_ITEMs = 45538
    else:
        raise NotImplementedError
    print("#items", UNIQUE_ITEMs)
    kg_file = os.path.join(args.data_name, "kg_final.txt")
    read_kg(kg_file, args.data_name, UNIQUE_ITEMs)

    """"
    train_file = os.path.join(args.data_name, "train.txt")
    test_file = os.path.join(args.data_name, "test.txt")
    original_train_pd, _ = load_interaction(train_file)
    print("Original--> #user: {}, #item: {} #pairs: {}".format(original_train_pd["src"].nunique(),
                                                              original_train_pd["dst"].nunique(),
                                                              original_train_pd.shape[0] ))
    train_user_dict, valid_user_dict = split_train_interaction(train_file, UNIQUE_ITEMs)
    train_data_pd = convert_dict2pd(train_user_dict)
    valid_data_pd = convert_dict2pd(valid_user_dict)
    print("Current --> #user: {}, #item: {}, #pairs: {}".format(train_data_pd["src"].nunique(),
                                                                train_data_pd["dst"].nunique(),
                                                                train_data_pd.shape[0] ))
    print("Valid --> #user: {}, #item: {}, #pairs: {}".format(valid_data_pd["src"].nunique(),
                                                              valid_data_pd["dst"].nunique(),
                                                              valid_data_pd.shape[0]))



    test_data_pd, test_user_dict = load_interaction(test_file)
    print("Test --> #user: {}, #item: {}, #pairs: {}".format(test_data_pd["src"].nunique(),
                                                             test_data_pd["dst"].nunique(),
                                                             test_data_pd.shape[0]))
    f_train = open(os.path.join(args.data_name, "train1.txt"), "w")
    for k, v in train_user_dict.items():
        str_v = " ".join(v)
        f_train.write("{} {}\n".format(k, str_v))
    f_train.close()

    f_valid = open(os.path.join(args.data_name, "valid1.txt"), "w")
    for k, v in valid_user_dict.items():
        str_v = " ".join(v)
        f_valid.write("{} {}\n".format(k, str_v))
    f_valid.close()

    # train_data_pd.to_csv(os.path.join(args.data_name, "train.pd"), index=False)
    # valid_data_pd.to_csv(os.path.join(args.data_name, "valid.pd"), index=False)
    # test_data_pd.to_csv(os.path.join(args.data_name, "test.pd"), index=False)
    # with open(os.path.join(args.data_name, "train.json"), "w") as json_file:
    #     json.dump(train_user_dict, json_file)
    # with open(os.path.join(args.data_name, "valid.json"), "w") as json_file:
    #     json.dump(valid_user_dict, json_file)
    # with open(os.path.join(args.data_name, "test.json"), "w") as json_file:
    #     json.dump(test_user_dict, json_file)
    """



