import numpy as np
import pandas as pd
import os
import argparse
import json

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



