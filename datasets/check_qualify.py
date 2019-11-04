import argparse
import os

def _load_interaction(file_name):
    pairs = []
    lines = open(file_name, 'r').readlines()
    for l in lines:
        tmps = l.strip()
        inters = [int(i) for i in tmps.split(' ')]
        if len(inters) > 1:
            user_id, item_ids = inters[0], inters[1:]
            item_ids = list(set(item_ids))
            for i_id in item_ids:
                pairs.append((user_id, i_id))
    return pairs

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Data name")
    parser.add_argument('--data_name', nargs='?', default='last-fm',
                        help='Choose a dataset from {yelp2018, last-fm, amazon-book}')
    args = parser.parse_args()

    train_file = os.path.join(args.data_name, 'train.txt')
    test_file = os.path.join(args.data_name, 'test.txt')

    train_pairs = _load_interaction(train_file)
    test_pairs = _load_interaction(test_file)

    train_dict = {}
    duplicate_num = 0
    for p in train_pairs:
        if p in train_dict:
            duplicate_num += 1
            print(p)
            train_dict[p] +=1
        else:
            train_dict[p] = 1

    appear_in_train = 0
    for p in test_pairs:
        if p in train_dict:
            appear_in_train += 1
    print("Total test pair: {}, seen in train: {} with {:.2f}%".format(len(test_pairs), appear_in_train,
                                                                  appear_in_train/len(test_pairs)))

