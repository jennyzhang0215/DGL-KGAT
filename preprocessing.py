import pandas as pd
import os
import numpy as np

data_name = "yelp2018"


data_dir = os.path.realpath(os.path.join(os.path.abspath(__file__), '..', "datasets", data_name))
kg_file = os.path.join(data_dir, "kg_final.txt")

kg_pd = pd.read_csv(kg_file, sep=" ", names=['h', "r", "t"], engine='python')
kg_pd = kg_pd.sort_values(by=['h'])
print(kg_pd)

num_hop = 3
item_ids = np.arange(45538).tolist()

for i in range(num_hop):
    print(i, "curr_items", len(item_ids))
    new_pd = kg_pd[kg_pd.h.isin(item_ids)]
    print(new_pd.shape)
    item_ids= np.unique(np.concatenate((new_pd['h'].values, new_pd['t'].values)))
    print("item_ids",item_ids.size)
    item_ids = item_ids.tolist()
