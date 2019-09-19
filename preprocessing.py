import pandas as pd
import os
import numpy as np

data_name = "yelp2018"


data_dir = os.path.realpath(os.path.join(os.path.abspath(__file__), '..', "datasets", data_name))
kg_file = os.path.join(data_dir, "kg_final.txt")

kg_pd = pd.read_csv(kg_file, sep=" ", names=['h', "r", "t"], engine='python')
kg_pd = kg_pd.sort_values(by=['h'])
print(kg_pd)
kg_pd = kg_pd.sort_values(by=['t'])
print(kg_pd)


num_hop = 2
item_ids = np.arange(45538).tolist()
print("item_ids", len(item_ids))

for i in range(num_hop):
    new_pd = kg_pd[kg_pd.h.isin(item_ids)]
    print(new_pd)
    ds = np.unique(np.concatenate((new_pd['h'].values, new_pd['t'].values)))
    print("item_ids", len(item_ids))
    item_ids = item_ids.tolist()
