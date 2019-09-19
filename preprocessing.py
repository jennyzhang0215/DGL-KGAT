import pandas as pd
import os
data_name = "yelp2018"


data_dir = os.path.realpath(os.path.join(os.path.abspath(__file__), '..', "datasets", data_name))
kg_file = os.path.join(data_dir, "kg_final.txt")

kg_pd = pd.read_csv(kg_file, sep=" ", names=['h', "r", "t"], engine='python')
kg_pd = kg_pd.sort_values(by=['h'])
print(kg_pd)
kg_pd = kg_pd.sort_values(by=['t'])
print(kg_pd)
