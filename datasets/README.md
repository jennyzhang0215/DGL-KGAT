# Datasets

The input data files are required to match the following input data formats. We also provide common preprocessing API for your convenience.

## Input file format
  - User-item interaction files (train, test, valid): 
    - File: `<user id> <item id> <interact type (optinal)> <timestamp (optinal)>`
  - Item data: 
    - Item-KG: `<head id> <relation id> <tail id>`

#### Requirements
  - The user/item ids should be consecutive and start from 0.
  - The user/item ids in item-KG and should be the same as those in the interaction files.

You can download and process the yelp2018 dataset via `process_kgat_data.py`. The processed datasets are stored in `[data_name]/data/`.
