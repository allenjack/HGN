# The HGN model for Sequential Recommendation
The implementation of the paper:

*Chen Ma, Peng Kang, and Xue Liu, "**Hierarchical Gating Networks for Sequential Recommendation**", in the 25th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (**KDD 2019**)* 

Arxiv: https://arxiv.org/abs/1906.09217

**Please cite our paper if you use our code. Thanks!**

Author: Chen Ma (allenmc1230@gmail.com)

**Feel free to send me an email if you have any questions.**

## Environments

- python 3.6
- PyTorch (version: 1.0.0)
- numpy (version: 1.15.0)
- scipy (version: 1.1.0)
- sklearn (version: 0.19.1)


## Dataset

In our experiments, the *movielens-20M* dataset is from https://grouplens.org/datasets/movielens/20m/, the *Amazon-CDs* and *Amazon-Books* datasets are from http://jmcauley.ucsd.edu/data/amazon/, the *GoodReads-Children* and *GoodReads-Comics* datasets are from https://sites.google.com/eng.ucsd.edu/ucsdbookgraph/home. (If you need the data after preprocessing, please send me an email).

The ```XXX_tem_sequences.pkl``` file is a list of lists that stores the inner item id of each user in a chronological order, e.g., ```user_records[0]=[item_id0, item_id1, item_id2,...]```.

The ```XXX_user_mapping.pkl``` file is a list that maps the user inner id to its original id, e.g., ```user_mapping[0]=A2SUAM1J3GNN3B```.

The ```XXX_item_mapping.pkl``` file is similar to ```XXX_user_mapping.pkl```.

## Example to run the code

Data preprocessing:

The code for data preprocessing is put in the ```/preprocessing``` folder. ```Amazon_CDs.ipynb``` provides an example on how to transform the raw data into the ```.pickle``` files that used in our program.

Train and evaluate the model (you are strongly recommended to run the program on a machine with GPU):

```
python run.py
```

## Acknowledgment
The sequence segmentation (interactions.py) is heavily built on [Spotlight](https://github.com/maciejkula/spotlight). Thanks for the amazing work.

