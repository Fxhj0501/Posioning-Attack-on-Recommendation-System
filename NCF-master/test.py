import numpy as np
import pandas as pd
import scipy.sparse as sp

import torch.utils.data as data

import config
def load_all(test_num=100):
    """ We load all the three file here to save time in each epoch. """
    train_data = pd.read_csv(
                 "/root/autodl-tmp/hetrec2011-lastfm-2k/user_artists.dat",
        sep='\t', header=None, names=['user', 'item'],
        usecols=[0, 1], dtype={0: np.int32, 1: np.int32})

    user_num = train_data['user'].max() + 1
    item_num = train_data['item'].max() + 1

    train_data = train_data.values.tolist()

    # load ratings as a dok matrix
    train_mat = sp.dok_matrix((user_num, item_num), dtype=np.float32)
    for x in train_data:
        train_mat[x[0], x[1]] = 1.0
    test_data = []
    # with open('/home/featurize/neural_collaborative_filtering-master/Data/ml-1m.test.negative', 'r') as fd:
    #     line = fd.readline()
    #     while line != None and line != '':
    #         arr = line.split('\t')
    #         u = eval(arr[0])[0]
    #         test_data.append([u, eval(arr[0])[1]])
    #         for i in arr[1:]:
    #             test_data.append([u, int(i)])
    #       line = fd.readline()
    #return train_data, test_data, user_num, item_num, train_mat
load_all()




