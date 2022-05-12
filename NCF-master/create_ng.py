import pandas as pd
import numpy as np
import math
from collections import defaultdict
import heapq
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch.backends.cudnn as cudnn
import os
import random
from torch.utils import data


def load_dataset(test_num=100):
     train_data = pd.read_csv("/root/autodl-tmp/hetrec2011-lastfm-2k/user_artists.dat", \
                              sep='\t', header=None, names=['user', 'item'], \
                              usecols=[0, 1], dtype={0: np.int32, 1: np.int32})
     user_num = train_data['user'].max() + 1
     item_num = train_data['item'].max() + 1

     train_data = train_data.values.tolist()
     train_data.append([user_num+1,100])
     train_data.append([user_num+2,100])
     #load ratings as a dok matrix
     train_mat = sp.dok_matrix((user_num+3,item_num),dtype=np.float32)
     for x in train_data:
         train_mat[x[0], x[1]] = 1.0

     test_data = []
     # 构造测试集合
     test_data = []
     #每个用户
     for u in range(2,user_num+2):
         # 每个用户100个测试样本，1个正样本，99个负样本
         for x in train_data:
             if x[0] == u:
                 test_data.append([u,x[1]])
                 break
         for i in range(0,test_num-1):
             test_temp = random.randint(1, item_num-1)
             while(bool(train_mat.get((u,test_temp))) or  ((u,test_temp)in test_data)):
                 test_temp = random.randint(1, item_num-1)
             test_data.append([u,test_temp])

     with open ('/root/autodl-tmp/hetrec2011-lastfm-2k/test_data.txt','w') as q:
         for i in test_data:
             q.write(str(i[0]))
             q.write('\t')
             q.write(str(i[1]))
             q.write('\n')


