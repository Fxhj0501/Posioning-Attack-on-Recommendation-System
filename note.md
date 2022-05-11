# CapStone

## 对于NCF模型用torch实现
### 针对ml-1m数据集
1. 加载数据的结果：
    - train_data:一个大列表，里面套用了巨多的小列表，每个小列表长度为2，是[user,item]这样的形式，表示user和哪些item交互过
      
    - user_num = 6040 （采样了6040名用户）
    - item_num = 3706  （这个似乎意义不大，就是电影最大编号是3706）
    - train_mat：稀疏矩阵存储数据，将交互过的统一标为1（交互记录是在ml-1m.train.rating）
    - test_data:形式和train_data一样，只不过是从`ml-1m.test.negative`数据集里面获取的