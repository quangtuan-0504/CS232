import numpy as np
from sklearn.neighbors import KDTree
#sklearn

def extract_class(img_name: str) -> str:
    return img_name.split('/')[-1].split('_')[0]


def kdtree(img_names,encode,vec_search,k):#img_names là 1 list các path ảnh, encode là một mảng np 2 chiều chứa các vector đặc trưng, path ảnh có index i trong img_names tưng ứng với vector có index i trong encode
    tree = KDTree(encode, leaf_size=8)
    query = tree.query(vec_search, k=k)
    dist, ind = query[0][0], query[1][0]# dist là list các khoảng cách tới k ảnh gần nhất, ind là index trog img_names tưng ứng của k ảnh này
    k_path=[img_names[0][i].split('/')[-1] for i in ind]
    return (k_path,dist)

