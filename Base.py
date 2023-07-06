from sklearn.metrics.pairwise import  cosine_similarity
import numpy as np
import pandas as pd
from feature_extrac_img import encode


def similarity(query_vec,encodes):
    query_vec=query_vec.reshape(1,-1)
    return cosine_similarity(query_vec,encodes)

def search_base(img_names,encodes,vector_search,k):
    cosine=similarity(vector_search,encodes)
    #print(cosine)
    #print(np.sort(cosine))
    #print(np.argsort(cosine))
    k_distance=np.sort(cosine)[0,:][-k:]
    k_idx=np.argsort(cosine)[0,:][-k:]

    k_path=[img_names[0][i].split('/')[-1] for i in k_idx]
    return k_path,k_distance

# img_names=pd.read_csv("Name_imgs_dataset/image_name_ggnet.txt",header=None)
# encodes = np.load("Features_dataset/dataset-ggnet.npy")
# vector_search=encode("Images_dataset/test_img/00000-146587939.png")
# print(search_base(img_names,encodes,vector_search,2))