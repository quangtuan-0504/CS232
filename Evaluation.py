from load_feature_dataset import load_Features_dataset
from feature_extrac_img import Resnet_feature_extrac,GgNet_feature_extrac
from KDtree import kdtree
from Base import search_base
import pandas as pd
from PIL import Image

query_imgs=['Images_dataset/dataset/0_8c54ccf63a12ee74.jpg','Images_dataset/dataset/1_02b09bb030cf5687.jpg']


#thực hiện search với 1 query
def imple_search(algorithm_search,encodes_name,num_result_per_qr,path_query):
    img = Image.open(path_query)
    if encodes_name=='Resnet':
        vec_search=Resnet_feature_extrac(img)
    elif encodes_name=='GgNet':
        vec_search = GgNet_feature_extrac(img)
    name_feature_extractor = encodes_name
    if encodes_name=='Resnet':
        img_names=pd.read_csv("Name_imgs_dataset/image_name_resnet50.txt",header=None)
    elif encodes_name=='GgNet':
        img_names=pd.read_csv("Name_imgs_dataset/image_name_ggnet.txt",header=None)
    encodes=load_Features_dataset(name_feature_extractor)
    if algorithm_search=='kdtree':

        k_path,distance=kdtree(img_names,encodes,vec_search,num_result_per_qr)

    elif algorithm_search=='Cluster':
        pass
    elif algorithm_search=='Base':

        k_path,distance=search_base(img_names,encodes,vec_search,num_result_per_qr)
    return k_path


def extract_class(img_name: str) -> str:
    return img_name.split('/')[-1].split('_')[0]
def cal_precision_recall_f1score(list_path_result_search,path_img_search):
    class_true=extract_class(path_img_search)
    num_relevant_retrieved=len([path for path in list_path_result_search if class_true==extract_class(path)])
    num_retrieved=len(list_path_result_search)
    precision=num_relevant_retrieved/num_retrieved
    recall=num_relevant_retrieved/10
    f1=2*precision*recall/(precision+recall)
    return precision,recall,f1
paths=imple_search('Base','Resnet',5,'Images_dataset/dataset/0_8c54ccf63a12ee74.jpg')
p,r,f=cal_precision_recall_f1score(paths,'Images_dataset/dataset/0_8c54ccf63a12ee74.jpg')
print(p,r,f)
def AP(algorithm_search,encodes_name,path_query,max_num_result):
    lst=[]
    for i in range(1,max_num_result):
        paths=imple_search(algorithm_search,encodes_name,i,path_query)
        p,r,f=cal_precision_recall_f1score(paths,path_query)
        lst.append(p)
        #print(p)
    return sum(lst)/len(lst)
print(AP('Base','Resnet','Images_dataset/dataset/0_8c54ccf63a12ee74.jpg',5))
def MAP(algorithm_search,encodes_name,max_num_result,query_imgs):#truyền vào những thứ j mà hàm trong hàm này sử dụng và những cái cần dùng
    lst=[]
    for query in query_imgs:
        ap=AP(algorithm_search,encodes_name,query,max_num_result)
        lst.append(ap)
    return sum(lst)/len(lst)
print(MAP('Base','Resnet',15,query_imgs))

