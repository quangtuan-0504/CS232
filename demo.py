#Mô tả giao diện
#   Mới mở lên sẽ hiển thị 10 ảnh trong kho dữ liệu
#   Có nơi chọn file ảnh cần tìm
#   Có nơi chọn số ảnh
#   Có nơi chọn thuật toán tìm kiếm và nút nhấn tìm
#streamlit,matplotlib


from load_feature_dataset import load_Features_dataset
from feature_extrac_img import Resnet_feature_extrac,GgNet_feature_extrac
from KDtree import kdtree
from Base import search_base
from Cluster import Cluster

import streamlit as st
from PIL import Image
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image
import time

def extract_class(img_name: str) -> str:
    return img_name.split('/')[-1].split('_')[0]

st.title(':blue[DEMO cuối kì môn tính toán đa phương tiện] :green_apple:')

# Hiển thị 10 hình ảnh có sẵn trong dataset khi mở giao diện
lst_path=['Images_dataset/test_img/'+name_img for name_img in os.listdir('Images_dataset/test_img')]
img_open=[]
for path in lst_path:
    img_=Image.open(path)
    img_open.append(img_)
st.image(img_open,width=200)

col1, col2= st.columns(2)
with col1:
    #tải hình lên
    uploaded_file = st.file_uploader("CHỌN FILE ẢNH")
    if uploaded_file is not None:
        #nhận hình ảnh tải lên và display
        img=Image.open(uploaded_file)
        st.image(img, caption='Hình ảnh của bạn',width=300)

with col2:
    #nhập số result trả về
    number = st.number_input('Result numbers',min_value=0,max_value=20,value=1)
    #chọn thuật toán tìm kiếm
    option = st.selectbox(
        'Search method',
        ('Base','kdtree', 'Cluster'))
    #chọn bộ feature extrac
    genre = st.radio(
        "Choose feature extractor model",
        ('Resnet', 'GgNet'))

    #nhấn tìm
    search=st.button("Search")

#Chạy thuật toán dc chọn
if uploaded_file is not None and search==True:
    #st.write(img)
    if genre=='Resnet':
        vec_search=Resnet_feature_extrac(img)
    elif genre=='GgNet':
        vec_search = GgNet_feature_extrac(img)
    name_feature_extractor = genre
    # hiển thị vector search
    #st.write(vec_search)

    #load feature_dataset  các path_ảnh đi kèm với vector đặc trưng tương ứng
    if genre=='Resnet':
        img_names=pd.read_csv("Name_imgs_dataset/image_name_resnet50.txt",header=None)
    elif genre=='GgNet':
        img_names=pd.read_csv("Name_imgs_dataset/image_name_ggnet.txt",header=None)
    encodes=load_Features_dataset(name_feature_extractor)
    # st.write(img_names[0][101])
    # st.write(encodes[101])
    #truyền vào vector search,list tên ảnh, vector_features=> k tên ảnh
    if option=='kdtree':
        start=time.time()
        k_path,distance=kdtree(img_names,encodes,vec_search,number)
        time_=time.time()-start
    elif option=='Cluster':
        pass
    elif option=='Base':
        start = time.time()
        k_path,distance=search_base(img_names,encodes,vec_search,number)
        time_ = time.time() - start
    # hiển thị kết quả

    st.title(":green[RESULT]")
    st.write('Search time',time_)
    col1,col2=st.columns(2)
    with col1:
        st.write(k_path)
    with col2:
        st.write(distance)

    lst_path_result = ['Images_dataset/dataset/' + name_img for name_img in k_path]
    img_open_result = []
    for path in lst_path_result:
        img = Image.open(path)
        img_open_result.append(img)
    st.image(img_open_result, width=200)


    fig,axs=plt.subplots(nrows=number//2+1,ncols=2)
    axes=axs.flatten()
    for i in range(len(lst_path_result)):
       img_plot=matplotlib.image.imread(lst_path_result[i])
       axes[i].imshow(img_plot)
       axes[i].set_title(f"{extract_class(lst_path_result[i])==uploaded_file.name.split('_')[0]};"\
                         f" groundtruth: class {extract_class(lst_path_result[i])}")
       axes[i].axis('off')
    st.pyplot(fig)




