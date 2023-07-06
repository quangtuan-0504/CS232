import numpy as np
def load_Features_dataset(name_feature_extractor):
    if name_feature_extractor=='GgNet':
        encode = np.load("Features_dataset/dataset-ggnet.npy")

    elif name_feature_extractor=="Resnet":
        encode = np.load("Features_dataset/dataset-resnet50.npy")


    return encode


def ham(a,b):
    return [a,b]
a,b=ham(1,2)
print(a,b)