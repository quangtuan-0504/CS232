import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
#pytorch, torchvision, chardet



##################################################
model_ggnet = models.googlenet(pretrained=True).to('cpu')
model_ggnet.eval()# thiếu dòng này 2 vector cảu cùng 1 hình sẽ bị encode khác nhau
# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def encode(img_path):
    # Load and preprocess dog image
    img = Image.open(img_path)
    #print(img)
    img = transform(img)
    img = img.unsqueeze(0).to('cpu')

    # Generate embeddings for dog and cat images
    with torch.no_grad():
        embeding = model_ggnet(img).cpu().detach().numpy()
    return embeding
def GgNet_feature_extrac(img):
    img = transform(img)
    img = img.unsqueeze(0).to('cpu')

    # Generate embeddings for dog and cat images
    with torch.no_grad():
        embeding = model_ggnet(img).cpu().detach().numpy()
    return embeding
#img=Image.open("Images_dataset/test_img/00000-146587939.png")
#print(GgNet_feature_extrac(img))
#vec1=encode("Images_dataset/test_img/00000-146587939.png")
#vec2=encode("Images_dataset/test_img/00000-146587939.png")
#print(cosine_similarity(vec1,vec2))

model_resnet = models.resnet50(pretrained=True).to('cpu')
model_resnet.eval()


def encode_(img_path):
    # Load and preprocess dog image
    img = Image.open(img_path)
    img = transform(img)
    img = img.unsqueeze(0).to('cpu')

    # Generate embeddings for dog and cat images
    with torch.no_grad():
        embeding = model_resnet(img).cpu().detach().numpy()
    return embeding
def Resnet_feature_extrac(img):
    image = transform(img)
    image = image.unsqueeze(0).to('cpu')

    with torch.no_grad():
        embeding = model_resnet(image).cpu().detach().numpy()
    return embeding
#print(encode_("Images_dataset/test_img/00000-146587939.png"))










################################################


# class GoogleNetNoAvgPool(torch.nn.Module):
#     def __init__(self):
#         super(GoogleNetNoAvgPool, self).__init__()
#         ggnet =  models.googlenet(pretrained=True)
#         ggnet.dropout = torch.nn.Identity()
#         ggnet.fc = torch.nn.Identity()
#         modules = list(ggnet.children())[:-1]
#         self.ggnet = torch.nn.Sequential(*modules)
#
#     def forward(self, x):
#         x = self.ggnet(x)
#         x = x.view(x.size(0), -1).cpu().detach().numpy()
#         return x
#
#
# ggnet_no_avgpool = GoogleNetNoAvgPool()
#
# transform = transforms.Compose([
#     transforms.Resize(256),
#     transforms.CenterCrop(224),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# ])
#
# def encode(img_path):
#     image = Image.open(img_path)
#     image = transform(image)
#     image = image.unsqueeze(0)
#
#     output_features = ggnet_no_avgpool(image)
#
#     return output_features
#
# #print(encode('Images_dataset/00000-146587939.png'))
# def GgNet_feature_extrac(img):
#     image = transform(img)
#     image = image.unsqueeze(0)#.to("cuda")
#
#     output_features = ggnet_no_avgpool(image)
#
#     return output_features
#
#
#
#
#
#
#
# class ResNetNoAvgPool(torch.nn.Module):
#     def __init__(self):
#         super(ResNetNoAvgPool, self).__init__()
#         resnet = models.resnet50(pretrained=True)
#         modules = list(resnet.children())[:-1]
#         self.resnet = torch.nn.Sequential(*modules)
#
#     def forward(self, x):
#         x = self.resnet(x)
#         x = x.view(x.size(0), -1).cpu().detach().numpy()
#         return x
#
# # =======
#
# resnet_no_avgpool = ResNetNoAvgPool()
#
# transform_1 = transforms.Compose([
#     transforms.Resize(256),
#     transforms.CenterCrop(224),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# ])
#
# def encode_1(img_path):
#     image = Image.open(img_path)
#     image = transform(image)
#     image = image.unsqueeze(0)
#
#     output_features = resnet_no_avgpool(image)
#
#     return output_features
#
#
# print(encode_1("Images_dataset/test_img/00000-146587939.png").shape)
#
# def Resnet_feature_extrac(img):
#     image = transform_1(img)
#     image = image.unsqueeze(0)#.to('cuda')
#
#     output_features = resnet_no_avgpool(image)
#
#     return output_features