import torch
import torch.nn as nn

import torchvision.models as models
from torchvision import transforms

from .dataloader_v3 import COCODataset
from torch.utils.data import DataLoader

from einops import rearrange


## ref.
# https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/03-advanced/image_captioning/model.py
# https://pytorch.org/vision/stable/models.html
class EncoderCNN_resNet(nn.Module):
    def __init__(self, n_channel):
        """Load the pretrained ResNet and replace top fc layer."""
        super(EncoderCNN_resNet, self).__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        modules = list(resnet.children())[:-2]  # delete avgpool and last fc layer
        self.resnet = nn.Sequential(*modules)

        # self.adaptive_pool = nn.AdaptiveAvgPool2d(
        #     (matrix_size, matrix_size)
        # )
        self.backbone_out_channels = 2048
        self.conv = nn.Conv2d(
            self.backbone_out_channels, n_channel, kernel_size=1
        )  # Convert to the desired number of channels

    def forward(self, images):
        """Extract feature vectors from input images."""
        with torch.no_grad():
            features = self.resnet(images)
        # features = self.adaptive_pool(features)
        features = self.conv(features)
        return features


##  resnet50에서 마지막만 -1 만 제거하면,   -> torch.Size([8, 2048, 1, 1])
##  resnet50에서 마지막만 -2 를 제거하면,   -> torch.Size([8, 2048, 16, 16])
### input(224x224) -> (nBatch, 7, 7, 2048) 224/32=7, 512 -> 16


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"{device} is available")

    batch_size = 8
    learning_rate = 0.00001
    num_epoch = 10

    ### dataload ###
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

    data_dir = "C:/Users/sar10/code/detr/cocodataset/"

    image_set = "val2017"  # "train2017"
    coco_train = COCODataset(
        data_dir, image_set=image_set, visualize=False, transform=transform
    )
    loader_train = DataLoader(coco_train, batch_size=batch_size, shuffle=False)
    ### end. dataload ###

    ### get image_features
    # input : 512x512 -> (512/32=16) -> output : 2048x16x16 -> 256X16x16
    # input : 640x640 -> (640/32=20) -> output : 2048x20x20 -> 256X20x20
    n_channel = 256
    cnn_model = EncoderCNN_resNet(n_channel).to(device)

    # # Loss and optimizer
    # criterion = nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(cnn_model.parameters(), lr=learning_rate)
    for i_batch, (image, label) in enumerate(loader_train):
        images = image.to(device)  # torch.Size([batch, 3, 640, 640])
        labels = label.to(device)

        # Forward pass
        image_features = cnn_model(images)
        print(image_features.shape)  # torch.Size([8, 256, 20, 20])

        image_features = rearrange(image_features, "b c h w -> b c (h w)")  # flatten(2)
        print(image_features.shape)  # torch.Size([8, 256, 400])

        # # Forward pass
        # loss = criterion(out_class, labels)

        # # Backward and optimize
        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()

        # if (i_batch + 1) % batch_size == 0:
        #     print(
        #         "Epoch [{}/{}], Loss: {:.4f}".format(e + 1, num_epoch, loss.item())
        #     )


# # ref. https://github.com/developer0hye/Custom-CNN-based-Image-Classification-in-PyTorch/blob/master/main.py
# class EncoderCNN(nn.Module):
#     def __init__(self, num_classes, embed_size):
#         super(EncoderCNN, self).__init__()
#         self.num_classes = num_classes

#         self.layer1 = self.conv_module(3, 16)
#         self.layer2 = self.conv_module(16, 32)
#         self.layer3 = self.conv_module(32, 64)
#         self.layer4 = self.conv_module(64, 128)
#         self.layer5 = self.conv_module(128, 256)
#         self.gap = self.global_avg_pool(256, self.num_classes)

#         self.linear = nn.Linear(256, embed_size)
#         self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)

#     def forward(self, x):
#         out = self.layer1(x)
#         out = self.layer2(out)
#         out = self.layer3(out)
#         out = self.layer4(out)
#         out = self.layer5(out)

#         out_features = self.linear(out)
#         out_features = self.bn(out_features)

#         out_class = self.gap(out)
#         out_class = out_class.view(-1, self.num_classes)

#         return out_class, out_features

#     def conv_module(self, in_num, out_num):
#         return nn.Sequential(
#             nn.Conv2d(in_num, out_num, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(out_num),
#             nn.LeakyReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#         )

#     def global_avg_pool(self, in_num, out_num):
#         return nn.Sequential(
#             nn.Conv2d(in_num, out_num, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(out_num),
#             nn.LeakyReLU(),
#             nn.AdaptiveAvgPool2d((1, 1)),
#         )


# if __name__ == "__main__":
#     # if __package__ is None:
#     #     import sys
#     #     from os import path
#     #     print(path.dirname( path.dirname( path.abspath(__file__) ) ))
#     # 	sys.path.append(path.dirname( path.dirname( path.abspath(__file__) ) ))
#     #     from dataloader import COCODataset, DataLoader
#     # else:
#     #     from ..dataloader import COCODataset, DataLoader

#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     print(f"{device} is available")

#     ### CNN - train paras. ###
#     batch_size = 8
#     learning_rate = 0.01
#     num_epoch = 10
#     ### end- CNN - train paras. ###

#     data_dir = "C:/Users/sar10/code/detr/cocodataset/"
#     image_set = "val2017"  # "train2017"
#     coco_train = COCODataset(data_dir, image_set=image_set, visualize=False)
#     loader_train = DataLoader(coco_train, batch_size=batch_size, shuffle=False)

#     # image_set = "val2017"
#     # coco_test = COCODataset(data_dir, image_set=image_set, visualize=False)
#     # loader_test = DataLoader(coco_test, batch_size=batch_size, shuffle=True)

#     ### CNN
#     num_classes = len(coco_train.classes)
#     cnn_model = EncoderCNN(num_classes=num_classes).to(device)

#     # Loss and optimizer
#     criterion = nn.CrossEntropyLoss()
#     optimizer = torch.optim.Adam(cnn_model.parameters(), lr=learning_rate)

#     for e in range(num_epoch):
#         for i_batch, (image, label) in enumerate(loader_train):
#             images = image.to(device)
#             labels = label.to(device)

#             # Forward pass
#             outputs = cnn_model(images)
#             loss = criterion(outputs, labels)

#             # Backward and optimize
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#             if (i_batch + 1) % batch_size == 0:
#                 print(
#                     "Epoch [{}/{}], Loss: {:.4f}".format(e + 1, num_epoch, loss.item())
#                 )

#     # # Test the model
#     # cnn_model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
#     # with torch.no_grad():
#     #     correct = 0
#     #     total = 0
#     #     for item in loader_test:
#     #         images = item["image"].to(device)
#     #         labels = item["label"].to(device)
#     #         outputs = cnn_model(images)
#     #         _, predicted = torch.max(outputs.data, 1)
#     #         total += len(labels)
#     #         correct += (predicted == labels).sum().item()

#     #     print(
#     #         "Test Accuracy of the model on the {} test images: {} %".format(
#     #             total, 100 * correct / total
#     #         )
#     #     )
