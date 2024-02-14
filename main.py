import torch
import torch.nn as nn
from einops import rearrange, repeat

import yaml

import torchvision.models as models
from torchvision import transforms
from TF_module.decoder import MultipleDecoder
from TF_module.encoder import MultipleEncoder
from backbone.EncoderCNN import EncoderCNN_resNet

from backbone.dataloader_v3 import COCODataset, DataLoader, make_transforms
from Embedding_PosEncoding import PositionalEncoding, PositionalEncoding_fix

# from backbone.dataloader_v4 import make_transforms


if __name__ == "__main__":
    # CUDA Version: 12.2
    # print(torch.__version__)  # 2.0.1+cpu -> 2.1.0+cu121

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"{device} is available")

    with open("config.yaml", "r") as stream:
        cfg = yaml.safe_load(stream)

    batch_size = cfg["batch_size"]
    learning_rate = cfg["learning_rate"]
    num_epoch = cfg["num_epoch"]

    data_dir = cfg["data_dir"]
    train_set = cfg["train_set"]
    # valid_set = cfg["valid_set"]

    n_channel = cfg["baackbone_params"]["n_channel"]

    n_dim = cfg["n_dim"]

    img_w = 320
    img_h = 320

    ###### dataload & backbond ######
    coco_train = COCODataset(
        data_dir,
        image_set=train_set,
        visualize=False,
        transform=make_transforms(train_set),
    )

    coco_train.update_ImageSize(img_w, img_h)
    loader_train = DataLoader(coco_train, batch_size=batch_size, shuffle=False)
    cnn_model = EncoderCNN_resNet(n_channel).to(device)

    # train 할 때 -> pos 고정 (그대로 쓰기)
    positionalEncoding = PositionalEncoding_fix(n_dim).to(device)
    pos = positionalEncoding(batch_size, img_w // 32, img_h // 32)

    for i_batch, (image, label, mask) in enumerate(loader_train):
        # ## Assuming CHW format, convert to HWC
        # img = rearrange(img, "c h w -> h w c")
        # labels : bbox_x1, bbox_y1, bbox_x2, bbox_y2, label = ann.astype(int)

        images = image.to(device)  # [batch, 3, 640, 640] : b c w h
        labels = label.to(device)

        # Forward pass
        image_features = cnn_model(images)  # [8, 256, 20, 20] : "b c h w"

        ### encoder ###
        multipleEncoder = MultipleEncoder(cfg["tf_model_params"]).to(device)
        en_output = multipleEncoder(image_features, pos)
        en_output = en_output.to(device)  # "b (h w) c"

        ### decoder ###
        multipleDecoder = MultipleDecoder(cfg=cfg["tf_model_params"]).to(device)
        de_output = multipleDecoder(en_output=en_output, pos=pos)
        print(de_output.shape)

        ### loss for training
        # gt_instances = [x["instances"].to(self.device) for x in batched_inputs]

        # targets = self.prepare_targets(gt_instances)
        # loss_dict = self.criterion(output, targets)
        # weight_dict = self.criterion.weight_dict
        # for k in loss_dict.keys():
        #     if k in weight_dict:
        #         loss_dict[k] *= weight_dict[k]
        # return loss_dict

    #################
    # for i_batch, (image, label, mask) in enumerate(loader_train):
    #     # ## Assuming CHW format, convert to HWC
    #     # img = rearrange(img, "c h w -> h w c")
    #     # labels : bbox_x1, bbox_y1, bbox_x2, bbox_y2, label = ann.astype(int)

    #     images = image.to(device)  # torch.Size([batch, 3, 640, 640])
    #     labels = label.to(device)

    #     # Forward pass
    #     image_features = cnn_model(images)  # torch.Size([8, 256, 20, 20])

    #     # positionalEncoding = PositionalEncoding_learnable(n_dim).to(device)
    #     positionalEncoding = PositionalEncoding(n_dim).to(device)

    #     ###### encoder input : image_fetaure_pos ######
    #     image_fetaure_pos = positionalEncoding(image_features)  #  b (h w) c
    #     pos = positionalEncoding.get_pos_encoding()

    #     ### encoder ###
    #     multipleEncoder = MultipleEncoder(cfg["tf_model_params"]).to(device)
    #     encoder_output = multipleEncoder(image_fetaure_pos)
    #     encoder_output = encoder_output.to(device)
    #     # print(en)  # torch.Size([8, 400, 256])

    #     ### decoder ###
    #     # de_input = torch.randn((8, 800, 256), device=device)
    #     # de_input = torch.zeros((labels.shape[0], labels.shape[1], 256), device=device)

    #     multipleDecoder = MultipleDecoder(cfg=cfg["tf_model_params"]).to(device)
    #     de = multipleDecoder(x=labels, e=encoder_output, pos=pos)
    #     print(de.shape)
