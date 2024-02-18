import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat, asnumpy

import yaml

from TF_module.decoder import MultipleDecoder
from TF_module.encoder import MultipleEncoder

from backbone.EncoderCNN import EncoderCNN_resNet
from Embedding_PosEncoding import  PositionalEncoding_fix

from scipy.optimize import linear_sum_assignment
from torchvision.ops import generalized_box_iou_loss

l1_loss = nn.L1Loss(reduction='none')

class DETR(nn.Module):
    def __init__(self, cfg):
        super(DETR, self).__init__()

        self.cfg = cfg

        self.cnn_model = EncoderCNN_resNet(cfg["backbone_params"]["n_channel"])
        self.positionalEncoding = PositionalEncoding_fix(cfg["n_dim"])
        self.multipleEncoder = MultipleEncoder(cfg["tf_model_params"])
        self.multipleDecoder = MultipleDecoder(cfg["tf_model_params"])
        self.class_predictor = nn.Linear(cfg["n_dim"], cfg["n_class"])   # 80 : class 개수 + 1 (background)
        self.bbox_predictor = nn.Linear(cfg["n_dim"], 4)  # 4 : bbox 개수    

    def forward(self, images, labels):
        decoder_input = torch.zeros(self.cfg["tf_model_params"]["decoder_params"]["x_shape"]).cuda() # (trg) input enbedding, query : for each iter., set to zero 

        # Forward pass
        image_features = self.cnn_model(images)  # [b, 256, 10, 10] : "b c H/32 W/32"

        pos = self.positionalEncoding(image_features.shape[0],image_features.shape[2],image_features.shape[3]).cuda()  # [b, 256, 10, 10] : "b c H W"

        ### encoder ###
        en_output = self.multipleEncoder(image_features, pos)
        en_output = en_output  # [b, 100, 256] :  "b (H/32 W/32) c"

        ### decoder ###
        de_output = self.multipleDecoder(tgt=decoder_input, en_output=en_output, en_pos=pos)
        # print(de_output.shape) # [b, 10, 256] :  "b N c"

        ### prediction ###
        predict_class = self.class_predictor(de_output) # [b, 10, 80] :  "b N class"
        prob_class = torch.softmax(predict_class, dim=2)
        predict_bbox = torch.sigmoid(self.bbox_predictor(de_output)) # [b, 10, 4] :  "b N bbox"

        ### loss for training
        loss = criterion(prob_class, predict_bbox, labels)

        return prob_class, predict_bbox, loss

# # ref. https://github.com/CoinCheung/pytorch-loss/blob/master/generalized_iou_loss.py
def generalized_iou_loss(gt_bboxes, pr_bboxes, reduction='none'):
    """
    gt_bboxes: tensor (-1, 4) xyxy
    pr_bboxes: tensor (-1, 4) xyxy
    loss proposed in the paper of giou
    """
    gt_area = (gt_bboxes[:, 2]-gt_bboxes[:, 0])*(gt_bboxes[:, 3]-gt_bboxes[:, 1])
    pr_area = (pr_bboxes[:, 2]-pr_bboxes[:, 0])*(pr_bboxes[:, 3]-pr_bboxes[:, 1])

    # iou
    lt = torch.max(gt_bboxes[:, :2], pr_bboxes[:, :2])
    rb = torch.min(gt_bboxes[:, 2:], pr_bboxes[:, 2:])
    TO_REMOVE = 1e-5
    wh = (rb - lt + TO_REMOVE).clamp(min=0)
    inter = wh[:, 0] * wh[:, 1]
    union = gt_area + pr_area - inter
    iou = inter / union
    # enclosure
    lt = torch.min(gt_bboxes[:, :2], pr_bboxes[:, :2])
    rb = torch.max(gt_bboxes[:, 2:], pr_bboxes[:, 2:])
    wh = (rb - lt + TO_REMOVE).clamp(min=0)
    enclosure = wh[:, 0] * wh[:, 1]

    giou = iou - (enclosure-union)/enclosure
    loss = 1. - giou
    if reduction == 'mean':
        loss = loss.mean()
    elif reduction == 'sum':
        loss = loss.sum()
    elif reduction == 'none':
        pass
    return loss

def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)

def criterion(prob_class, predict_bbox, labels):
    
    # predict_bbox = predict_bbox/320
    labels[:,:,:4] = labels[:,:,:4]/320

    loss_mat = torch.zeros(prob_class.shape[0], prob_class.shape[1], labels.shape[1]).cuda() # [b, N, n] : "b N(pred) class(true)"
    for n_pred in range(prob_class.shape[1]):
        for n_true in range(labels.shape[1]):
            true_class = labels[:,n_true, 4].long()
            class_nll = F.nll_loss(prob_class[:,n_pred,:].log(), true_class, reduction='none')
            # if torch.isnan(class_nll).any():
            #     print('class_nll', class_nll)
            
            ## Bounding box loss.
            no_obj_massking = (true_class != 80) * 1
            bbox_l1_loss =  l1_loss(predict_bbox[:,n_pred,:], labels[:,n_true,:4]).mean(dim=1)


            xyxy_label = box_cxcywh_to_xyxy(labels[:,n_true,:4])
            xyxy_pred = box_cxcywh_to_xyxy(predict_bbox[:,n_pred,:])

            bbox_giou_loss = generalized_box_iou_loss(xyxy_label,xyxy_pred)
            bbox_losss = (5 * bbox_l1_loss + 2 * bbox_giou_loss) * no_obj_massking
            # if torch.isnan(bbox_losss).any():
            #     print('bbox_losss', bbox_losss)

            loss_mat[:,n_pred,n_true] = class_nll + bbox_losss
            
    loss = 0
    for i in range(loss_mat.shape[0]):
        match_idx = linear_sum_assignment(asnumpy(loss_mat[i]))
        loss += loss_mat[:,match_idx[0],match_idx[1]].mean()

    return loss
