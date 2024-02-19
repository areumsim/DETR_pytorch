import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import asnumpy, rearrange

from TF_module.decoder import MultipleDecoder
from TF_module.encoder import MultipleEncoder

from backbone.EncoderCNN import EncoderCNN_resNet
from Embedding_PosEncoding import  PositionalEncoding_fix

from scipy.optimize import linear_sum_assignment
from torchvision.ops import generalized_box_iou_loss


import util.box_ops as box_ops


l1_loss = nn.L1Loss(reduction='none')

class DETR(nn.Module):
    def __init__(self, cfg):
        super(DETR, self).__init__()

        self.cfg = cfg

        self.cnn_model = EncoderCNN_resNet(cfg["backbone_params"]["n_channel"])
        self.positionalEncoding = PositionalEncoding_fix(cfg["tf_model_params"])
        self.multipleEncoder = MultipleEncoder(cfg["tf_model_params"])
        self.multipleDecoder = MultipleDecoder(cfg["tf_model_params"])
        
        # self.class_predictor = nn.Linear(cfg["tf_model_params"]["n_dim"], cfg["data"]["n_class"])  
        # self.bbox_predictor = nn.Linear(cfg["tf_model_params"]["n_dim"], 4)  # 4 : bbox 개수    
        self.class_predictor = nn.Linear(cfg["tf_model_params"]["d_model"], cfg["data"]["n_class"])  
        self.bbox_predictor = nn.Linear(cfg["tf_model_params"]["d_model"], 4)  # 4 : bbox 개수    


        self.obj_query_shape = [1, cfg["tf_model_params"]['n_obj'], cfg["tf_model_params"]["d_model"]] 

    def forward(self, images, labels):

        decoder_input = torch.zeros(self.obj_query_shape).cuda() # (trg) input enbedding, query : for each iter., set to zero 
        # decoder_input = torch.zeros(self.cfg["tf_model_params"]["decoder_params"]["x_shape"]).cuda() # (trg) input enbedding, query : for each iter., set to zero 

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
        
        loss = criterion(prob_class, predict_bbox, labels, self.cfg )

        return prob_class, predict_bbox, loss


def criterion(prob_class, predict_bbox, labels, cfg):
    n_real_class = cfg["data"]["n_class"]-1
    image_w = cfg["data"]["image_width"]
    image_h = cfg["data"]["image_height"]

    ### predict_bbox = predict_bbox/320
    # labels[:,:,:4] = labels[:,:,:4]/image_w
    labels[:,:, [0, 2]] /= image_w
    labels[:,:, [1, 3]] /= image_h  

    ### batch합쳐서 matrix 계산
    # L_matching
    cost_mat = torch.zeros(prob_class.shape[0], prob_class.shape[1], labels.shape[1]).cuda() # [b, N, n] : "b N(pred) class(true)"
    # L_hungarian
    loss_mat = torch.zeros(prob_class.shape[0], prob_class.shape[1], labels.shape[1]).cuda() # [b, N, n] : "b N(pred) class(true)"

    flat_prob_class = rearrange(prob_class,'b n c -> (b n) c')
    flat_predict_bbox = rearrange(predict_bbox,'b n c -> (b n) c')

    flat_labels = rearrange(labels,'b n c -> (b n) c')
    flat_class = flat_labels[:,4].long()
    flat_bbox = flat_labels[:,:4]

    ## classification cost
    class_cost = flat_prob_class[:,flat_class]
    log_class_loss = torch.log(class_cost)
    class_cost = -1*class_cost

    ## bbox cost
    boxes1 = box_ops.box_cxcywh_to_xyxy(flat_predict_bbox)
    boxes2 = box_ops.box_cxcywh_to_xyxy(flat_bbox)

    l1cost_bbox = torch.cdist(flat_predict_bbox, flat_bbox, p=1)
    if torch.isnan(boxes1).any():
        print('l1cost_bbox', boxes1)
    iou_cost = box_ops.generalized_box_iou(boxes1, boxes2)

    ## total cost
    cost = class_cost + 5*l1cost_bbox + 2*iou_cost
    cost = cost*((flat_class != 80) * 1).unsqueeze(0)  # hungarian index 계산에 사용할때, label이 없는애는 class와 bbox 둘다 제거 
    multi_cost_matrix = rearrange(cost, '(b n) (d m) -> b n m d', b=prob_class.shape[0], d=prob_class.shape[0])

    for i in range(multi_cost_matrix.shape[0]):
        cost_mat[i] = multi_cost_matrix[i,:,:,i]

    ## total loss
    bbox_loss = 5*l1cost_bbox + 2*iou_cost
    bbox_loss = bbox_loss*((flat_class != 80) * 1).unsqueeze(0)   # 실제 loss 계산 할때는, label이 없는애 bbox 만 제거 
    loss = -log_class_loss +  bbox_loss

    multi_loss_matrix = rearrange(loss, '(b n) (d m) -> b n m d', b=prob_class.shape[0], d=prob_class.shape[0])
    for i in range(multi_loss_matrix.shape[0]):
        loss_mat[i] = multi_loss_matrix[i,:,:,i]


    ### batch 별 matrix 계산  (중복 for문)
    # for n_pred in range(prob_class.shape[1]):
    #     for n_true in range(labels.shape[1]):
    #         true_class = labels[:,n_true, 4].long() # 'n_true'번째 이미지의 true classes
    #         class_nll = F.nll_loss(prob_class[:,n_pred,:].log(), true_class, reduction='none')
    #         # if torch.isnan(class_nll).any():
    #         #     print('class_nll', class_nll)
            
    #         ## Bounding box loss.
    #         no_obj_massking = (true_class != n_real_class) * 1
    #         bbox_l1_loss =  l1_loss(predict_bbox[:,n_pred,:], labels[:,n_true,:4]).mean(dim=1)

    #         xyxy_label = box_ops.box_cxcywh_to_xyxy(labels[:,n_true,:4])
    #         xyxy_pred = box_ops.box_cxcywh_to_xyxy(predict_bbox[:,n_pred,:])

    #         bbox_giou_loss = generalized_box_iou_loss(xyxy_label,xyxy_pred)
    #         bbox_losss = (5 * bbox_l1_loss + 2 * bbox_giou_loss) * no_obj_massking
    #         # if torch.isnan(bbox_losss).any():
    #         #     print('bbox_losss', bbox_losss)

    #         loss_mat[:,n_pred,n_true] = class_nll + bbox_losss
            
    loss = 0
    for i in range(loss_mat.shape[0]):
        match_idx = linear_sum_assignment(asnumpy(cost_mat[i]))
        loss += loss_mat[:,match_idx[0],match_idx[1]].mean()

    return loss
