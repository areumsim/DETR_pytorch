import cv2
from matplotlib import pyplot as plt
import torch
import yaml
from einops import rearrange
import numpy as np

from backbone.dataloader_v3 import COCODataset
from torch.utils.data import DataLoader

from detr import DETR
from tqdm import tqdm

from util.box_ops import box_cxcywh_to_xyxy

if __name__ == "__main__":
    # CUDA Version: 12.2
    # print(torch.__version__)  # 2.0.1+cpu -> 2.1.0+cu121

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"{device} is available")

    with open("config.yaml", "r") as stream:
        cfg = yaml.safe_load(stream)

    batch_size = cfg['train_params']["batch_size"]
    num_epoch = cfg['train_params']["num_epoch"]

    ###### dataload & backbond ######
    coco_train = COCODataset(
        cfg['data']
    )
    loader_train = DataLoader(coco_train, batch_size=batch_size, shuffle=True)


    ###### model ######
    detr_model = DETR(cfg).cuda()

    ###### optimizer ######
    optimizer = torch.optim.AdamW(detr_model.parameters(), lr=cfg['train_params']["learning_rate"], weight_decay=cfg['train_params']['weight_decay'])


    epoch_losses = [] 
    for epoch in range(num_epoch):
        bar = tqdm(loader_train)
        batch_losses = []
        for i_batch, (image, label, mask) in enumerate(bar):
            # ## Assuming CHW format, convert to HWC
            # img = rearrange(img, "c h w -> h w c")
            # labels : bbox_x1, bbox_y1, bbox_x2, bbox_y2, label = ann.astype(int)

            images = image.to(device).float()  # [b, 3, 320, 320] : b c H W
            labels = label.to(device).float()

            optimizer.zero_grad()
            prob_class, predict_bbox, loss = detr_model(images, labels.clone())

            loss.backward()
            optimizer.step()

            bar.set_postfix(loss=loss.item())
            # print(f"loss: {loss.item()}")
            
            batch_losses.append(loss.item())

            # for evaluation, draw a predicted bbox and class label
            if i_batch % 50 == 0:
                i = 0
                img = images[i].cpu().numpy().copy()
                img *= np.array([0.229, 0.224, 0.225])[:,None,None]
                img += np.array([0.485, 0.456, 0.406])[:,None,None]
                img = rearrange(img, "c h w -> h w c")
                img = img * 255
                img = img.astype(np.uint8)

                lb = labels[i].cpu()
                lb[:,:4] = box_cxcywh_to_xyxy(lb[:,:4])
                lb = lb.numpy()

                _predict_bbox = predict_bbox[i].cpu().detach().numpy()
                _prob_class = prob_class[i].cpu().detach().numpy()

                # draw bbox
                image_w = cfg['data']["image_width"]
                image_h = cfg['data']["image_height"]
                for j in range(len(_predict_bbox)):
                    # bbox = _predict_bbox[j]*320
                    bbox = _predict_bbox[j]
                    bbox[[0, 2]] *= image_w
                    bbox[[1, 3]] *= image_h

                    bbox = rearrange(bbox, "c -> () c")
                    bbox = bbox[0].astype(int)
                    x, y, w, h = bbox
                    # bbox to xyxy
                    x1, y1, x2, y2 = int(x-w*0.5), int(y-h*0.5), int(x+w*0.5), int(y+h*0.5)

                    img = cv2.rectangle(img.copy(), (x1, y1), (x2, y2), (0, 0, 255), 2)

                # draw class label
                for j in range(len(_prob_class.argmax(1))):
                    score = _prob_class.argmax(1)[j]
                    score = score.astype(int)
                    img = cv2.putText(img, str(score), (bbox[0], bbox[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

                # draw ground truth bbox
                for ann in lb:
                    bbox_x1, bbox_y1, bbox_x2, bbox_y2, lab = (ann).astype(int)


                    c = (
                        (np.random.random((1, 3)) * 0.6 * 255 + 0.4 * 255)
                        .astype(np.uint8)
                        .tolist()[0]
                    )
                    img = cv2.rectangle(
                        img.copy(), (bbox_x1, bbox_y1), (bbox_x2, bbox_y2), color=c, thickness=2
                    )

                # cv2.imshow("", cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                cv2.imwrite(f"./result_image/result_{epoch}_{i_batch}.jpg", cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                cv2.waitKey(0)

        

        # save and show loss
        epoch_losses.append(np.mean(batch_losses))
        # plt.plot(np.array(loss), 'r')

        plt.plot(np.arange(len(epoch_losses)), epoch_losses, marker='.', c='red', label='Trainset_loss')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        # plt.show()
        plt.savefig(f"./result_image/loss_{epoch}.png")
        
        with open(f"./result_model/loss_{epoch}.txt", 'w') as f:
            f.write('\n'.join(map(str, epoch_losses)))

    ## save model and loss
    torch.save(detr_model.state_dict(), f"./result_model/detr_model_final({epoch})_0219_1.pth")
    torch.save(loss, f"./result_model/loss_final({epoch})_0219_1.txt")
