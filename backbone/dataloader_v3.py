import cv2

import os
import random
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset

from pycocotools.coco import COCO
from PIL import Image

from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

from torchvision.io import read_image
from torchvision import transforms

import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage

from einops import rearrange

from util.box_ops import box_xyxy_to_cxcywh

## pycocotools  --
#  getAnnIds  - Get ann ids that satisfy given filter conditions.
#  getCatIds  - Get cat ids that satisfy given filter conditions.
#  getImgIds  - Get img ids that satisfy given filter conditions.
#  loadAnns   - Load anns with the specified ids.
#  loadCats   - Load cats with the specified ids.
#  loadImgs   - Load imgs with the specified ids.
#  showAnns   - Display the specified annotations.


def make_transforms(image_set):
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


# def box_xyxy_to_cxcywh(x):
#     x0, y0, x1, y1 = x.unbind(-1)
#     b = [(x0 + x1) / 2, (y0 + y1) / 2,
#          (x1 - x0), (y1 - y0)]
#     return torch.stack(b, dim=-1)


class COCODataset(Dataset):
    def __init__(
        self, cfg, transform=None, visualize=False, collate_fn=None
    ):
        super(COCODataset, self).__init__()
        self.cfg = cfg

        self.data_dir = cfg["data_dir"]
        self.image_set = cfg["train_set"]

        self.width = cfg["image_width"]
        self.height = cfg["image_height"]

        self.n_class = cfg["n_class"] # real_class + no_obj


        self.image_folrder = os.path.join(self.data_dir, self.image_set)
        self.anno_file = os.path.join(
            self.data_dir.replace("/", "\\"),
            "annotations",
            "instances_" + self.image_set + ".json",
        )

        self.transform = make_transforms(self.image_set)
        self.visualize = visualize

        self.getMask = True

        self.coco = COCO(self.anno_file)

        # to remove not annotated image idx
        self.image_ids = []
        whole_image_ids = self.coco.getImgIds()
        no_anno_list = []
        for idx in whole_image_ids:
            annotations_ids = self.coco.getAnnIds(imgIds=idx, iscrowd=False)
            if len(annotations_ids) == 0:
                no_anno_list.append(idx)
            else:
                self.image_ids.append(idx)

        self.load_classes()  # read class information

        ### category Info
        # self.catIds = self.coco.getCatIds(catNms=["person"])
        # self.catCnt = len(self.coco.getCatIds())
        # self.imgIds = self.coco.getImgIds(catIds=self.catIds)
        self.seq = iaa.Sequential(
            [
                iaa.Resize((0.4, 0.5)),
                iaa.SomeOf(
                    1,
                    [
                        iaa.AdditiveLaplaceNoise(scale=(0, 0.2 * 255)),
                        iaa.Fliplr(0.5),
                        iaa.Add(50, per_channel=True),
                        iaa.Sharpen(alpha=0.5),
                        # iaa.CropAndPad(percent=(-0.25, 0.25)),
                    ],
                ),
                # iaa.PadToFixedSize(
                #     width=self.width, height=self.height, position="center"
                # ),
                iaa.Resize(
                    {"height": self.height, "width": self.width}
                ),
                
            ]
        )

    def __len__(self):
        return len(self.image_ids)

    def load_image(self, image_idx):  #
        image_info = self.coco.loadImgs(self.image_ids[image_idx])[0]
        image_file_path = os.path.join(self.image_folrder, image_info["file_name"])
        image = Image.open(image_file_path).convert("RGB")

        image = np.array(image)
        return image

    def load_annotations(self, image_idx):
        annotations_ids = self.coco.getAnnIds(
            imgIds=self.image_ids[image_idx], iscrowd=False
        )
        # shuffle ids  : use only #max_boxes boxes
        random.shuffle(annotations_ids)
        annotations = np.ones((0, 5)) * -1

        # some images appear to miss annotations (like image with id 257034)
        if len(annotations_ids) == 0:
            return annotations

        # parse annotations
        coco_annotations = self.coco.loadAnns(annotations_ids)
        for idx, a in enumerate(coco_annotations):
            # some annotations have basically no width / height, skip them
            if a["bbox"][2] < 1 or a["bbox"][3] < 1:
                continue

            annotation = np.zeros((1, 5))
            annotation[0, :4] = a["bbox"]
            annotation[0, 4], _ = self.cocoLabel_to_label(a["category_id"])
            # annotation[0, 5] = a["category_id"]  ## self.labels[annotation[0, 4]]

            annotations = np.append(annotations, annotation, axis=0)

        ##############
        # pad annotations to have a consistent size
        max_boxes = self.cfg["max_boxes"] # maximum number of boxes, consider y also as a set of size N padded with (no object).
        padded_annotations = np.ones((max_boxes, 5)) * -1
        num_boxes = min(len(annotations), max_boxes)
        padded_annotations[:num_boxes, :] = annotations[:num_boxes, :]
        ##############

        # transform from [x, y, w, h] to [x1, y1, x2, y2]
        padded_annotations[:, 2] = padded_annotations[:, 0] + padded_annotations[:, 2]
        padded_annotations[:, 3] = padded_annotations[:, 1] + padded_annotations[:, 3]

        return padded_annotations

    def load_classes(self):
        # load class names (name -> label)
        categories = self.coco.loadCats(self.coco.getCatIds())
        categories.sort(key=lambda x: x["id"])

        self.classes = {}
        self.cocoLabels = {}
        self.cocoLabels_invserse = {}

        for i, c in enumerate(categories):
            self.cocoLabels[i] = c["id"]
            self.cocoLabels_invserse[c["id"]] = i
            self.classes[c["name"]] = i

        # also load the reverse (label -> name)
        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key

    def cocoLabel_to_label(self, cocoLabel):
        label_number = self.cocoLabels_invserse[cocoLabel]
        class_name = self.labels[label_number]
        return label_number, class_name

    def show_Anns(self, image, anns, mask=None):
        ax = plt.gca()
        ax.imshow(image)
        ax.set_autoscale_on(False)
        polygons, colors = [], []

        for ann in anns:
            bbox_x1, bbox_y1, bbox_x2, bbox_y2, label = ann
            if label == (self.n_class-1):  # -1, No Obj #TODO : or Show No Obj
                continue

            c = (np.random.random((1, 3)) * 0.6 + 0.4).tolist()[0]
            poly = [
                [bbox_x1, bbox_y1],
                [bbox_x1, bbox_y2],
                [bbox_x2, bbox_y2],
                [bbox_x2, bbox_y1],
            ]
            np_poly = np.array(poly).reshape((4, 2))
            polygons.append(Polygon(np_poly))
            colors.append(c)
            ax.text(bbox_x1, bbox_y1, self.labels[label], color=c)

        p = PatchCollection(polygons, facecolor="none", edgecolors=colors, linewidths=2)
        ax.add_collection(p)
        plt.axis("off")
        plt.xticks([])
        plt.yticks([])

        if mask is not None:
            color_mask = np.zeros_like(mask)
            color_mask[mask[:, :, 0] == 255] = [255, 0, 0]  # masking -> red
            color_mask[mask[:, :, 0] == 0] = [0, 255, 0]  # image -> green

            plt.imshow(color_mask, alpha=0.3)

        plt.show()

    # def update_ImageSize(self, width, height):
    #     self.width = width
    #     self.height = height

    def __getitem__(self, idx):  # 인덱스에 접근할 때 호출
        image = self.load_image(idx)
        annotation = self.load_annotations(idx)

        # https://imgaug.readthedocs.io/
        # # iaa.Resize(0.5) -> resize size 직접입력
        # 비율로 resize하면 float으로 나와서 나중에 mask를 못 씌움
        # resize_ratio = 0.5
        # resize_height, resize_width = int(image.shape[0] * resize_ratio), int(
        #     image.shape[1] * resize_ratio
        # )
        original_img_pos = [0, 0, image.shape[1], image.shape[0], -99]
        annotation = np.vstack([annotation, original_img_pos])
        bbs = BoundingBoxesOnImage(
            [
                BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)
                for x1, y1, x2, y2, _ in annotation
            ],
            shape=image.shape,
        )

        # Augment BBs and images.
        image, bbs = self.seq(image=image, bounding_boxes=bbs)
        annotation[:, :-1] = np.array(
            [[bb.x1, bb.y1, bb.x2, bb.y2] for bb in bbs.bounding_boxes]
        )

        new_img_pos = np.array(annotation[-1], dtype=int)  # x1, y1, x2, y2
        annotation = annotation[:-1]

        # padding_mask = np.zeros_like(image, dtype=np.uint8)
        padding_mask = np.ones_like(image, dtype=np.uint8) * 255
        if self.getMask:
            # 사각형 부분을 0으로 설정
            padding_mask[
                new_img_pos[1] : new_img_pos[3], new_img_pos[0] : new_img_pos[2], :
            ] = 0
            # (for visualization purposes)
            # self.show_Anns(image, annotation, padding_mask)

        if self.visualize:
            self.show_Anns(image, annotation)

        if self.transform is not None:
            image = self.transform(image)

        # boxes = torch.FloatTensor(annotation[:, :4])
        # labels = torch.LongTensor(annotation[:, 4])
        padding_mask = torch.tensor(rearrange(padding_mask, "h w c -> c h w"))
        annotation[:,4][annotation[:,4]==-1] = (self.n_class-1)

        # box_xyxy_to_cxcywh
        # x0 = annotation[:,0]
        # y0 = annotation[:,1]
        # x1 = annotation[:,2]
        # y1 = annotation[:,3]

        # b = [(x0 + x1) / 2, (y0 + y1) / 2,
        #     (x1 - x0), (y1 - y0)]
        # annotation[:,:4] = np.array(b).T
        new_box = box_xyxy_to_cxcywh(torch.tensor(annotation[:,:4])).numpy()
        annotation[:,:4] = new_box

        return image, annotation, padding_mask


if __name__ == "__main__":
    from einops import rearrange, asnumpy

    # Image preprocessing, normalization for the pretrained resnet
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    data_dir = "C:/Users/sar10/code/detr/cocodataset/"
    # image_set = "train2017"
    image_set = "val2017"
    coco = COCODataset(
        data_dir, image_set=image_set, visualize=False, transform=transform
    )
    coco.update_ImageSize(width=640, height=640)
    print()

    ### Show image. tensor to cv2.imshow (numpy) ###
    img, labels, mask = coco[0]

    ## Assuming CHW format, convert to HWC
    img = rearrange(img, "c h w -> h w c")

    ## denormalize :  IMAGENET 형식으로 normalize 된 경우
    IMAGENET_MEAN, IMAGENET_STD = torch.tensor([0.485, 0.456, 0.406]), torch.tensor(
        [0.229, 0.224, 0.225]
    )
    ## tensor -> np.uint8
    img = (
        asnumpy(torch.clip(255.0 * (img * IMAGENET_STD + IMAGENET_MEAN), 0, 255))
        .astype(np.uint8)
        .copy()
    )

    polygons, colors = [], []
    for ann in labels:
        bbox_x1, bbox_y1, bbox_x2, bbox_y2, label = ann.astype(int)
        c = (
            (np.random.random((1, 3)) * 0.6 * 255 + 0.4 * 255)
            .astype(np.uint8)
            .tolist()[0]
        )
        img = cv2.rectangle(
            img, (bbox_x1, bbox_y1), (bbox_x2, bbox_y2), color=c, thickness=2
        )

    cv2.imshow("", cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    ### end. Show image. tensor to cv2.imshow (numpy) ###

    # Show the mask (for visualization purposes)
    cv2.imshow("Padding Mask", mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
