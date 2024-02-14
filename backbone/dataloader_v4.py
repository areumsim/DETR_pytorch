import cv2

import os
import random
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader

from pycocotools.coco import COCO
from PIL import Image

from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

from torchvision.io import read_image
from torchvision import transforms as T
import torch.nn.functional as F

import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage

import transformation_ as T_


## pycocotools  --
#  getAnnIds  - Get ann ids that satisfy given filter conditions.
#  getCatIds  - Get cat ids that satisfy given filter conditions.
#  getImgIds  - Get img ids that satisfy given filter conditions.
#  loadAnns   - Load anns with the specified ids.
#  loadCats   - Load cats with the specified ids.
#  loadImgs   - Load imgs with the specified ids.
#  showAnns   - Display the specified annotations.


class COCODataset(Dataset):
    def __init__(
        self, data_dir, image_set, transform=None, visualize=True, collate_fn=None
    ):
        super(COCODataset, self).__init__()

        self.data_dir = data_dir
        self.image_set = image_set

        self.image_folrder = os.path.join(self.data_dir, self.image_set)
        self.anno_file = os.path.join(
            data_dir.replace("/", "\\"),
            "annotations",
            "instances_" + self.image_set + ".json",
        )

        self.transform = transform
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

        self.width = 640
        self.height = 640

        ### category Info
        # self.catIds = self.coco.getCatIds(catNms=["person"])
        # self.catCnt = len(self.coco.getCatIds())
        # self.imgIds = self.coco.getImgIds(catIds=self.catIds)

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
        max_boxes = 20  # or any suitable maximum number of boxes
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

    def show_Anns(self, image, anns):
        ax = plt.gca()
        ax.imshow(image)
        ax.set_autoscale_on(False)
        polygons, colors = [], []

        for ann in anns:
            bbox_x1, bbox_y1, bbox_x2, bbox_y2, label = ann
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

    def update_ImageSize(self, width, height):
        self.width = width
        self.height = height

    # def __getitem__(self, idx):  # 인덱스에 접근할 때 호출
    #     image = self.load_image(idx)
    #     annotation = self.load_annotations(idx)

    #     original_image_shape = image.shape

    #     # https://imgaug.readthedocs.io/
    #     bbs = BoundingBoxesOnImage(
    #         [
    #             BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)
    #             for x1, y1, x2, y2, _ in annotation
    #             # if x1 >= 0
    #         ],
    #         # shape=(image.shape[1], image.shape[2], image.shape[0]),
    #         shape=image.shape,
    #     )

    #     seq = iaa.Sequential(
    #         [
    #             iaa.Resize(0.5),
    #             iaa.PadToFixedSize(
    #                 width=self.width, height=self.height, position="center"
    #             ),
    #             iaa.SomeOf(
    #                 1,
    #                 [
    #                     iaa.Affine(rotate=15),
    #                     iaa.AdditiveGaussianNoise(scale=0.2 * 255),
    #                     iaa.Add(50, per_channel=True),
    #                     iaa.Sharpen(alpha=0.5),
    #                 ],
    #             ),  # translate by 40/60px on x/y axis, and scale to 50-70%, affects BBs
    #         ]
    #     )

    #     # Augment BBs and images.
    #     image, bbs = seq(image=image, bounding_boxes=bbs)
    #     annotation[:, :-1] = np.array(
    #         [[bb.x1, bb.y1, bb.x2, bb.y2] for bb in bbs.bounding_boxes]
    #     )

    #     padding_mask = np.zeros_like(image, dtype=np.uint8)
    #     if self.getMask:
    #         original_height, original_width = original_image_shape[:2]
    #         original_height, original_width = (
    #             original_height * 0.5,
    #             original_width * 0.5,
    #         )

    #         # Assume 'image' is the augmented/padded image after the transformations
    #         padded_height, padded_width = image.shape[:2]

    #         # Calculate the padded region
    #         left_pad = int((padded_width - original_width) // 2)
    #         right_pad = int(padded_width - original_width - left_pad)
    #         top_pad = int((padded_height - original_height) // 2)
    #         bottom_pad = int(padded_height - original_height - top_pad)

    #         # Create a mask for the padded region
    #         padding_mask = cv2.rectangle(
    #             padding_mask,
    #             (left_pad, top_pad),
    #             (padded_width - right_pad, padded_height - bottom_pad),
    #             (255, 255, 255),  # White color as mask
    #             thickness=cv2.FILLED,
    #         )

    #         # # Show the mask (for visualization purposes)
    #         # cv2.imshow("Padding Mask", padding_mask)
    #         # cv2.waitKey(0)
    #         # cv2.destroyAllWindows()

    #         mask = rearrange(mask, "b h w c -> b c h w")

    #     if self.visualize:
    #         self.show_Anns(image, annotation)
    #         plt.show()

    #     if self.transform is not None:
    #         image = self.transform(image)

    #     # boxes = torch.FloatTensor(annotation[:, :4])
    #     # labels = torch.LongTensor(annotation[:, 4])
    #     return image, annotation, padding_mask

    def __getitem__(self, idx):  # 인덱스에 접근할 때 호출
        image = self.load_image(idx)
        annotation = self.load_annotations(idx)

        original_image_shape = image.shape

        image_t, annotation_t = self.transform(image, annotation)

        # if self.transform is not None:
        #     image = self.transform(image)

        # mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]

        padding_mask = np.zeros_like(image, dtype=np.uint8)
        if self.getMask:
            original_height, original_width = original_image_shape[:2]
            original_height, original_width = (
                original_height * 0.5,
                original_width * 0.5,
            )

            # Assume 'image' is the augmented/padded image after the transformations
            padded_height, padded_width = image.shape[:2]

            # Calculate the padded region
            left_pad = int((padded_width - original_width) // 2)
            right_pad = int(padded_width - original_width - left_pad)
            top_pad = int((padded_height - original_height) // 2)
            bottom_pad = int(padded_height - original_height - top_pad)

            # Create a mask for the padded region
            padding_mask = cv2.rectangle(
                padding_mask,
                (left_pad, top_pad),
                (padded_width - right_pad, padded_height - bottom_pad),
                (255, 255, 255),  # White color as mask
                thickness=cv2.FILLED,
            )

            # # Show the mask (for visualization purposes)
            # cv2.imshow("Padding Mask", padding_mask)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            mask = rearrange(mask, "b h w c -> b c h w")

        if self.visualize:
            self.show_Anns(image, annotation)
            plt.show()

        # boxes = torch.FloatTensor(annotation[:, :4])
        # labels = torch.LongTensor(annotation[:, 4])
        return image, annotation, padding_mask


def make_transforms(image_set):
    normalize = T.Compose(
        [T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    )

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    # if image_set == "train":
    transforms = T.Compose(
        [
            T.RandomHorizontalFlip(),
            T_.RandomSelect(
                T_.RandomResize(scales, max_size=1333),
                T.Compose(
                    [
                        T_.RandomResize([400, 500, 600]),
                        T_.RandomSizeCrop(384, 600),
                        T_.RandomResize(scales, max_size=1333),
                    ]
                ),
            ),
            normalize,
        ]
    )

    # if image_set == "val":
    #     transforms = T.Compose(
    #         [
    #             T.RandomResize([800], max_size=1333),
    #             normalize,
    #         ]
    #     )
    # raise ValueError(f"unknown {image_set}")

    return transforms


if __name__ == "__main__":
    from einops import rearrange, asnumpy

    # Image preprocessing, normalization for the pretrained resnet
    data_dir = "C:/Users/sar10/code/detr/cocodataset/"
    # image_set = "train2017"
    image_set = "val2017"
    coco = COCODataset(
        data_dir,
        image_set=image_set,
        visualize=False,
        transform=make_transforms(image_set),
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
