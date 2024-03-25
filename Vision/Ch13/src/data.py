import os 
import numpy as np 
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader
import lightning as pl

from PIL import Image,ImageFile
import random
from typing import Any

# ImageFile.LOAD_TRUNCATED_IMAGES= True

from .config import (
    MAP_IOU_THRESH,
    P_MOSIC,
    BATCH_SIZE,
    NUM_WORKERS,
    PIN_MEMORY,
    SHUFFLE,
    IMAGE_SIZE,
    TRAIN_TRASFORMS, TEST_TRANSFORMS,
    IMAGE_DIR,
    LABEL_DIR,
    ANCHORS
)

from .helpers import(
    cells_to_bboxes,
    iou_width_height as iou,
    non_max_suppression as nms,
    plot_image,
    xywhn2xyxy,
    xyxy2xywhn
)

class YOLODataset(Dataset):
    def __init__(self,
                 csv_file:str, 
                 img_dir:str, 
                 label_dir:str, 
                 anchors:list, 
                 image_size:int=416, 
                 S:list=[13,26,52],
                 C:int=20,
                 transform=None
    ) -> None:
        '''
            Construction of YOLODataset
            - args:
                - csv_file : csv_file location where (img,label) mapping done.
                - img_dir  : image directory location
                - label_dir: label_directory location 
                - anchors  : list of anchor boxes in our case, [ anchors[0], anchors[1], anchors[3] ] i.e) each have 3 anchors centroid
                - transform: Albumentation transforms
                - S        : list of scaled Predictions image_size [13x13, 26x26, 52x52]
                - C        : number of prediction classes
            - functions:
                - __len__
                - __getitem__
                - load_mosaic              

        '''
        super().__init__()
        self.annotations: pd.DataFrame = pd.read_csv(csv_file)
        self.image_dir:str             = img_dir
        self.label_dir:str             = label_dir
        self.image_size:int            = image_size
        self.mosaic_borders:list       = [image_size//2,image_size//2]
        self.transform                 = transform
        self.S:list                    = S                         
        self.C:int                     = C


        # [3`prediction layers`,  3`anchor boxes for each layer`, 2`centroid points(x,y) for each anchor`]
        # [3,3,2] -> [[9,2]]
        self.anchors                   = torch.tensor(
            anchors[0] + anchors[1] + anchors[2]
        ) # for all 3 scales  

        self.num_anchors:int          = self.anchors.shape[0]
        self.num_anchors_per_scale    = self.num_anchors //3 
        self.ignore_iou_thresh        = MAP_IOU_THRESH


    def __len__(self):
        return len(self.annotations)
    

    def load_mosaic(self, index):
        '''
            function returns `mosiac images` Loads 1 image + 3 random Images
        '''
        labels4 = []
        s = self.image_size
        yc, xc = (
            int(random.uniform(x, 2 * s - x)) for x in self.mosaic_borders
        )  # mosaic center x, y
        indices = [index] + random.choices(
            range(len(self)), k=3
        )  # 3 additional image indices
        random.shuffle(indices)
        for i, index in enumerate(indices):
            # Load image
            label_path = os.path.join(self.label_dir, self.annotations.iloc[index, 1])
            bboxes = np.roll(
                np.loadtxt(fname=label_path, delimiter=" ", ndmin=2), 4, axis=1
            ).tolist()
            img_path = os.path.join(self.image_dir, self.annotations.iloc[index, 0])
            img = np.array(Image.open(img_path).convert("RGB"))

            h, w = img.shape[0], img.shape[1]
            labels = np.array(bboxes)

            ############################
            ## place image in imag4
            #############################
            if i == 0:  # top left
                img4 = np.full(
                    (s * 2, s * 2, img.shape[2]), 114,  dtype=np.uint16 #dtype=np.uint8
                )  # base image with 4 tiles
                x1a, y1a, x2a, y2a = (
                    max(xc - w, 0),
                    max(yc - h, 0),
                    xc,
                    yc,
                )  # xmin, ymin, xmax, ymax (large image)
                x1b, y1b, x2b, y2b = (
                    w - (x2a - x1a),
                    h - (y2a - y1a),
                    w,
                    h,
                )  # xmin, ymin, xmax, ymax (small image)
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

            img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
            padw = x1a - x1b
            padh = y1a - y1b

            # Labels
            if labels.size:
                labels[:, :-1] = xywhn2xyxy(
                    labels[:, :-1], w, h, padw, padh
                )  # normalized xywh to pixel xyxy format
            labels4.append(labels)

        # Concat/clip labels
        labels4 = np.concatenate(labels4, 0)
        for x in (labels4[:, :-1],):
            np.clip(x, 0, 2 * s, out=x)  # clip when using random_perspective()
        # img4, labels4 = replicate(img4, labels4)  # replicate
        labels4[:, :-1] = xyxy2xywhn(labels4[:, :-1], 2 * s, 2 * s)
        labels4[:, :-1] = np.clip(labels4[:, :-1], 0, 1)
        labels4 = labels4[labels4[:, 2] > 0]
        labels4 = labels4[labels4[:, 3] > 0]
        return img4, labels4


    def __getitem__(self, index) -> Any:
        if random.random() >= P_MOSIC:
            image, bboxes = self.load_mosaic(index)
        else:
            label_path = os.path.join( self.label_dir, self.annotations.iloc[index,1])
            bboxes     = np.roll(
                np.loadtxt(fname=label_path, delimiter=" ", ndmin=2), 4, axis=1
            ).tolist()
            img_path = os.path.join( self.image_dir, self.annotations.iloc[index,0] )
            image = np.array(Image.open(img_path).convert("RGB"))

        if self.transform:
            image = np.uint8(image)
            augmentations = self.transform(image=image,bboxes=bboxes)
            image = augmentations['image']
            bboxes = augmentations['bboxes']
        # Below assumes 3 scale predictions (as paper) and same num of anchors per scale
        targets = [torch.zeros((self.num_anchors // 3, S, S, 6)) for S in self.S]
        for box in bboxes:
            iou_anchors = iou(torch.tensor(box[2:4]), self.anchors)   #self.anchors:: (9,2)
            anchor_indices = iou_anchors.argsort(descending=True, dim=0)
            x, y, width, height, class_label = box
            has_anchor = [False] * 3  # each scale should have one anchor
            for anchor_idx in anchor_indices:
                scale_idx = anchor_idx // self.num_anchors_per_scale
                anchor_on_scale = anchor_idx % self.num_anchors_per_scale
                S = self.S[scale_idx]
                i, j = int(S * y), int(S * x)  # which cell
                anchor_taken = targets[scale_idx][anchor_on_scale, i, j, 0]
                if not anchor_taken and not has_anchor[scale_idx]:
                    targets[scale_idx][anchor_on_scale, i, j, 0] = 1
                    x_cell, y_cell = S * x - j, S * y - i  # both between [0,1]
                    width_cell, height_cell = (
                        width * S,
                        height * S,
                    )  # can be greater than 1 since it's relative to cell
                    box_coordinates = torch.tensor(
                        [x_cell, y_cell, width_cell, height_cell]
                    )
                    targets[scale_idx][anchor_on_scale, i, j, 1:5] = box_coordinates
                    targets[scale_idx][anchor_on_scale, i, j, 5] = int(class_label)
                    has_anchor[scale_idx] = True

                elif (
                    not anchor_taken
                    and iou_anchors[anchor_idx] > self.ignore_iou_thresh
                ):
                    targets[scale_idx][
                        anchor_on_scale, i, j, 0
                    ] = -1  # ignore prediction

        return image, tuple(targets)



class PASCALDataModule(pl.LightningDataModule):
    def __init__(
            self, 
            train_csv_path:str=None, 
            test_csv_path:str=None
    ) -> None:
        super().__init__()
        self.train_csv_path = train_csv_path
        self.test_csv_path  = test_csv_path
        self.batch_size     = BATCH_SIZE
        self.num_workers    = NUM_WORKERS
        self.shuffle        = SHUFFLE
        self.IMAGE_SIZE     = IMAGE_SIZE
        self.anchors        = ANCHORS
        self.pin_memory     = PIN_MEMORY
        self.IMAGE_SIZE     = IMAGE_SIZE

        
    def prepare_data(self) -> None:
        pass 

    def setup(self,stage=None):
        self.train_dataset:YOLODataset = YOLODataset(
            self.train_csv_path,
            transform= TRAIN_TRASFORMS,
            S= [self.IMAGE_SIZE//32, self.IMAGE_SIZE//16, self.IMAGE_SIZE//8],
            img_dir=IMAGE_DIR,
            label_dir= LABEL_DIR,
            anchors= self.anchors
        )
        self.val_dataset:YOLODataset = YOLODataset(
            self.test_csv_path,
            transform= TEST_TRANSFORMS,
            S= [self.IMAGE_SIZE//32, self.IMAGE_SIZE//16, self.IMAGE_SIZE//8],
            img_dir=IMAGE_DIR,
            label_dir= LABEL_DIR,
            anchors= self.anchors
        )

        self.test_dataset:YOLODataset = YOLODataset(
            self.test_csv_path,
            transform= TEST_TRANSFORMS,
            S= [self.IMAGE_SIZE//32, self.IMAGE_SIZE//16, self.IMAGE_SIZE//8],
            img_dir=IMAGE_DIR,
            label_dir= LABEL_DIR,
            anchors= self.anchors
        )
        
    def train_dataloader(self) -> Any:
        return DataLoader(
            dataset= self.train_dataset,
            batch_size= self.batch_size,
            num_workers= self.num_workers,
            pin_memory= self.pin_memory,
            shuffle=True
        )  

    def val_dataloader(self) -> Any:
        return DataLoader(
            dataset= self.test_dataset,
            batch_size= self.batch_size,
            num_workers= self.num_workers,
            pin_memory= self.pin_memory,
            shuffle=False
        ) 

    def test_dataloader(self) -> Any:
        return DataLoader(
            dataset= self.test_dataset,
            batch_size= self.batch_size,
            num_workers= self.num_workers,
            pin_memory= self.pin_memory,
            shuffle=False
        ) 
