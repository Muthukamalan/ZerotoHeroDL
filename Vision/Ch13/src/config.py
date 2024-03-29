import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2 
import os
import torch




DATASET       = "../../data/PASCAL_VOC/"
DEVICE        = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
DEVICE_COUNT  = torch.cuda.device_count()
NUM_WORKERS   = os.cpu_count()-1
BATCH_SIZE    = 256#512
SHUFFLE       = True
IMAGE_SIZE    = 416
NUM_CLASSES   = 20
LEARNING_RATE = 1e-3
WEIGHT_DECAY  = 1e-4
NUM_EPOCHS    = 40
CONF_THRESHOLD= 0.05
MAP_IOU_THRESH= 0.5
NMS_OPU_THRESH= 0.45
S             = [IMAGE_SIZE//32, IMAGE_SIZE//16, IMAGE_SIZE//8]
PIN_MEMORY    = True
LOAD_MODEL    = False
SAVE_MODEL    = True
CHKPT_FILE    = 'checkpoint.pth.tar'
IMAGE_DIR     = DATASET+'/images/'
LABEL_DIR     = DATASET+'/labels/'
P_MOSIC       = 0.05


ANCHORS = [
    [(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)],
    [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)],
    [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)]
] # Note these have been rescaled to be  between [0,1]   
# [3,3,2] 
# 3 prediction layers
# 3 anchor boxes for each layer, so 3x3=9 boxes
# each box has 2 points (centroid)




MEANS = [0.485, 0.456, 0.406]
SCALE = 1.1


TRAIN_TRASFORMS = A.Compose(
    [
        A.LongestMaxSize(max_size=int(IMAGE_SIZE*SCALE)),
        A.PadIfNeeded(
            min_height = int(IMAGE_SIZE*SCALE),
            min_width  = int(IMAGE_SIZE*SCALE),
            border_mode= cv2.BORDER_CONSTANT,
        ),
        A.Rotate(limit=10,interpolation=1,border_mode=4),
        A.RandomCrop(width=IMAGE_SIZE, height=IMAGE_SIZE),
        A.ColorJitter(brightness=0.6, contrast=0.6,saturation=0.6,hue=0.4,p=0.4),
        A.OneOf([
            A.ShiftScaleRotate(rotate_limit=20,p=0.5,border_mode=cv2.BORDER_CONSTANT),
        ],p=1.0),
        A.HorizontalFlip(p=0.5),
        A.Blur(p=0.1),
        A.CLAHE(p=0.1),
        A.Posterize(p=0.1),
        A.ToGray(p=0.1),
        # A.ChannelShuffle(p=0.05),
        A.Normalize(
            mean=[0, 0, 0],
            std=[1, 1, 1],
            max_pixel_value=255,
        ),
        ToTensorV2(),
    ],
    bbox_params=A.BboxParams(format="yolo",min_visibility=0.4,label_fields=[])
)

TEST_TRANSFORMS = A.Compose([
        A.LongestMaxSize(max_size=IMAGE_SIZE),
        A.PadIfNeeded(
            min_height=IMAGE_SIZE, min_width=IMAGE_SIZE, border_mode=cv2.BORDER_CONSTANT
        ),
        A.Normalize(
            mean=[0, 0, 0],
            std=[1, 1, 1],
            max_pixel_value=255,
        ),
        ToTensorV2(),
    ],
    bbox_params=A.BboxParams(format="yolo", min_visibility=0.4, label_fields=[]),
)



PASCAL_CLASSES = [
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]