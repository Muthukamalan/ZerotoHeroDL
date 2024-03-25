from .config import (
    ANCHORS,
    S,
    PASCAL_CLASSES,
    NUM_CLASSES,
    CHKPT_FILE,
    DEVICE,
    NUM_WORKERS,
    SAVE_MODEL,
    LEARNING_RATE,
    WEIGHT_DECAY,
    NUM_EPOCHS,
    CONF_THRESHOLD,
    LOAD_MODEL,
    PIN_MEMORY,
    IMAGE_SIZE,
    BATCH_SIZE
)
from .loss import YOLOLoss
from .data import PASCALDataModule
from .helpers import one_cycle_lr,check_class_accuracy


import lightning as pl
from torch import nn 
from lightning.pytorch.utilities.types import STEP_OUTPUT
import torch
from typing import Any,AnyStr



# Model Architecture Config
# tuple: Conv Block
# list:  Residual Block represent "B" and number of repetations
# string:
#   S: Scaled Prediction (compute_loss) 
#   U: Upscale featuremap and concatenate with previous layer
config = [
    (32, 3, 1),
    (64, 3, 2),
    ["B", 1],
    (128, 3, 2),
    ["B", 2],
    (256, 3, 2),
    ["B", 8],
    (512, 3, 2),
    ["B", 8],
    (1024, 3, 2),
    ["B", 4],  # To this point is Darknet-53
    (512, 1, 1),
    (1024, 3, 1),
    "S",
    (256, 1, 1),
    "U",
    (256, 1, 1),
    (512, 3, 1),
    "S",
    (128, 1, 1),
    "U",
    (128, 1, 1),
    (256, 3, 1),
    "S",
]



class CNNBlock(pl.LightningModule):
    def __init__(self,in_channels:int,out_channels:int,bn_act:bool=True,**kwargs:Any) -> None:
        '''
            # ConvBlock 
            args:
            - in_channels:  input_channels of conv layer
            - out_channels: output_channels of conv layer
            - bn_act:       wanna use batch_norm and activation function
            - **kwargs:     will take care  
                - kernel_size:int (required),
                - padding:int=0` , 
                - stride:int=1, 
                - dilation:int =1 and 
                - groups:int =1
        '''
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels,out_channels=out_channels,bias=False,**kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(0.1)
        self.use_bn_act = bn_act

    def forward(self,x)->torch.Tensor:
        '''
            x be the input tensor
        '''
        if self.use_bn_act:
            return self.relu(self.bn(self.conv(x)))
        else:
            return self.conv(x)
        



class ResidualBlock(pl.LightningModule):
    def __init__(self,channels:int, use_residual:bool=True,num_repeats:int=1, **kwargs: Any) -> None:
        '''
            # Residual Block
            args:
            - channels:      number of input channels for CNNBlock
            - use_residual:  wanna add `input_tensor + residual`
            - num_repeats:   number of time residual block gonna repeat

            ## Note:
            - In Residual Block Internel, 
                - we reduce number of channels, by 1x1 kernel
                - do Feature-MAP by 3x3 kernel and increase number of channels

            - In Residual Block Construction,
                - construct layer by number of times
                - (0-num_repeats): num_repeats x Sequentials
                    (
                        - [0] CNNBlock
                        - [1] CNNBlock
                    )
                    if residue: x+residue(x)
                    else: x
        '''
        super().__init__()
        self.num_repeats:int =  num_repeats
        self.use_residual:bool = use_residual 

        self.layers = nn.ModuleList()
        for repeat in range(num_repeats):
            self.layers += [
                nn.Sequential(
                    CNNBlock(
                        in_channels = channels, 
                        out_channels= channels//2,        # reduce channels by half by 1x1
                        kernel_size=1
                    ),
                    CNNBlock(
                        in_channels= channels//2, 
                        out_channels= channels,           # Features + increase channels as same 3x3
                        kernel_size = 3,
                        padding =1 
                    )
                )
            ]
    def forward(self,x):
        for layer in self.layers:
            if self.use_residual:
                x += layer(x)
            else:
                x = layer(x)
        return(x)
    

class ScalePrediction(nn.Module):
    def __init__(self, in_channels, num_classes) -> None:
        super().__init__()

        self.num_classes = num_classes
        self.pred = nn.Sequential(
            CNNBlock(in_channels, 2*in_channels, bn_act=True,kernel_size=3,padding=1),  # increase channels by 2X
            CNNBlock(2*in_channels, (self.num_classes +5)*3, bn_act=False, kernel_size=1 )    # prediction_layer: num_anchor_box=3, coordinates+objectness_score=4+1 , number_of_classes
        )


    def forward(self,x):
        # x-> (bs,c,h,w)
        # print(x.shape)
        # x = self.pred(x)
        # print(x.shape)
        # x = x.reshape(  x.shape[0], 3, self.num_classes+5,  x.shape[2],  x.shape[3])
        # print(x.shape)
        # x = x.permute(0,1,3,4,2)
        # print(x.shape)
        # return x

        return(
            self.pred(x)\
                    .reshape(
                            x.shape[0], 
                            3, 
                            self.num_classes+5, 
                            x.shape[2], 
                            x.shape[3] )\
                    .permute(0,1,3,4,2)
        )
    


class YOLOV3(pl.LightningModule):
    def __init__(
            self,
            in_channel=3,
            num_classes=NUM_CLASSES,
            epochs:int=NUM_EPOCHS,
            loss_fn=YOLOLoss,
            data_module = PASCALDataModule,
            learning_rate=LEARNING_RATE,
            weight_decay = WEIGHT_DECAY,
            maxlr = None,
            scheduler_steps = None,
            device_count=NUM_WORKERS,
            ) -> None:
        super().__init__()
        self.num_classes   = num_classes
        self.in_channels   = in_channel
        self.epochs        = epochs
        self.loss          = loss_fn       #instance returned
        self.data_module   = data_module  #directly passing values of PASCALDataModule
        self.learning_rate = learning_rate
        self.weight_decay  = weight_decay
        self.max_lr        = maxlr
        self.scheduler_step= scheduler_steps
        self.device_count  = device_count

        self.layers        = self._create_conv_layers()

        # S (scaled frame)
        # [[[13, 13], [13, 13], [13, 13]],
        #  [[26, 26], [26, 26], [26, 26]],
        #  [[52, 52], [52, 52], [52, 52]]]
        self.scaled_anchors= torch.tensor(ANCHORS) * torch.tensor(S).unsqueeze(1).unsqueeze(1).repeat(1,3,2)

    def return_layers(self):
        return self.layers
    
    def forward(self,x):
        outputs           = []       # for each scale
        route_connections = [] 
        for layer in self.layers:
            if isinstance(layer,ScalePrediction):
                outputs.append(layer(x))
                continue
            
            x = layer(x)

            if isinstance(layer,ResidualBlock) and layer.num_repeats==8:   # runs only 2 times
                route_connections.append(x)
            elif isinstance(layer,nn.Upsample):
                x = torch.cat([x,route_connections[-1]],dim=1)
                route_connections.pop()

        return outputs


    def _create_conv_layers(self):
        layers = nn.ModuleList()
        in_channels = self.in_channels

        for module in config:
            # CONV
            if isinstance(module,tuple):
                out_channels, kernel_size,stride = module
                layers.append(
                    CNNBlock(
                        in_channels,
                        out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding = 1 if kernel_size==3 else 0
                        )
                ) 
                in_channels = out_channels
            
            # RESIDUAL
            if isinstance(module,list):
                num_repeats = module[1]
                layers.append(
                    ResidualBlock(in_channels,num_repeats=num_repeats)
                ) 

            # ScaledPrediction / UpScaling
            if isinstance(module,str):
                if module=="S":
                    layers+= [
                            ResidualBlock(in_channels,use_residual=False,num_repeats=1),
                            CNNBlock(in_channels,in_channels//2,kernel_size=1),
                            ScalePrediction(in_channels//2,num_classes=self.num_classes)
                        ]
                    
                    in_channels = in_channels//2
                elif module=="U":
                    layers.append( nn.Upsample(scale_factor=2) )
                    # increase channel after upscaling
                    in_channels = in_channels*3 


        return layers



    def configure_optimizers(self)->dict:
        optimizer = torch.optim.Adam(
                            self.parameters(),
                            lr = self.learning_rate,
                            weight_decay= self.weight_decay
                        )
        scheduler = one_cycle_lr(
            optimizer=optimizer,
            maxlr=self.max_lr,steps=self.scheduler_step,epochs=self.epochs
        )
        return {
            "optimizer":optimizer,
            "lr_scheduler":{
                "scheduler":scheduler,
                "interval":"step"
            }
        }
    
    def _common_step(self,batch,batch_idx):
        self.scaled_anchors = self.scaled_anchors.to(self.device)
        x,y = batch                    # len(y)=3  
        y0, y1, y2 = y[0], y[1], y[2]  # y[i] = (BS, 3, anchor_resolution_{13,26,52},anchor_resolution_{13,26,52}, 6 )

        out = self(x)   #model(x)
        loss = (
            self.loss(out[0],y0, self.scaled_anchors[0])
            + self.loss(out[1],y1, self.scaled_anchors[1])
            + self.loss(out[2],y2, self.scaled_anchors[2])
        )
        return loss
    
    def training_step(self,batch,batch_idx):
        loss = self._common_step(batch,batch_idx)
        self.log(name="trian_loss",value=loss,on_step=True,on_epoch=True,prog_bar=True)
        return loss 
    
    def validation_step(self,batch,batch_idx):
        loss = self._common_step(batch,batch_idx)
        self.log(name="val_loss",value=loss,on_step=True,on_epoch=True,prog_bar=True)
        return loss 
    
    def test_step(self,batch,batch_idx):
        class_acc , noobj_acc, obj_acc = check_class_accuracy(model=self,loader=self.data_module.test_dataloader(), threshold=CONF_THRESHOLD  ,device=DEVICE )
        self.log_dict(
            {
                "class_acc":class_acc,
                "noobj_acc":noobj_acc,
                "obj_acc":obj_acc
            },
            prog_bar=True
        )
        

if __name__=='__main__':
    model = YOLOV3(num_classes=NUM_CLASSES).to(DEVICE)
    IMAGE_SIZE
    x = torch.randn((2, 3, IMAGE_SIZE, IMAGE_SIZE)).to(DEVICE)
    out = model(x)
    assert model(x)[0].shape == (
        2,
        3,
        IMAGE_SIZE // 32,
        IMAGE_SIZE // 32,
        NUM_CLASSES + 5,
    ),"Dimension Mismatch in ScaledPrediction Zero block"
    assert model(x)[1].shape == (
        2,
        3,
        IMAGE_SIZE // 16,
        IMAGE_SIZE // 16,
        NUM_CLASSES + 5,
    ),"Dimension Mismatch in ScaledPrediction First block"
    assert model(x)[2].shape == (
        2,
        3,
        IMAGE_SIZE // 8,
        IMAGE_SIZE // 8,
        NUM_CLASSES + 5,
    ),"Dimension Mismatch in ScaledPrediction Thired block"
    print(f"model is ready to serve!!")