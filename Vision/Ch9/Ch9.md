# ADVANCED CONVOLUTIONS

## Normal Convolution
<!-- ![Conv](../assets/convolution.gif) -->
<img src="../assets/convolution.gif" alt="Conv" width="500"/>


## MultiChannel Convolution
<!-- ![ConceptofChannels](../assets/concept-of-channels.png) -->
<img src="../assets/concept-of-channels.png" alt="ConceptofChannels" width="500"/>



## [Receptive Fields](https://distill.pub/2019/computing-receptive-fields/)
- benefits of going beyond RF of image  (Edges/Gradient >> Pattern >> Parts of Image >> Image >> ??)

## [Stride & Checkerboard Issues](https://distill.pub/2016/deconv-checkerboard/)



Q: `How do we get 5x5 RF with one 3x3 Kernels?`
## [Atrous or Dilated Convolution](https://ezyang.github.io/convolution-visualizer/)
- increase the receptive view (global view) of the network exponentially and linear parameter accretion. 
- usage: **dense predicition**
    - Semantic Segmentation
    - Instance Segmentation
    - Panoptic Segmentation
    - Image Super-Resolution
    - Generative Art with Reference Image
    - Facial recognition
    - Post-Estimation

<!-- ![Dilated Conv](../assets/Atrous%20Conv.gif) -->
<img src="../assets/Atrous%20Conv.gif" alt="Dilated" width="500"/>


Q: `How do we increase channels size after convolution`
## Transpose Convolution or Deconvolution (Bilinear Interpolation)
<!-- ![Deconv](../assets/deconv.gif) -->
<img src="../assets/deconv.gif" alt="Deconv" width="500"/>


`Key:` 
- Dilated Conv is NOT to make features. 
- It's for Detect/Integration of features.

## Pixel Shuffle
<!-- ![pixel shuffle](../assets/PixelShuffle.jpg) -->
<img src="../assets/PixelShuffle.jpg" alt="Pixel shuffle" width="500"/>

## Depthwise Seperable Convolution
<!-- ![Depthwise Pointwise](../assets/Depthwise-Pointwise%20Conv.png) -->
<img src="../assets/Depthwise-Pointwise%20Conv.png" alt="Depthwise Pointwise" width="500"/>


## Grouped Convolutions
<img src="../assets/groupedConv.png" alt="grouped convolutions" width="500"/>


# ARCHITECTURE
## RESNET & RESNEXT 
<img src="../assets/resnet_resnext.png" alt="RESNET & RESNEXT ARCHITECTURE" width="500"/>


## INCEPTION 
<img src="../assets/inception.png" alt="Inception ARCHITECTURE" width="500"/>


# DATA AUGMENTATION
[The performance on vision tasks increases logarithmically based on the volume of training data size with preserving labels]()


- PMDA ( Poor Man's Data Augmentation Strategies)
    - Scale
    - Translation; 
    - Rotation; 
    - Blurring; 
    - Image Mirroring; 
    - Color Shifting / Whitening.
- MMDA ( Middle-Class Man's )
    - CutOut

    <img src="../assets/cutout.png" alt="Cutout Aug" width="500"/>

    - Mixup
    
    <img src="../assets/mixup.png" alt="Mixup Aug" width="500"/>

    - Elastic Distortions
    - RICAP
    
    <img src="../assets/ricap aug.PNG" alt="RICAP Aug" width="500"/>

- RMDA ( Rich Man's ) 
    - Reinforcement Learning or AutoAugmentations


# Dense VS Sparse Problems

- [X] Invariance   e.g) classification
- [X] Equivariance e.g) segmentation
 
<img src="../assets/sparse%20vs%20dense%20problem.png" alt="Dense VS Sparse Problem" width="500"/>