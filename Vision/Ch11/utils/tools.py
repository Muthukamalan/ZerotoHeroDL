from math import sqrt,floor,ceil 
from typing import Iterable,List,Tuple 
from matplotlib import pyplot as plt 
import numpy as np 
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix
import torch
from torch import Tensor 


def denormalize(img:Tensor):
    '''
        normalize_img $= {(img - mean) \over std}$
        img $={normalize\_img*std}+mean$
    '''
    channel_means = (0.4914, 0.4822, 0.4465 )
    channel_stds  = (0.2470, 0.2435, 0.2616 )

    # IMAGE (C*H*W)
    img = img.astype(dtype=np.float32)

    for i in range(img.shape[0]):
        img[i] = (img[i]*channel_stds[i] + channel_means[i])

    # Return (H*W*C)
    return np.transpose(img,(1,2,0))




def get_rows_cols(num: int) -> Tuple[int, int]:
    '''helper for visualize_data function to display'''
    cols = floor(sqrt(num))
    rows = ceil(num / cols)
    return rows, cols



def visualize_data( loader:torch.utils.data.DataLoader,num_figures: int =12,label:str = "",classes:List[str] = [] ):
    '''
        helper function to visualize data
        with setting,
            col=$abs|\sqrt(num)|$ and 
            row=$abs|{num \over col}|$
    '''

    # Loop 1st batch Images
    batch_data, batch_labels = next(iter(loader))

    fig = plt.figure()
    fig.suptitle(label)

    rows,cols = get_rows_cols(num=num_figures)  
    for i in range(num_figures):
        plt.subplot(rows,cols,i+1)
        plt.tight_layout()

        # denormalized images with shape H*W*C
        npimg = denormalize(batch_data[i].cpu().numpy().squeeze())
        label = (
            classes[batch_labels[i]] if batch_labels[i]<len(classes) else batch_labels[i]
        )
        plt.imshow(npimg,cmap='gray')
        plt.title(label)
        plt.xticks([])
        plt.yticks([])



def show_misclassified_images(
        images: List[Tensor],
        predictions:List[int],
        labels:List[int],
        classes:List[str]
    ):
    '''helper function to show misclassified imgs'''
    assert len(images)==len(predictions) == len(labels)

    fig = plt.figure(figsize=(20,10))
    for i in range(len(images)):
        sub = fig.add_subplot(len(images)//5, 5, i+1)
        img = images[i]
        npimg = denormalize(img.cpu().numpy().squeeze())
        plt.imshow(npimg,cmap='gray')
        predicted = classes[predictions[i]]
        correct   = classes[labels[i]]

        sub.set_title("Correct class: {}\nPredicted class: {}".format(correct,predicted))
    plt.tight_layout()
    plt.show()




def plot_class_distribution(loader,classes):
    class_counts={}
    for cname in classes:
        class_counts[cname]=0
    for _,labels in loader:
        for lbl in labels:
            class_counts[
                    classes[ lbl.item() ]
            ]+=1
    fig = plt.figure()
    plt.suptitle('Class Distribution')
    plt.bar(
        range( len(class_counts) ),
        list(class_counts.values())
    )
    plt.xticks(
        range( len(class_counts) ),
        list(class_counts.keys()),
        rotation=90
     )
    plt.tight_layout()
    plt.show()



def plot_confusion_matrix(model,test_loader,device,classes):
    model.eval()
    predictions = []
    labels = []  # TRUTH
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, preds = torch.max(output, 1)
            predictions.extend(preds.cpu().numpy())
            labels.extend(target.cpu().numpy())
    
    cm= confusion_matrix(y_true= [i.item() for i in labels],y_pred=[i.item() for i in predictions])
    columns ={}
    for i,v in enumerate(classes):
        columns[i]=v
    plt.figure(figsize=(9,6))
    sns.heatmap(pd.DataFrame(cm).rename(columns=columns,index=columns),annot=True,fmt='',cmap="crest")
    plt.xlabel('y_pred')
    plt.ylabel('y_true')
    plt.show()




def plot_curves(train_losses,train_acc,test_losses,test_acc):
    fig, axs = plt.subplots(2,2,figsize=(15,10))
    
    axs[0, 0].plot(train_losses)
    axs[0, 0].set_title("Training Loss")
    
    axs[1, 0].plot(train_acc)
    axs[1, 0].set_title("Training Accuracy")
    
    axs[0, 1].plot(test_losses)
    axs[0, 1].set_title("Test Loss")
    
    axs[1, 1].plot(test_acc)
    axs[1, 1].set_title("Test Accuracy")
    plt.show()


def lr_curve(lr_history):
    plt.plot(lr_history)
    plt.title("Learning Rate")
    plt.show()




from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image,deprocess_image,preprocess_image
import cv2
def show_grad_cam_images(
        model,
        target_layers,
        misclassified_images,
        misclassified_predictions,
        correct_labels,
        class_names
)->None:
    """
    args:
    - model
    - target_layers
    - misclassified_images
    - misclassified_predictions
    - correct_labels
    - class_names
    """
    assert( len(misclassified_images)==20,"greater or equal to atleast 20 images")
    fig,axes = plt.subplots(20,1,figsize=(20,55),tight_layout=True);
    plt.axis('off')
    with GradCAM(model=model, target_layers=target_layers) as cam:
        for (mimg,mpred,mlabel),ax in zip(
                zip(
                      misclassified_images[:20],
                      misclassified_predictions[:20],
                      correct_labels[:20]
                ), axes.flatten()
        ):
            grayscale_cams = cam(
                            input_tensor=mimg.unsqueeze(0),                                                                    # BS,C,H,W
                            targets=[ClassifierOutputTarget(mpred.item())]
                        ) 
            color_cams  = show_cam_on_image(
                                mimg.permute(1,2,0).float().detach().cpu().numpy().astype(np.float32) / 255,                    # H,W,C
                                grayscale_cams.squeeze(),                                                                       # H,W
                                use_rgb=True
                    )
            grayscale_cams = np.uint8(255*grayscale_cams[0, :])
            grayscale_cams = cv2.merge([grayscale_cams, grayscale_cams, grayscale_cams])
            grad_img = np.concatenate([ deprocess_image(mimg.permute(1,2,0).detach().cpu().numpy()), grayscale_cams, color_cams],axis=1)
            ax.imshow(grad_img)
            ax.set_title("Predicted class: {}\nCorrect class: {}".format(class_names[mpred.item()],class_names[mlabel.item()]))
            ax.axis('off')
    fig.show()