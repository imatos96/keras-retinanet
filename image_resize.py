#izvor: https://github.com/asetkn/Tutorial-Image-and-Multiple-Bounding-Boxes-Augmentation-for-Deep-Learning-in-4-Steps/blob/master/Tutorial-Image-and-Multiple-Bounding-Boxes-Augmentation-for-Deep-Learning-in-4-Steps.ipynb



import imgaug as ia
ia.seed(1)
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from imgaug import augmenters as iaa 
import imageio
import pandas as pd
import numpy as np
import re
import os
import glob
import shutil
from csv import reader, writer

folders = glob.glob('D:\\tsr\\train\\*')
images = []

def searchImagesFolder(list, str):
    for folder in folders:
        for index, file in enumerate(glob.glob(folder+str)):
            images.append(imageio.imread(file))
    return images

def bbs_obj_to_df(bbs_object):
    bbs_array = bbs_object.to_xyxy_array()
    df_bbs = pd.DataFrame(bbs_array, columns=['xmin', 'ymin', 'xmax', 'ymax'])
    df_bbs.round(0).astype(int)
    return df_bbs


def resize_imgaug(df, images_path, aug_images_path, image_prefix):
    aug_bbs_xy = pd.DataFrame(columns=
                              ['filename', 'xmin', 'ymin', 'xmax', 'ymax', 'label']
                             )
    grouped = df.groupby('filename')    
    
    for filename in df['filename'].unique():
        group_df = grouped.get_group(filename)
        group_df = group_df.reset_index()
        group_df = group_df.drop(['index'], axis=1)

        image = imageio.imread(filename)
        height=image.shape[0]
        width=image.shape[1]
       
        fil=filename.split('/')
        fil1=fil[0]
        fil2=fil[1]

        if(group_df['label'].isnull().values.any()):
            image = imageio.imread(images_path+filename)
            image_aug = height_resize(image=image)
            imageio.imwrite(aug_images_path+'/'+fil1+'/'+image_prefix+fil2, image_aug)  

            group_df['xmin'] =''
            group_df['xmax'] =''
            group_df['ymin'] =''
            group_df['ymax'] =''
            info_df = group_df

            info_df['filename'] = info_df['filename'].apply(lambda x: aug_images_path+fil1+'/'+image_prefix+fil2)
            aug_bbs_xy = pd.concat([aug_bbs_xy, info_df])

           

        else:
            image = imageio.imread(images_path+filename)
            bb_array = group_df.drop(['filename', 'label'], axis=1).values
            bbs = BoundingBoxesOnImage.from_xyxy_array(bb_array, shape=image.shape)
            image_aug, bbs_aug = height_resize(image=image, bounding_boxes=bbs)
            imageio.imwrite(aug_images_path+'/'+fil1+'/'+image_prefix+fil2, image_aug)  
            info_df = group_df.drop(['xmin', 'ymin', 'xmax', 'ymax'], axis=1)   
            info_df['filename'] = info_df['filename'].apply(lambda x: aug_images_path+fil1+'/'+image_prefix+fil2)
            bbs_df = bbs_obj_to_df(bbs_aug)
            aug_df = pd.concat([info_df, bbs_df], axis=1)
            aug_bbs_xy = pd.concat([aug_bbs_xy, aug_df])

    aug_bbs_xy = aug_bbs_xy.reset_index()
    aug_bbs_xy = aug_bbs_xy.drop(['index'], axis=1)
    return aug_bbs_xy


print('We have {} images'.format(len(images)))

labels_df = pd.read_csv("Train.csv", header=None, names=['filename', 'xmin', 'ymin', 'xmax', 'ymax', 'label']) 

height_resize = iaa.Sequential([ 
    iaa.Resize({"height": 600, "width": 800})
])

width_resize = iaa.Sequential([ 
    iaa.Resize({"height": 600, "width": 800})
])

resized_images_df = resize_imgaug(labels_df, 'C:/Users/vova/Desktop/tsr/train/', 'augImg/', 'resizeImg')
resized_images_df.to_csv(('newTrainAnotationResizing.csv'),index=False, header = False)


