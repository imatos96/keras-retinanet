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
import cv2

folders = glob.glob('D:\\tsr\\train\\*')
images = []

counter=0
counter2=0
counter3=0

aug = iaa.SomeOf(2, [    
    iaa.Affine(rotate=(-5, 5)),
    iaa.AffineCv2(translate_px=8, mode=["replicate"]),
    iaa.Affine(shear=(-8, 8),cval=(0, 255)),
    iaa.AffineCv2(scale=(0.5, 1.5), order=[0, 1]),
    iaa.Multiply((0.5, 1.5)),
    iaa.AverageBlur(k=(2, 5)),
    iaa.MedianBlur(k=(3, 7)),
    iaa.GaussianBlur(sigma=(1.0, 3.0)),
    iaa.Sharpen(alpha=(0, 0.6), lightness=(0.75, 1.5)),
    iaa.Emboss(alpha=(0, 0.85), strength=(0, 1.5)),
    iaa.LinearContrast((0.5, 2.0), per_channel=0.5),
    iaa.AdditiveGaussianNoise(scale=(0.03*255, 0.05*255))
])



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




def image_aug(df, images_path, aug_images_path, image_prefix, augmentor):
    aug_bbs_xy = pd.DataFrame(columns=
                              ['filename', 'xmin', 'ymin', 'xmax', 'ymax', 'label']
                             )
    grouped = df.groupby(['label'])    
    grouped2 = df.groupby(['filename'])  

    for label in grouped.size().index:
        print(label)
        group_df = grouped.get_group(label)
        group_df = group_df.reset_index()
        group_df = group_df.drop(['index'], axis=1)
        counter=0
        counter3=0
        counter2=0
        while (group_df['filename'].unique().size+counter)<1000:
            if((group_df['filename'].unique().size+counter)>999):
                 break

            for filename in group_df['filename'].unique(): 
                 if((group_df['filename'].unique().size+counter)>999):
                    break


                 group_df2=grouped2.get_group(filename)
                 group_df2 = group_df2.reset_index()
                 group_df2= group_df2.drop(['index'], axis=1)

                 counter+=1
                 yolo=str(counter)

                 fil=filename.split('/')
                 fil1=fil[0]
                 fil2=fil[1]
                 fil3=fil[2]

                 if (group_df['label'].isnull().values.any()):
                      image = imageio.imread(filename)
                      image_aug = augmentor(image=image)
                      
                      yolo=str(counter)
                      imageio.imwrite(aug_images_path+'/'+image_prefix+yolo+'_'+fil3, image_aug)
                      group_df2['xmin'] =''
                      group_df2['xmax'] =''
                      group_df2['ymin'] =''
                      group_df2['ymax'] =''
                      info_df = group_df2

                      info_df['filename'] = info_df['filename'].apply(lambda x: aug_images_path+fil1+'/'+image_prefix+fil2)
                      aug_bbs_xy = pd.concat([aug_bbs_xy, info_df])
                      counter2+=1

                 else:
        
                    image = imageio.imread(filename)     
                    bb_array = group_df2.drop(['filename', 'label'], axis=1).values
                    bbs = BoundingBoxesOnImage.from_xyxy_array(bb_array, shape=image.shape)
                    image_aug, bbs_aug = augmentor(image=image, bounding_boxes=bbs)
                    bbs_aug = bbs_aug.remove_out_of_image()
                    bbs_aug = bbs_aug.clip_out_of_image()

                    if re.findall('Image...', str(bbs_aug)) == ['Image([]']:
                         pass

                    else:
                        imageio.imwrite(aug_images_path+fil2+'/'+image_prefix+yolo+'_'+fil3, image_aug)  
                        counter3+=1
                  
                        info_df = group_df2.drop(['xmin', 'ymin', 'xmax', 'ymax'], axis=1)      
                      
                        info_df['filename'] = info_df['filename'].apply(lambda x: aug_images_path+fil2+'/'+image_prefix+yolo+'_'+fil3)
                     
                        bbs_df = bbs_obj_to_df(bbs_aug)
                  
                        aug_df = pd.concat([info_df, bbs_df], axis=1)
                  
                        aug_bbs_xy = pd.concat([aug_bbs_xy, aug_df])
                     

   
    aug_bbs_xy = aug_bbs_xy.reset_index()
    aug_bbs_xy = aug_bbs_xy.drop(['index'], axis=1)
    return aug_bbs_xy


resized_images_df = pd.read_csv("newTrainInt.csv", header=None, names=['filename', 'xmin', 'ymin', 'xmax', 'ymax', 'label']) 

augmented_images_df = image_aug(resized_images_df, 'resizeImg_/', 'augImg/', 'aug_', aug)
print(augmented_images_df)
augmented_images_df.to_csv(('augmentedTrain.csv'),index=False, header = False)


