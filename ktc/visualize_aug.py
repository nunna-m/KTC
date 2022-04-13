import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sys
import yaml
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import numpy as np
import dataset

def plot_image_grid(images, ncols=None, cmap='gray'):
    '''Plot a grid of images'''
    if not ncols:
        factors = [i for i in range(1, len(images)+1) if len(images) % i == 0]
        ncols = factors[len(factors) // 2] if len(factors) else len(images) // 4 + 1
    nrows = int(len(images) / ncols) + int(len(images) % ncols)
    imgs = [images[i] if len(images) > i else None for i in range(nrows * ncols)]
    f, axes = plt.subplots(nrows, ncols, figsize=(3*ncols, 2*nrows))
    axes = axes.flatten()[:len(imgs)]
    for img, ax in zip(imgs, axes.flatten()): 
        if np.any(img):
            if len(img.shape) > 2 and img.shape[2] == 1:
                img = img.squeeze()
            ax.axis('off')
            ax.imshow(img, cmap=cmap)
    plt.show()
base_path = '/home/maanvi/LAB/Datasets'
phase = 'dc'
path = os.path.join(base_path,'kt_new_trainvaltest',phase,'5CV/allSubjectPaths0.yaml')
with open(path, 'r') as file:
    data = yaml.safe_load(file)
    subject_path = data['train'][0]

print(subject_path)

img_file = os.path.join(subject_path,phase,'1.png')
img = cv2.imread(img_file)
#plt.imshow(img)
#plt.show()
label = 0
output_size = (224,224)
aug_functions = [
    dataset.flip_leftright,
    dataset.rotate90,
    dataset.rotate180,
    dataset.rotate270,
    dataset.up_rotate90,
    dataset.up_rotate180,
    dataset.up_rotate270,
    dataset.random_horizontalflip,
    dataset.random_verticalflip,
    dataset.random_rotation,
    dataset.random_brightness,
]

images = []
for aug in aug_functions:
    image = dataset.parse_subject(subject_path, output_size, [phase], tumor_region_only=True)[phase]['1']
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image = tf.expand_dims(image, axis=-1)
    image = tf.repeat(image, repeats=[3],axis=-1)
    images.append(aug(image,label)[0])
    # plt.imshow(image,cmap="gray")
    # plt.show()

plot_image_grid(images)