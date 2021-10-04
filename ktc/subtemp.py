import os
from numpy.core.numeric import indices
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tensorflow as tf
import numpy as np
import glob
from functools import partial
import cv2
import sys
AUTOTUNE = tf.data.experimental.AUTOTUNE

def prep_combined_modalities(subject, output_size, modalities, tumor_region_only):
    #tf.print("entered prep_combined_modalities function")
    if isinstance(subject, str): pass
    elif isinstance(subject, tf.Tensor): subject = subject.numpy().decode()
    else: raise NotImplementedError
    subject_data = parse_subject(subject, output_size, modalities=modalities, tumor_region_only=tumor_region_only)
    slice_names = subject_data[modalities[0]].keys()
    appending_shape = list(output_size)+[len(modalities)+1]
    if subject_data['clas']=='AML':
        clas_tensor = tf.zeros(appending_shape,dtype=tf.uint8)
    elif subject_data['clas']=='CCRCC':
        clas_tensor = tf.ones(appending_shape,dtype=tf.uint8)
    else:
        raise NotImplementedError
    
    temp_slices = [tf.stack([subject_data[type_][slice_] for type_ in modalities], axis=-1) for slice_ in slice_names]
    print(clas_tensor)
    indices = tf.constant([[2]])

    slices = [
        tf.tensor_scatter_nd_add(
            clas_tensor,
            indices=indices,
            updates=temp_slices[i],
            ) for i in range(4)
    ]

    # slices = tf.stack([clas_tensor], axis=-1)
    # slices = tf.stack(
    #     [
    #         tf.stack(
    #             [
    #                 subject_data[type_][slice_] for type_ in modalities
    #             ], axis=-1) for slice_ in slice_names
    #     ]+clas_tensor)
    print("working slices shape: ",slices.shape)
    
    # for i in range(subject_data['num_slices_per_modality']):
    #     print("before: ",slices[i].shape)
    #     slices[i] = tf.stack([slices[i],clas_tensor], axis=-1)
    #     print("after: ",slices[i].shape)
            
    #print(slices)
    #slices = tf.stack([slices[i] for i in range(slices.shape[0]),clas_tensor],axis=-1)

    return dict(
        stacked_modality_slices=slices,
        clas=subject_data['clas'],
        ID=subject_data['ID'],
        subject_path=subject_data['subject_path'],
    )

def parse_subject(subject_path, output_size, modalities,tumor_region_only, decoder=tf.image.decode_image, resize=tf.image.resize):
    subject_data = {'subject_path': subject_path}
    subject_data['clas'], subject_data['ID'] = get_class_ID_subjectpath(subject_path)
    gathered_modalities_paths = {
            modality: set(os.listdir(os.path.join(subject_path,modality)))
            for modality in modalities
        }
    
    same_named_slices = set.intersection(*map(
        lambda slices: set(
            map(lambda name: os.path.splitext(name)[0], slices)),
        gathered_modalities_paths.values(),
        ))
    
    assert same_named_slices, f'Not enough slices with same name in {subject_path}'

    for modality in modalities:
        gathered_modalities_paths[modality] = list(
            filter(lambda x: os.path.splitext(x)[0],
            gathered_modalities_paths[modality])
        )
    subject_data['num_slices_per_modality']=len(same_named_slices)

    def image_decoder(decode_func):
        def wrapper(img):
            img = tf.io.read_file(img)
            img = decode_func(img)
            return img
        return wrapper
    decoder = image_decoder(decoder)

    def image_resizer(resize_func):
        def wrapper(img, output_size):
            img = img[... , tf.newaxis]
            img = resize_func(img, output_size, antialias=True, method='bilinear')
            img = tf.reshape(img, tf.shape(tf.squeeze(img)))
            img = tf.cast(img, dtype=tf.uint8)
            return img
        return wrapper
    resize = image_resizer(resize)


    if tumor_region_only:
        for modality, names in gathered_modalities_paths.items():
            subject_data[modality] = {
                os.path.splitext(name)[0]: crop(decoder(os.path.join(subject_path, modality, name))[:, :, 2] ,output_size, get_tumor_boundingbox(os.path.join(subject_path, modality, name),os.path.join(subject_path, modality+'L', name))) for name in names
            } 
    else:
        for modality, names in gathered_modalities_paths.items():
            subject_data[modality] = {
                os.path.splitext(name)[0]: resize(decoder(os.path.join(subject_path, modality, name))[:, :, 0], output_size)for name in names
            }
    
    #print("Subject datat: ",subject_data)
    return subject_data

def get_class_ID_subjectpath(subject):
    splitup = subject.split(os.path.sep)
    ID = splitup[-1]
    clas = splitup[-2]
    assert clas in ('AML', 'CCRCC'), f'Classification category{clas} extracted from : {subject} unknown'
    return clas, ID

def crop(img, resize_shape, crop_dict):
    
    tumor_center = {
            'y':crop_dict['y']+(crop_dict['height']//2),
            'x':crop_dict['x']+(crop_dict['width']//2),
    }
    
    img = img.numpy()
    (h, w) = resize_shape    
    y1 = tumor_center['y'] - h//2
    x1 = tumor_center['x'] - w//2
    y2 = tumor_center['y'] + h//2
    x2 = tumor_center['x'] + w//2
    for i in [y1,x1,y2,x2]:
        assert i>0, f'height or width going out of bounds'
    
    img = tf.convert_to_tensor(img, dtype=tf.uint8)
    #tf.cast(img, dtype=)
    return img

def get_tumor_boundingbox(imgpath, labelpath):
    (orig_height, orig_width) = cv2.imread(imgpath)[:,:,2].shape
    image = cv2.imread(labelpath)[:,:,2]
    image = cv2.resize(image, (orig_width, orig_height))
    backup = image.copy()
    lower_red = np.array([220])
    upper_red = np.array([255])
    mask = cv2.inRange(image, lower_red, upper_red)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2. CHAIN_APPROX_NONE)
    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    crop_info = {
        'y': y,
        'x': x,
        'height': h,
        'width': w,
        'labelpath':labelpath,
    }
    return crop_info

d_ret = prep_combined_modalities(
    subject='/home/maanvi/LAB/Datasets/kidney_tumor_trainvaltest/am_tm/train/AML/86049119', output_size = (224,224), modalities=['am','tm'], tumor_region_only=False
)

#print(d_ret['stacked_modality_slices'])
