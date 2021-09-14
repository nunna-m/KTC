import os
from numpy.lib.type_check import _nan_to_num_dispatcher
import tensorflow as tf
import numpy as np
import glob
from functools import partial, wraps
import cv2
from tensorflow.python.framework.ops import reset_default_graph


def parse_subject(subject_path, modalities,tumor_region_only, decoder=tf.image.decode_image, crop=tf.image.crop_to_bounding_box):
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
            return decode_func(img)
        return wrapper
    decoder = image_decoder(decoder)

    def image_crop(crop_func):
        def wrapper(img, crop_dict):
            img = tf.convert_to_tensor(img, dtype=tf.uint8)
            img = crop_func(img, crop_dict['y'], crop_dict['x'], crop_dict['height'], crop_dict['width'])
            return img
        return wrapper
    crop = image_crop(crop)

    if tumor_region_only:
        for modality, names in gathered_modalities_paths.items():
            subject_data[modality] = {
                os.path.splitext(name)[0]: crop(decoder(os.path.join(subject_path, modality, name))[:, :, 0] ,get_tumor_boundingbox(os.path.join(subject_path, modality+'L', name))) for name in names
            } 
    else:
        for modality, names in gathered_modalities_paths.items():
            subject_data[modality] = {
                os.path.splitext(name)[0]: decoder(os.path.join(subject_path, modality, name))[:, :, 0] for name in names
            }
    
    return subject_data

def get_class_ID_subjectpath(subject):
    splitup = subject.split(os.path.sep)
    ID = splitup[-1]
    clas = splitup[-2]
    assert clas in ('AML', 'CCRCC'), f'Classification category{clas} extracted from : {subject} unknown'
    return clas, ID



def get_tumor_boundingbox(path):
    image = cv2.imread(path)[:,:,0]
    backup = image.copy()
    lower_red = np.array([220])
    upper_red = np.array([255])
    mask = cv2.inRange(image, lower_red, upper_red)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2. CHAIN_APPROX_NONE)
    for i, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(backup, (x,y), (x+w,y+h),(0,0,255),2)
    #cv2.imwrite("result.png",backup)
    crop_info = {
        'y': y,
        'x': x,
        'height': h,
        'width': w,
    }
    return crop_info

subject_data = parse_subject(subject_path='/home/maanvi/LAB/Datasets/kidney_tumor_trainvaltest/am_tm/train/AML/87345564', modalities=['am','tm'], tumor_region_only=True)

print(subject_data)