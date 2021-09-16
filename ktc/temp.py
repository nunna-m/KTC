import os
from numpy.lib.type_check import _nan_to_num_dispatcher
import tensorflow as tf
import numpy as np
import glob
from functools import partial, wraps
import cv2
from tensorflow.python.framework.ops import reset_default_graph
from tensorflow.python.ops.gen_array_ops import shape


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

subject_data = parse_subject(subject_path='/home/maanvi/LAB/Datasets/kidney_tumor_trainvaltest/am_tm/train/AML/87345564', output_size = (224,224), modalities=['am','tm'], tumor_region_only=False)

print(subject_data)


# backup = img.numpy()
# print("backup shape: ",backup.shape)
# cv2.rectangle(backup, (crop_dict['x'],crop_dict['y']), (crop_dict['x']+crop_dict['width'],crop_dict['y']+crop_dict['height']),(0,0,255),2)
# # cv2.imwrite('result.png', backup)
# img = tf.convert_to_tensor(img, dtype=tf.uint8)
# img = tf.reshape(img,  shape=(1,img.shape[0], img.shape[1],1))
# NUM_BOXES = 1
# boxes = tf.constant([crop_dict['y'], crop_dict['x'], crop_dict['y']+crop_dict['height'], crop_dict['x']+crop_dict['width']],shape=(NUM_BOXES,4))
# boxes = tf.cast(boxes, dtype=tf.float32)
# box_indices = tf.random.uniform(shape=(NUM_BOXES,), minval=0, maxval=1, dtype=tf.int32)
# img = crop_func(img, boxes, box_indices, resize_shape, method='nearest')
# print("image shape before reshape: ",img.shape)
# img = tf.reshape(img, shape=(img.shape[1], img.shape[2]))
# cv2.imwrite('result.png', img[0:,:,:,0].numpy())



#zoom_out = False
#h, w = img.shape[:2]
# if zoom_out:
#     height_add = (h - crop_dict['height'])//10
#     width_add = (w - crop_dict['width'])//10
#     y1 = crop_dict['y'] - height_add
#     x1 = crop_dict['x'] - width_add
#     y2 = crop_dict['y'] + crop_dict['height'] + height_add
#     x2 = crop_dict['x'] + crop_dict['width'] + width_add
# else:
#     y1 = crop_dict['y']
#     x1 = crop_dict['x']
#     y2 = crop_dict['y'] + crop_dict['height']
#     x2 = crop_dict['x'] + crop_dict['width']


#cv2.imwrite(os.path.splitext(crop_dict['labelpath'])[0]+'cropped_resized_zoomout.png', img)
#cv2.imwrite(os.path.splitext(crop_dict['labelpath'])[0]+'cropped_resized.png', img)