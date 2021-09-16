'''
abstraction for tf.data.Dataset API
'''
import os
from numpy.lib.type_check import _nan_to_num_dispatcher
import tensorflow as tf
import numpy as np
import glob
from functools import partial, wraps
import cv2

from tensorflow.python.framework.ops import reset_default_graph
from ktc.utils import data as dataops

def train_dataset(
    data_root,
    batch_size,
    buffer_size,
    repeat = True,
    modalities=('am','tm','dc','ec','pc'),
    output_size=(224,224),
    aug_configs=None,
    tumor_region_only=False,
):
    traindir = os.path.join(data_root,'_'.join(modalities),'train')
    dataset = load_raw(
        traindir,
        modalities=modalities,
        output_size=output_size,
        tumor_region_only = tumor_region_only
    )

    dataset = custom_augmentation(
        dataset,
        aug_configs,
    )

    dataset = image_label(dataset, modalities=modalities)
    dataset = configure_dataset(
        dataset,
        batch_size,
        buffer_size,
        repeat=repeat
    )

    return dataset

def load_raw(traindir, modalities=('am','tm','dc','ec','pc'), output_size=(224,224), tumor_region_only=False, dtype=tf.float32):
    
    training_subject_paths = glob.glob(os.path.join(traindir,*'*'*2))
    dataset = tf.data.Dataset.from_tensor_slices(training_subject_paths)
    dataset = dataset.interleave(tf.data.Dataset.list_files)
    dataset = dataset.interleave(
        partial(
            combine_modalities,
            output_size=output_size,
            modalities=modalities,
            tumor_region_only=tumor_region_only,
        ),
        cycle_length=dataops.count(dataset),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )

    if output_size is not None and tumor_region_only==False: 
        dataset = dataset.map(
            lambda image: tf.image.crop_to_bounding_box(
                image,
                ((tf.shape(image)[:2] - output_size) // 2)[0],
                ((tf.shape(image)[:2] - output_size) // 2)[1],
                *output_size,
            ),
            tf.data.experimental.AUTOTUNE,
        )
    dataset = dataset.map(lambda x: tf.reshape(x, [*x.shape[:-1], len(modalities)]), tf.data.experimental.AUTOTUNE)
    dataset = dataset.map(lambda x: tf.cast(x, dtype=dtype), tf.data.experimental.AUTOTUNE)
    dataset = dataset.map(lambda x: x / 255.0, tf.data.experimental.AUTOTUNE)


    return dataset

def filter_modalities(paths,modalities):
    assert isinstance(paths,list)
    labels = list(map(lambda x: x+'L',modalities))
    modalities = list(modalities) + labels
    paths = list(
        filter(
            lambda x: os.path.split(x)[0].rsplit(os.path.sep,1)[1] in modalities, 
        paths))
    return paths

def combine_modalities(subject, output_size, modalities=('am','tm','dc','ec','pc'), tumor_region_only=False):
    return tf.py_function(
        lambda x: partial(prep_combined_modalities,
        output_size,
        modalities=modalities, tumor_region_only=tumor_region_only)(x)['modalities'],
        [subject],
        tf.uint8,
    )

def prep_combined_modalities(subject, output_size, modalities, tumor_region_only):
    if isinstance(subject, str): pass
    else: raise NotImplementedError
    subject_data = parse_subject(subject, output_size, modalities=modalities, tumor_region_only=tumor_region_only)
    slice_names = subject_data[modalities[0]].keys()

    slices = tf.stack([tf.stack([subject_data[type_][slice_] for type_ in modalities], axis=-1) for slice_ in slice_names])
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


def custom_augmentation(dataset, aug_configs):
    return dataset 
    

def image_label(dataset, modalities=('am','tm','dc','ec','pc')):
    return dataset

def configure_dataset(dataset, batch_size, buffer_size, repeat=False):
    return dataset 

