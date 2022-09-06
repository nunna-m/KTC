'''
abstraction for tf.data.Dataset API
'''
from email.errors import NoBoundaryInMultipartDefect
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tensorflow as tf
import numpy as np
import glob
from functools import partial
import cv2
import sys
import matplotlib.pyplot as plt
import tensorflow_addons as tfa
import yaml

TL_num = 3

def train_ds(
    data_root,
    modalities,
    batch_size,
    buffer_size,
    repeat = True,
    output_size=(224,224),
    aug_configs=None,
    tumor_region_only=False,
):
    '''
    create train dataset

    Args:
        data_root: file with training and testing subject filenames for a specific CV fold, 'train' key filenames indicating train data is selected
        modalities: ct or mri or both modalities
        batch_size: batch size
        buffer_size: buffer_size
        repeat: flag indicating if dataset be repeated defaults True
        output_size: size of images in the dataset defaults (224,224)
        aug_configs (dict): augmentation configurations
        tumor_region_only: flag indicating cropped to rectangular box around tumor defaults False
    '''
    if aug_configs is None:
        aug_configs = {
            'random_crop':{},
        }
    default_aug_configs = {
        random_crop_img: dict(output_size=output_size),
        random_horizontalflip_img: {},
        random_verticalflip_img: {},
        random_contrast_img:  dict(channels=list(range(len(modalities)))),
        random_brightness_img: {},
        random_hue_img: {},
        random_saturation_img: {},
        random_rotation_img: {},
        random_shear_img: {},
        flip_leftright_img: {},
        rotate90_img: {},
        rotate180_img: {},
        rotate270_img: {},
        up_rotate90_img: {},
        up_rotate180_img: {},
        up_rotate270_img: {},
    }
    with open(data_root,'r') as file:
        data = yaml.safe_load(file)
    traindir = data['train']
    dataset = load_raw(
        traindir,
        modalities=modalities,
        output_size=output_size,
        tumor_region_only = tumor_region_only,
    )
    dataset = augmentation(
        dataset,
        methods=parse_aug_configs(aug_configs,
                                    default_aug_configs),
    )

    dataset = configure_dataset(
        dataset,
        batch_size,
        buffer_size,
        repeat=repeat
    )
    print("Final dataset:  ",dataset)
    return dataset

def train_ds_registered(
    data_root,
    modalities,
    batch_size,
    buffer_size,
    repeat = True,
    output_size=(224,224),
    aug_configs=None,
    tumor_region_only=False,
):
    '''
    create train dataset

    Args:
        data_root: file with training and testing subject filenames for a specific CV fold, 'train' key filenames indicating train data is selected
        modalities: ct or mri or both modalities
        batch_size: batch size
        buffer_size: buffer_size
        repeat: flag indicating if dataset be repeated defaults True
        output_size: size of images in the dataset defaults (224,224)
        aug_configs (dict): augmentation configurations
        tumor_region_only: flag indicating cropped to rectangular box around tumor defaults False
    '''
    if aug_configs is None:
        aug_configs = {
            'random_crop':{},
        }
    default_aug_configs = {
        random_crop_img: dict(output_size=output_size),
        random_horizontalflip_img: {},
        random_verticalflip_img: {},
        random_contrast_img:  dict(channels=list(range(len(modalities)))),
        random_brightness_img: {},
        random_hue_img: {},
        random_saturation_img: {},
        random_rotation_img: {},
        random_shear_img: {},
        flip_leftright_img: {},
        rotate90_img: {},
        rotate180_img: {},
        rotate270_img: {},
        up_rotate90_img: {},
        up_rotate180_img: {},
        up_rotate270_img: {},
    }
    with open(data_root,'r') as file:
        data = yaml.safe_load(file)
    traindir = data['train']
    dataset = load_raw_registered(
        traindir,
        modalities=modalities,
        output_size=output_size,
        tumor_region_only = tumor_region_only,
    )
    dataset = augmentation(
        dataset,
        methods=parse_aug_configs(aug_configs,
                                    default_aug_configs),
    )

    dataset = configure_dataset(
        dataset,
        batch_size,
        buffer_size,
        repeat=repeat
    )
    print("Final dataset:  ",dataset)
    return dataset

def eval_ds(
    data_root,
    modalities,
    batch_size,
    output_size=(224,224),
    tumor_region_only=False,
):
    '''
    create validation dataset (not in use when crossvalidation being used)

    Args:
        data_root: file with validation subject filenames
        modalities: ct or mri or both modalities
        batch_size: batch size
        output_size: size of images in the dataset defaults (224,224)
        tumor_region_only: flag indicating cropped to rectangular box around tumor defaults False
    '''
    evaldir = os.path.join(data_root,'_'.join(modalities),'val')
    ds = load_raw(
        evaldir,
        modalities=modalities,
        output_size=output_size,
        tumor_region_only = tumor_region_only
    )
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
    return ds

def eval_ds_registered(
    data_root,
    modalities,
    batch_size,
    output_size=(224,224),
    tumor_region_only=False,
):
    '''
    create validation dataset (not in use when crossvalidation being used)

    Args:
        data_root: file with validation subject filenames
        modalities: ct or mri or both modalities
        batch_size: batch size
        output_size: size of images in the dataset defaults (224,224)
        tumor_region_only: flag indicating cropped to rectangular box around tumor defaults False
    '''
    evaldir = os.path.join(data_root,'_'.join(modalities),'val')
    ds = load_raw_registered(
        evaldir,
        modalities=modalities,
        output_size=output_size,
        tumor_region_only = tumor_region_only
    )
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
    return ds

def predict_ds(data_root,
         modalities,
         batch_size,
         output_size=(224,224),
         tumor_region_only=False,
):
    '''
    create testing dataset

    Args:
        data_root: file with training and testing subject filenames for a specific CV fold, 'test' key filenames indicating test data is selected
        modalities: ct or mri or both modalities
        batch_size: batch size
        output_size: size of images in the dataset defaults (224,224)
        tumor_region_only: flag indicating cropped to rectangular box around tumor defaults False
    '''
    with open(data_root,'r') as file:
        data = yaml.safe_load(file)
    testdir = data['test']
    ds = load_raw(testdir,modalities=modalities, output_size=output_size, tumor_region_only=tumor_region_only)
    ds = ds.batch(batch_size)
    return ds

def predict_ds_registered(data_root,
         modalities,
         batch_size,
         output_size=(224,224),
         tumor_region_only=False,
):
    '''
    create testing dataset

    Args:
        data_root: file with training and testing subject filenames for a specific CV fold, 'test' key filenames indicating test data is selected
        modalities: ct or mri or both modalities
        batch_size: batch size
        output_size: size of images in the dataset defaults (224,224)
        tumor_region_only: flag indicating cropped to rectangular box around tumor defaults False
    '''
    with open(data_root,'r') as file:
        data = yaml.safe_load(file)
    testdir = data['test']
    ds = load_raw_registered(testdir,modalities=modalities, output_size=output_size, tumor_region_only=tumor_region_only)
    ds = ds.batch(batch_size)
    return ds

def load_raw(traindir, 
            modalities=('am','tm','dc','ec','pc'), 
            output_size=(224,224), 
            tumor_region_only=False, 
            dtype=tf.float32
):
    '''
    generate the basic raw dataset

    Args:
        traindir list[str]: list of training subject directories
        modalities: ct or mri or both modalities
        tumor_region_only: flag indicating cropped to rectangular box around tumor defaults False
        dtype: type to convert to defaults float32
    '''
    training_subject_paths = traindir
    ds = tf.data.Dataset.from_tensor_slices(training_subject_paths)
    label_ds = ds.interleave(
            partial(
                tf_combine_labels,
                modalities=modalities,
                return_type='dataset',
            ),
            cycle_length=count(ds),
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )
    #convert to one_hot because using categorical crossentropy loss function
    label_ds = label_ds.map(convert_one_hot)
    feature_ds = ds.interleave(
            partial(
                tf_combine_modalities,
                output_size=output_size,
                modalities=modalities,
                tumor_region_only=tumor_region_only,
                return_type='dataset',
            ),
            cycle_length=count(ds),
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )
    
    if output_size is not None: feature_ds = feature_ds.map(
            lambda image: tf.image.crop_to_bounding_box(
                image,
                ((tf.shape(image)[:2] - output_size) // 2)[0],
                ((tf.shape(image)[:2] - output_size) // 2)[1],
                *output_size,
            ),
            tf.data.experimental.AUTOTUNE,
        )
    ds = tf.data.Dataset.zip((feature_ds, label_ds))
    if len(modalities) <= 3:
        norm = 3
    else:
        norm = len(modalities)
    ds = ds.map(lambda image, label: tf_reshape_cast_normalize(image, label, num_mod=norm, dtype=dtype), tf.data.experimental.AUTOTUNE)
    return ds

def load_raw_registered(traindir, 
            modalities=('am','tm','dc','ec','pc'), 
            output_size=(224,224), 
            tumor_region_only=False, 
            dtype=tf.float32
):
    '''
    generate the basic raw dataset

    Args:
        traindir list[str]: list of training subject directories
        modalities: ct or mri or both modalities
        tumor_region_only: flag indicating cropped to rectangular box around tumor defaults False
        dtype: type to convert to defaults float32
    '''
    training_subject_paths = traindir
    ds = tf.data.Dataset.from_tensor_slices(training_subject_paths)
    label_ds = ds.interleave(
            partial(
                tf_combine_labels_registered,
                modalities=modalities,
                return_type='dataset',
            ),
            cycle_length=count(ds),
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )
    #convert to one_hot because using categorical crossentropy loss function
    label_ds = label_ds.map(convert_one_hot)
    feature_ds = ds.interleave(
            partial(
                tf_combine_modalities_registered,
                output_size=output_size,
                modalities=modalities,
                tumor_region_only=tumor_region_only,
                return_type='dataset',
            ),
            cycle_length=count(ds),
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )
    
    if output_size is not None: feature_ds = feature_ds.map(
            lambda image: tf.image.crop_to_bounding_box(
                image,
                ((tf.shape(image)[:2] - output_size) // 2)[0],
                ((tf.shape(image)[:2] - output_size) // 2)[1],
                *output_size,
            ),
            tf.data.experimental.AUTOTUNE,
        )
    ds = tf.data.Dataset.zip((feature_ds, label_ds))
    if len(modalities) <= 3:
        norm = 3
    else:
        norm = len(modalities)
    ds = ds.map(lambda image, label: tf_reshape_cast_normalize(image, label, num_mod=norm, dtype=dtype), tf.data.experimental.AUTOTUNE)
    return ds

def convert_one_hot(label):
    '''
    convert binary labels to one-hot encoding labels
    '''
    label = tf.one_hot(label,2, dtype=tf.int32)
    return label

def tf_reshape_cast_normalize(image, label, num_mod, dtype):
    '''
    Normalize the images in tf.data.dataset format
    Reshape to required size
    cast to correct data type
    mean standardize
    '''
    print("in tf_reshape: ",image.shape)
    image = tf.reshape(image, [*image.shape[:-1], num_mod])
    image = tf.cast(image, dtype=dtype)
    image = (image / 127.5)
    image = (image-1)
    return image, label

def tf_combine_labels(subject_path, modalities,return_type='array'):
    '''
    outer function to stack different labels of each modality for every subject
    '''
    return_type  =return_type.lower()
    modality = modalities[0]
    if return_type == 'array':
        return tf.py_function(
            lambda x: partial(get_label,
            modality=modality)(x),
            [subject_path],
            tf.int32,
        )
    elif return_type == 'dataset':
        return tf.data.Dataset.from_tensor_slices(
            tf_combine_labels(subject_path=subject_path,
            modalities=modalities, return_type='array')
        )
    else: raise NotImplementedError

def tf_combine_labels_registered(subject_path, modalities,return_type='array'):
    '''
    outer function to stack different labels of each modality for every subject
    '''
    return_type  =return_type.lower()
    modality = modalities[0]
    if return_type == 'array':
        return tf.py_function(
            lambda x: partial(get_label_registered,
            modality=modality)(x),
            [subject_path],
            tf.int32,
        )
    elif return_type == 'dataset':
        return tf.data.Dataset.from_tensor_slices(
            tf_combine_labels_registered(subject_path=subject_path,
            modalities=modalities, return_type='array')
        )
    else: raise NotImplementedError

def get_label(subject, modality):
    '''
    get label of image from file or directory name
    convert into number tensor and create an array of repeated label values for as many number of slices
    '''
    if isinstance(subject, str): 
        pass
    elif isinstance(subject, tf.Tensor): 
        subject = subject.numpy().decode()
    else: raise NotImplementedError
    clas, _ = get_class_ID_subjectpath(subject)
    required_path = os.path.join(subject, modality)
    num = len([name for name in os.listdir(required_path) if os.path.isfile(os.path.join(required_path,name))])
    num_slices = tf.constant([num], tf.int32)
    if clas=='AML':
        label = tf.constant([0], tf.int32)
    elif clas=='CCRCC':
        label = tf.constant([1], tf.int32)
    final = tf.tile(label, num_slices)
    return final

def get_label_registered(subject, modality):
    '''
    get label of image from file or directory name
    convert into number tensor and create an array of repeated label values for as many number of slices
    '''
    if isinstance(subject, str): 
        pass
    elif isinstance(subject, tf.Tensor): 
        subject = subject.numpy().decode()
    else: raise NotImplementedError
    clas, _ = get_class_ID_subjectpath(subject)
    num = len([name for name in os.listdir(subject) if os.path.isfile(os.path.join(subject,name))])
    num_slices = tf.constant([num], tf.int32)
    if clas=='AML':
        label = tf.constant([0], tf.int32)
    elif clas=='CCRCC':
        label = tf.constant([1], tf.int32)
    final = tf.tile(label, num_slices)
    return final

def tf_combine_modalities(subject_path, output_size, modalities, tumor_region_only,return_type='array'):
    '''
    outer function to stack different images of each modality for every subject
    '''
    return_type  =return_type.lower()
    if return_type == 'array':
        return tf.py_function(
            lambda x: partial(combine_modalities, output_size=output_size,
            modalities=modalities,
            tumor_region_only=tumor_region_only)(x)['slices'],
            [subject_path],
            tf.uint8,
        )
    elif return_type == 'dataset':
        return tf.data.Dataset.from_tensor_slices(
            tf_combine_modalities(subject_path=subject_path,output_size=output_size,
            modalities=modalities,
            tumor_region_only=tumor_region_only, return_type='array')
        )
    else: raise NotImplementedError

def tf_combine_modalities_registered(subject_path, output_size, modalities, tumor_region_only,return_type='array'):
    '''
    outer function to stack different images of each modality for every subject
    '''
    return_type  =return_type.lower()
    if return_type == 'array':
        return tf.py_function(
            lambda x: partial(combine_modalities_registered, output_size=output_size,
            modalities=modalities,
            tumor_region_only=tumor_region_only)(x)['slices'],
            [subject_path],
            tf.uint8,
        )
    elif return_type == 'dataset':
        return tf.data.Dataset.from_tensor_slices(
            tf_combine_modalities_registered(subject_path=subject_path,output_size=output_size,
            modalities=modalities,
            tumor_region_only=tumor_region_only, return_type='array')
        )
    else: raise NotImplementedError

def combine_modalities(subject, output_size, modalities, tumor_region_only):
    '''
    inner function to stack different labels of each modality for every subject
    if number of modalities is 1, create a copy of same image 3 times to make 3D image
    if number of modalities is 2, concatenate with empty array
    if number of modalities >= 3, use as is
    '''
    if isinstance(subject, str): pass
    elif isinstance(subject, tf.Tensor): subject = subject.numpy().decode()
    else: raise NotImplementedError
    subject_data = parse_subject(subject, output_size, modalities=modalities, tumor_region_only=tumor_region_only)
    slice_names = subject_data[modalities[0]].keys()
    
    slices = []
    for slice_ in slice_names:
        modals = []
        for type_ in modalities:
            img = subject_data[type_][slice_]
            modals.append(img)
        modals = tf.stack(modals, axis=-1)
        if len(modalities)<3:
            diff = 3-len(modalities)
            if len(modalities)==2:
                zeros = tf.zeros((img.shape[0],img.shape[1],diff),dtype=tf.uint8)
                modals = tf.concat([modals,zeros], axis=-1)
            elif len(modalities)==1:
                modals = tf.repeat(modals, repeats=[3],axis=-1)
        slices.append(modals)
    slices = tf.stack(slices, axis=0)

    return dict(
        slices=slices,
        subject_path=subject_data['subject_path'],
    )

def combine_modalities_registered(subject, output_size, modalities, tumor_region_only):
    '''
    generate registered image based on modalities (NOT for single modalities, modalities>=2)
    tumor_region_only set to False because we want box crop (as registered images are not labeled)
    NOTE: some registered images are transformed (translated, rotated, sheared etc) during the registration process hence rendering the exact labels of the tumors useless for cropping out pixel perfect regions
    finally take the registered image duplicate thrice (to emulate rgb) and send back with class label
    '''
    if isinstance(subject, str): pass
    elif isinstance(subject, tf.Tensor): subject = subject.numpy().decode()
    else: raise NotImplementedError
    subject_data = parse_subject_registered(subject, output_size, modalities=modalities, tumor_region_only=tumor_region_only)
    slice_names = subject_data['registered_images'].keys()
    assert len(modalities) > 1
    images = []
    for name in slice_names:
        diff = 3
        img = tf.expand_dims(subject_data['registered_images'][name], axis=-1)
        final_image = tf.repeat(img, repeats=[diff],axis=-1)
        images.append(final_image)
    slices = tf.stack(images, axis=0)
    #print(f"Slice.shape: {slices.shape}")
    return dict(
        slices=slices,
        subject_path=subject_data['subject_path'],
    )

def parse_subject(subject_path, 
                    output_size, 
                    modalities,
                    tumor_region_only, 
                    decoder=tf.image.decode_image, 
                    resize=tf.image.resize):
    '''
    take directory of subject and return decoded images stack them according to image sample number for each modality
    
    Args:
        subject_path: subject directory with the images of various modalites
        output_size: final size to which images are resized before returning
        modalities: ct or mri or both
        tumor_region_only (bool): true indicates cropping to a rectangular box around the tumor region
        decoder: tensorflow decode_image function that is used to read the image into a tensor, used when tumor_region_only is False, when it is True, opencv imgread is used
        resize: tensorflow resize function to perform resize operation
    '''
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
    
    for temp in modalities:
        gathered_modalities_paths[temp] = {k+'.png' for k in same_named_slices}

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
                os.path.splitext(name)[0]: get_exact_tumor(os.path.join(subject_path, modality, name),os.path.join(subject_path, modality+'L', name)) for name in names
            } 
    else:
        for modality, names in gathered_modalities_paths.items():
            subject_data[modality] = {
                os.path.splitext(name)[0]: get_tumor_boundingbox(os.path.join(subject_path, modality, name),os.path.join(subject_path, modality+'L', name)) for name in names
            }
        # for modality, names in gathered_modalities_paths.items():
        #     subject_data[modality] = {
        #         os.path.splitext(name)[0]: resize(decoder(os.path.join(subject_path, modality, name))[:, :, 0], output_size)for name in names
        #     }
    return subject_data

def parse_subject_registered(subject_path, 
                    output_size, 
                    modalities,
                    tumor_region_only,):
    '''
    take directory of subject and return decoded images
    
    Args:
        subject_path: subject directory with the images of various modalites
        output_size: final size to which images are resized before returning
        modalities: ct or mri or both
        tumor_region_only (bool): true indicates cropping to a pixel perfect region false: rectangular box around the tumor region
    '''
    registered_subject_path = subject_path
    registered_subject_path_label = subject_path.replace('kt_registered','kt_registered_labels')
    #print(registered_subject_path_label)
    subject_data = {'subject_path': subject_path, 'registered_subject_path': registered_subject_path}
    
    subject_data['clas'], subject_data['ID'] = get_class_ID_subjectpath(subject_path)
    sliceNames = os.listdir(registered_subject_path)
    subject_data['num_slices_per_modality']=len(sliceNames)

    subject_data['registered_images'] = dict()
    if tumor_region_only:
        for name in sliceNames:
            subject_data['registered_images'][name] = get_exact_tumor_registered(os.path.join(registered_subject_path,name),os.path.join(registered_subject_path_label,name))
    else:
        for name in sliceNames:
            subject_data['registered_images'][name] = get_tumor_boundingbox_registered(os.path.join(registered_subject_path,name),os.path.join(registered_subject_path_label,name))

    #print(f"Subject data shape: {subject_data['registered_images']['1.png'].shape}")
    return subject_data

def get_class_ID_subjectpath(subject):
    '''
    based on subject folder path extract tumor class and subject ID
    '''
    splitup = subject.split(os.path.sep)
    ID = splitup[-1]
    clas = splitup[-2]
    assert clas in ('AML', 'CCRCC'), f'Classification category{clas} extracted from : {subject} unknown'
    return clas, ID

def get_exact_tumor_registered(imgpath, labelpath):
    '''
    get the exact segmented tumor region (pixel perfect) based on label already provided
    '''
    #print(imgpath, labelpath)
    # orig_image = cv2.imread(imgpath)[:,:,0]
    # (orig_height, orig_width) = orig_image.shape
    # #cv2.imwrite(f'/home/maanvi/registered_{imgpath[-5]}.png',orig_image)
    # mask = cv2.imread(labelpath)[:,:,0]
    # # print(labelpath)
    # # print(f'Mask.shape: {mask.shape}')
    # # print(f'Image shape: {(orig_height,orig_width)}')
    # #gaussian standardizes only modality am
    # mean, std = orig_image.mean(), orig_image.std()
    # orig_image = (orig_image - mean)/std
    # mean, std = orig_image.mean(), orig_image.std()
    # orig_image = np.clip(orig_image, -1.0, 1.0)
    # orig_image = (orig_image + 1.0) / 2.0
    # orig_image *= 255
    # #cv2.imwrite('/home/maanvi/Desktop/mask.png',mask)
    # ret, thresh1 = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)
    # #print(thresh1.shape)
    # orig_image[thresh1==0] = 0
    # out = np.zeros_like(orig_image)
    # out[mask == 255] = orig_image[mask == 255]
    # #crop out
    # #print(np.where(mask==255))
    # (y, x) = np.where(mask == 255)
    # (topy, topx) = (np.min(y), np.min(x))
    # (bottomy, bottomx) = (np.max(y), np.max(x))
    # out = out[topy:bottomy+1, topx:bottomx+1]
    # out = cv2.resize(out, (224,224), interpolation=cv2.INTER_CUBIC)
    #cv2.imwrite(f'/home/maanvi/registered_resize_exact{imgpath[-5]}.png',out)
    new_path = imgpath.replace('kt_registered','kt_registered_exact')
    out = cv2.imread(new_path)[:,:,0]
    out = tf.convert_to_tensor(out, dtype=tf.uint8)
    return out

def get_tumor_boundingbox_registered(imgpath, labelpath):
    '''
    get the bounding box coordinates around tumor
    first calculate center of tumor based on segmentation label
    then calculate bounding box around it after zooming out by a factor of 0.3 on both heigth and width (just to be sure of including the entire region of the tumor)
    am modality is gaussian standardized also
    '''
    # orig_image = cv2.imread(imgpath)[:,:,0]
    # #cv2.imwrite(f'/home/maanvi/registered_{imgpath[-5]}.png',orig_image)
    # (orig_height, orig_width) = cv2.imread(imgpath)[:,:,0].shape
    # mask = cv2.imread(labelpath)[:,:,0]
    # contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2. CHAIN_APPROX_NONE)
    # c = max(contours, key=cv2.contourArea)
    # x, y, w, h = cv2.boundingRect(c)
    # assert (w,h)<(224,224)
    # assert (x,y)>=(0,0)
    # const = 0.3
    # diff_x = int(const*w)
    # diff_y = int(const*h)
    # if (x-diff_x)<0:
    #     x1 = 0
    # else:
    #     x1 = x-diff_x
    # if (y-diff_y)<0:
    #     y1 = 0
    # else:
    #     y1 = y-diff_y
    # if (x+w+diff_x)>=orig_width:
    #     x2 = orig_width
    # else:
    #     x2 = x+diff_x+w
    # if (y+diff_y+h)>=orig_height:
    #     y2 = orig_height
    # else:
    #     y2 = y+diff_y+h


    # #gaussian standardizes only modality am
    # tmp = imgpath.rsplit(os.path.sep,2)[1]
    # if tmp=='am':
    #     mean, std = orig_image.mean(), orig_image.std()
    #     orig_image = (orig_image - mean)/std
    #     mean, std = orig_image.mean(), orig_image.std()
    #     orig_image = np.clip(orig_image, -1.0, 1.0)
    #     orig_image = (orig_image + 1.0) / 2.0
    #     orig_image *= 255
    # backup = orig_image[y1:y2,x1:x2]
    # backup = cv2.resize(backup, (224,224),interpolation = cv2.INTER_LINEAR)
    # #cv2.imwrite(f'/home/maanvi/registered_resize{imgpath[-5]}.png',backup)
    new_path = imgpath.replace('kt_registered','kt_registered_box')
    backup = cv2.imread(new_path)[:,:,0]
    backup = tf.convert_to_tensor(backup, dtype=tf.uint8)
    #print(backup.shape)
    return backup
def get_exact_tumor(imgpath, labelpath):
    '''
    get the exact segmented tumor region (pixel perfect) based on label already provided
    '''
    orig_image = cv2.imread(imgpath)[:,:,0]
    (orig_height, orig_width) = cv2.imread(imgpath)[:,:,0].shape
    image = cv2.imread(labelpath)
    image = cv2.resize(image, (orig_width, orig_height))
    backup = image.copy()
    #gaussian standardizes only modality am
    tmp = imgpath.rsplit(os.path.sep,2)[1]
    if tmp=='am':
        mean, std = orig_image.mean(), orig_image.std()
        orig_image = (orig_image - mean)/std
        mean, std = orig_image.mean(), orig_image.std()
        orig_image = np.clip(orig_image, -1.0, 1.0)
        orig_image = (orig_image + 1.0) / 2.0
        orig_image *= 255
    lower_red = np.array([0,0,50])
    upper_red = np.array([0,0,255])
    mask = cv2.inRange(image, lower_red, upper_red)
    #cv2.imwrite('/home/maanvi/Desktop/mask.png',mask)
    ret, thresh1 = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)
    orig_image[thresh1==0] = 0
    out = np.zeros_like(orig_image)
    out[mask == 255] = orig_image[mask == 255]
    #crop out
    (y, x) = np.where(mask == 255)
    (topy, topx) = (np.min(y), np.min(x))
    (bottomy, bottomx) = (np.max(y), np.max(x))
    out = out[topy:bottomy+1, topx:bottomx+1]
    out = cv2.resize(out, (224,224), interpolation=cv2.INTER_CUBIC)
    #cv2.imwrite(f'/home/maanvi/registered_resize_exact{imgpath[-5]}.png',out)
    out = tf.convert_to_tensor(out, dtype=tf.uint8)
    return out
def get_tumor_boundingbox(imgpath, labelpath):
    '''
    get the bounding box coordinates around tumor
    first calculate center of tumor based on segmentation label
    then calculate bounding box around it after zooming out by a factor of 0.3 on both heigth and width (just to be sure of including the entire region of the tumor)
    am modality is gaussian standardized also
    '''
    orig_image = cv2.imread(imgpath)[:,:,0]
    (orig_height, orig_width) = cv2.imread(imgpath)[:,:,0].shape
    image = cv2.imread(labelpath)
    image = cv2.resize(image, (orig_width, orig_height))
    backup = image.copy()
    lower_red = np.array([0,0,50])
    upper_red = np.array([0,0,255])
    mask = cv2.inRange(image, lower_red, upper_red)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2. CHAIN_APPROX_NONE)
    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    assert (w,h)<(224,224)
    assert (x,y)>=(0,0)
    const = 0.3
    diff_x = int(const*w)
    diff_y = int(const*h)
    if (x-diff_x)<0:
        x1 = 0
    else:
        x1 = x-diff_x
    if (y-diff_y)<0:
        y1 = 0
    else:
        y1 = y-diff_y
    if (x+w+diff_x)>=orig_width:
        x2 = orig_width
    else:
        x2 = x+diff_x+w
    if (y+diff_y+h)>=orig_height:
        y2 = orig_height
    else:
        y2 = y+diff_y+h


    #gaussian standardizes only modality am
    tmp = imgpath.rsplit(os.path.sep,2)[1]
    if tmp=='am':
        mean, std = orig_image.mean(), orig_image.std()
        orig_image = (orig_image - mean)/std
        mean, std = orig_image.mean(), orig_image.std()
        orig_image = np.clip(orig_image, -1.0, 1.0)
        orig_image = (orig_image + 1.0) / 2.0
        orig_image *= 255
    backup = orig_image[y1:y2,x1:x2]
    backup = cv2.resize(backup, (224,224),interpolation = cv2.INTER_LINEAR)
    cv2.imwrite('/home/maanvi/registered_resize.png',backup)
    backup = tf.convert_to_tensor(backup, dtype=tf.uint8)
    return backup

def parse_aug_configs(configs, default_configs=None):
    '''
    update dictionary of configs with values provided in the augmnentation config file
    '''
    if default_configs is None:
        default_configs = {}
    updated_conf = {}
    for name, conf in configs.items():
        if conf is None: conf = {}
        func = globals()[f'{name}_img']
        if func in default_configs:
            new_conf = default_configs[func].copy()
            new_conf.update(conf)
            conf = new_conf
        updated_conf[func] = conf
    return updated_conf    
    
def augmentation(dataset, methods=None):
    '''
    call on the functions specified in the augmentation config file
    '''
    if methods is None:
        methods = {
        random_crop_img: {},
        random_horizontalflip_img: {},
        random_verticalflip_img: {},
        random_contrast_img:  {},
        random_brightness_img: {},
        random_hue_img: {},
        random_saturation_img: {},
        random_rotation_img: {},
        random_shear_img: {},
        flip_leftright_img: {},
        rotate90_img: {},
        rotate180_img: {},
        rotate270_img: {},
        up_rotate90_img: {},
        up_rotate180_img: {},
        up_rotate270_img: {},
        }
    else:
        assert isinstance(methods, dict)
        methods = dict(map(
            lambda name, config: (augmentation_method(name),config),
            methods.keys(), methods.values()
        ))

    for operation, config in methods.items():
        print('Applying augmentation: ', operation, config)
        dataset = operation(dataset, **config)
    
    return dataset

def augmentation_method(method_in_str):
    '''
    call on a specific augmentation method
    '''
    if callable(method_in_str): return method_in_str
    method_in_str.endswith('_img')
    method = vars[method_in_str]
    return method

def flip_leftright_img(dataset):
    '''
    flip image left right
    '''
    dataset = dataset.map(
        lambda image, label: flip_leftright(image, label),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )
    return dataset

def flip_leftright(image, label):
    image = tf.image.flip_left_right(image)
    return image, label

def rotate90_img(dataset):
    '''
    rotate image 90 degrees
    '''
    dataset = dataset.map(
        lambda img, label: rotate90(img, label),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )
    return dataset

def rotate90(img, label):
    img = tf.image.rot90(img, k=1)
    return img, label

def rotate180_img(dataset):
    '''
    rotate image 180 degrees, similar to flip left right
    '''
    dataset = dataset.map(
        lambda img, label: rotate180(img, label),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )
    return dataset

def rotate180(img, label):
    img = tf.image.rot90(img, k=2)
    return img, label

def rotate270_img(dataset):
    '''
    rotate image 270 degrees
    '''
    dataset = dataset.map(
        lambda img, label: rotate270(img, label),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )
    return dataset

def rotate270(img, label):
    img = tf.image.rot90(img, k=3)
    return img, label

def up_rotate90_img(dataset):
    '''
    mirror the image and rotate 90 degrees
    '''
    dataset = dataset.map(
        lambda img, label: up_rotate90(img, label),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )
    return dataset

def up_rotate90(img, label):
    img = tf.image.flip_up_down(img)
    img = tf.image.rot90(img, k=1)
    return img, label

def up_rotate180_img(dataset):
    '''
    mirror the image and rotate 180 degrees
    '''
    dataset = dataset.map(
        lambda img, label: up_rotate180(img, label),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )
    return dataset

def up_rotate180(img, label):
    img = tf.image.flip_up_down(img)
    img = tf.image.rot90(img, k=2)
    return img, label

def up_rotate270_img(dataset):
    dataset = dataset.map(
        lambda img, label: up_rotate270(img, label),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )
    return dataset

def up_rotate270(img, label):
    '''
    mirror the image and rotate 270 degrees
    '''
    img = tf.image.flip_up_down(img)
    img = tf.image.rot90(img, k=3)
    return img, label

def random_crop_img(dataset, **configs):
    '''
    randomly crop the image from the center
    '''
    dataset = dataset.map(
        lambda image,label: random_crop(image, label,**configs),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )
    return dataset

def random_crop(img, label, output_size=(224,224), stddev=4, max_=6, min_=-6):
    threshold = tf.clip_by_value(tf.cast(tf.random.normal([2],stddev=stddev), tf.int32), min_, max_)
    diff = (tf.shape(img)[:2] - output_size) // 2 + threshold
    img = tf.image.crop_to_bounding_box(
        img,
        diff[0],
        diff[1],
        *output_size,
    )
    return img, label

def random_horizontalflip_img(dataset):
    '''
    horizontally flip image randomly (flip or not every image)
    '''
    dataset = dataset.map(
        lambda image, label: random_horizontalflip(image, label),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )
    return dataset

def random_horizontalflip(image, label):
    image = tf.image.random_flip_left_right(image)
    return image, label

def random_verticalflip_img(dataset):
    '''
    vertically flip image randomly (flip or not every image)
    '''
    dataset = dataset.map(
        lambda image, label:
        random_verticalflip(image, label),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )
    return dataset

def random_verticalflip(image, label):
    image = tf.image.random_flip_up_down(image)
    return image, label

def random_contrast_img(dataset, channels, lower=0.8, upper=1.2):
    '''
    change image contrast randomly
    '''
    dataset = dataset.map(
        lambda image, label: random_contrast(image, label,lower=lower, upper=upper, channels=channels),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )
    return dataset

def random_contrast(img, label, lower, upper, channels):
    skip_channels = [i for i in range(img.shape[-1]) if i not in channels]

    picked_channels_img = tf.gather(img, channels, axis=2)
    skipped_channels_img = tf.gather(img, skip_channels, axis=2)
    final_img = tf.image.random_contrast(picked_channels_img, lower=lower, upper=upper)
    img = tf.concat([final_img, skipped_channels_img], axis=2)
    indices = list(map(
        lambda CW: CW[1],
        sorted(zip(channels+skip_channels, range(1000)),
        key=lambda CW:CW[0],
                ),
    ))
    print("INDICESS----:",indices)
    img = tf.gather(img, indices, axis=2)
    return img, label

def random_brightness_img(dataset, max_delta=0.2):
    '''
    change image brightness randomly(decide to change the brightness or keep image as is -- meaning of random)
    '''
    dataset = dataset.map(
        lambda img,label: random_brightness(img,label,
        max_delta=max_delta),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )
    return dataset

def random_brightness(image, label, max_delta=0.2):
    image = tf.image.random_brightness(image, max_delta=max_delta)
    return image, label

def random_saturation_img(dataset, lower=5, upper=10):
    '''
    change image saturation randomly
    '''
    dataset = dataset.map(
        lambda img, label: random_saturation(img, label,
        lower=lower, upper=upper),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )
    return dataset

def random_saturation(image, label, lower, upper):
    if image.shape[-1]<3:
        return image, label
    image = tf.image.random_saturation(image,
        lower=lower, upper=upper)
    return image, label

def random_hue_img(dataset, max_delta=0.2):
    '''
    change image hue randomly
    '''
    dataset = dataset.map(
        lambda img, label: random_hue(img, label,max_delta=max_delta),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )
    return dataset

def random_hue(image, label, max_delta):
    if image.shape[-1]<3:
        return image, label
    image = tf.image.random_hue(image, max_delta=max_delta)
    return image, label

def random_rotation_img(dataset):
    '''
    rotate image at a random angle within a default range
    '''
    dataset = dataset.map(
        lambda img, label: random_rotation(img, label),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )
    return dataset

def random_rotation(img, label, angle_range=(-5,5),interpolation='bilinear',fill_mode='nearest'):
    angle = tf.random.uniform(shape=[1], minval=angle_range[0], maxval=angle_range[1])
    img = tfa.image.rotate(img, angles=angle,
    interpolation=interpolation,
    fill_mode=fill_mode)
    return img, label

def random_shear_img(dataset, x=(-10,10), y=(-10,10)):
    '''
    shear image at a random x units and random y units within a default range
    '''
    x_axis = tf.random.uniform(shape=(), minval=y[0], maxval=y[1])
    y_axis = tf.random.uniform(shape=(), minval=x[0], maxval=x[1])
    dataset = dataset.map(
        lambda img, label: random_shear_x_y(img, label, x_axis=x_axis, y_axis=y_axis)
    )
    return dataset

def random_shear_x_y(image, label, x_axis, y_axis):
    image = tfa.image.shear_x(image, y_axis, [1])
    image = tfa.image.shear_y(image, x_axis, [1])
    return image, label

def configure_dataset(dataset, batch_size, buffer_size, repeat=False):
    '''
    shuffle
    repeat
    batch
    prefectch
    call the above functions on every tf.data.dataset generated
    '''
    dataset = dataset.shuffle(buffer_size)
    if repeat:
        print("entering repeat")
        dataset = dataset.repeat(None)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset

def count(ds):
    '''
    count the number of records in the dataset
    '''
    size = 0
    for _ in ds: 
        size += 1
    return size


# print(parse_subject_registered('/home/maanvi/LAB/Datasets/kt_new_trainvaltest/am_tm/train/CCRCC/16659607', 
#                     (224,224), 
#                     ('ec','tm'),
#                     tumor_region_only=True,))
