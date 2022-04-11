'''
abstraction for tf.data.Dataset API
'''
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
    if aug_configs is None:
        aug_configs = {
            'random_crop':{},
        }
    default_aug_configs = {
        flip_leftright_img: {},
        rotate90_img: {},
        rotate180_img: {},
        rotate270_img: {},
        up_rotate90_img: {},
        up_rotate180_img: {},
        up_rotate270_img: {},
    }
    print("calling the new dataset file")
    with open(data_root,'r') as file:
        data = yaml.safe_load(file)
    traindir = data['train']
    # if cv:
    #     with open(data_root,'r') as file:
    #         data = yaml.safe_load(file)
    #     traindir = data['train']
    # else:
    #     traindir = os.path.join(data_root,'_'.join(modalities),'train')
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

def eval_ds(
    data_root,
    modalities,
    batch_size,
    include_meta=False,
    output_size=(224,224),
    tumor_region_only=False,
):
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

def predict_ds(data_root,
         modalities,
         batch_size,
         output_size=(224,224),
         tumor_region_only=False,):
    with open(data_root,'r') as file:
        data = yaml.safe_load(file)
    testdir = data['test']
    print("Len of test: ",len(testdir))
    # if cv:
    #     #print("entered cv in predict ds")
    #     with open(data_root,'r') as file:
    #         data = yaml.safe_load(file)
    #     testdir = data['test']
    #     #print(testdir, len(testdir))
    # else:
    #     testdir = os.path.join(data_root,'_'.join(modalities),'test')
    ds = load_raw(testdir,modalities=modalities, output_size=output_size, tumor_region_only=tumor_region_only)
    ds = ds.batch(batch_size)
    return ds

def load_raw(traindir, modalities=('am','tm','dc','ec','pc'), output_size=(224,224), tumor_region_only=False, dtype=tf.float32):
    training_subject_paths = traindir
    multiclass = True
    # if cv:
    #     training_subject_paths = traindir
    #     multiclass = True
    # else:
    #     training_subject_paths = glob.glob(os.path.join(traindir,*'*'*2))
    #     multiclass = False
    
    #print("data len: ",len(training_subject_paths))
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
    # counting = 0
    # for ele in label_ds.as_numpy_iterator():
    #     #print(ele)
    #     counting+=1
    # print("test data label count inside load raw: ",counting)
    if multiclass:
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
    norm = 3
    ds = ds.map(lambda image, label: tf_reshape_cast_normalize(image, label, num_mod=norm, dtype=dtype), tf.data.experimental.AUTOTUNE)
    return ds

def convert_one_hot(label):
    label = tf.one_hot(label,2, dtype=tf.int32)
    return label

def tf_reshape_cast_normalize(image, label, num_mod, dtype):
    print("in tf_reshape: ",image.shape)
    image = tf.reshape(image, [*image.shape[:-1], num_mod])
    image = tf.cast(image, dtype=dtype)
    image = (image / 127.5)
    image = (image-1)
    #label.set_shape([1])
    return image, label

def tf_crop_bounding_box(image, label, output_size):
    image = tf.image.crop_to_bounding_box(
        image,
        ((tf.shape(image)[:2] - output_size) // 2)[0],
        ((tf.shape(image)[:2] - output_size) // 2)[1],
        *output_size,
    )
    #print(image.shape, label.shape)
    return image, label

def tf_combine_labels(subject_path, modalities,return_type='array'):
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

def get_label(subject, modality):
    #print("inside get label")
    #print(subject, modality)
    if isinstance(subject, str): 
        pass
    elif isinstance(subject, tf.Tensor): 
        subject = subject.numpy().decode()
        #modality = modality.numpy().decode()
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
    #print("labels in get_label: ",final.numpy())
    return final
    #yield final

def tf_combine_modalities(subject_path, output_size, modalities, tumor_region_only,return_type='array'):
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

def combine_modalities(subject, output_size, modalities, tumor_region_only):
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
                os.path.splitext(name)[0]: get_tumor_boundingbox(os.path.join(subject_path, modality, name),os.path.join(subject_path, modality+'L', name)) for name in names
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
    #tf.cast(img, dtype=)
    return img

def get_tumor_boundingbox(imgpath, labelpath):
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
    backup = tf.convert_to_tensor(backup, dtype=tf.uint8)
    return backup

def parse_aug_configs(configs, default_configs=None):
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
    if methods is None:
        methods = {
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
    if callable(method_in_str): return method_in_str
    method_in_str.endswith('_img')
    method = vars[method_in_str]
    return method

def flip_leftright_img(dataset):
    dataset = dataset.map(
        lambda image, label: flip_leftright(image, label),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )
    return dataset

def flip_leftright(image, label):
    image = tf.image.flip_left_right(image)
    return image, label

def rotate90_img(dataset):
    dataset = dataset.map(
        lambda img, label: rotate90(img, label),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )
    return dataset

def rotate90(img, label):
    img = tf.image.rot90(img, k=1)
    return img, label

def rotate180_img(dataset):
    dataset = dataset.map(
        lambda img, label: rotate180(img, label),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )
    return dataset

def rotate180(img, label):
    img = tf.image.rot90(img, k=2)
    return img, label

def rotate270_img(dataset):
    dataset = dataset.map(
        lambda img, label: rotate270(img, label),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )
    return dataset

def rotate270(img, label):
    img = tf.image.rot90(img, k=3)
    return img, label

def up_rotate90_img(dataset):
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
    img = tf.image.flip_up_down(img)
    img = tf.image.rot90(img, k=3)
    return img, label



def configure_dataset(dataset, batch_size, buffer_size, repeat=False):
    dataset = dataset.shuffle(buffer_size)
    if repeat:
        print("entering repeat")
        dataset = dataset.repeat(None)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset

def count(ds):
    size = 0
    for _ in ds: 
        size += 1
    return size
