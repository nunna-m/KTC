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
import tensorflow_addons as tfa

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

    if aug_configs is None: aug_configs = {'random_crop': {}}
    default_aug_configs = {
        augment_random_crop: dict(output_size=output_size),
        augment_random_flip: {},
        augment_random_contrast: dict(target_channels=list(range(len(modalities)))),
        augment_random_warp: {},
    }

    traindir = os.path.join(data_root,'_'.join(modalities),'train')
    dataset = load_raw(
        traindir,
        modalities=modalities,
        output_size=output_size,
        tumor_region_only = tumor_region_only
    )

    dataset = custom_augmentation(
        dataset,
        methods=parse_augment_options(aug_configs, default_aug_configs),
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
                #os.path.splitext(name)[0]: resize(decoder(os.path.join(subject_path, modality, name))[:, :, 0], output_size)for name in names
                os.path.splitext(name)[0]: decoder(os.path.join(subject_path, modality, name))[:, :, 0]for name in names
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


def custom_augmentation(dataset, methods=None):
    if methods is None:
        methods = {
            augment_random_crop: {},
            augment_random_flip: {},
            augment_random_contrast: {},
            augment_random_warp: {},
        }
    else:
        assert isinstance(methods, dict)
        methods = dict(map(
            lambda conf_name, conf_value: (solve_augment_method(conf_name),conf_value),
            methods.keys(), methods.values(),
        ))
    
    for operation, config in methods.items():
        print('Augment: applying', operation, config)
        dataset = operation(dataset, **config)
    return dataset 
    
def parse_augment_options(augs, default_augs=None):
    if default_augs is None:
        default_augs = {}
    data = {}
    for conf, value in augs.items():
        if value is None:
            conf = {}
        func = globals()[f'augment_{conf}']
        if func in default_augs:
            new_value = default_augs[func].copy()
            new_value.update(value)
        data[func] = value
    return data

def solve_augment_method(method_str):
    '''
    check if the specified augment method exists
    and if it's really an augment method.
    '''
    if callable(method_str): return method_str
    method_str.startswith('augment_')
    method = vars[method_str]
    return method


def augment_random_contrast(ds, target_channels, lower=0.8, upper=1.2):
    ds = ds.map(
        lambda image: random_contrast(image, lower=lower, upper=upper, target_channels=target_channels),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )
    return ds


def random_contrast(image, lower, upper, target_channels):
    non_target_channels = [i for i in range(image.shape[-1]) if i not in target_channels]

    target = tf.gather(image, target_channels, axis=2)
    non_target = tf.gather(image, non_target_channels, axis=2)
    target_out = tf.image.random_contrast(target, lower=lower, upper=upper)
    image = tf.concat([target_out, non_target], axis=2)
    indices = list(map(
        lambda xy: xy[1],
        sorted(
            zip(target_channels + non_target_channels, range(1000)),
            key=lambda xy: xy[0],
        ),
    ))
    image = tf.gather(image, indices, axis=2)
    return image


def augment_random_hue(ds, max_delta=0.2):
    ds = ds.map(
        lambda image: tf.image.random_hue(image, max_delta=max_delta),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )
    return ds


def augment_random_flip(ds):
    ds = ds.map(
        tf.image.random_flip_left_right,
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )
    return ds


def augment_random_warp(ds: tf.data.Dataset, process_in_batch=10, **options) -> tf.data.Dataset:
    '''apply augmentation based on image warping

    Args:
        process_in_batch: the number of images to apply warping in a batch
            None to disable this feature
        options: options to be passed to random_warp function
    '''
    if process_in_batch is not None:
        ds = ds.batch(process_in_batch)
    ds = ds.map(
        lambda image: random_warp(image, process_in_batch=process_in_batch, **options),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )
    if process_in_batch is not None:
        ds = ds.unbatch()
    return ds


def augment_random_crop(ds, **options):
    '''apply augmentation based on image warping'''
    ds = ds.map(
        lambda image: random_crop(image, **options),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )
    return ds


def augment_random_intrachannelwarp(ds: tf.data.Dataset, paired=((0, -1),), **options) -> tf.data.Dataset:
    '''warps each channel indenpently.

    As oppposed to `augment_random_warp`, this will introduce misalignments between
    channels. For this reason, this function is not intended to used as a regular training,
    rather for experiment purpose to see how a model robust against misalignments between slices.

    Args:
        paired (list[tuple[int, int]]): list of paired channels. Paired channels will be applied
            warping operation together, keeping them consistent.
            This is meant to be used to align a channel to a label, while still introducing
            artificial misalignments between channels.
        options: options to be passed to random_warp function
    '''
    ds = ds.map(
        lambda image: random_intrachannelwarp(image, paired=paired, **options),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )
    return ds


def random_crop(image, output_size=(512, 512), stddev=4, max_=6, min_=-6):
    """
    performs augmentation by cropping/resizing
    given image
    """
    diff = tf.clip_by_value(tf.cast(tf.random.normal([2], stddev=stddev), tf.int32), min_, max_)
    image = tf.image.crop_to_bounding_box(
        image,
        ((tf.shape(image)[:2] - output_size) // 2 + diff)[0],
        ((tf.shape(image)[:2] - output_size) // 2 + diff)[1],
        *output_size,
    )
    return image


def random_intrachannelwarp(image, n_points=100, max_diff=5, stddev=2.0, paired=()):
    paired = list(map(
        lambda channel_list: list(map(
            lambda channel: channel if channel >= 0 else image.get_shape()[-1] + channel,
            channel_list,
        )),
        paired,
    ))
    non_paired = list(map(
        lambda x: [x],
        set(range(image.get_shape()[-1])) - set([index for indices in paired for index in indices]),
    ))
    channel_groups = paired + non_paired
    warped_groups = list(map(
        lambda group: random_warp(tf.gather(image, group, axis=-1), n_points=100, max_diff=max_diff, stddev=stddev),
        channel_groups,
    ))
    warped_groups = [channel for channels in warped_groups for channel in tf.unstack(channels, axis=-1)]
    indices = [index for indices in channel_groups for index in indices]
    image = tf.stack(
        list(map(lambda x: x[1], sorted(zip(indices, warped_groups), key=lambda x: x[0]))),
        axis=-1
    )
    return image


@tf.function
def random_warp(image, n_points=100, max_diff=5, stddev=2.0, process_in_batch=None):
    '''
    this function will perfom data augmentation
    using Non-affine transformation, namely
    image warping.
    Currently, only square images are supported

    Args:
        image: input image
        n_points: the num of points to take for image warping
        max_diff: maximum movement of pixels
    Return:
        warped image
    '''
    if process_in_batch is not None:
        width_index, height_index, n_images = 1, 2, process_in_batch
        image = tf.reshape(image, [n_images, *image.get_shape()[1:]])
    else:
        width_index, height_index, n_images = 0, 1, 1

    width = tf.shape(image)[width_index]
    height = tf.shape(image)[height_index]

    with tf.control_dependencies([tf.assert_equal(width, height)]):
        raw = tf.random.uniform([n_images, n_points, 2], 0.0, tf.cast(width, tf.float32), dtype=tf.float32)
        diff = tf.random.normal([n_images, n_points, 2], mean=0.0, stddev=stddev, dtype=tf.float32)
        # ensure that diff is not too big
        diff = tf.clip_by_value(diff, tf.cast(-max_diff, tf.float32), tf.cast(max_diff, tf.float32))

    if process_in_batch is None:
        # expand dimension to meet the requirement of sparse_image_warp
        image = tf.expand_dims(image, 0)

    image = tfa.image.sparse_image_warp(
        image=image,
        source_control_point_locations=raw,
        dest_control_point_locations=raw + diff,
    )[0]
    # sparse_image_warp function will return a tuple
    # (warped image, flow_field)

    if process_in_batch is None:
        # shrink dimension
        image = image[0, :, :, :]
    return image

def image_label(dataset, modalities=('am','tm','dc','ec','pc')):
    return dataset

def configure_dataset(dataset, batch_size, buffer_size, repeat=False):
    return dataset 

