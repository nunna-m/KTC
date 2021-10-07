import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tensorflow as tf
import numpy as np
import glob
from functools import partial
import cv2
import sys
AUTOTUNE = tf.data.experimental.AUTOTUNE

def count(ds):
    size = 0
    for _ in ds: 
        size += 1
    return size

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
    if aug_configs is None:
        aug_configs = {
            'random_crop':{},
        }
    default_aug_configs = {
        random_crop_img: dict(output_size=output_size),
        random_horizontalflip_img: {},
    }
    
    traindir = os.path.join(data_root,'_'.join(modalities),'train')
    dataset = load_raw(
        traindir,
        modalities=modalities,
        output_size=output_size,
        tumor_region_only = tumor_region_only
    )
    dataset = augmentation(
        dataset,
        methods=parse_aug_configs(aug_configs,
                                    default_aug_configs),
    )
    len_of_dataset = tf.data.experimental.cardinality(dataset)
    dataset = image_label(dataset, modalities, len_of_dataset)
    dataset = configure_dataset(
        dataset,
        batch_size,
        buffer_size,
        repeat=repeat
    )
    return dataset

def load_raw(traindir, modalities=('am','tm','dc','ec','pc'), output_size=(224,224), tumor_region_only=False, dtype=tf.float32):
    
    training_subject_paths = glob.glob(os.path.join(traindir,*'*'*2))
    print(training_subject_paths)
    ds = tf.data.Dataset.from_tensor_slices(training_subject_paths)
    #ds = ds.interleave(tf.data.Dataset.list_files(shuffle=False))
    label_ds = ds.interleave(
            lambda subject_path: tf.data.Dataset.from_generator(
                get_label, args=(subject_path,),
                output_signature=tf.TensorSpec(shape=(), dtype=tf.int32)),
            cycle_length=count(ds),
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )
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
    
    ds = tf.data.Dataset.zip((feature_ds, label_ds))
    ds = ds.map(
        lambda feature,label: duplicate_label(feature,label),
        num_parallel_calls=tf.data.experimental.AUTOTUNE)
    for ele in ds.as_numpy_iterator():
        print(ele[0].shape, ele[1].shape)
    return feature_ds
    tf.data.Dataset.partial_map = partial_map
    if output_size is not None and tumor_region_only==False: 
        ds= ds.map(
        lambda subject_data: {
            'slices': tf.map_fn(
                lambda image: tf.image.crop_to_bounding_box(
                    image,
                    ((tf.shape(image)[:2] - output_size) // 2)[0],
                    ((tf.shape(image)[:2] - output_size) // 2)[1],
                    *output_size,),
                subject_data['stacked_modality_slices'],
            ),
            'labels':subject_data['labels'],
            'clas':subject_data['clas'],
            'ID':subject_data['ID'],
            'subject_path':subject_data['subject_path'],
        },
        AUTOTUNE,
        )
    else:
        ds = ds.map(
        lambda subject_data: {
            'slices': subject_data['stacked_modality_slices'],
            'labels':subject_data['labels'],
            'clas':subject_data['clas'],
            'ID':subject_data['ID'],
            'subject_path':subject_data['subject_path'],
        },
        AUTOTUNE,
        )

    ds = ds.partial_map('slices', lambda x: tf.reshape(x, [*x.shape[:-1], len(modalities)]))
    ds = ds.partial_map('slices', lambda x: tf.cast(x, dtype=dtype))
    ds = ds.partial_map('slices', lambda x: x / 255.0)

    return ds

def partial_map(dataset, key, function):
    def wrapper(data):
        data.update(
            {
                key: function(data[key])
            }
        )
        return data
    dataset = dataset.map(wrapper, AUTOTUNE)
    return dataset

def convert_to_dict(data):
    print("DATA: ",data)
    return dict(
        slices=tf.cast(data[0], dtype=tf.uint8), 
        labels=tf.cast(data[1], dtype=tf.uint8)
        )

def tf_combine_modalities(subject_path, output_size, modalities, tumor_region_only,return_type='array'):
    return_type  =return_type.lower()
    if return_type == 'array':
        return tf.py_function(
            lambda x: partial(combine_modalities, output_size=output_size,
            modalities=modalities,
            tumor_region_only=tumor_region_only)(x),
            [subject_path],
            Tout=[tf.uint8],
        )
        return partial()
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
    
    slices = tf.stack([tf.stack([subject_data[type_][slice_] for type_ in modalities], axis=-1) for slice_ in slice_names])
    #labels = get_label(subject_data['clas'],slices.shape[0])
    return slices
    return dict(
        stacked_modality_slices=slices,
        labels=labels,
        features_labels=(slices,labels),
        subject_path=subject_data['subject_path'],
    )

def get_label(subject):
    if isinstance(subject, str): 
        pass
    elif isinstance(subject, bytes): 
        subject = subject.decode()
    else: raise NotImplementedError
    clas, _ = get_class_ID_subjectpath(subject)
    if clas=='AML':
        yield 0
        #labels = tf.zeros(shape=(num_slices,1), dtype=tf.int32)
    elif clas=='CCRCC':
        yield 1
        #labels = tf.ones(shape=(num_slices,1), dtype=tf.int32)
    #return labels

def duplicate_label(feature, label):
    print(feature, label)
    (feature_shape, label_shape) = (tf.shape(feature), tf.shape(label))
    print(feature_shape, label_shape)
    label = tf.tile(
        tf.convert_to_tensor(label, dtype=tf.int32), [feature_shape[0],1])
    print(label)
    return (feature,label)

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
        random_crop_img: {},
        random_horizontalflip_img: {},
        }
    else:
        assert isinstance(methods, dict)
        method = dict(map(
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

def random_crop_img(dataset, **configs):
    dataset = dataset.map(
        lambda image: random_crop(image, **configs),
        num_parallel_calls=AUTOTUNE,
    )
    return dataset

def random_crop(img, output_size=(224,224), stddev=4, max_=6, min_=-6):
    threshold = tf.clip_by_value(tf.cast(tf.random.normal([2],stddev=stddev), tf.int32), min_, max_)
    diff = (tf.shape(img)[:2] - output_size) // 2 + threshold
    img = tf.image.crop_to_bounding_box(
        img,
        diff[0],
        diff[1],
        *output_size,
    )
    return img

def random_horizontalflip_img(dataset):
    dataset = dataset.map(
        tf.image.random_flip_left_right,
        num_parallel_calls=AUTOTUNE,
    )
    return dataset

def image_label(dataset, modalities, lendata):
    slice_indices = [i for i in range(len(modalities))]
    def convert(data):
        combined_slices = data
        feature = tf.gather(combined_slices, slice_indices, axis=-1)
        label = 0.0
        return feature, label
    dataset = dataset.map(convert, AUTOTUNE)
    return dataset

def configure_dataset(dataset, batch_size, buffer_size, repeat=False):
    dataset = dataset.shuffle(buffer_size)
    if repeat:
        dataset = dataset.repeat(None)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(AUTOTUNE)
    return dataset

final_dataset = train_dataset(data_root='/home/maanvi/LAB/Datasets/sample_kt',batch_size=4,buffer_size=10,repeat=True,modalities=('am','tm'),output_size=(224,224),aug_configs=None,tumor_region_only=False)

print("done generating dataset")
# for item in final_dataset.take(1):
#     print(item.numpy())
# prep = tf.data.experimental.get_single_element(final_dataset)

# print("prep: ",prep)


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

# subject_data = parse_subject(subject_path='/home/maanvi/LAB/Datasets/kidney_tumor_trainvaltest/am_tm/train/AML/87345564', output_size = (224,224), modalities=['am','tm'], tumor_region_only=False)

# print(subject_data)