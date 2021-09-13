'''
abstraction for tf.data.Dataset API
'''
import os
from numpy.lib.type_check import _nan_to_num_dispatcher
import tensorflow as tf
import numpy as np
import glob
from functools import partial
import re

from tensorflow.python.framework.ops import reset_default_graph

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

def load_raw(traindir, modalities=('am','tm','dc','ec','pc'), output_size=(224,224), tumor_region_only=False):
    
    training_subject_paths = glob.glob(os.path.join(traindir,*'*'*2))
    #training_image_paths = filter_modalities(training_image_paths,modalities)
    dataset = tf.data.Dataset.from_tensor_slices(training_subject_paths)
    dataset = dataset.interleave(tf.data.Dataset.list_files)
    dataset = dataset.interleave(
        partial(
            combine_modalities,
            modalities=modalities,
            tumor_region_only=tumor_region_only,
        ),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )

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

def combine_modalities(subject, modalities=('am','tm','dc','ec','pc'), tumor_region_only=False):
    return tf.py_function(
        lambda x: partial(prep_combined_modalities,
        modalities=modalities, tumor_region_only=tumor_region_only)(x)['modalities'],
        [subject],
        tf.uint8,
    )

def prep_combined_modalities(subject, modalities, tumor_region_only):
    if isinstance(subject, str): pass
    else: raise NotImplementedError
    subject_data = parse_subject(subject, modalities=modalities, tumor_region_only=tumor_region_only)
    slice_names = subject_data['TRA'].keys()

    slices = tf.stack([tf.stack([subject_data[type_][slice_] for type_ in modalities], axis=-1) for slice_ in slice_names])
    return dict(
        slices=slices,
        category=subject_data['category'],
        patientID=subject_data['patientID'],
        examID=subject_data['examID'],
        path=subject_data['path'],
    )

def parse_subject(subject, modalities,tumor_region_only, decoder=tf.image.decode_image):
    subject_data = {'subject_path': subject}
    subject_data['clas'], subject_data['ID'] = get_class_ID_subjectpath(subject)
    gathered_modalities_paths = {
            modality: set(os.listdir(os.path.join(subject,modality)))
            for modality in modalities
        }
    
    same_named_slices = set.intersection(*map(
        lambda slices: set(
            map(lambda name: os.path.splitext(name)[0], slices)),
        gathered_modalities_paths.values(),
        ))
    
    assert same_named_slices, f'Not enough slices with same name in {subject}'

    for modality in modalities:
        gathered_modalities_paths[modality] = list(
            filter(lambda x: os.path.splitext(x)[0],
            gathered_modalities_paths[modality])
        )
    subject_data['num_slices_per_modality']=len(same_named_slices)

    def wrapper(func):
        def _func(x):
            x = tf.io.read_file(x)
            return func(x)
        return _func
    decoder = wrapper(decoder)

    if tumor_region_only:
        coords = get_tumor_boundingbox(subject, modalities, gathered_modalities_paths)


    for modality, names in gathered_modalities_paths.items():
        subject_data[modality] = {
            os.path.splitext(name)[0]: decoder(os.path.join(subject, modality, name))[:, :, 0] for name in names
        }
    
    return subject_data

def get_class_ID_subjectpath(subject):
    splitup = subject.split(os.path.sep)
    ID = splitup[-1]
    clas = splitup[-2]
    assert clas in ('AML', 'CCRCC'), f'Classification category{clas} extracted from : {subject} unknown'
    return clas, ID

def get_tumor_boundingbox(subject, modalities, slice_names):
    return


def custom_augmentation(dataset, aug_configs):
    return dataset 
    

def image_label(dataset, modalities=('am','tm','dc','ec','pc')):
    return dataset

def configure_dataset(dataset, batch_size, buffer_size, repeat=False):
    return dataset 

