'''
abstraction for tf.data.Dataset API
'''
import pathlib
import tensorflow as tf

def get_label_names(data_root):
    label_names = sorted(item.name for item in data_root.glob('*/') if item.is_dir())
    return label_names

data_root = '/home/maanvi/LAB/Datasets/kidney_tumor'

#Load all the file paths in the directory root
all_image_paths = list(data_root.glob('*/*'))
all_image_paths = [str(path) for path in all_image_paths]

#Gather the list of labels and create a labelmap
label_names = sorted(item.name for item in data_root.glob('*/') if item.is_dir())
label_to_index = dict((name, index) for index, name in enumerate(label_names))

#Use the label map to fetch all categorical labels
all_image_labels = [label_to_index[pathlib.Path(path).parent.name]
                    for path in all_image_paths]

path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)
label_ds = tf.data.Dataset.from_tensor_slices(all_image_labels)

def preprocess_image(path):
    image = tf.io.read_file(path)
    image = tf.image.decode_image(image, channels = 3)
    image = tf.image.resize(image, [224,224])
    image /= 255.0 
    return image

image_ds = path_ds.map(preprocess_image)
image_label_ds = tf.data.Dataset.zip((image_ds, label_ds))

BATCH_SIZE = 32
ds = image_label_ds.shuffle(
    buffer_size=len(all_image_paths),
).repeat().batch(BATCH_SIZE)

steps_per_epoch = tf.math.ceil(len(all_image_paths)/BATCH_SIZE).numpy()

model.fit(ds,epochs=1,steps_per_epoch=steps_per_epoch)