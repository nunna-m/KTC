import tensorflow as tf
from tensorflow.keras.models import save_model, Sequential
import onnx
from onnx2keras import onnx_to_keras
import numpy as np
import cv2

#sample data to check if the output of tf model is same as pytorch/onnx model
data = np.load('/home/maanvi/LAB/code/organamnist.npz', allow_pickle=True)
test_image1 = data['test_images'][2]
print(data['test_labels'][2])
# test_image1 = np.repeat(test_image1[:,:,np.newaxis], 3, axis=-1)
# test_image1 = cv2.resize(test_image1, (224,224))
# tf_test_image = test_image1[:]
# test_image1 = np.moveaxis(test_image1, -1, 0)
# test_image1 = np.asarray(test_image1,dtype=np.float32)[np.newaxis,:,:,:]
# print(test_image1.shape)
# print(tf_test_image.shape)
# test_image1 = tf.convert_to_tensor(test_image1)

# model_path = '/home/maanvi/LAB/code/vgg16_pretrained.pb'
# model = tf.saved_model.load(model_path)
# infer = model.signatures['serving_default']
# outputs = list(infer.structured_outputs)
# print(outputs)

# y = infer(test_image1)[outputs[0]].numpy()
# print(y)

# onnx_model = onnx.load('/home/maanvi/LAB/code/vgg16_pretrained.onnx')
# keras_model = onnx_to_keras(onnx_model, input_names=['input.1'], name_policy='renumerate',change_ordering=False)
# tf_test_image = tf.convert_to_tensor(np.asarray(tf_test_image,dtype=np.float32)[np.newaxis,:,:,:])
# print(tf_test_image.shape)
# #outputs = keras_model.predict(tf_test_image)
# outputs = keras_model.predict(test_image1)
# print(np.argmax(outputs))
# # print(outputs.shape)
# #outputs = tf.transpose(outputs, (0, 3, 1, 2))
# #print(outputs)
# keras_model.save('/home/maanvi/LAB/code/vgg16_pretrained_CHW.h5')