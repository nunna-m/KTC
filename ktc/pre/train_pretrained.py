import tensorflow as tf
import onnx
from onnx_tf.backend import prepare
import numpy as np
import cv2

data = np.load('/home/maanvi/LAB/code/organamnist.npz', allow_pickle=True)
test_image1 = data['test_images'][0]
test_image1 = np.repeat(test_image1[:,:,np.newaxis], 3, axis=-1)
print(test_image1.shape)
#print(test_image1.shape)
test_image1 = cv2.resize(test_image1, (224,224))
test_image1 = np.moveaxis(test_image1, -1, 0)
print(test_image1.shape)
#cv2.imwrite('organ.png',test_image1)

# read_data = np.asarray(cv2.imread('organ.png'),dtype=np.float32)[np.newaxis,:,:,:]
# print(read_data.shape)
model = onnx.load('/home/maanvi/LAB/code/vgg16_pretrained.onnx')
tf_rep = prepare(model)


output = tf_rep.run(np.asarray(test_image1,dtype=np.float32)[np.newaxis,:,:,:])
print(output)