#%%
import SimpleITK as sitk
import cv2
import numpy as np
#OUTPUT_DIR = '/home/maanvi/LAB/Output'
OUTPUT_DIR = 'D:/01_Maanvi/LABB/simpleitk_output'
#%%
%matplotlib inline
import matplotlib.pyplot as plt

from ipywidgets import interact, fixed
from IPython.display import clear_output


# %%
# Callback invoked by the interact IPython method for scrolling through the image stacks of
# the two images (moving and fixed).
def display_images(fixed_image_z, moving_image_z, fixed_npa, moving_npa):
    # Create a figure with two subplots and the specified size.
    plt.subplots(1, 2, figsize=(10, 8))

    # Draw the fixed image in the first subplot.
    plt.subplot(1, 2, 1)
    plt.imshow(fixed_npa[fixed_image_z, :, :], cmap=plt.cm.Greys_r)
    plt.title("fixed image")
    plt.axis("off")

    # Draw the moving image in the second subplot.
    plt.subplot(1, 2, 2)
    plt.imshow(moving_npa[moving_image_z, :, :], cmap=plt.cm.Greys_r)
    plt.title("moving image")
    plt.axis("off")

    plt.show()
# %%
# Callback invoked by the IPython interact method for scrolling and modifying the alpha blending
# of an image stack of two images that occupy the same physical space.
def display_images_with_alpha(image_z, alpha, fixed, moving):
    img = (1.0 - alpha) * fixed[:, :, image_z] + alpha * moving[:, :, image_z]
    plt.imshow(sitk.GetArrayViewFromImage(img), cmap=plt.cm.Greys_r)
    plt.axis("off")
    plt.show()
# %%
# Callback invoked when the StartEvent happens, sets up our new data.
def start_plot():
    global metric_values, multires_iterations

    metric_values = []
    multires_iterations = []
# %%
# Callback invoked when the EndEvent happens, do cleanup of data and figure.
def end_plot():
    global metric_values, multires_iterations

    del metric_values
    del multires_iterations
    # Close figure, we don't want to get a duplicate of the plot latter on.
    plt.close()
# %%
# Callback invoked when the IterationEvent happens, update our data and display new figure.
def plot_values(registration_method):
    global metric_values, multires_iterations

    metric_values.append(registration_method.GetMetricValue())
    # Clear the output area (wait=True, to reduce flickering), and plot current data
    clear_output(wait=True)
    # Plot the similarity metric values
    plt.plot(metric_values, "r")
    plt.plot(
        multires_iterations,
        [metric_values[index] for index in multires_iterations],
        "b*",
    )
    plt.xlabel("Iteration Number", fontsize=12)
    plt.ylabel("Metric Value", fontsize=12)
    plt.show()
# %%
# Callback invoked when the sitkMultiResolutionIterationEvent happens, update the index into the
# metric_values list.
def update_multires_iterations():
    global metric_values, multires_iterations
    multires_iterations.append(len(metric_values))
# %%
fixed_image_path = 'D:/01_Maanvi/LABB/datasets/kt_new_trainvaltest/dc_pc/train/AML/15630104/dc/1.png'
moving_image_path = 'D:/01_Maanvi/LABB/datasets/kt_new_trainvaltest/dc_pc/train/AML/15630104/pc/1.png'
fixed_arr = np.array(cv2.cvtColor(cv2.imread(fixed_image_path),cv2.COLOR_BGR2RGB))
moving_arr = np.array(cv2.cvtColor(cv2.imread(moving_image_path),cv2.COLOR_BGR2RGB))
print(fixed_arr.shape)

#%%
sitk.Show(sitk.ReadImage(fixed_image_path,imageIO="PNGImageIO"))
#%%
use_affine = False
fixed_image = sitk.GetImageFromArray(fixed_arr)
moving_image = sitk.GetImageFromArray(moving_arr)
transform = sitk.AffineTransform(2) if use_affine else sitk.ScaleTransform(2)
initial_transform = sitk.CenteredTransformInitializer(sitk.Cast(fixed_image, moving_image.GetPixelID()),moving_image,transform,sitk.CenteredTransformInitializerFilter.GEOMETRY)
ff_img = sitk.Cast(fixed_image, sitk.sitkFloat32)
mv_img = sitk.Cast(moving_image, sitk.sitkFloat32)
registration_method = sitk.ImageRegistrationMethod()

registration_method.SetMetricAsMeanSquares()

# sample_per_axis = 12
# registration_method.SetOptimizerAsExhaustive([sample_per_axis // 2, 0, 0])
# registration_method.SetOptimizerScales([2.0 * 3.14 / sample_per_axis, 1.0, 1.0])

registration_method.SetInterpolator(sitk.sitkLinear)

registration_method.SetOptimizerAsGradientDescent(learningRate=1.0,numberOfIterations=200,convergenceMinimumValue=1e-6,convergenceWindowSize=10)

registration_method.SetOptimizerScalesFromPhysicalShift()

registration_method.SetInitialTransform(initial_transform, inPlace=False)
final_transform_v1 = registration_method.Execute(ff_img, mv_img)

resample = sitk.ResampleImageFilter()
resample.SetReferenceImage(fixed_image)

resample.SetInterpolator(sitk.sitkBSpline)
resample.SetTransform(final_transform_v1)
resample.Execute(moving_image)
# %%
