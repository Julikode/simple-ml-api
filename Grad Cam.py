# %% [markdown]
# <center><a target="_blank" href="https://academy.constructor.org/"><img src=https://lh3.googleusercontent.com/d/1fypIr9T-7ntcsVQFmC2_iMPcsm7h8jXg width="500" style="background:none; border:none; box-shadow:none;" /></a> </center>
# <hr />
# 
# 
# # <h1 align="center"> Grad-CAM (Helper Notebook) </h1>
# 
# <hr />
# <center>Constructor Nexademy, 2026</center>

# %% [markdown]
# ##  Grad-CAM: Understanding CNN layers
# 
# __Part 1: An introduction to Grad-CAM__
# 
# 

# %% [markdown]
# ## Introduction
# 
# 
# Grad-CAM is an algorithm proposed by the paper `Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization` https://arxiv.org/abs/1610.02391. It's a very useful tool for understanding deep CNN models better and being able to improve them. It is a class-discriminative localization technique that generates visual explanations for any CNN-based network without requiring architectural changes or re-training. The image below is an overview of how Grad-CAM highlights signals picked up by our model.
# 
# 
# <img src=https://storage.googleapis.com/prod-alemira-alms-ztool-public/rich-text-attachments/0d13f755-453c-479a-abf4-3e622d9662c6/50eda2d1-7349-44f6-8981-0f7f4fb34dcc/grad_cam.jpg width=1500 class="center">
# 
# With Grad-CAM we can:
# - understand the reason behind wrong predictions.
# - identify biases in datasets.
# - discern a ‘stronger’ network from a ‘weaker’ one, even when both make identical prediction.
# 
# In order to obtain the class-discriminative localization map Grad-CAM, we first compute the gradient of the scores for each class, with respect to the feature map activations of a convolutional layer. These gradients are then used to obtain the neuron importance of the weights.
# 
# For more theoretical understanding, please refer to the links below:
# - https://arxiv.org/pdf/1610.02391.pdf
# - https://www.pyimagesearch.com/2020/03/09/grad-cam-visualize-class-activation-maps-with-keras-tensorflow-and-deep-learning
# 
# 
# Grad-CAM gained very recently attention and is for example part of the `tf-explain` package: https://tf-explain.readthedocs.io/en/latest/index.html.
# 
# 

# %% [markdown]
# ## Task
# 
# 
# - Follow along with this guided tutorial understanding the key code snippets involved in the Grad-CAM methodology
# - We provide you with a notebook that already can be run in Google Colab.
# - Test Grad-CAM on the provided example images.
# - Search for more images in the web that might challenge the network, like a cat and a dog in the same picture.
# - Compare the performance of different pretrained networks (VGG, ResNet-50, ...). Do you see differences in the performances? Can you find obvious errors which are probably caused by bias in the training data?
# 

# %% [markdown]
# Grad-CAM works by
# 1. finding the final convolutional layer in the network
# 2. examining the gradient information flowing into that layer
# 
# The output of Grad-CAM is a heatmap visualization for a given class label (either the top, predicted label or an arbitrary label we select for debugging).
# 
# We can use this heatmap to __visually verify__ where in the image the CNN is looking.

# %% [markdown]
# ## Utility functions


# %%
from functools import wraps

# function to add class methods dynamically
def add_method(cls):
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            return func(self, *args, **kwargs)
        setattr(cls, func.__name__, wrapper)
        # Note we are not binding func, but wrapper which accepts self but does exactly the same as func
        return func # returning func means func can still be used normally
    return decorator

# %% [markdown]
# ## Building GradCAM

# %% [markdown]
# ## Import libraries

# %%
from tensorflow.keras.models import Model
import tensorflow as tf
import numpy as np
import cv2 # open cv

# %% [markdown]
# ## Create an empty GradCAM Class
# 
# We will add functionality to this empty class step by step with detailed info about each function

# %%
class GradCAM:
    pass

# %% [markdown]
# ## Creating the Constructor
# 
# The constructor accepts and stores:
# 
# - A TensorFlow model which we’ll use to compute a heatmap
# - The classIdx — a specific class index that we’ll use to measure our class activation heatmap
# - An optional CONV layerName of the model in case we want to visualize the heatmap of a specific layer of our CNN; otherwise, if a specific layer name is not provided, we will automatically infer on the final CONV/POOL layer of the model architecture

# %%
@add_method(GradCAM)
def __init__(self, model, classIdx, layerName=None):

    # store the model, the class index used to measure the class
    # activation map, and the layer to be used when visualizing
    # the class activation map
    self.model = model
    self.classIdx = classIdx
    self.layerName = layerName

    # if the layer name is None, attempt to automatically find
    # the target output layer
    if self.layerName is None:
        self.layerName = self.find_target_layer()

# %% [markdown]
# ## Adding function to find target layer
# 
# The `find_target_layer()` function loops over all layers in the network in reverse order, during which time it checks to see if the current layer has a 4D output (implying a CONV or POOL layer).
# 
# If find such a 4D output, we return that layer name.
# 
# Otherwise, if the network does not have a 4D output, then we cannot apply Grad-CAM, at which point, we raise a ValueError exception, causing our program to stop

# %%
@add_method(GradCAM)
def find_target_layer(self):

    # attempt to find the final convolutional layer in the network
    # by looping over the layers of the network in reverse order
    for layer in reversed(self.model.layers):
        # check to see if the layer has a 4D output
        if len(layer.output_shape) == 4:
            return layer.name

    # otherwise, we could not find a 4D layer so the GradCAM
    # algorithm cannot be applied
    raise ValueError("Could not find 4D layer. Cannot apply GradCAM.")

# %% [markdown]
# ## Adding function to construct GradCAM heatmap
# 
# The compute_heatmap method, which is the heart of our Grad-CAM model. Let’s take this implementation one step at a time to learn how it works.
# 
# <br/>
# 
# __Step 1:__
# 
# Our Grad-CAM requires that we pass in the image for which we want to visualize class activations mappings for.
# 
# From there, we construct our gradModel, which consists of both an input and an output:
# 
# - inputs: The standard image input to the model
# - outputs: The outputs of the layerName class attribute used to generate the class activation mappings.
# 
# Notice how we call `get_layer()` on the model itself while also grabbing the output of that specific layer.
# 
# <br/>
# 
# __Step 2:__
# 
# Once our gradient model is constructed, we’ll proceed to compute gradients.
# 
# Going forward, we need to understand the definition of automatic differentiation and what TensorFlow calls a gradient tape.
# 
# _Automatic differentiation_ is the process of computing a value and computing derivatives of that value.
# 
# > _TensorFlow provides the `tf.GradientTape` API for automatic differentiation — computing the gradient of a computation with respect to its input variables. TensorFlow “records” all operations executed inside the context of a `tf.GradientTape` onto a “tape”. TensorFlow then uses that tape and the gradients associated with each recorded operation to compute the gradients of a “recorded” computation using reverse mode differentiation ([TensorFlow’s Automatic differentiation and gradient tape Tutorial](https://www.tensorflow.org/tutorials/customization/autodiff))_
# 
# - In this step, we accept the input image and cast it to a 32-bit floating point type
# - A forward pass through the gradient model produces the convOutputs and predictions of the layerName layer
# - We then extract the loss associated with our predictions and specific classIdx we are interested in. Notice that our inference stops at the specific layer we are concerned about. We do not need to compute a full forward pass
# - Finally we use automatic differentiation to compute the gradients that we will call grads
# 
# <br/>
# 
# __Step 3:__
# 
# Given our gradients, we’ll now compute guided gradients
# 
# First, we find all outputs and gradients with a value > 0 and cast them from a binary mask to a 32-bit floating point data type
# 
# Then we compute the guided gradients by multiplication
# 
# Keep in mind that both `castConvOutputs` and `castGrads` contain only values of 1’s and 0’s; therefore, during this multiplication if any of `castConvOutputs`, `castGrads`, and `grads` are zero, then the output value for that particular index in the volume will be zero.
# 
# Essentially, what we are doing here is finding positive values of both `castConvOutputs` and `castGrads`, followed by multiplying them by the gradient of the differentiation — this operation will allow us to visualize where in the volume the network is activating __(i.e highlight regions of the image \ feature map being activated based on the guided gradients)__
# 
# The convolution and guided gradients have a batch dimension that we don’t need, hence we discard it.
# 
# <br/>
# 
# __Step 4:__
# 
# Here. we compute the weights of the gradient values by computing the mean of `guidedGrads`, which is essentially a 1 x 1 x N average across the volume.
# 
# We then take those weights and sum the ponderated (i.e., mathematically weighted) maps into the Grad-CAM visualization (`cam`).
# 
# <br/>
# 
# __Step 5:__
# 
# Our final step is to generate the output heatmap associated with our image.
# 
# We grab the original dimensions of input image and scale our `cam` mapping to the original image dimensions.
# 
# From there, we perform min-max rescaling to the range [0, 1] and then convert the pixel values back to the range [0, 255].
# 
# Finally, the last step of our `compute_heatmap()` method returns the heatmap to the caller.
# 
# 

# %%
@add_method(GradCAM)
def compute_heatmap(self, image, eps=1e-8, verbose=False):

    # Step 1:
    # construct our gradient model by supplying
    # (1) the inputs to our pre-trained model
    # (2) the output of the (presumably) final 4D layer in the network
    # (3) the output of the softmax activations from the model
    gradModel = Model(
        inputs=[self.model.inputs],
        outputs=[self.model.get_layer(self.layerName).output,
            self.model.output])
    if verbose:
        print('\nStep 1:')
        print('Model:', self.model.name)
        print('Conv layerName:', self.layerName)
        print('Conv Layer Shape:', gradModel.outputs[0].shape)


    # Step 2:
    # record operations for automatic differentiation
    with tf.GradientTape() as tape:

        # cast the image tensor to a float-32 data type, pass the
        # image through the gradient model, and grab the loss
        # associated with the specific class index
        inputs = tf.cast(image, tf.float32)
        (convOutputs, predictions) = gradModel(inputs)
        loss = predictions[:, self.classIdx]

    # use automatic differentiation to compute the gradients
    grads = tape.gradient(loss, convOutputs)
    if verbose:
        print('\nStep 2:')
        print('loss:', loss)
        print('convOutputs shape;', convOutputs.shape)
        print('gradients shape:', grads.shape)


    # Step 3:
    # compute the guided gradients
    castConvOutputs = tf.cast(convOutputs > 0, "float32")
    castGrads = tf.cast(grads > 0, "float32")
    guidedGrads = castConvOutputs * castGrads * grads
    if verbose:
        print('\nStep 3')
        print('guidedGrads shape', guidedGrads.shape)

    # the convolution and guided gradients have a batch dimension
    # (which we don't need) so let's grab the volume itself and
    # discard the batch
    convOutputs = convOutputs[0]
    guidedGrads = guidedGrads[0]
    if verbose:
        print('guidedGrads shape batch removal', guidedGrads.shape)
        print('convOutputs shape batch removal', convOutputs.shape)


    # Step 4:
    # compute the average of the gradient values, and using them
    # as weights, compute the ponderation of the filters with
    # respect to the weights
    weights = tf.reduce_mean(guidedGrads, axis=(0, 1))
    cam = tf.reduce_sum(tf.multiply(weights, convOutputs), axis=-1)
    if verbose:
        print('\nStep 4:')
        print('weights shape:', weights.shape)
        print('cam shape:', cam.shape)


    # Step 5:
    # grab the spatial dimensions of the input image and resize
    # the output class activation map to match the input image
    # dimensions
    (w, h) = (image.shape[2], image.shape[1])
    heatmap = cv2.resize(cam.numpy(), (w, h),
                         interpolation=cv2.INTER_CUBIC)

    # normalize the heatmap such that all values lie in the range
    # [0, 1], scale the resulting values to the range [0, 255],
    # and then convert to an unsigned 8-bit integer
    numer = heatmap - np.min(heatmap)
    denom = (heatmap.max() - heatmap.min()) + eps
    heatmap = numer / denom
    heatmap = (heatmap * 255).astype("uint8")
    if verbose:
        print('\nStep 5:')
        print('heatmap shape:', heatmap.shape)

    # return the resulting heatmap to the calling function
    return heatmap

# %% [markdown]
# ## Adding function to overlay heatmap on the image
# 
# Our heatmap produced by the previous compute_heatmap function is a single channel, grayscale representation of where the network activated in the image — larger values correspond to a higher activation, smaller values to a lower activation.
# 
# In order to overlay the heatmap, we first need to apply a pseudo/false-color to the heatmap. To do so, we will use OpenCV’s built in `VIRIDIS` colormap (i.e., `cv2.COLORMAP_VIRIDIS`).

# %%
@add_method(GradCAM)
def overlay_heatmap(self, heatmap, image, alpha=0.5,
                    colormap=cv2.COLORMAP_JET):

    # apply the supplied color map to the heatmap and then
    # overlay the heatmap on the input image
    heatmap = cv2.applyColorMap(heatmap, colormap)
    output = cv2.addWeighted(image, alpha, heatmap, 1 - alpha, 0)

    # return a 2-tuple of the color mapped heatmap and the output,
    # overlaid image
    return (heatmap, output)

# %% [markdown]
# ## Applying GradCAM

# %% [markdown]
# ## Import libraries

# %%
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.applications import imagenet_utils
import numpy as np
import cv2
import matplotlib.pyplot as plt

# %%


# %% [markdown]
# ## Load Pre-trained Models

# %%
vgg_model = VGG16(weights='imagenet')
vgg_pre_process_fn = tf.keras.applications.vgg16.preprocess_input

resnet50_model = ResNet50(weights='imagenet')
resnet50_pre_process_fn = tf.keras.applications.resnet50.preprocess_input

# %% [markdown]
# ## Helper Functions - Image Utilities

# %%
# helper function for image pre-processing
def image_processing_function(im_path, input_img_dims, pre_process_function):
  # load the original image from gdrive (in OpenCV format)
  # resize the image to its target dimensions

  # load original image for plotting later on
  orig = load_img(im_path)
  orig_arr = img_to_array(orig).astype("uint8")

  # load the input image from gdrive (in Keras/TensorFlow format)
  # basic image pre-processing especially for passing to the CNN
  image = load_img(im_path, target_size=input_img_dims)
  image_arr = img_to_array(image)
  image_arr = np.expand_dims(image_arr, axis=0)
  image_arr = pre_process_function(image_arr) # image_arr / 255. if you have a CNN from scratch
  return image_arr, orig_arr


# helper function to get predicted classes
# use only for imagenet models directly not for finetuned or models from scratch
# use your own labels and ids then
def get_class_predictions(preds, class_rank):
  rank = class_rank
  classes_ranked = np.argsort(preds[0])[::-1]
  i = classes_ranked[rank] # finds the class ID with highest probability

  # helps in getting class IDs, the predicted probability
  # and the class label name of the imagenet data (1000 classes)
  # top 10 classes with highest prob.
  # gets it from this mapping: https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a
  decoded = imagenet_utils.decode_predictions(preds, 10)
  (imagenetID, label, prob) = decoded[0][0]

  label = "{}: {:.2f}%".format(label, prob * 100)
  print('Class with highest probability:')
  print("{}".format(label))

  return i, decoded



# %% [markdown]
# ## VGG16 Interpretation Flow

# %% [markdown]
# ### Get Image

# %%
!curl https://i.imgur.com/tkOmvsl.png -o dog.png

# %% [markdown]
# ### Load and Pre-process Image for VGG16 interpretation

# %%
# define path to image
im_path = './dog.png'

# use helper function for image pre-processing
input_img_dims = (224, 224)
image, orig = image_processing_function(im_path, input_img_dims, vgg_pre_process_fn)

# %%
plt.imshow(orig)

# %%
orig.shape

# %%
image.shape

# %% [markdown]
# ### Run VGG Inference

# %%
# use the network to make predictions on the input image and find
preds = vgg_model.predict(image)

# %%
preds.shape

# %%
np.argmax(preds)

# %%
preds[0][213]

# %%
preds[0][107]

# %%
# get classes
idx, decoded = get_class_predictions(preds, 0)

# %%
idx

# %%
# lets look at other possible classes
decoded

# %%
idx

# %% [markdown]
# ### Initialize GradCAM instance

# %%
for layer in vgg_model.layers:
    print(layer.name, layer)


# %%
vgg_gc = GradCAM(model=vgg_model, classIdx=213, layerName="block5_pool")

# %% [markdown]
# ### Get GradCAM heatmap

# %%
heatmap = vgg_gc.compute_heatmap(image, verbose=True) # heatmap size is size of preprocessed image - 224 x 224
# resize the resulting heatmap to the original input image dimensions which is 325 x 441
# overlay heatmap on top of the image
heatmap = cv2.resize(heatmap, (orig.shape[1], orig.shape[0]),
                     interpolation=cv2.INTER_CUBIC)
(heatmap, output) = vgg_gc.overlay_heatmap(heatmap, orig, alpha=0.45)
# change color image alignment to RGB from BGR which is done automatically by open cv
heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)

# %%
f, ax = plt.subplots(1,3, figsize=(20, 10))

plt.axis('off');
ax[0].imshow(orig);
ax[1].imshow(heatmap);
ax[2].imshow(output);
[axi.set_axis_off() for axi in ax.ravel()];

# %%
f, ax = plt.subplots(1,3, figsize=(20, 10))

plt.axis('off');
ax[0].imshow(orig);
ax[1].imshow(heatmap);
ax[2].imshow(output);
[axi.set_axis_off() for axi in ax.ravel()];

# %% [markdown]
# ## Now repeat the same steps for the ResNet50 model

# %%
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import cv2
import matplotlib.pyplot as plt

print("Бібліотеки успішно завантажені!")

# %%
# Завантажуємо модель ResNet50
resnet_model = ResNet50(weights="imagenet", include_top=True)

# Отримуємо функцію попередньої обробки, специфічну для ResNet
resnet_pre_process_fn = tf.keras.applications.resnet50.preprocess_input

print("Модель ResNet50 готова до роботи!")

# %%
def image_processing_function(im_path, input_img_dims, pre_process_function):
    # Завантажуємо оригінал для подальшого виведення на екран
    orig = load_img(im_path)
    orig_arr = img_to_array(orig).astype("uint8")

    # Завантажуємо зображення у потрібному розмірі для моделі
    image = load_img(im_path, target_size=input_img_dims)
    image_arr = img_to_array(image)
    image_arr = np.expand_dims(image_arr, axis=0)
    image_arr = pre_process_function(image_arr)

    return image_arr, orig_arr

# %%
# Вкажіть точну назву вашого файлу, який ви перетягли у панель файлів зліва в Colab
# Наприклад: 'my_dog.jpg', 'photo.png' тощо
im_path = './Gemini_Generated_Image_hjdrfnhjdrfnhjdr.jpeg'

input_img_dims = (224, 224)

# Викликаємо функцію обробки
image, orig = image_processing_function(im_path, input_img_dims, resnet_pre_process_fn)
print("Ваше локальне зображення успішно завантажено та підготовлено!")

# %%


# %%
plt.imshow(orig)

# %%
orig.shape

# %%
image.shape

# %%
# use the network to make predictions on the input image and find
preds = resnet_model.predict(image)

# %%
preds.shape

# %%
np.argmax(preds)

# %%
preds[0][213]

# %%
preds[0][107]

# %%
# get classes
idx, decoded = get_class_predictions(preds, 0)

# %%
idx

# %%
# lets look at other possible classes
decoded

# %%
for layer in resnet_model.layers:
    print(layer.name, layer)

# %%
resnet = GradCAM(model=resnet_model, classIdx=225, layerName="conv5_block3_out")

# %%
heatmap = resnet.compute_heatmap(image, verbose=True) # heatmap size is size of preprocessed image - 224 x 224
# resize the resulting heatmap to the original input image dimensions which is 325 x 441
# overlay heatmap on top of the image
heatmap = cv2.resize(heatmap, (orig.shape[1], orig.shape[0]),
                     interpolation=cv2.INTER_CUBIC)
(heatmap, output) = resnet.overlay_heatmap(heatmap, orig, alpha=0.45)
# change color image alignment to RGB from BGR which is done automatically by open cv
heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)

# %%
f, ax = plt.subplots(1,3, figsize=(20, 10))

plt.axis('off');
ax[0].imshow(orig);
ax[1].imshow(heatmap);
ax[2].imshow(output);
[axi.set_axis_off() for axi in ax.ravel()];

# %%
f, ax = plt.subplots(1,3, figsize=(20, 10))

plt.axis('off');
ax[0].imshow(orig);
ax[1].imshow(heatmap);
ax[2].imshow(output);
[axi.set_axis_off() for axi in ax.ravel()];

# %%
def image_processing_function(im_path, input_img_dims, pre_process_function):
    # Завантажуємо оригінал для подальшого виведення на екран
    orig = load_img(im_path)
    orig_arr = img_to_array(orig).astype("uint8")

    # Завантажуємо зображення у потрібному розмірі для моделі
    image = load_img(im_path, target_size=input_img_dims)
    image_arr = img_to_array(image)
    image_arr = np.expand_dims(image_arr, axis=0)
    image_arr = pre_process_function(image_arr)

    return image_arr, orig_arr

# %%
# Вкажіть точну назву вашого файлу, який ви перетягли у панель файлів зліва в Colab
# Наприклад: 'my_dog.jpg', 'photo.png' тощо
im_path = '/content/dog 3.jpeg'

input_img_dims = (224, 224)

# Викликаємо функцію обробки з правильною назвою функції передобробки resnet_pre_process_fn
image, orig = image_processing_function(im_path, input_img_dims, resnet_pre_process_fn)
print("Ваше локальне зображення успішно завантажено та підготовлено!")

# %%
plt.imshow(orig)

# %%
orig.shape

# %%
image.shape

# %%
# use the network to make predictions on the input image and find
preds = resnet_model.predict(image)

# %%
preds.shape

# %%
np.argmax(preds)

# %%
preds[0][213]

# %%
preds[0][107]

# %%
# get classes
idx, decoded = get_class_predictions(preds, 0)

# %%
idx

# %%
decoded

# %%
for layer in resnet_model.layers:
    print(layer.name, layer)

# %%
resnet = GradCAM(model=resnet_model, classIdx=225, layerName="conv5_block3_out")

# %%
heatmap = resnet.compute_heatmap(image, verbose=True) # heatmap size is size of preprocessed image - 224 x 224
# resize the resulting heatmap to the original input image dimensions which is 325 x 441
# overlay heatmap on top of the image
heatmap = cv2.resize(heatmap, (orig.shape[1], orig.shape[0]),
                     interpolation=cv2.INTER_CUBIC)
(heatmap, output) = resnet.overlay_heatmap(heatmap, orig, alpha=0.45)
# change color image alignment to RGB from BGR which is done automatically by open cv
heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)

# %%
f, ax = plt.subplots(1,3, figsize=(20, 10))

plt.axis('off');
ax[0].imshow(orig);
ax[1].imshow(heatmap);
ax[2].imshow(output);
[axi.set_axis_off() for axi in ax.ravel()];

# %%
# use the network to make predictions on the input image and find
preds = resnet_model.predict(image)

# %%


# %%


# %% [markdown]
# ## Try more examples

# %% [markdown]
# Now lets see what happens when we have two classes in one image (cat and dog)

# %% [markdown]
# ## VGG16 Explanation

# %%
!wget -O cat_dog.png https://i.imgur.com/BlyglPM.png

# %%


# %%


# %%


# %% [markdown]
# For our second image, if you implemented things correctly, it will look like our VGG16 model pays more attention to the dog than the cat. Woof woof! Try the same with the other model. What do you see?

# %% [markdown]
# ## Try a ResNet50 Interpretation

# %%


# %%


# %%


# %%


# %%


# %% [markdown]
# ## Now try to visualize what VGG16 sees for the same image as a cat
# 
# Hint: Use classIdx of 281

# %%
vgg_gc = GradCAM(<...FILL IN CODE...>)

<...YOUR CODE HERE...>

# %%


# %% [markdown]
# ## Bias in Datasets
# 
# Try to run the following image through both the models, what do you see?

# %%
!wget -O train.png https://i.imgur.com/UAKXs0e.png

# %% [markdown]
# ### VGG16 Explanation

# %%


# %%


# %% [markdown]
# ### ResNet50 Explanation

# %%


# %%



