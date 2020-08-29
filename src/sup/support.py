import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import signal
import numpy as np
import random
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img


def path_update(root_dir, composing_dirs):
    composed_dirs = []
    for path in composing_dirs:
        if type(path) == list:
            dirs = [os.path.join(root_dir, i) for i in path]
            composed_dirs.extend(dirs)
        else:
            dirs = os.path.join(root_dir, path)
            composed_dirs.append(dirs)
    return composed_dirs


def visualize_model(model_path,model_weight_path, img_paths):
    model = load_model(model_path)
    model.load_weights(model_weight_path)

    successive_outputs = [layer.output for layer in model.layers[1:]]
    # visualization_model = Model(img_input, successive_outputs)
    visualization_model = tf.keras.models.Model(inputs=model.input, outputs=successive_outputs)
    # prepare a random input image from the training set.
    #img_path = random.choice(img_paths)                                   ## comment for enter only one perticular pic.

    img = load_img(img_paths, target_size=(224, 224))  # this is a PIL image
    x = img_to_array(img)  # Numpy array with shape (150, 150, 3)
    x = x.reshape((1,) + x.shape)  # Numpy array with shape (1, 150, 150, 3)

    # Rescale by 1/255
    x /= 255.

    # run our image through our network, thus obtaining all
    # intermediate representations for this image.
    successive_feature_maps = visualization_model.predict(x)

    # These are the names of the layers, so can have them as part of our plot
    layer_names = [layer.name for layer in model.layers[1:]]

    # display our representations
    for layer_name, feature_map in zip(layer_names, successive_feature_maps):
        if len(feature_map.shape) == 4:
            n_features = feature_map.shape[-1]  # number of features in feature map
            # The feature map has shape (1, size, size, n_features)
            size = feature_map.shape[1]
            display_grid = np.zeros((size, size * n_features))
            for i in range(n_features):
                # Postprocess the feature to make it visually palatable
                x = feature_map[0, :, :, i]
                x -= x.mean()
                x /= x.std()
                x *= 64
                x += 128
                x = np.clip(x, 0, 255).astype('uint8')
                # We'll tile each filter into this big horizontal grid
                display_grid[:, i * size: (i + 1) * size] = x
            # Display the grid
            scale = 20. / n_features
            plt.figure(figsize=(scale * n_features, scale))
            plt.title(layer_name)
            plt.grid(False)
            plt.imshow(display_grid, aspect='auto', cmap='viridis')


def plot_sample_of_img(nrows, ncols, img_paths):
    fig = plt.gcf()
    fig.set_size_inches(ncols * 4, nrows * 4)

    for i, img_path in enumerate(img_paths):
        # Set up subplot; subplot indices start at 1
        sp = plt.subplot(nrows, ncols, i + 1)
        sp.axis('Off')  # Don't show axes (or gridlines)
        img = mpimg.imread(img_path)
        plt.imshow(img)
    plt.show()


def save(model, name):
    model_save_path = '../../h5_files/models/'
    weight_save_path = '../../h5_files/weights/'
    if not os.path.exists(model_save_path):
        os.mkdir(model_save_path)
    if not os.path.exists(weight_save_path):
        os.mkdir(weight_save_path)
    model.save(os.path.join(model_save_path, name))
    model.save_weights(os.path.join(weight_save_path, name))
    return os.path.join(model_save_path, name),os.path.join(weight_save_path, name)


def rnd_predict(model_path, model_weight_path, img_path, clToInt_dict):
    model = load_model(model_path)
    model.load_weights(model_weight_path)

    x_img = load_img(img_path, target_size=(224, 224))
    x = img_to_array(x_img)
    x = np.expand_dims(x, axis=0)
    result = model.predict(x)
    img_class = np.argmax(result[0])
    str_img_class = clToInt_dict[img_class]

    plt.imshow(x_img)
    plt.title(str_img_class)
    plt.show()

    return str_img_class, img_class


#os.kill(os.getpid(), signal.SIGKILL)
