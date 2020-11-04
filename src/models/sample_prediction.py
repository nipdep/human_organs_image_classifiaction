
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

class Sampling:

    def __call__(self, model_path,model_weight_path,numOfImages,img_path,img_root="",
                 clToInt_dict={0: 'Brain', 1: 'Eye', 2: 'Heart', 3: 'Kidney', 4: 'Other', 5: 'Skeleton'}):
        if numOfImages == 1:
            self.pred_sample(model_path,model_weight_path,img_path,clToInt_dict)
        else:
            self.pred_samples(model_path,model_weight_path,img_root,img_path,clToInt_dict)

    def pred_sample(self,model_path, model_weight_path, img_path, clToInt_dict):
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

        return

    def pred_samples(self,model_path, model_weight_path, img_root_path,image_name_list, clToInt_dict):
        model = load_model(model_path)
        model.load_weights(model_weight_path)

        num_of_rows = len(image_name_list)//4
        fig = plt.gcf()
        fig.set_size_inches(4 * 4, num_of_rows * 4)

        for i, img_path in enumerate(image_name_list):
            x_img = load_img(os.path.join(img_root_path,img_path), target_size=(224, 224))
            x = img_to_array(x_img)
            x = np.expand_dims(x, axis=0)
            result = model.predict(x)
            img_class = np.argmax(result[0])
            str_img_class = clToInt_dict[img_class]

            sp = plt.subplot(4, num_of_rows, i + 1)
            sp.axis('Off')  # Don't show axes (or gridlines)
            img = mpimg.imread(img_path)
            plt.imshow(img)
            plt.title(str_img_class)
        plt.show()

        return

sample = Sampling()
