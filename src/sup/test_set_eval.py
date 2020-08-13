
import numpy as np
from .support import path_update
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def test_eval(model,classes):
    test_dir = '../../datasets/test/'
    #te_heart_dir, te_brain_dir, te_eye_dir, te_kidney_dir, te_skull_dir = path_update(test_dir,classes)
    test_gen_tmp = ImageDataGenerator(rescale=1/225.)
    test_gen = test_gen_tmp.flow_from_directory(test_dir,
                                                target_size=(300, 300),
                                                color_mode='rgb',
                                                class_mode='categorical',
                                                batch_size=100,
                                                shuffle=False,
                                                seed=42)


    STEP_SIZE_TEST = test_gen.n // test_gen.batch_size
    test_gen.reset()
    pred = model.predict_generator(test_gen,
                                   steps=STEP_SIZE_TEST,
                                   verbose=1)

    predicted_class_indices = np.argmax(pred, axis=1)

    labels = (test_gen.class_indices)
    class_to_int = dict((v, k) for k, v in labels.items())

    img_classes = test_gen.filenames()
    test_y = [class_to_int[j] for j in img_classes]

    return predicted_class_indices,test_y


