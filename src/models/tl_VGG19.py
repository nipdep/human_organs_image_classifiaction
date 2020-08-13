from tensorflow import keras
from datetime import datetime

from ..sup.evaluation import *
from ..sup.support import *
from ..sup.test_set_eval import *

model_name = ""

classes = ['heart','brain','eye','kidney','skull','other']
root_dir = '../../datasets/'
train_dir = os.path.join(root_dir,'train')
validation_dir = os.path.join(root_dir,'validation')
tr_heart_dir,tr_brain_dir,tr_eye_dir,tr_kidney_dir,tr_skull_dir = path_update(train_dir,classes)
vl_heart_dir,vl_brain_dir,vl_eye_dir,vl_kidney_dir,vl_skull_dir = path_update(validation_dir,classes)


plot_sample_of_img(4,4,os.listdir(tr_heart_dir)+os.listdir(tr_eye_dir))

train_gen_tmp = ImageDataGenerator(rescale=1./255,
      rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')

validation_gen_tmp = ImageDataGenerator(rescale=1/225.)

train_gen = train_gen_tmp.flow_from_directory(train_dir,
                                              target_size=(300,300),
                                              color_mode='rgb',
                                              class_mode='categorical',
                                              batch_size= 100,
                                              shuffle=True,
                                              seed=42)

validation_gen = validation_gen_tmp.flow_from_directory(validation_dir,
                                              target_size=(300,300),
                                              color_mode='rgb',
                                              class_mode='categorical',
                                              batch_size= 100,
                                              shuffle=True,
                                              seed=42)

STEP_SIZE_TRAIN=train_gen.n//train_gen.batch_size
STEP_SIZE_VALID=validation_gen.n//validation_gen.batch_size

clToInt_dict = train_gen.class_indices
clToInt_dict = dict((v,k) for v,k in clToInt_dict.items())

model = keras.models.Sequential()

model.compile()

# Define the Keras TensorBoard callback.
logdir="logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

history = model.fit()

#visualize_model(model,img_path)

acc_n_loss(history)

model.evaluate_generator(validation_gen,
                         steps=STEP_SIZE_VALID)

y_pred,y_test = test_eval(model,classes)
plot_confusion_metrix(y_test,y_pred,classes)
ROC_classes(6,y_test,y_pred,classes)

model_path,model_weight_path = save(model,datetime.now()+model_name)

#rnd_predict(model_path,model_weight_path,img_path,clToInt_dict)
