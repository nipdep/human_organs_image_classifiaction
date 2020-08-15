
import os
import cv2
import numpy

def img_resize(current_dir,save_dir,size_x=300,size_y=300,):
    img_list = os.listdir(current_dir)

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    #os.chdir(save_dir)

    for name in img_list:
        print(name)
        img = cv2.imread(os.path.join(current_dir,name),1)
        #cv2.imshow('',img)
        res = cv2.resize(img, dsize=(size_x, size_y), interpolation=cv2.INTER_CUBIC)


        ind = name.rfind('.')
        img_name,ext = name[:ind],name[ind+1:]
        if ext != 'jpg':
            name = img_name+'.jpg'

        cv2.imwrite(os.path.join(save_dir,name),res)


img_resize('../../datasets/training/Other','../../datasets/training/others')
