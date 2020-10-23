import os

import numpy as np
import PIL
from PIL import Image

dir_input = 'dataset/DRIVE/training/1st_manual_raw/'
dir_output = 'dataset/DRIVE/training/1st_manual_real_resized/'
os.makedirs(dir_output, exist_ok=True)

def resize(dir_input,dir_output):
    count = 1
    for i in range(21, 41):
        print("doing "+str(i))

        if i<10:
            prefix = "0"+str(i)
        else:
            prefix = str(i)
        img_path = dir_input + prefix + "_manual1.gif"  # 拼出图片路径和文件名
        img = PIL.Image.open(img_path)  # 读入图片  [565,584]
        img = np.array(img) #[584,565]
        img2 = np.empty([512,512],dtype=int)
        x_len = img.shape[0] #584
        y_len = img.shape[1] #565
        # print(img[1,1])
        #every pix in new
        for j in range(512):
            for k in range(512):
                decision = 0
                #area of raw
                for x in range(int(x_len/512*j),int(x_len/512*(j+1))):
                    for y in range(int(y_len / 512 * k), int(y_len / 512 * (k + 1))):
                        # print(str(x)+" " +str(y))
                        if(img[x,y] == 255):
                            decision = 255
                        # print(str(int(568/512*j))+" "+str(int(568/512*(j+1)))+" "+str(img[int(568/512*j),100]) +" "+str(img[int(568/512*(j+1)),100]))
                img2[j,k] = decision
        img2 = Image.fromarray(img2)
        img2.save(dir_output + prefix + "_manual1.gif")
        # print(img2.shape)

        # for j in range(1, 30):
        #     img3, img4 = rotate_pair(img1, img)
        #     img3, img4 = flip_pair(img3, img4)
        #     img3, img4 = jitter_pair(img3, img4)
        #     img3.save(outpath_training+ str(count)+ '_training.tif')
        #     img4.save(outpath_manual+ str(count)+ '_manual1.gif')
        #     count = count + 1
        # # 生成图片先不归一化
        # print('done:' + str(i))  # 打印状态提示
        count = count + 1

resize(dir_input,dir_output)