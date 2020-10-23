import os

# path = 'D:\Desktop\ECE SR\GANproject\DRIVE\combined'

path_training = "D:/Desktop/ECE SR/GANproject/augmented/training/"
path_manual = "D:/Desktop/ECE SR/GANproject/augmented/manual/"

# def iterbrowse(path):
#     for home, dirs, files in os.walk(path):
#         for filename in files:
#             yield os.path.join(home, filename)


for i,j,k in os.walk(path_training):
    for l in k:
        print(l)

