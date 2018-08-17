import os
import shutil

folder = "output/M_ff/ModelDINFFM_v3_r"
res_folder = "output/M_ff/test2"
os.mkdir(res_folder)

subfolders = [os.path.join(folder,sf) for sf in os.listdir(folder)]

for subfolder in subfolders:
    if os.path.isdir(subfolder):
        for fn in os.listdir(subfolder):
            if fn.find("his_test2")!=-1:
                src = os.path.join(subfolder,fn)
                des = os.path.join(res_folder,fn)
                shutil.move(src,des)
