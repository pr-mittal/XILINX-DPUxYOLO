import torch
import os
import sys
import shutil
import yolov5.models
sys.path.insert(0, './yolov5')
rootdir = 'yolov5/runs/train'
str_exps=[x.replace("exp","").replace("exp","") for x in os.listdir(rootdir)]
exps=map(lambda x: 0 if x=="" else int(x),str_exps)
# model = torch.hub.load('ultralytics/yolov5', 'custom', path=rootdir+"/exp"+str(max(exps))+"/weights/best.pt", trust_repo=True,source='local')
model=torch.load(rootdir+"/exp"+str(max(exps))+"/weights/best.pt")
# print(model['model'].state_dict())
# save the trained model
float_model = './build/float_model'
shutil.rmtree(float_model, ignore_errors=True)    
os.makedirs(float_model)   
save_path = os.path.join(float_model, 'f_model.pt')
# torch.save(model['model'].state_dict(), save_path, _use_new_zipfile_serialization=False)
torch.save(model, os.path.join(float_model,"f_model.pt"), _use_new_zipfile_serialization=False)
print('Trained model written to',save_path)