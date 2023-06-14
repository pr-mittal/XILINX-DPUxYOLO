import torch
import os
import sys
import shutil

# #ultralytics model
# import yolov5.models
# sys.path.insert(0, './yolov5')
# rootdir = 'yolov5/runs/train'
# str_exps=[x.replace("exp","").replace("exp","") for x in os.listdir(rootdir)]
# exps=map(lambda x: 0 if x=="" else int(x),str_exps)
# # model = torch.hub.load('ultralytics/yolov5', 'custom', path=rootdir+"/exp"+str(max(exps))+"/weights/best.pt", trust_repo=True,source='local')
# model=torch.load(rootdir+"/exp"+str(max(exps))+"/weights/best.pt")
# # print(model['model'].state_dict())
# # save the trained model
# float_model = './build/float_model'
# shutil.rmtree(float_model, ignore_errors=True)    
# os.makedirs(float_model)   
# save_path = os.path.join(float_model, 'f_model.pt')
# # torch.save(model['model'].state_dict(), save_path, _use_new_zipfile_serialization=False)
# torch.save(model, os.path.join(float_model,"f_model.pt"), _use_new_zipfile_serialization=False)
# print('Trained model written to',save_path)

# # ckpt = {'epoch': epoch,
# #                     'best_fitness': best_fitness,
# #                     'best_epoch': best_epoch,
# #                     'mAP': results[-1],
# #                     'fitness': fi,
# #                     'model': deepcopy(ori_model),
# #                     'ema': deepcopy(ori_model_ema),
# #                     'qat_model_quant_info': get_quant_info(deployema_path),
# #                     'qat_model_quant_info_test': get_quant_info(deployema_path +'/test'),
# #                     'qat_model_state_dict': model.state_dict(),
# #                     'qat_ema_state_dict': ema.ema.state_dict(),
# #                     'updates': ema.updates,
# #                     'optimizer': None if final_epoch else optimizer.state_dict()}

def compile_model(model="yolov3"):
    if(model=="yolov3"):
        from models.models import ofa_yolo_0
        pretrained_ofa_model =  "build/float_model/yolo_base_0.pth"
        with open(pretrained_ofa_model, 'rb') as f:
            checkpoint = torch.load(f, map_location='cpu')
        anchors_weight = checkpoint['anchors']
        # print(anchors_weight)
        # print(checkpoint.keys())
        for x in ['module_240.weight', 'module_240.bias', 'module_241.weight', 'module_241.bias', 'module_242.weight', 'module_242.bias']:
            # print("BEFORE ",checkpoint[x].size() , type(checkpoint[x]))
            if(x.find("weight")!=-1):
                checkpoint[x]=torch.rand(36,checkpoint[x].size()[1],checkpoint[x].size()[2],checkpoint[x].size()[3])
            else:
                checkpoint[x]=torch.rand(36)
            # print("AFTER ",checkpoint[x].size() , type(checkpoint[x]))
        torch.save(checkpoint,pretrained_ofa_model)

        # torch.Size([255, 192, 1, 1]) <class 'torch.Tensor'>
        # torch.Size([255]) <class 'torch.Tensor'>
        # torch.Size([255, 384, 1, 1]) <class 'torch.Tensor'>
        # torch.Size([255]) <class 'torch.Tensor'>
        # torch.Size([255, 768, 1, 1]) <class 'torch.Tensor'>
        # torch.Size([255]) <class 'torch.Tensor'>

        anchors = [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]]
        model = ofa_yolo_0(anchors,anchors_weight)
        # im = torch.zeros(1, 3, 640, 640)  # image size(1,3,320,192) BCHW iDetection
        # model(im)
        del checkpoint['anchors']
        model.model.load_state_dict(checkpoint, strict=True)
    if(model=="yolox"):
        # update
        pretrained_model =  "float/yolox.pth"
        with open(pretrained_model, 'rb') as f:
            checkpoint = torch.load(f, map_location='cpu')
        # checkpoint['start_epoch']=0
        # print(checkpoint['start_epoch'])

        import torch.nn as nn

        from yolox.models.yolox_deploy import YOLOX
        from yolox.models.yolo_pafpn_deploy import YOLOPAFPN
        from yolox.models.yolo_head_deploy import YOLOXHead

        def init_yolo(M):
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03
        in_channels = [256, 512, 1024]
        num_classes = 7
        depth = 0.33
        width = 0.50
        act = 'lrelu'
        backbone = YOLOPAFPN(depth, width, in_channels=in_channels, act=act)
        head = YOLOXHead(num_classes, width, in_channels=in_channels, act=act)
        model = YOLOX(backbone, head)

        model.apply(init_yolo)
        model.head.initialize_biases(1e-2)
        # print(model.state_dict())

        # #update
        # for keys in checkpoint['model'].keys():
        #     if(keys.split(".")[0]=="head"):
        #         checkpoint['model'][keys]=model.state_dict()[keys]
        torch.save(checkpoint,pretrained_model)
        
        # #test
        # model.load_state_dict(checkpoint["model"])

if __name__=="__main__":
    compile_model("yolox")