import xml.etree.ElementTree as ET
import os
import shutil
import random
from math import floor
import numpy as np
import json

model="yolox"


if(model=="yolox"):
    pathlib="/home/siliconmerc/git/dac_sdc_2023/scripts/model2/code/datasets"
else:
    pathlib = '/home/siliconmerc/git/dac_sdc_2023/scripts/model/dataset'
labels={"Motor Vehicle":0,"Non-motorized Vehicle":1,"Pedestrian":2,"Traffic Light-Red Light":3,"Traffic Light-Yellow Light":4,"Traffic Light-Green Light":5,"Traffic Light-Off":6}

def divide_dataset(model="yolov3"):
    if(model=="yolox"):
        if(os.path.exists(pathlib+"/COCO")):
            shutil.rmtree(pathlib+"/COCO/")
        os.mkdir(pathlib+"/COCO")
        os.mkdir(pathlib+"/COCO/test2017")
        os.mkdir(pathlib+"/COCO/train2017")
        os.mkdir(pathlib+"/COCO/val2017")
        if(os.path.exists(pathlib+"/COCO/annotations")):
            shutil.rmtree(pathlib+"/COCO/annotations/")
        os.mkdir(pathlib+"/COCO/annotations")
        os.mkdir(pathlib+"/COCO/annotations/train2017")
        os.mkdir(pathlib+"/COCO/annotations/test2017")
        os.mkdir(pathlib+"/COCO/annotations/val2017")

        files=os.listdir(pathlib+"/images/JPEGImages")
        random.shuffle(files)
        data={"train2017":files[0:round(0.8*len(files))],"test2017":files[round(0.8*len(files)):round(0.9*len(files))],"val2017":files[round(0.9*len(files)):]}
        for key,val in data.items():
            if(os.path.exists(pathlib+"/"+key)):
                shutil.rmtree(pathlib+"/"+key)
            os.mkdir(pathlib+"/"+key)
            # os.mkdir(pathlib+"/"+key+"/images")
            os.mkdir(pathlib+"/"+key+"/Annotations")
            for file in val:
                shutil.copy(pathlib+"/images/JPEGImages/"+file, pathlib+"/COCO/"+key+"/"+file)
                # shutil.copy(pathlib+"/images/label/"+file.replace("jpg","json"), pathlib+"/"+key+"/Annotations/"+file.replace("jpg","json"))
                shutil.copy(pathlib+"/images/Annotations/"+file.replace("jpg","xml"), pathlib+"/COCO/annotations/"+key+"/"+file.replace("jpg","xml"))

    else:
        files=os.listdir(pathlib+"/images/JPEGImages")
        random.shuffle(files)
        data={"train":files[0:round(0.8*len(files))],"test":files[round(0.8*len(files)):round(0.9*len(files))],"val":files[round(0.9*len(files)):]}
        for key,val in data.items():
            if(os.path.exists(pathlib+"/"+key)):
                shutil.rmtree(pathlib+"/"+key)
            os.mkdir(pathlib+"/"+key)
            os.mkdir(pathlib+"/"+key+"/images")
            os.mkdir(pathlib+"/"+key+"/Annotations")
            for file in val:
                shutil.copy(pathlib+"/images/JPEGImages/"+file, pathlib+"/"+key+"/images/"+file)
                shutil.copy(pathlib+"/images/Annotations/"+file.replace("jpg","xml"), pathlib+"/"+key+"/Annotations/"+file.replace("jpg","xml"))

def create_labels(model="yolov3"):
    if(model=="yolox"): 
        if(not os.path.exists(pathlib+"/COCO/annotations")):
            return
        annotations={'info':{'description': 'COCO 2017 Dataset', 'url': 'http://cocodataset.org', 'version': '1.0', 'year': 2017, 'contributor': 'COCO Consortium', 'date_created': '2017/09/01'}
                ,'licenses':[], 'images':[], 'annotations':[], 'categories':[]}
        for key,val in labels.items():
            annotations['categories'].append({'supercategory': 'object', 'id': val+1, 'name': key})
        for folder in ["train2017","test2017","val2017"]:
            path=pathlib+"/COCO/annotations/"+folder
            label_id=1
            for filename in os.listdir(path):
                if not filename.endswith('.xml'): continue
                fullname = os.path.join(path,filename)
                tree = ET.parse(fullname)
                # print(tree)
                # <annotation>
                # <folder>GuoJiDaShuJu</folder>
                # <filename>00001.jpg</filename>
                # <size>
                #     <width>1920</width>
                #     <height>1080</height>
                #     <depth>3</depth>
                # </size>
                # <segmented>0</segmented>
                # <object>
                #     <name>Motor Vehicle</name>
                #     <pose>Unspecified</pose>
                #     <truncated>0</truncated>
                #     <difficult>0</difficult>
                #     <bndbox>
                #         <xmin>1168</xmin>
                #         <ymin>639</ymin>
                #         <xmax>1625</xmax>
                #         <ymax>884</ymax>
                #     </bndbox>
                # </object>
                # </annotation>
                id=int(filename.replace('.xml',''))
                root=tree.getroot()
                for child in root:
                    if(child.tag=="filename"):
                        jpgname=child.text
                    if(child.tag=="size"):
                        for dim in child:
                            if(dim.tag=="width"):
                                width=int(dim.text)
                            if(dim.tag=="height"):
                                height=int(dim.text)
                    if(child.tag=="object"):
                        class_obj=0
                        for prop in child:
                            if(prop.tag=="name"):
                                # print("|"+prop.text+"|")
                                class_obj=labels[prop.text]+1
                                # print(class_obj)
                            if(prop.tag=="bndbox"):
                                for dim in prop:
                                    if(dim.tag=="xmin"):
                                        xmin=int(dim.text)
                                    elif(dim.tag=="ymin"):
                                        ymin=int(dim.text)
                                    elif(dim.tag=="xmax"):
                                        xmax=int(dim.text)
                                    elif(dim.tag=="ymax"):
                                        ymax=int(dim.text)
                        annotations['annotations'].append({'segmentation': [[xmin,ymin,xmin,ymax,xmax,ymax,xmax,ymin]], 
                                                            'area': (xmax-xmin)*(ymax-ymin),
                                                            'iscrowd': 0,
                                                            'image_id': id, 'bbox': [xmin,ymin,(xmax-xmin),(ymax-ymin)], 'category_id': class_obj, 'id': label_id}) 
                        label_id=label_id+1   
                annotations['licenses'].append({'url': 'http://creativecommons.org/licenses/by-nc-sa/2.0/', 'id': id, 'name': 'Attribution-NonCommercial-ShareAlike License'})
                annotations['images'].append({'license': 4, 'file_name': jpgname, 'coco_url': 'na', 'height': height, 'width': width, 'date_captured': 'na', 'flickr_url': 'na', 'id': id})
            with open(os.path.join(pathlib+"/COCO/annotations","instances_"+folder+".json"), "w") as outfile:
                json.dump(annotations, outfile)
    else:
        for folder in ["train","test","val"]:
            path=pathlib+"/"+folder
            if(os.path.exists(path+"/labels")):
                shutil.rmtree(path+"/labels/")

            os.mkdir(path+"/labels") 
            for filename in os.listdir(path+"/Annotations"):
                if not filename.endswith('.xml'): continue
                fullname = os.path.join(path+"/Annotations", filename)
                tree = ET.parse(fullname)
                file = open(path+"/labels/"+filename[:-4]+".txt","w")
                # print(tree)
                # <annotation>
                # <folder>GuoJiDaShuJu</folder>
                # <filename>00001.jpg</filename>
                # <size>
                #     <width>1920</width>
                #     <height>1080</height>
                #     <depth>3</depth>
                # </size>
                # <segmented>0</segmented>
                # <object>
                #     <name>Motor Vehicle</name>
                #     <pose>Unspecified</pose>
                #     <truncated>0</truncated>
                #     <difficult>0</difficult>
                #     <bndbox>
                #         <xmin>1168</xmin>
                #         <ymin>639</ymin>
                #         <xmax>1625</xmax>
                #         <ymax>884</ymax>
                #     </bndbox>
                # </object>
                # </annotation>
                root=tree.getroot()
                width=1920
                height=1080
                for child in root:
                    if(child.tag=="size"):
                        for dim in child:
                            if(dim.tag=="width"):
                                width=1920
                            if(dim.tag=="height"):
                                height=1080
                    if(child.tag=="object"):
                        class_obj=0
                        for prop in child:
                            if(prop.tag=="name"):
                                # print("|"+prop.text+"|")
                                class_obj=labels[prop.text]
                                # print(class_obj)
                            if(prop.tag=="bndbox"):
                                for dim in prop:
                                    if(dim.tag=="xmin"):
                                        xmin=int(dim.text)
                                    elif(dim.tag=="ymin"):
                                        ymin=int(dim.text)
                                    elif(dim.tag=="xmax"):
                                        xmax=int(dim.text)
                                    elif(dim.tag=="ymax"):
                                        ymax=int(dim.text)
                        # class x_center y_center width height
                        file.write(f'{class_obj} {((xmin+xmax)/2)/width} {((ymin+ymax)/2)/height} {(xmax-xmin)/width} {(ymax-ymin)/height}\n')
                # print(tree[0])
                file.close()
                # break

if __name__ =="__main__":
    divide_dataset(model)
    create_labels(model)