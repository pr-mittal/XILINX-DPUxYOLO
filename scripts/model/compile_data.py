import xml.etree.ElementTree as ET
import os
import shutil
import random
from math import floor
pathlib = '/home/siliconmerc/git/dac_sdc_2023/scripts/model/dataset'
labels={"Motor Vehicle":0,"Non-motorized Vehicle":1,"Pedestrian":2,"Traffic Light-Red Light":3,"Traffic Light-Yellow Light":4,"Traffic Light-Green Light":5,"Traffic Light-Off":6}

def divide_dataset():
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
divide_dataset()

def create_labels():
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
create_labels()