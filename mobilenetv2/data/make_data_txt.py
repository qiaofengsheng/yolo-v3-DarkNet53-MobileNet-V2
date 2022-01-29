import math
import xml.etree.ElementTree as ET
import os
from PIL import Image
class_dict={
    'aeroplane': 0,
    'bicycle': 1,
    'bird': 2,
    'boat': 3,
    'bottle': 4,
    'bus': 5,
    'car': 6,
    'cat': 7,
    'chair': 8,
    'cow': 9,
    'diningtable': 10,
    'dog': 11,
    'horse': 12,
    'motorbike': 13,
    'person': 14,
    'pottedplant': 15,
    'sheep': 16,
    'sofa': 17,
    'train': 18,
    'tvmonitor': 19
}
train_xml_path=r'G:\data\voc\voc_train\VOC2007\Annotations'
train_img_path=r'G:\data\voc\voc_train\VOC2007\JPEGImages'
test_xml_path=r'G:\data\voc\voc_test\VOC2007\Annotations'
test_img_path=r'G:\data\voc\voc_test\VOC2007\JPEGImages'

xml_files=os.listdir(train_xml_path)
with open('train_data.txt','a') as f:
    for xml_file in xml_files:
        tree=ET.parse(os.path.join(train_xml_path,xml_file))
        root=tree.getroot()
        image_name=root.find('filename')
        class_name=root.findall('object/name')
        boxes=root.findall('object/bndbox')
        filename=image_name.text
        temp=max(Image.open(os.path.join(train_img_path,filename)).size)
        data=[]
        data.append(filename)
        for cls,box in zip(class_name,boxes):
            cls=class_dict[cls.text]
            cx,cy=math.floor((int(box[0].text)+int(box[2].text))/2),math.floor((int(box[1].text)+int(box[3].text))/2)
            w,h=(int(box[2].text)-int(box[0].text)),(int(box[3].text)-int(box[1].text))
            obj=f"{cls},{math.floor(cx*416/temp)},{math.floor(cy*416/temp)},{math.floor(w*416/temp)},{math.floor(h*416/temp)}"
            data.append(obj)
        str=''
        for i in data:
            str=str+i+','
        str=str.replace(',',' ').strip()
        f.write(str+'\n')



