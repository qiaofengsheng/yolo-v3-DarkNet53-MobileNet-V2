import numpy as np
from PIL import Image,ImageDraw
import os
f=open('data.txt','r')
datas=f.readlines()
for data in datas:
    data=data.strip().split()
    img_path=os.path.join('images',data[0])
    img=Image.open(img_path)
    w,h=img.size
    case=416/max(w,h)
    _boxes=np.array([float(x) for x in data[1:]])
    boxes=np.split(_boxes,len(_boxes)//5)
    draw=ImageDraw.Draw(img)
    for box in boxes:
        cls,cx,cy,w,h=box
        x1,y1,x2,y2=cx/case-0.5*w/case,cy/case-0.5*h/case,cx/case+0.5*w/case,cy/case+0.5*h/case
        draw.rectangle((x1,y1,x2,y2),outline='red',width=2)

    img.show()