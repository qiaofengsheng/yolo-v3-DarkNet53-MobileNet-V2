import torch

from darbnet53_module import *
import cfg
from PIL import Image,ImageDraw
from darbnet53_module import *
from dataset import *
import os
class_dict={
    0:'person',
    1:'horse',
    2:'bicycle',
}
class Detector(torch.nn.Module):

    def __init__(self):
        super(Detector, self).__init__()

        self.net = Darknet53()
        self.net.load_state_dict(torch.load('darknet_params/net597.pt'))
        self.net.eval()

    def forward(self, input, thresh, anchors,case):
        output_13, output_26, output_52 = self.net(input)

        idxs_13, vecs_13 = self._filter(output_13, thresh)
        boxes_13 = self._parse(idxs_13, vecs_13, 32, anchors[13],case)

        idxs_26, vecs_26 = self._filter(output_26, thresh)
        boxes_26 = self._parse(idxs_26, vecs_26, 16, anchors[26],case)

        idxs_52, vecs_52 = self._filter(output_52, thresh)
        boxes_52 = self._parse(idxs_52, vecs_52, 8, anchors[52],case)
        boxes=torch.cat([boxes_13, boxes_26, boxes_52], dim=0)
        # rst=[]
        # for i in range(3):
        #     bs=boxes[boxes[...,6]==i]
        #     for j in bs
        #     bs = nms(bs, 0.9, mode="inter")
        #     rst.append(bs)
        boxes=nms(boxes, 0.5, mode='inter')
        return boxes


    def _filter(self, output, thresh):
        output = output.permute(0, 2, 3, 1)
        output = output.reshape(output.size(0), output.size(1), output.size(2), 3, -1)
        #mask:N,H,W,3,15

        mask = torch.sigmoid(output[..., 0]) > thresh

        idxs = mask.nonzero()
        vecs = output[mask]
        return idxs, vecs

    def _parse(self, idxs, vecs, t, anchors,case):
        anchors = torch.Tensor(anchors)

        n = idxs[:, 0]  # 所属的图片
        a = idxs[:, 3]  # 建议框

        cy = (idxs[:, 1].float() + vecs[:, 2]) * t /case # 原图的中心点y
        cx = (idxs[:, 2].float() + vecs[:, 1]) * t /case # 原图的中心点x

        w = anchors[a, 0] * torch.exp(vecs[:, 3])/case
        h = anchors[a, 1] * torch.exp(vecs[:, 4])/case
        p=vecs[:,0]
        cls_p=vecs[:,5:]
        cls_p=torch.softmax(cls_p,dim=1)
        cls_index = torch.argmax(cls_p, dim=1)
        return torch.stack([n.float(), torch.sigmoid(p),cx, cy, w, h,cls_index], dim=1)


if __name__ == '__main__':
    detector = Detector()
    # y = detector(torch.randn(3, 3, 416, 416), 0.3, cfg.ANCHORS_GROUP,0.5)
    # print(y.shape)
    for i in os.listdir('images'):
        img=Image.open('images/'+i)
        _img = make_image_data('images/'+i)
        w, h = _img.size[0], _img.size[1]
        case = 416 / w
        # print(case)
        _img = _img.resize((416, 416))  # 此处要等比缩放
        _img_data = transforms(_img)
        _img_data=torch.unsqueeze(_img_data,dim=0)
        # print(_img_data.shape)
        result=detector(_img_data, 0.2, cfg.ANCHORS_GROUP,case)
        draw=ImageDraw.Draw(img)
        for rst in result:
            if len(rst)==0:
                continue
            else:
                # rst=rst[0]
                x1,y1,x2,y2=rst[2]-0.5*rst[4],rst[3]-0.5*rst[5],rst[2]+0.5*rst[4],rst[3]+0.5*rst[5]
                print(f'置信度：{str(rst[1].item())[:4]} 坐标点：{x1,y1,x2,y2} 类别：{class_dict[int(rst[6].item())]}')
                draw.text((x1,y1),class_dict[int(rst[6].item())]+str(rst[1].item())[:4])
                draw.rectangle((x1,y1,x2,y2),width=1,outline='red')
        img.show()
