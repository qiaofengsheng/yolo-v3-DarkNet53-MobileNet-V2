
IMG_HEIGHT = 416
IMG_WIDTH = 416

CLASS_NUM = 3

ANCHORS_GROUP = {
    13: [[270, 254], [291, 179], [162, 304]],
    26: [[175, 222], [112, 235], [175, 140]],
    52: [[81, 118], [53, 142], [44, 28]]
}

ANCHORS_GROUP_AREA = {
    13: [x * y for x, y in ANCHORS_GROUP[13]],
    26: [x * y for x, y in ANCHORS_GROUP[26]],
    52: [x * y for x, y in ANCHORS_GROUP[52]],
}
