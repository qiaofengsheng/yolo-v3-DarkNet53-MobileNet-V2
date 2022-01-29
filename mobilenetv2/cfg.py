
IMG_HEIGHT = 416
IMG_WIDTH = 416

CLASS_NUM = 3

ANCHORS_GROUP = {
    13: [[311, 247], [159, 232], [200, 117]],
    26: [[89, 159], [91, 74], [47, 97]],
    52: [[48, 34], [25, 55], [15, 21]]
}

ANCHORS_GROUP_AREA = {
    13: [x * y for x, y in ANCHORS_GROUP[13]],
    26: [x * y for x, y in ANCHORS_GROUP[26]],
    52: [x * y for x, y in ANCHORS_GROUP[52]],
}
