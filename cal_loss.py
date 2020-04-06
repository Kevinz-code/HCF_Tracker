from __future__ import print_function
import os
import xml.dom.minidom as XML


def get_bbox_classes(idx, path):
    xml_list = sorted(os.listdir(path))
    xml_path = os.path.join(path, xml_list[idx])
    DomTree = XML.parse(xml_path)
    Root = DomTree.documentElement

    obj_all = Root.getElementsByTagName("object")
    boxes = []
    classes = []
    for obj in obj_all:
        # get the classes
        obj_name = obj.getElementsByTagName('name')[0]
        one_class = obj_name.childNodes[0].data
        classes.append(one_class)

        # get the box
        one_box = []
        obj_bbox = 0
        for child in obj.childNodes:
            if child.nodeName == 'bndbox':
                obj_bbox = child
                break
        for i in range(1, 8, 2):
            dot = obj_bbox.childNodes[i].childNodes[0].data
            one_box.append(float(dot))
        boxes.append(one_box)

    boxes = boxes[0]  # [[x1, y1, x2, y2]]
    b = []
    for i in boxes:
        b.append(2*i)

    return b

# cal loss

acu_error = 0
with open("../KCF/results.txt", "r") as f:
    for id, line in enumerate(f.readlines(), 0):
        line = line.strip().split(',')
        cal_cx, cal_cy = list(map(int, line))

        x1, y1, x2, y2 = get_bbox_classes(idx=id, path="res/xml")
        real_cx, real_cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        error = abs(real_cx - cal_cx) + abs(real_cy - cal_cy)
        acu_error += error
        print(id, acu_error)
print(acu_error / 273)
'''

os.remove("res/groundtruth.txt")
# write groudtruth
with open("res/groundtruth.txt", "a") as f:
    for i in range(273):
        x1, y1, x2, y2 = get_bbox_classes(idx=i, path="res/xml")
        real_cx, real_cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        real_cx, real_cy = int(real_cx), int(real_cy)
        line = "%d,%d\n"%(real_cx, real_cy)
        f.writelines(line)


'''