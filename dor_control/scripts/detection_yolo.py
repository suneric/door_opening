#!/usr/bin/env python
# door handle detection with yolo and opencv
import cv2
import numpy as np
import os
import rospy
from math import ceil
from camera import RSD435

def get_output_layers(net):
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    return output_layers

def split_image(image):
    width = image.shape[1]
    height = image.shape[0]
    result = []
    if width > height:
        n_image = ceil(width/height*2)
        left = 0
        for i in range(int(n_image)):
            if left + height > width:
                left = width - height
            result.append((left, 0, height, height))
            left += int(height/2)
    else:
        n_image = ceil(height/width*2)
        top = 0
        for i in range(int(n_image)):
            if top + width > height:
                top = height - width
            result.append((0, top, width, width))
            top += int(width/2)
    return result

def detection_output(img,net):
    # return info of detection
    class_ids=[]
    confidences = []
    boxes = []

    W = img.shape[1]
    H = img.shape[0]

    # defect whole image
    scale = 0.00392 # 1/255
    blob = cv2.dnn.blobFromImage(img,scale,(416,416),(0,0,0),True,crop=False)
    net.setInput(blob)
    outs = net.forward(get_output_layers(net))
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * W)
                center_y = int(detection[1] * H)
                w = int(detection[2] * W)
                h = int(detection[3] * H)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x,y,w,h])

    # crop square subimage and do further detection
    sub_img_list = split_image(img)
    for s in sub_img_list:
        sub_img = img[s[1]:s[1]+s[3], s[0]:s[0]+s[2]]
        blob = cv2.dnn.blobFromImage(sub_img, scale, (416,416), (0,0,0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(get_output_layers(net))
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x = int(detection[0] * s[2]) + s[0]
                    center_y = int(detection[1] * s[3]) + s[1]
                    w = int(detection[2] * s[2])
                    h = int(detection[3] * s[3])
                    x = center_x - w / 2
                    y = center_y - h / 2
                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([x, y, w, h])

    return class_ids, confidences, boxes,


def detect_door_handle(sensor,net,classes,colors):
    img = sensor.color_image()
    if len(img) == 0:
        return

    H,W = sensor.image_size()

    class_ids, confidences, boxes = detection_output(img,net)
    #print(class_ids, confidences)
    conf_threshold = 0.5
    nms_threshold = 0.4
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    text_horizontal = 0
    for i in indices:
        i = i[0]
        box = boxes[i]
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        class_id = class_ids[i]
        label = str(classes[class_id])
        color = colors[class_id]
        cv2.rectangle(img, (x,y), (x+w,y+h), color, 2)
        cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        u = x+w/2
        v = y+h/2
        point3d = sensor.point3d(u,v)
        # information will be displayed on the frame
        info = [
            ("x", "{:.2f}".format(point3d[0])),
            ("y", "{:.2f}".format(point3d[1])),
            ("z", "{:.2f}".format(point3d[2]))
        ]
        # loop over the info tuples and draw them on our frame
        for (i, (k, v)) in enumerate(info):
            text = "{}: {}".format(k, v)
            cv2.putText(img, text, (10 + text_horizontal*100, H - ((i * 20) + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        text_horizontal += 1

    cv2.namedWindow("door handle measurement")
    img = cv2.resize(img, None, fx=1.0, fy=1.0)
    cv2.imshow('door handle measurement',img)
    cv2.waitKey(3)

if __name__ == '__main__':
    rospy.init_node("door_handle_detection", anonymous=True, log_level=rospy.INFO)
    sensor = RSD435()
    # load classfier
    dir = os.path.dirname(os.path.realpath(__file__))
    config = os.path.join(dir,'../classifier/yolo/yolo-door.cfg')
    weights = os.path.join(dir, '../classifier/yolo/yolo-obj.weights')
    # classes = os.path.join(dir,'../classifer/yolo/door.names')
    # load trained network model from weights
    net = cv2.dnn.readNet(weights,config)

    classes = ['door', 'handle', 'cabinet door', 'refrigerator door']
    colors = np.random.uniform(0, 255, size=(len(classes), 3))

    rate = rospy.Rate(10)
    try:
        while not rospy.is_shutdown():
            if sensor.ready():
                detect_door_handle(sensor,net,classes,colors)
            rate.sleep()
    except rospy.ROSInterruptException:
        pass
    cv2.destroyAllWindows()
