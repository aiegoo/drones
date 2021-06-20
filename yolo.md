[github](https://github.com/durner/yolo-autonomous-drone)

YOLO Autonomous Drone - Deep Learning Person Detection
===================

Table of Contents
__TOC__
1. [Requirements](#Requirements)
2. [yolo network](#YOLO-Network)
3. [Run the project](#run-the-project)
4. [Switching between interfaces](#switching-between-interfaces)
5. [Autonomous Interface](#autonomous-interface)
6. [Manual Interface](#manual-interface)
7. [Files](#files)
   + [drone.py](#dronepy)
   + [pid.py](#pidpy)
   + [yolo.py](#yolopy)
   + [actuator.py](#actuatorpy)
8. [Utils folder](#utils-folder)
   + [DarkNet.py](#darknetpy)
   + [MeasureAccuracy.py](#measureaccuracypy)
   + [pythonReader.py](#pythonreaderpy)
   + [pythonPascalVoc2.py](#pythonpascalvoc2py)
   + [ReadPascalforLoc.py](#readpascalforlocpy)
   + [TinyYoloNet.py](#tinyyolonetpy)
   + [TinyYoloNetforLoc.py](#tinyyolonetforlocpy)
   + [__init__.py](#--init--py)
   + [crop.py](#croppy)
   + [readImgFile.py](#readimgfilepy)
- [--------------------------------------------------------](#--------------------------------------------------------)
> 

The YOLO Drone localizes and follows people with the help of the YOLO Deep Network. Often, more than just one person might be in the picture of the drone’s camera so a standard deep learning people/body recognition cannot deliver sufficient results. This is why we chose the color of the shirt worn by the respective person to be a second criterion. Hence, we require the "operator" of the drone to wear a shirt with a distinct yellow color. This turns out to be a suitable solution to the aforementioned problem. 

## Requirements
To run this project Keras and Theano are needed for the deeplearning part. Furthermore, a working libardrone must be installed. For shirt detection opencv must be installed on the system.

> **Requirements list (+ all dependencies!) (python2.7):**
> - keras (http://www.keras.io)
> - theano (http://deeplearning.net/software/theano/)
> - libardrone (https://github.com/venthur/python-ardrone)
> - opencv (http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_tutorials.html)

## YOLO Network
For the YOLO network we tweaked the original implementation of https://github.com/sunshineatnoon/Darknet.keras. To run the Network with pretrained weights we suggest to use http://pjreddie.com/media/files/yolo-tiny.weights.

## Run the project
If you have all requirements as defined above you can simple run the project by entering:
```
$ python drone.py
```
This contains the main file of the drone. Please make sure that you have an active connection to the drone via wifi.

## Switching between interfaces
If you want to switch between autonomous and manual flight you can simply change the main definition of drone.py by flipping the manual argument
```
def main():
    drone = YOLODrone(manual=False)
    drone.start()
```

## Autonomous Interface

![Detection 1](pictures/detection_1.png?raw=true "Detection 1") ![Detection 2](pictures/detection_2.png?raw=true "Detection 2")

As already described, the drone is looking for persons. The interface marks persons / groups of persons with red boxes. Additionally, a yellow t-shirt determines the real operator of the drone which is also highlighted in the interface. If more than one person wears a yellow shirt in the picture, the drone chooses the red box (person) that has the highest amount of yellow in them and continues to follow this particular person.

## Manual Interface
If you don't press any key the drone will hover at its position. Use following keys to control the drone.

Key     | Function
------- | ------- 
t       | takeoff
(space) | land
w       | move forward
s       | move backward
d       | move right
a       | move left
8       | move up
2       | move down
e       | turn right
q       | turn left
c       | stop flight

## Contributers
 - [Dominik Durner](https://github.com/durner)
 - [Christopher Helm](https://github.com/chrishelm)

## Upstream Repository


## Files
- drone.py : Main file of the project. Includes the manual interface, the glue code to the autonomous interface between YOLO Network and Actuators. All multithreading and OpenCV pre-processing is handled.
- PID.py : simple PID controller interface to easily control the movements of the drone (incl. smoothing of the movements).
- YOLO.py : Set up of the YOLO Deep network in python. The subfolder utils include further needed files for the YOLO net.
- actuators.py : With the help of the localized operator the actuators calculate how the drone needs to move to center the operator and follow him. Uses PID controllers for calculating the movements.

## drone.py

```python
import os
import libardrone.libardrone as libardrone
import time
from threading import Thread, Lock, Condition
import cv2
import numpy
import keras
from YOLO import SimpleNet, convert_yolo_detections, do_nms_sort
from actuators import Actuator
from utils.TinyYoloNet import ReadTinyYOLONetWeights


class YOLODrone(object):
    def __init__(self, manual=True):
        self.key = None
        self.stop = False
        self.mutex = None
        self.manuel = manual
        self.PID = None
        self.boxes = None
        self.condition = Condition()
        self.update = False
        self.contours = None
        self.boxes_update = False
        self.image = None

        self.labels = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
                  "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
        yoloNet = ReadTinyYOLONetWeights(os.path.join(os.getcwd(), 'weights/yolo-tiny.weights'))
        # reshape weights in every layer
        for i in range(yoloNet.layer_number):
            l = yoloNet.layers[i]
            if (l.type == 'CONVOLUTIONAL'):
                weight_array = l.weights
                n = weight_array.shape[0]
                weight_array = weight_array.reshape((n // (l.size * l.size), (l.size * l.size)))[:, ::-1].reshape((n,))
                weight_array = numpy.reshape(weight_array, [l.n, l.c, l.size, l.size])
                l.weights = weight_array
            if (l.type == 'CONNECTED'):
                weight_array = l.weights
                weight_array = numpy.reshape(weight_array, [l.input_size, l.output_size])
                l.weights = weight_array

        self.model = SimpleNet(yoloNet)
        sgd = keras.optimizers.SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
        self.model.compile(optimizer=sgd, loss='categorical_crossentropy')

    def start(self):
        self.drone = libardrone.ARDrone(True)
        self.drone.reset()

        if self.manuel:
            try:
                self.mutex = Lock()
                t1 = Thread(target=self.getKeyInput, args=())
                t2 = Thread(target=self.getVideoStream, args=())
                t3 = Thread(target=self.getBoundingBoxes, args=())
                t1.start()
                t2.start()
                t3.start()
                t1.join()
                t2.join()
                t3.join()
            except:
                print "Error: unable to start thread"
        else:
            try:
                self.mutex = Lock()
                t1 = Thread(target=self.autonomousFlight, args=(448, 448, 98, 0.1, self.labels,))
                t2 = Thread(target=self.getVideoStream, args=())
                t3 = Thread(target=self.getBoundingBoxes, args=())
                t1.start()
                t2.start()
                t3.start()
                t1.join()
                t2.join()
                t3.join()
            except:
                print "Error: unable to start thread"


        print("Shutting down...")
        cv2.destroyAllWindows()
        self.drone.land()
        time.sleep(0.1)
        self.drone.halt()
        print("Ok.")


    def getKeyInput(self):
        while not self.stop:  # while 'bedingung true'
            time.sleep(0.1)


            if self.key == "t":  # if 'bedingung true'
                self.drone.takeoff()
            elif self.key == " ":
                self.drone.land()
            elif self.key == "0":
                self.drone.hover()
            elif self.key == "w":
                self.drone.move_forward()
            elif self.key == "s":
                self.drone.move_backward()
            elif self.key == "a":
                self.drone.move_left()
            elif self.key == "d":
                self.drone.move_right()
            elif self.key == "q":
                self.drone.turn_left()
            elif self.key == "e":
                self.drone.turn_right()
            elif self.key == "8":
                self.drone.move_up()
            elif self.key == "2":
                self.drone.move_down()
            elif self.key == "c":
                self.stop = True
            else:
                self.drone.hover()

            if self.key != " ":
                self.key = ""

    def getVideoStream(self, img_width=448, img_height=448):
        while not self.stop:
            img = self.image
            if img != None:
                nav_data = self.drone.get_navdata()
                nav_data = nav_data[0]
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_size = 0.5

                cv2.putText(img, 'Altitude: %.0f' % nav_data['altitude'], (5, 15), font, font_size, (255, 255, 255))
                cv2.putText(img, 'Battery: %.0f%%' % nav_data['battery'], (5, 30), font, font_size, (255, 255, 255))

                cv2.drawContours(img, self.contours, -1, (0, 255, 0), 3)
                thresh = 0.2
                self.mutex.acquire()
                if self.boxes_update:
                    self.boxes_update = False
                    for b in self.boxes:
                        max_class = numpy.argmax(b.probs)
                        prob = b.probs[max_class]
                        if (prob > thresh and self.labels[max_class] == "person"):
                            left = (b.x - b.w / 2.) * img_width
                            right = (b.x + b.w / 2.) * img_width

                            top = (b.y - b.h / 2.) * img_height
                            bot = (b.y + b.h / 2.) * img_height

                            cv2.rectangle(img, (int(left), int(top)), (int(right), int(bot)), (0, 0, 255), 3)
                self.mutex.release()
                cv2.imshow('frame', img)

                l = cv2.waitKey(150)
                if l < 0:
                    self.key = ""
                else:
                    self.key = chr(l)
                    if self.key == "c":
                        self.stop = True

    def variance_of_laplacian(self, image):
        # compute the Laplacian of the image and then return the focus
        #  measure, which is simply the variance of the Laplacian
        return cv2.Laplacian(image, cv2.CV_64F).var()

    def getBoundingBoxes(self):
        newest = time.time()
        while not self.stop:
            try:
                pixelarray = self.drone.get_image()
                pixelarray = cv2.cvtColor(pixelarray, cv2.COLOR_BGR2RGB)

                # Check for Blurry
                gray = cv2.cvtColor(pixelarray, cv2.COLOR_RGB2GRAY)
                fm = self.variance_of_laplacian(gray)
                if fm < 10:
                    continue

                if pixelarray != None:
                    # ima = pixelarray[120:540]
                    ima = cv2.resize(pixelarray, (448, 448))

                    image = cv2.cvtColor(ima, cv2.COLOR_RGB2BGR)

                    image = numpy.rollaxis(image, 2, 0)
                    image = image / 255.0
                    image = image * 2.0 - 1.0
                    image = numpy.expand_dims(image, axis=0)

                    out = self.model.predict(image)
                    predictions = out[0]
                    boxes = convert_yolo_detections(predictions)

                    self.mutex.acquire()
                    self.boxes = do_nms_sort(boxes, 98)
                    self.image = ima
                    self.update = True
                    self.mutex.release()

            except:
                pass

    def autonomousFlight(self, img_width, img_height, num, thresh, labels):
        actuator = Actuator(self.drone, img_width, img_width * 0.5)

        print self.drone.navdata
        while not self.stop:
            if self.update == True:
                self.update = False

                hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
                image = cv2.medianBlur(hsv, 3)

                # Filter by color red
                lower_red_1 = numpy.array([15, 150, 150])
                upper_red_1 = numpy.array([35, 255, 255])

                image = cv2.inRange(image, lower_red_1, upper_red_1)

                # Put on median blur to reduce noise
                image = cv2.medianBlur(image, 11)

                # Find contours and decide if hat is one of them
                contours, hierarchy = cv2.findContours(image.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

                self.contours = contours

                boxes = self.boxes

                best_prob = -99999
                best_box = -1
                best_contour = None

                self.mutex.acquire()
                for i in range(num):
                    # for each box, find the class with maximum prob
                    max_class = numpy.argmax(boxes[i].probs)
                    prob = boxes[i].probs[max_class]

                    temp = boxes[i].w
                    boxes[i].w = boxes[i].h
                    boxes[i].h = temp

                    if prob > thresh and labels[max_class] == "person":

                        for contour in contours:
                            x, y, w, h = cv2.boundingRect(contour)

                            left = (boxes[i].x - boxes[i].w / 2.) * img_width
                            right = (boxes[i].x + boxes[i].w / 2.) * img_width

                            top = (boxes[i].y - boxes[i].h / 2.) * img_height
                            bot = (boxes[i].y + boxes[i].h / 2.) * img_height

                            if not (x + w < left or right < x or y + h < top or bot < y):
                               if best_prob < prob and w > 30:
                                    print "prob found"
                                    best_prob = prob
                                    best_box = i
                                    best_contour = contour

                self.boxes_update = True
                if best_box < 0:
                    # print "No Update"
                    self.mutex.release()
                    self.drone.at(libardrone.at_pcmd, False, 0, 0, 0, 0)
                    continue

                b = boxes[best_box]

                left = (b.x - b.w / 2.) * img_width
                right = (b.x + b.w / 2.) * img_width

                top = (b.y - b.h / 2.) * img_height
                bot = (b.y + b.h / 2.) * img_height


                if (left < 0): left = 0;
                if (right > img_width - 1): right = img_width - 1;
                if (top < 0): top = 0;
                if (bot > img_height - 1): bot = img_height - 1;

                width = right - left
                height = bot - top
                x, y, w, h = cv2.boundingRect(best_contour)

                actuator.step(right - width/2., width)
                self.mutex.release()


def main():
    drone = YOLODrone(manual=False)
    drone.start()


if __name__ == '__main__':
    main()
```




## pid.py

```python
class PID(object):
    def __init__(self, K_p=0.4, K_d=0.00, K_i=0.00, dt=0.5):
        self.K_p = K_p
        self.K_d = K_d
        self.K_i = K_i
        self.dt = dt
        self.w = 0
        self.velocity = 0
        self.errorsum = 0
        self.actual_previous = 0

    def step(self, desired, actual):
        self.errorsum = (desired - actual) * self.dt
        self.velocity = (actual - self.actual_previous) / self.dt
        u = self.K_p * (desired - actual) + self.K_d * (self.w - self.velocity) + self.K_i * self.errorsum
        self.actual_previous = actual
        return u
```

## yolo.py

```python
import os
import numpy as np

from keras.models import Sequential
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.core import Flatten, Dense, Activation, Reshape

from math import pow
import theano

from PIL import Image
from PIL import ImageDraw


class box:
    def __init__(self,classes):
        self.x = 0
        self.y = 0
        self.h = 0
        self.w = 0
        self.class_num = 0
        self.probs = np.zeros((classes,1))

def SimpleNet(yoloNet):
    model = Sequential()

    #Convolution Layer 2 & Max Pooling Layer 3
    model.add(ZeroPadding2D(padding=(1,1),input_shape=(3,448,448)))
    model.add(Convolution2D(16, 3, 3, weights=[yoloNet.layers[1].weights,yoloNet.layers[1].biases],border_mode='valid',subsample=(1,1)))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    #Use a for loop to replace all manually defined layers
    for i in range(3,yoloNet.layer_number):
        l = yoloNet.layers[i]
        if(l.type == "CONVOLUTIONAL"):
            model.add(ZeroPadding2D(padding=(l.size//2,l.size//2,)))
            model.add(Convolution2D(l.n, l.size, l.size, weights=[l.weights,l.biases],border_mode='valid',subsample=(1,1)))
            model.add(LeakyReLU(alpha=0.1))
        elif(l.type == "MAXPOOL"):
            model.add(MaxPooling2D(pool_size=(2, 2),border_mode='valid'))
        elif(l.type == "FLATTEN"):
            model.add(Flatten())
        elif(l.type == "CONNECTED"):
            model.add(Dense(l.output_size, weights=[l.weights,l.biases]))
        elif(l.type == "LEAKY"):
            model.add(LeakyReLU(alpha=0.1))
        elif(l.type == "DROPOUT"):
            pass
        else:
            print "Error: Unknown Layer Type",l.type
    return model

def get_activations(model, layer, X_batch):
    get_activations = theano.function([model.layers[0].input], model.layers[layer].get_output(train=False), allow_input_downcast=True)
    activations = get_activations(X_batch) # same result as above
    return activations

def convert_yolo_detections(predictions,classes=20,num=2,square=True,side=7,w=1,h=1,threshold=0.2,only_objectness=0):
    boxes = []
    probs = np.zeros((side*side*num,classes))
    for i in range(side*side):
        row = i / side
        col = i % side
        for n in range(num):
            index = i*num+n
            p_index = side*side*classes+i*num+n
            scale = predictions[p_index]
            box_index = side*side*(classes+num) + (i*num+n)*4

            new_box = box(classes)
            new_box.x = (predictions[box_index + 0] + col) / side * w
            new_box.y = (predictions[box_index + 1] + row) / side * h
            new_box.h = pow(predictions[box_index + 2], 2) * w
            new_box.w = pow(predictions[box_index + 3], 2) * h

            for j in range(classes):
                class_index = i*classes
                prob = scale*predictions[class_index+j]
                if(prob > threshold):
                    new_box.probs[j] = prob
                else:
                    new_box.probs[j] = 0
            if(only_objectness):
                new_box.probs[0] = scale

            boxes.append(new_box)
    return boxes

def prob_compare(boxa,boxb):
    if(boxa.probs[boxa.class_num] < boxb.probs[boxb.class_num]):
        return 1
    elif(boxa.probs[boxa.class_num] == boxb.probs[boxb.class_num]):
        return 0
    else:
        return -1

def do_nms_sort(boxes,total,classes=20,thresh=0.5):
    for k in range(classes):
        for box in boxes:
            box.class_num = k
        sorted_boxes = sorted(boxes,cmp=prob_compare)
        for i in range(total):
            if(sorted_boxes[i].probs[k] == 0):
                continue
            boxa = sorted_boxes[i]
            for j in range(i+1,total):
                boxb = sorted_boxes[j]
                if(boxb.probs[k] != 0 and box_iou(boxa,boxb) > thresh):
                    boxb.probs[k] = 0
                    sorted_boxes[j] = boxb
    return sorted_boxes

def overlap(x1,w1,x2,w2):
    l1 = x1 - w1/2;
    l2 = x2 - w2/2;
    if(l1 > l2):
        left = l1
    else:
        left = l2
    r1 = x1 + w1/2;
    r2 = x2 + w2/2;
    if(r1 < r2):
        right = r1
    else:
        right = r2
    return right - left;

def box_intersection(a, b):
    w = overlap(a.x, a.w, b.x, b.w);
    h = overlap(a.y, a.h, b.y, b.h);
    if(w < 0 or h < 0):
         return 0;
    area = w*h;
    return area;

def box_union(a, b):
    i = box_intersection(a, b);
    u = a.w*a.h + b.w*b.h - i;
    return u;

def box_iou(a, b):
    return box_intersection(a, b)/box_union(a, b);

def draw_detections(impath,num,thresh,boxes,classes,labels,save_name):
    """
    Args:
        impath: The image path
        num: total number of bounding boxes
        thresh: boxes prob beyond this thresh will be drawn
        boxes: boxes predicted by the network
        classes: class numbers of the objects
    """
    img = Image.open(impath)
    drawable = ImageDraw.Draw(img)
    ImageSize = img.size
    for i in range(num):
        #for each box, find the class with maximum prob
        max_class = np.argmax(boxes[i].probs)
        prob = boxes[i].probs[max_class]
        if(prob > thresh and labels[max_class] == "person"):
            b = boxes[i]

            temp = b.w
            b.w = b.h
            b.h = temp

            left  = (b.x-b.w/2.)*ImageSize[0];
            right = (b.x+b.w/2.)*ImageSize[0];
            top   = (b.y-b.h/2.)*ImageSize[1];
            bot   = (b.y+b.h/2.)*ImageSize[1];

            if(left < 0): left = 0;
            if(right > ImageSize[0]-1): right = ImageSize[0]-1;
            if(top < 0): top = 0;
            if(bot > ImageSize[1]-1): bot = ImageSize[1]-1;

            # print "The four cords are: ",left,right,top,bot
            drawable.rectangle([left,top,right,bot],outline="red")
            img.save("results/" + save_name)
            # print labels[max_class],": ",boxes[i].probs[max_class]
```


## actuator.py

```python
import time
from libardrone import libardrone

from PID import PID

class Actuator(object):
    def __init__(self, drone, picture_width, desired_move):
        self.turn = PID(K_p=0.6, K_d=0.1)
        self.move = PID(K_p=0.15, K_d=0.01)
        self.height = PID(K_p=0.2, K_d=0.00)
        self.picture_width = picture_width
        self.desired_move = desired_move
        self.drone = drone
        time.sleep(0.05)
        self.drone.takeoff()
        time.sleep(0.05)

    def step(self, wdithmid, width):
        desired_turn = self.picture_width / 2
        actual_turn = wdithmid
        actual_move = width

        ut = self.turn.step(desired_turn, actual_turn)

        um = self.move.step(self.desired_move, actual_move)

        height = 550
        nav_data = self.drone.get_navdata()
        nav_data = nav_data[0]
        uh = self.height.step(height, nav_data['altitude'])

        self.drone.at(libardrone.at_pcmd, True, 0, self.moveDrone(um), self.heightDrone(uh), self.turnDrone(ut))

    def turnDrone(self, u):
        speed = - u / (self.picture_width / 2.)
        print "move horizontal to" + str(speed)
        return speed

    def moveDrone(self, u):
        speed = - u / (self.picture_width / 2.)
        print "move near to" + str(speed)
        return speed

    def heightDrone(self, u):
        speed = u / 500
        print "height near to" + str(speed)
        return speed

```

### Utils folder
- DarkNet.py
- pythonReader.py
- pythonPascalVoc2.py
- ReadPascalforLoc.py
- TinyYoloNet.py
- TinyYoloNetforLoc.py
- __init__.py
- crop.py
- readImgFile.py
- timer.py

## DarkNet.py

```python
import numpy as np
from enum import Enum
import os

class layer:
    def __init__(self,size,c,n,h,w,type):
        self.size = size
        self.c = c
        self.n = n
        self.h = h
        self.w = w
        self.type = type

class convolutional_layer(layer):
    def __init__(self,size,c,n,h,w):
        layer.__init__(self,size,c,n,h,w,"CONVOLUTIONAL")
        self.biases = np.zeros(n)
        self.weights = np.zeros((size*size,c,n))

class connected_layer(layer):
    def __init__(self,size,c,n,h,w,input_size,output_size):
        layer.__init__(self,size,c,n,h,w,"CONNECTED")
        self.output_size = output_size
        self.input_size = input_size

class DarkNet:
    layers = []
    layer_number = 18
    def __init__(self):
        self.layers.append(layer(0,0,0,0,0,"CROP"))
        self.layers.append(convolutional_layer(3,3,16,224,224))
        self.layers.append(layer(0,0,0,0,0,"MAXPOOL"))
        self.layers.append(convolutional_layer(3,16,32,112,112))
        self.layers.append(layer(0,0,0,0,0,"MAXPOOL"))
        self.layers.append(convolutional_layer(3,32,64,56,56))
        self.layers.append(layer(0,0,0,0,0,"MAXPOOL"))
        self.layers.append(convolutional_layer(3,64,128,28,28))
        self.layers.append(layer(0,0,0,0,0,"MAXPOOL"))
        self.layers.append(convolutional_layer(3,128,256,14,14))
        self.layers.append(layer(0,0,0,0,0,"MAXPOOL"))
        self.layers.append(convolutional_layer(3,256,512,7,7))
        self.layers.append(layer(0,0,0,0,0,"MAXPOOL"))
        self.layers.append(convolutional_layer(3,512,1024,4,4))
        self.layers.append(layer(0,0,0,0,0,"AVGPOOL"))
        self.layers.append(connected_layer(0,0,0,0,0,1024,1000))
        self.layers.append(layer(0,0,0,0,0,"SOFTMAX"))
        self.layers.append(layer(0,0,0,0,0,"COST"))

def ReadDarkNetWeights(weight_path):
    darkNet = DarkNet()
    type_string = "(3)float32,i4,"
    for i in range(darkNet.layer_number):
        l = darkNet.layers[i]
        if(l.type == "CONVOLUTIONAL"):
            bias_number = l.n
            weight_number = l.n*l.c*l.size*l.size
            type_string = type_string +"("+ str(bias_number) + ")float32,(" + str(weight_number) + ")float32,"
        elif(l.type == "CONNECTED"):
             bias_number = l.output_size
             weight_number = l.output_size * l.input_size
             type_string = type_string + "("+ str(bias_number) + ")float32,("+ str(weight_number)+")float32"
    #dt = np.dtype((+str(64)+")float32"))
    #type_string = type_string + ",i1"
    dt = np.dtype(type_string)
    testArray = np.fromfile(weight_path,dtype=dt)
    #write the weights read from file to GoogleNet biases and weights

    count = 2
    for i in range(0,darkNet.layer_number):
        l = darkNet.layers[i]
        if(l.type == "CONVOLUTIONAL" or l.type == "CONNECTED"):
            l.biases = np.asarray(testArray[0][count])
            count = count + 1
            l.weights = np.asarray(testArray[0][count])
            count = count + 1
            darkNet.layers[i] = l
            if(l.type == 'CONNECTED'):
                weight_array = l.weights
                weight_array = np.reshape(weight_array,[l.input_size,l.output_size])
                weight_array = weight_array.transpose()
            #print i,count

    #write back to file and see if it is the same

    write_fp = open('reconstruct.weights','w')
    write_fp.write((np.asarray(testArray[0][0])).tobytes())
    write_fp.write((np.asarray(testArray[0][1])).tobytes())
    for i in range(0,darkNet.layer_number):
        l = darkNet.layers[i]
        if(l.type == "CONVOLUTIONAL" or l.type == "CONNECTED"):
            write_fp.write(l.biases.tobytes())
            write_fp.write(l.weights.tobytes())


    write_fp.close()

    return darkNet

if __name__ == '__main__':
    darkNet = ReadDarkNetWeights('/home/xuetingli/Documents/YOLO.keras/weights/darknet.weights')
    for i in range(darkNet.layer_number):
        l = darkNet.layers[i]
        print l.type
        if(l.type == 'CONNECTED'):
            print l.weights.shape
```

## MeasureAccuracy.py

```python
from keras.models import model_from_json
import theano.tensor as T

from utils.readImgFile import readImg
from utils.crop import crop_detection
from utils.ReadPascalVoc2 import prepareBatch

import os
import numpy as np

def Acc(imageList,model,sample_number=5000,thresh=0.3):
    correct = 0
    object_num = 0

    count = 0
    for image in imageList:
	count += 1
        #Get prediction from neural network
        img = crop_detection(image.imgPath,new_width=448,new_height=448)
        img = np.expand_dims(img, axis=0)
        out = model.predict(img)
        out = out[0]

        for i in range(49):
            preds = out[i*25:(i+1)*25]
            if(preds[24] > thresh):
                object_num += 1
            	row = i/7
            	col = i%7
                '''
            	centerx = 64 * col + 64 * preds[0]
            	centery = 64 * row + 64 * preds[1]

            	h = preds[2] * preds[2]
            	h = h * 448.0
            	w = preds[3] * preds[3]
            	w = w * 448.0

            	left = centerx - w/2.0
            	right = centerx + w/2.0
            	up = centery - h/2.0
            	down = centery + h/2.0

            	if(left < 0): left = 0
            	if(right > 448): right = 447
            	if(up < 0): up = 0
            	if(down > 448): down = 447
                '''
            	class_num = np.argmax(preds[4:24])

                #Ground Truth
                box = image.boxes[row][col]
                if(box.has_obj):
                    for obj in box.objs:
                        true_class = obj.class_num
                        if(true_class == class_num):
                            correct += 1
	                    break


    return correct*1.0/object_num

def Recall(imageList,model,sample_number=5000,thresh=0.3):
    correct = 0
    obj_num = 0
    count = 0
    for image in imageList:
        count += 1
        #Get prediction from neural network
        img = crop_detection(image.imgPath,new_width=448,new_height=448)
        img = np.expand_dims(img, axis=0)
        out = model.predict(img)
        out = out[0]
        #for each ground truth, see we have predicted a corresponding result
        for i in range(49):
            preds = out[i*25:i*25+25]
            row = i/7
            col = i%7
            box = image.boxes[row][col]
            if(box.has_obj):
                for obj in box.objs:
                    obj_num += 1
                    true_class = obj.class_num
                    #see what we predict
                    if(preds[24] > thresh):
                        predcit_class = np.argmax(preds[4:24])
                        if(predcit_class == true_class):
                            correct += 1
    return correct*1.0/obj_num

def MeasureAcc(model,sample_number,vocPath,imageNameFile):
    imageList = prepareBatch(0,sample_number,imageNameFile,vocPath)
    acc = Acc(imageList,model)
    re = Recall(imageList,model)

    return acc,re
```

## pythonReader.py

```python
import numpy as np
from enum import Enum
import os

class layer:
    def __init__(self,size,c,n,h,w,type):
        self.size = size
        self.c = c
        self.n = n
        self.h = h
        self.w = w
        self.type = type

class convolutional_layer(layer):
    def __init__(self,size,c,n,h,w):
        layer.__init__(self,size,c,n,h,w,"CONVOLUTIONAL")
        self.biases = np.zeros(n)
        self.weights = np.zeros((size*size,c,n))

class connected_layer(layer):
    def __init__(self,size,c,n,h,w,input_size,output_size):
        layer.__init__(self,size,c,n,h,w,"CONNECTED")
        self.output_size = output_size
        self.input_size = input_size

class GoogleNet:
    layers = []
    layer_number = 29
    def __init__(self):
        self.layers.append(layer(0,0,0,0,0,"CROP"))
        self.layers.append(convolutional_layer(7,3,64,224,224))
        self.layers.append(layer(0,0,0,0,0,"MAXPOOL"))
        self.layers.append(convolutional_layer(3,64,192,56,56))
        self.layers.append(layer(0,0,0,0,0,"MAXPOOL"))
        self.layers.append(convolutional_layer(1,192,128,28,28))
        self.layers.append(convolutional_layer(3,128,256,28,28))
        self.layers.append(convolutional_layer(1,256,256,28,28))
        self.layers.append(convolutional_layer(3,256,512,28,28))
        self.layers.append(layer(0,0,0,0,0,"MAXPOOL"))
        self.layers.append(convolutional_layer(1,512,256,14,14))
        self.layers.append(convolutional_layer(3,256,512,14,14))
        self.layers.append(convolutional_layer(1,512,256,14,14))
        self.layers.append(convolutional_layer(3,256,512,14,14))
        self.layers.append(convolutional_layer(1,512,256,14,14))
        self.layers.append(convolutional_layer(3,256,512,14,14))
        self.layers.append(convolutional_layer(1,512,256,14,14))
        self.layers.append(convolutional_layer(3,256,512,14,14))
        self.layers.append(convolutional_layer(1,512,512,14,14))
        self.layers.append(convolutional_layer(3,512,1024,14,14))
        self.layers.append(layer(0,0,0,0,0,"MAXPOOL"))
        self.layers.append(convolutional_layer(1,1024,512,7,7))
        self.layers.append(convolutional_layer(3,512,1024,7,7))
        self.layers.append(convolutional_layer(1,1024,512,7,7))
        self.layers.append(convolutional_layer(3,512,1024,7,7))
        self.layers.append(layer(0,0,0,0,0,"AVGPOOL"))
        self.layers.append(connected_layer(0,0,0,0,0,1024,1000))
        self.layers.append(layer(0,0,0,0,0,"SOFTMAX"))
        self.layers.append(layer(0,0,0,0,0,"COST"))

def ReadGoogleNetWeights(weight_path):
    googleNet = GoogleNet()
    type_string = "(3)float32,i4,"
    for i in range(googleNet.layer_number):
        l = googleNet.layers[i]
        if(l.type == "CONVOLUTIONAL"):
            bias_number = l.n
            weight_number = l.n*l.c*l.size*l.size
            type_string = type_string +"("+ str(bias_number) + ")float32,(" + str(weight_number) + ")float32,"
        elif(l.type == "CONNECTED"):
             bias_number = l.output_size
             weight_number = l.output_size * l.input_size
             type_string = type_string + "("+ str(bias_number) + ")float32,("+ str(weight_number)+")float32"
    #dt = np.dtype((+str(64)+")float32"))
    #type_string = type_string + ",i1"
    dt = np.dtype(type_string)
    testArray = np.fromfile(weight_path,dtype=dt)
    #print len(testArray[0])
    #write the weights read from file to GoogleNet biases and weights

    count = 2
    for i in range(0,googleNet.layer_number):
        l = googleNet.layers[i]
        if(l.type == "CONVOLUTIONAL" or l.type == "CONNECTED"):
            l.biases = np.asarray(testArray[0][count])
            count = count + 1
            l.weights = np.asarray(testArray[0][count])
            count = count+1
            googleNet.layers[i] = l
            #print i,count

    #write back to file and see if it is the same
    '''
    write_fp = open('reconstruct.weights','w')
    write_fp.write((np.asarray(testArray[0][0])).tobytes())
    write_fp.write((np.asarray(testArray[0][1])).tobytes())
    for i in range(0,googleNet.layer_number):
        l = googleNet.layers[i]
        if(l.type == "CONVOLUTIONAL" or l.type == "CONNECTED"):
            write_fp.write(l.biases.tobytes())
            write_fp.write(l.weights.tobytes())

    write_fp.close()
    '''
    return googleNet
```

## pythonPascalVoc2.py

```python
import os
import xml.etree.ElementTree as ET
from crop import crop_detection
import numpy as np
from PIL import Image
from PIL import ImageDraw
import scipy
import random


vocPath = os.path.abspath(os.path.join(os.getcwd(),os.path.pardir,'dataset'))

class objInfo():
    """
    objInfo saves the information of an object, including its class num, its cords
    """
    def __init__(self,x,y,h,w,class_num):
        self.x = x
        self.y = y
        self.h = h
        self.w = w
        self.class_num = class_num

class Cell():
    """
    A cell is a grid cell of an image, it has a boolean variable indicating whether there are any objects in this cell,
    and a list of objInfo objects indicating the information of objects if there are any
    """
    def __init__(self):
        self.has_obj = False
        self.objs = []

class image():
    """
    Args:
       side: An image is divided into side*side grids
    Each image class has two variables:
       imgPath: the path of an image on my computer
       bboxes: a side*side matrix, each element in the matrix is cell
    """
    def __init__(self,side,imgPath):
        self.imgPath = imgPath
        self.boxes = []
        for i in range(side):
            rows = []
            for j in range(side):
                rows.append(Cell())
            self.boxes.append(rows)

    def parseXML(self,xmlPath,labels,side):
        """
        Args:
          xmlPath: The path of the xml file of this image
          labels: label names of pascal voc dataset
          side: an image is divided into side*side grid
        """
        tree = ET.parse(xmlPath)
        root = tree.getroot()

        width = int(root.find('size').find('width').text)
        height = int(root.find('size').find('height').text)

        for obj in root.iter('object'):
            class_num = labels.index(obj.find('name').text)
            bndbox = obj.find('bndbox')
            xmin = int(bndbox.find('xmin').text)
            ymin = int(bndbox.find('ymin').text)
            xmax = int(bndbox.find('xmax').text)
            ymax = int(bndbox.find('ymax').text)
            h = ymax-ymin
            w = xmax-xmin
            #objif = objInfo(xmin/448.0,ymin/448.0,np.sqrt(ymax-ymin)/448.0,np.sqrt(xmax-xmin)/448.0,class_num)

            #which cell this obj falls into
            centerx = (xmax+xmin)/2.0
            centery = (ymax+ymin)/2.0
            newx = (448.0/width)*centerx
            newy = (448.0/height)*centery

            h_new = h * (448.0 / height)
            w_new = w * (448.0 / width)

            cell_size = 448.0/side
            col = int(newx / cell_size)
            row = int(newy / cell_size)
           # print "row,col:",row,col,centerx,centery

            cell_left = col * cell_size
            cell_top = row * cell_size
            cord_x = (newx - cell_left) / cell_size
            cord_y = (newy - cell_top)/ cell_size

            objif = objInfo(cord_x,cord_y,np.sqrt(h_new/448.0),np.sqrt(w_new/448.0),class_num)
            self.boxes[row][col].has_obj = True
            self.boxes[row][col].objs.append(objif)

def prepareBatch(start,end,imageNameFile,vocPath):
    """
    Args:
      start: the number of image to start
      end: the number of image to end
      imageNameFile: the path of the file that contains image names
      vocPath: the path of pascal voc dataset
    Funs:
      generate a batch of images from start~end
    Returns:
      A list of end-start+1 image objects
    """
    imageList = []
    labels = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train","tvmonitor"]
    file = open(imageNameFile)
    imageNames = file.readlines()
    for i in range(start,end):
        imgName = imageNames[i].strip('\n')
        imgPath = os.path.join(vocPath,'JPEGImages',imgName)+'.jpg'
        xmlPath = os.path.join(vocPath,'Annotations',imgName)+'.xml'
        img = image(side=7,imgPath=imgPath)
        img.parseXML(xmlPath,labels,7)
        imageList.append(img)

    return imageList

#Prepare training data
def generate_batch_data(vocPath,imageNameFile,batch_size,sample_number):
    """
    Args:
      vocPath: the path of pascal voc data
      imageNameFile: the path of the file of image names
      batchsize: batch size, sample_number should be divided by batchsize
    Funcs:
      A data generator generates training batch indefinitely
    """
    class_num = 20
    #Read all the data once and dispatch them out as batches to save time
    TotalimageList = prepareBatch(0,sample_number,imageNameFile,vocPath)

    while 1:
        batches = sample_number // batch_size
        for i in range(batches):
            images = []
            boxes = []
            sample_index = np.random.choice(sample_number,batch_size,replace=True)
            #sample_index = [3]
            for ind in sample_index:
                image = TotalimageList[ind]
                #print image.imgPath
                image_array = crop_detection(image.imgPath,new_width=448,new_height=448)
                #image_array = np.expand_dims(image_array,axis=0)

                y = []
                for i in range(7):
                    for j in range(7):
                        box = image.boxes[i][j]
                        '''
                        ############################################################
                        #x,y,h,w,one_hot class label vector[0....0],objectness{0,1}#
                        ############################################################
                        '''
                        if(box.has_obj):
                            obj = box.objs[0]

                            y.append(obj.x)
                            y.append(obj.y)
                            y.append(obj.h)
                            y.append(obj.w)

                            labels = [0]*20
                            labels[obj.class_num] = 1
                            y.extend(labels)
                            y.append(1) #objectness
                        else:
                            y.extend([0]*25)
                y = np.asarray(y)
                #y = np.reshape(y,[1,y.shape[0]])

                images.append(image_array)
                boxes.append(y)
            #return np.asarray(images),np.asarray(boxes)
            yield np.asarray(images),np.asarray(boxes)

if __name__ == '__main__':
    imageNameFile='/home/media/Documents/YOLO.keras/dataset/train_val/SingleImageNameFile.txt'
    vocPath='/home/media/Documents/YOLO.keras/dataset/train_val'
    '''
    imageList = prepareBatch(0,2,imageNameFile,vocPath)
    for i in range(0,2):
        img = imageList[i]
        print img.imgPath
        boxes = img.boxes
        for i in range(7):
            for j in range(7):
                if(boxes[i][j].has_obj):
                    print i,j
                    objs = boxes[i][j].objs
                    for obj in objs:
                        print obj.class_num
                        print obj.x
                        print obj.y
                        print
    '''
    image_array,y = generate_batch_data(vocPath,imageNameFile,1,sample_number=16)
    print image_array.shape,y.shape
    #print image_array[0,...,...,...].shape
    #let's see if we read correctly
    image_array = image_array[0,...,...,...]
    #scipy.misc.imsave('recovered.jpg', image_array)
    print image_array.shape
    image_array = (image_array + 1.0) / 2.0 * 225.0
    image_array = np.rollaxis(image_array,2,0)
    image_array = np.rollaxis(image_array,2,0)
    print image_array.shape

    scipy.misc.imsave('recovered.jpg', image_array)
    # center should be in (3,3)
    labels = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train","tvmonitor"]
    out = y[0]

    imgPath = os.path.join(os.getcwd(),'recovered.jpg')
    img = Image.open(imgPath)
    image_arr,img_resize = crop_detection(imgPath,448,448,save=True)
    drawable = ImageDraw.Draw(img_resize)
    #Draw orignal bounding boxes

    count = 0
    for i in range(49):
        preds = out[i*25:(i+1)*25]
        if(preds[24] > 0.3):
            count = count + 1
            #print preds[0:4],preds[24]
            row = i/7
            col = i%7
            print row,col
            centerx = 64 * col + 64 * preds[0]
            centery = 64 * row + 64 * preds[1]

            h = preds[2] * preds[2]
            h = h * 448.0
            w = preds[3] * preds[3]
            w = w * 448.0

            left = centerx - w/2.0
            right = centerx + w/2.0
            up = centery - h/2.0
            down = centery + h/2.0

            if(left < 0): left = 0
            if(right > 448): right = 447
            if(up < 0): up = 0
            if(down > 448): down = 447

            drawable.rectangle([left,up,right,down],outline='red')
            print 'Class is: ',labels[np.argmax(preds[4:24])]
            print np.max(preds[4:24])
    print count
    img_resize.save(os.path.join(os.getcwd(),'recover.jpg'))
```


## ReadPascalforLoc.py

```python
import os
import xml.etree.ElementTree as ET
from crop import crop_detection
import numpy as np
from PIL import Image
from PIL import ImageDraw
import scipy
import random


vocPath = os.path.abspath(os.path.join(os.getcwd(),os.path.pardir,'dataset'))

class box:
    def __init__(self,x,y,h,w):
        self.x = x
        self.y = y
        self.h = h
        self.w = w

def overlap(x1,w1,x2,w2):
    l1 = x1 - w1/2;
    l2 = x2 - w2/2;
    if(l1 > l2):
        left = l1
    else:
        left = l2
    r1 = x1 + w1/2;
    r2 = x2 + w2/2;
    if(r1 < r2):
        right = r1
    else:
        right = r2
    return right - left;

def box_intersection(a, b):
    w = overlap(a.x, a.w, b.x, b.w);
    h = overlap(a.y, a.h, b.y, b.h);
    if(w < 0 or h < 0):
         return 0;
    area = w*h;
    return area;

def box_union(a, b):
    i = box_intersection(a, b);
    u = a.w*a.h + b.w*b.h - i;
    return u;

def box_iou(a, b):
    return box_intersection(a, b)/box_union(a, b);

class image():
    """
    Args:
       side: An image is divided into side*side grids
    Each image class has two variables:
       imgPath: the path of an image on my computer
       bboxes: a side*side matrix, each element in the matrix is cell
    """
    def __init__(self,side,imgPath):
        self.imgPath = imgPath
        self.boxes = np.zeros((side,side))

    def parseXML(self,xmlPath,labels,side):
        """
        Args:
          xmlPath: The path of the xml file of this image
          labels: label names of pascal voc dataset
          side: an image is divided into side*side grid
        """
        tree = ET.parse(xmlPath)
        root = tree.getroot()

        width = int(root.find('size').find('width').text)
        height = int(root.find('size').find('height').text)

        for obj in root.iter('object'):
            class_num = labels.index(obj.find('name').text)
            bndbox = obj.find('bndbox')
            left = int(bndbox.find('xmin').text)
            top = int(bndbox.find('ymin').text)
            right = int(bndbox.find('xmax').text)
            down = int(bndbox.find('ymax').text)

            #trans the coords to 448*448
            left = left*1.0 / width * 448
            right = right*1.0 / width * 448
            top = top*1.0 / height * 448
            down = down*1.0 / height * 448

            boxa = box((left+right)/2,(top+down)/2,down-top,right-left)
            for i in range(int(left/64),int(right/64)+1):
                for j in range(int(top/64),int(down/64)+1):
                    box_left = i*64
                    box_right = (i+1)*64
                    box_top = j*64
                    box_down = (j+1)*64
                    boxb = box((box_left+box_right)/2,(box_top+box_down)/2,64,64)
                    iou = box_intersection(boxa,boxb)
                    if(iou/(64*64) > 0.25):
                        self.boxes[j][i] = 1

def prepareBatch(start,end,imageNameFile,vocPath):
    """
    Args:
      start: the number of image to start
      end: the number of image to end
      imageNameFile: the path of the file that contains image names
      vocPath: the path of pascal voc dataset
    Funs:
      generate a batch of images from start~end
    Returns:
      A list of end-start+1 image objects
    """
    imageList = []
    labels = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train","tvmonitor"]
    file = open(imageNameFile)
    imageNames = file.readlines()
    for i in range(start,end):
        imgName = imageNames[i].strip('\n')
        imgPath = os.path.join(vocPath,'JPEGImages',imgName)+'.jpg'
        xmlPath = os.path.join(vocPath,'Annotations',imgName)+'.xml'
        img = image(side=7,imgPath=imgPath)
        img.parseXML(xmlPath,labels,7)
        imageList.append(img)

    return imageList

#Prepare training data
def generate_batch_data(vocPath,imageNameFile,batch_size,sample_number):
    """
    Args:
      vocPath: the path of pascal voc data
      imageNameFile: the path of the file of image names
      batchsize: batch size, sample_number should be divided by batchsize
    Funcs:
      A data generator generates training batch indefinitely
    """
    class_num = 20
    #Read all the data once and dispatch them out as batches to save time
    TotalimageList = prepareBatch(0,sample_number,imageNameFile,vocPath)

    while 1:
        batches = sample_number // batch_size
        for i in range(batches):
            images = []
            boxes = []
            sample_index = np.random.choice(sample_number,batch_size,replace=True)
            #sample_index = [3]
            for ind in sample_index:
                image = TotalimageList[ind]
                #print image.imgPath
                image_array = crop_detection(image.imgPath,new_width=448,new_height=448)
                #image_array = np.expand_dims(image_array,axis=0)

                images.append(image_array)
                boxes.append(np.reshape(image.boxes,-1))
            #return np.asarray(images),np.asarray(boxes)
            yield np.asarray(images),np.asarray(boxes)

if __name__ == '__main__':
    img,boxes = generate_batch_data('/home/media/Documents/YOLO.keras/dataset/train_val/','/home/media/Documents/YOLO.keras/utils/image_name',1,1)
    #labels = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train","tvmonitor"]
    #img = image(side=7,imgPath='/home/media/Documents/YOLO.keras/dataset/train_val/JPEGImages/000011.jpg')
    #img.parseXML(xmlPath='/home/media/Documents/YOLO.keras/dataset/VOCdevkit/VOC2007/Annotations/000011.xml',labels=labels,side=7)
    print boxes
```

## TinyYoloNet.py

```python
import numpy as np
from enum import Enum
import os

class layer:
    def __init__(self,size,c,n,h,w,type):
        self.size = size
        self.c = c
        self.n = n
        self.h = h
        self.w = w
        self.type = type

class convolutional_layer(layer):
    def __init__(self,size,c,n,h,w):
        layer.__init__(self,size,c,n,h,w,"CONVOLUTIONAL")
        self.biases = np.zeros(n)
        self.weights = np.zeros((size*size,c,n))

class connected_layer(layer):
    def __init__(self,size,c,n,h,w,input_size,output_size):
        layer.__init__(self,size,c,n,h,w,"CONNECTED")
        self.output_size = output_size
        self.input_size = input_size
        self.biases = np.zeros(output_size)
        self.weights = np.zeros((output_size*input_size))

class Tiny_YOLO:
    layers = []
    layer_number = 22
    def __init__(self):
        self.layers.append(layer(0,0,0,0,0,"CROP"))
        self.layers.append(convolutional_layer(3,3,16,448,448))
        self.layers.append(layer(0,0,0,0,0,"MAXPOOL"))
        self.layers.append(convolutional_layer(3,16,32,224,224))
        self.layers.append(layer(0,0,0,0,0,"MAXPOOL"))
        self.layers.append(convolutional_layer(3,32,64,112,112))
        self.layers.append(layer(0,0,0,0,0,"MAXPOOL"))
        self.layers.append(convolutional_layer(3,64,128,56,56))
        self.layers.append(layer(0,0,0,0,0,"MAXPOOL"))
        self.layers.append(convolutional_layer(3,128,256,28,28))
        self.layers.append(layer(0,0,0,0,0,"MAXPOOL"))
        self.layers.append(convolutional_layer(3,256,512,14,14))
        self.layers.append(layer(0,0,0,0,0,"MAXPOOL"))
        self.layers.append(convolutional_layer(3,512,1024,7,7))
        self.layers.append(convolutional_layer(3,1024,1024,7,7))
        self.layers.append(convolutional_layer(3,1024,1024,7,7))
        self.layers.append(layer(0,0,0,0,0,"FLATTEN"))
        self.layers.append(connected_layer(0,0,0,0,0,50176,256))
        self.layers.append(connected_layer(0,0,0,0,0,256,4096))
        self.layers.append(layer(0,0,0,0,0,"DROPOUT"))
        self.layers.append(layer(0,0,0,0,0,"LEAKY"))
        self.layers.append(connected_layer(0,0,0,0,0,4096,1470))

def ReadTinyYOLONetWeights(weight_path):
    YOLO = Tiny_YOLO()
    type_string = "(3)float32,i4,"
    for i in range(YOLO.layer_number):
        l = YOLO.layers[i]
        if(l.type == "CONVOLUTIONAL"):
            bias_number = l.n
            weight_number = l.n*l.c*l.size*l.size
            type_string = type_string +"("+ str(bias_number) + ")float32,(" + str(weight_number) + ")float32,"
        elif(l.type == "CONNECTED"):
             bias_number = l.output_size
             weight_number = l.output_size * l.input_size
             type_string = type_string + "("+ str(bias_number) + ")float32,("+ str(weight_number)+")float32"
             if(i != YOLO.layer_number-1):
                 type_string = type_string + ","
    #dt = np.dtype((+str(64)+")float32"))
    #type_string = type_string + ",i1"
    dt = np.dtype(type_string)
    testArray = np.fromfile(weight_path,dtype=dt)
    #write the weights read from file to GoogleNet biases and weights

    count = 2
    for i in range(0,YOLO.layer_number):
        l = YOLO.layers[i]
        if(l.type == "CONVOLUTIONAL" or l.type == "CONNECTED"):
            l.biases = np.asarray(testArray[0][count])
            count = count + 1
            l.weights = np.asarray(testArray[0][count])
            count = count + 1
            YOLO.layers[i] = l

    #write back to file and see if it is the same
    '''
    write_fp = open('reconstruct.weights','w')
    write_fp.write((np.asarray(testArray[0][0])).tobytes())
    write_fp.write((np.asarray(testArray[0][1])).tobytes())
    for i in range(0,YOLO.layer_number):
        l = YOLO.layers[i]
        if(l.type == "CONVOLUTIONAL" or l.type == "CONNECTED"):
            write_fp.write(l.biases.tobytes())
            write_fp.write(l.weights.tobytes())


    write_fp.close()
    '''
    return YOLO

if __name__ == '__main__':
    YOLO = ReadTinyYOLONetWeights('/home/xuetingli/Documents/YOLO.keras/weights/yolo-tiny.weights')
    for i in range(YOLO.layer_number):
        l = YOLO.layers[i]
        print l.type
```

## TinyYoloNetforLoc.py

```python
import numpy as np
from enum import Enum
import os

class layer:
    def __init__(self,size,c,n,h,w,type):
        self.size = size
        self.c = c
        self.n = n
        self.h = h
        self.w = w
        self.type = type

class convolutional_layer(layer):
    def __init__(self,size,c,n,h,w):
        layer.__init__(self,size,c,n,h,w,"CONVOLUTIONAL")
        self.biases = np.zeros(n)
        self.weights = np.zeros((size*size,c,n))

class connected_layer(layer):
    def __init__(self,size,c,n,h,w,input_size,output_size):
        layer.__init__(self,size,c,n,h,w,"CONNECTED")
        self.output_size = output_size
        self.input_size = input_size
        self.biases = np.zeros(output_size)
        self.weights = np.zeros((output_size*input_size))

class Tiny_YOLO:
    layers = []
    layer_number = 22
    def __init__(self):
        self.layers.append(layer(0,0,0,0,0,"CROP"))
        self.layers.append(convolutional_layer(3,3,16,448,448))
        self.layers.append(layer(0,0,0,0,0,"MAXPOOL"))
        self.layers.append(convolutional_layer(3,16,32,224,224))
        self.layers.append(layer(0,0,0,0,0,"MAXPOOL"))
        self.layers.append(convolutional_layer(3,32,64,112,112))
        self.layers.append(layer(0,0,0,0,0,"MAXPOOL"))
        self.layers.append(convolutional_layer(3,64,128,56,56))
        self.layers.append(layer(0,0,0,0,0,"MAXPOOL"))
        self.layers.append(convolutional_layer(3,128,256,28,28))
        self.layers.append(layer(0,0,0,0,0,"MAXPOOL"))
        self.layers.append(convolutional_layer(3,256,512,14,14))
        self.layers.append(layer(0,0,0,0,0,"MAXPOOL"))
        self.layers.append(convolutional_layer(3,512,1024,7,7))
        self.layers.append(convolutional_layer(3,1024,1024,7,7))
        self.layers.append(convolutional_layer(3,1024,1024,7,7))
        self.layers.append(layer(0,0,0,0,0,"FLATTEN"))
        self.layers.append(connected_layer(0,0,0,0,0,50176,256))
        self.layers.append(connected_layer(0,0,0,0,0,256,4096))
        self.layers.append(layer(0,0,0,0,0,"DROPOUT"))
        self.layers.append(layer(0,0,0,0,0,"LEAKY"))
        self.layers.append(connected_layer(0,0,0,0,0,4096,49))

def ReadTinyYOLONetWeights(weight_path):
    YOLO = Tiny_YOLO()
    type_string = "(3)float32,i4,"
    for i in range(YOLO.layer_number):
        l = YOLO.layers[i]
        if(l.type == "CONVOLUTIONAL"):
            bias_number = l.n
            weight_number = l.n*l.c*l.size*l.size
            type_string = type_string +"("+ str(bias_number) + ")float32,(" + str(weight_number) + ")float32,"
        elif(l.type == "CONNECTED"):
             bias_number = l.output_size
             weight_number = l.output_size * l.input_size
             type_string = type_string + "("+ str(bias_number) + ")float32,("+ str(weight_number)+")float32"
             if(i != YOLO.layer_number-1):
                 type_string = type_string + ","
    #dt = np.dtype((+str(64)+")float32"))
    #type_string = type_string + ",i1"
    dt = np.dtype(type_string)
    testArray = np.fromfile(weight_path,dtype=dt)
    #write the weights read from file to GoogleNet biases and weights

    count = 2
    for i in range(0,YOLO.layer_number):
        l = YOLO.layers[i]
        if(l.type == "CONVOLUTIONAL" or l.type == "CONNECTED"):
            l.biases = np.asarray(testArray[0][count])
            count = count + 1
            l.weights = np.asarray(testArray[0][count])
            count = count + 1
            YOLO.layers[i] = l

    #write back to file and see if it is the same
    '''
    write_fp = open('reconstruct.weights','w')
    write_fp.write((np.asarray(testArray[0][0])).tobytes())
    write_fp.write((np.asarray(testArray[0][1])).tobytes())
    for i in range(0,YOLO.layer_number):
        l = YOLO.layers[i]
        if(l.type == "CONVOLUTIONAL" or l.type == "CONNECTED"):
            write_fp.write(l.biases.tobytes())
            write_fp.write(l.weights.tobytes())


    write_fp.close()
    '''
    return YOLO

if __name__ == '__main__':
    YOLO = ReadTinyYOLONetWeights('/home/xuetingli/Documents/YOLO.keras/weights/yolo-tiny.weights')
    for i in range(YOLO.layer_number):
        l = YOLO.layers[i]
        print l.type
```

## __init__.py

```python

```


## crop.py

```python
from PIL import Image
import numpy as np
from scipy import misc
import os

def crop(imPath,resize_width=256,resize_height=256,new_width=224,new_height=224):
    im = Image.open(imPath)
    im = im.resize((resize_width,resize_height),Image.ANTIALIAS)

    #central crop 224,224
    width, height = im.size   # Get dimensions

    left = (width - new_width)/2
    top = (height - new_height)/2
    right = (width + new_width)/2
    bottom = (height + new_height)/2

    im = im.crop((left, top, right, bottom))

    im.save("frames/cropped.jpg")

    image_array = np.array(im)
    image_array = np.rollaxis(image_array,2,0)
    image_array = image_array/255.0
    image_array = image_array * 2.0 - 1.0
    return image_array

def crop_detection(imPath,new_width=448,new_height=448,save=False,test=False):
    im = Image.open(imPath)
    im = im.resize((new_width,new_height),Image.ANTIALIAS)

    image_array = np.array(im)
    image_array = np.rollaxis(image_array,2,0)
    image_array = image_array/255.0
    image_array = image_array * 2.0 - 1.0

    if(test):
        image_array = (image_array + 1.0) / 2.0 * 225.0
        image_array = np.rollaxis(image_array,2,0)
        image_array = np.rollaxis(image_array,2,0)
        print image_array.shape

        misc.imsave('recovered.jpg', image_array)

    if(save):
        return image_array,im
    else:
        return image_array

```

## readImgFile.py

```python
import numpy as np
import os

def readImg(imgPath,h=224,w=224):
    dt = np.dtype("float32")
    testArray = np.fromfile(imgPath,dtype=dt)

    image = np.reshape(testArray,[3,h,w])
    return image
```

##timer.py
# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import time

class Timer(object):
    """A simple timer."""
    def __init__(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            return self.average_time
        else:
            return self.diff
```
