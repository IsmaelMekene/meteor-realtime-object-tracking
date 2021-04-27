import cv2
import model as retinanet_models
from utils.image import read_image_bgr, preprocess_image, resize_image
from utils.viz import draw_box, draw_caption
import matplotlib.pyplot as plt
import matplotlib.image as pltimage
import numpy as np
import tensorflow as tf
import os
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input
from PIL import Image
import colorsys
import warnings

from utils.helpers import tiny_yolov3_main, yolov3_main, letterbox_image, yolo_eval, preprocess_input, retrieve_yolo_detections, draw_boxes, padded_fragment, box_iou2, write_each_second

class ObjectDetection_Me:
    """
    This is the object detection class for images in the ImageAI library. It provides support for RetinaNet
        , YOLOv3 and TinyYOLOv3 object detection networks . After instantiating this class, you can set it's properties and
    make object detections using it's pre-defined functions.
    The following functions are required to be called before object detection can be made
    * setModelPath()
    * At least of of the following and it must correspond to the model set in the setModelPath()
    [setModelTypeAsRetinaNet(), setModelTypeAsYOLOv3(), setModelTypeAsTinyYOLOv3()]
    * loadModel() [This must be called once only before performing object detection]
    Once the above functions have been called, you can call the detectObjectsFromImage() function of
    the object detection instance object at anytime to obtain observable objects in any image.
    """

    def __init__(self):
        self.__modelType = ""
        self.modelPath = ""
        self.__modelPathAdded = False
        self.__modelLoaded = False
        self.__model_collection = []

        # Instance variables for RetinaNet Model
        self.__input_image_min = 1333
        self.__input_image_max = 800

        self.numbers_to_names = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus',
                                 6: 'train',
                                 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign',
                                 12: 'parking meter',
                                 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow',
                                 20: 'elephant',
                                 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag',
                                 27: 'tie',
                                 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball',
                                 33: 'kite',
                                 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard',
                                 38: 'tennis racket',
                                 39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon',
                                 45: 'bowl',
                                 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot',
                                 52: 'hot dog',
                                 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant',
                                 59: 'bed',
                                 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote',
                                 66: 'keyboard',
                                 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink',
                                 72: 'refrigerator',
                                 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear',
                                 78: 'hair dryer',
                                 79: 'toothbrush'}

        # Unique instance variables for YOLOv3 and TinyYOLOv3 model
        self.__yolo_iou = 0.45
        self.__yolo_score = 0.1
        self.__nms_thresh = 0.45
        self.__yolo_anchors = [[116,90,  156,198,  373,326],  [30,61, 62,45,  59,119], [10,13,  16,30,  33,23]]
        self.__yolo_model_image_size = (416, 416)
        self.__yolo_boxes, self.__yolo_scores, self.__yolo_classes = "", "", ""
        self.__tiny_yolo_anchors = [[81, 82, 135, 169, 344, 319], [10, 14, 23, 27, 37, 58]]
        self.__box_color = (112, 19, 24)
        

    def setModelTypeAsRetinaNet(self):
        """
        'setModelTypeAsRetinaNet()' is used to set the model type to the RetinaNet model
        for the video object detection instance instance object .
        :return:
        """
        self.__modelType = "retinanet"

    def setModelTypeAsYOLOv3(self):
        """
                'setModelTypeAsYOLOv3()' is used to set the model type to the YOLOv3 model
                for the video object detection instance instance object .
                :return:
                """

        self.__modelType = "yolov3"

    def setModelTypeAsTinyYOLOv3(self):
        """
                        'setModelTypeAsTinyYOLOv3()' is used to set the model type to the TinyYOLOv3 model
                        for the video object detection instance instance object .
                        :return:
                        """

        self.__modelType = "tinyyolov3"

    def setModelPath(self, model_path):
        """
         'setModelPath()' function is required and is used to set the file path to a RetinaNet
          object detection model trained on the COCO dataset.
          :param model_path:
          :return:
        """

        if (self.__modelPathAdded == False):
            self.modelPath = model_path
            self.__modelPathAdded = True

    def loadModel(self, detection_speed="normal"):
        """
                'loadModel()' function is required and is used to load the model structure into the program from the file path defined
                in the setModelPath() function. This function receives an optional value which is "detection_speed".
                The value is used to reduce the time it takes to detect objects in an image, down to about a 10% of the normal time, with
                 with just slight reduction in the number of objects detected.
                * prediction_speed (optional); Acceptable values are "normal", "fast", "faster", "fastest" and "flash"
                :param detection_speed:
                :return:
        """

        if (self.__modelType == "retinanet"):
            if (detection_speed == "normal"):
                self.__input_image_min = 800
                self.__input_image_max = 1333
            elif (detection_speed == "fast"):
                self.__input_image_min = 400
                self.__input_image_max = 700
            elif (detection_speed == "faster"):
                self.__input_image_min = 300
                self.__input_image_max = 500
            elif (detection_speed == "fastest"):
                self.__input_image_min = 200
                self.__input_image_max = 350
            elif (detection_speed == "flash"):
                self.__input_image_min = 100
                self.__input_image_max = 250
        elif (self.__modelType == "yolov3"):
            if (detection_speed == "normal"):
                self.__yolo_model_image_size = (416, 416)
            elif (detection_speed == "fast"):
                self.__yolo_model_image_size = (320, 320)
            elif (detection_speed == "faster"):
                self.__yolo_model_image_size = (208, 208)
            elif (detection_speed == "fastest"):
                self.__yolo_model_image_size = (128, 128)
            elif (detection_speed == "flash"):
                self.__yolo_model_image_size = (96, 96)

        elif (self.__modelType == "tinyyolov3"):
            if (detection_speed == "normal"):
                self.__yolo_model_image_size = (832, 832)
            elif (detection_speed == "fast"):
                self.__yolo_model_image_size = (576, 576)
            elif (detection_speed == "faster"):
                self.__yolo_model_image_size = (416, 416)
            elif (detection_speed == "fastest"):
                self.__yolo_model_image_size = (320, 320)
            elif (detection_speed == "flash"):
                self.__yolo_model_image_size = (272, 272)

        if (self.__modelLoaded == False):
            if (self.__modelType == ""):
                raise ValueError("You must set a valid model type before loading the model.")
            elif (self.__modelType == "retinanet"):
                model = retinanet_models.load_model(self.modelPath, backbone_name='resnet50')
                self.__model_collection.append(model)
                self.__modelLoaded = True
            elif (self.__modelType == "yolov3" or self.__modelType == "tinyyolov3"):

                input_image = Input(shape=(None, None, 3))

                if self.__modelType == "yolov3":
                    model = yolov3_main(input_image, len(self.__yolo_anchors),
                                    len(self.numbers_to_names.keys()))
                else:
                    model = tiny_yolov3_main(input_image, 3,
                                 len(self.numbers_to_names.keys()))

                model.load_weights(self.modelPath)
               
                self.__model_collection.append(model)
                self.__modelLoaded = True

    def detectObjectsFromImage_Me(self, input_image="", output_image_path="", input_type="file", output_type="file",
                               extract_detected_objects=False, minimum_percentage_probability=50,
                               display_percentage_probability=True, display_object_name=True,
                               display_box=True, thread_safe=False, custom_objects=None):
        """
            'detectObjectsFromImage()' function is used to detect objects observable in the given image path:
                    * input_image , which can be a filepath, image numpy array or image file stream
                    * output_image_path (only if output_type = file) , file path to the output image that will contain the detection boxes and label, if output_type="file"
                    * input_type (optional) , file path/numpy array/image file stream of the image. Acceptable values are "file", "array" and "stream"
                    * output_type (optional) , file path/numpy array/image file stream of the image. Acceptable values are "file" and "array"
                    * extract_detected_objects (optional) , option to save each object detected individually as an image and return an array of the objects' image path.
                    * minimum_percentage_probability (optional, 50 by default) , option to set the minimum percentage probability for nominating a detected object for output.
                    * display_percentage_probability (optional, True by default), option to show or hide the percentage probability of each object in the saved/returned detected image
                    * display_display_object_name (optional, True by default), option to show or hide the name of each object in the saved/returned detected image
                    * thread_safe (optional, False by default), enforce the loaded detection model works across all threads if set to true, made possible by forcing all Tensorflow inference to run on the default graph.
            The values returned by this function depends on the parameters parsed. The possible values returnable
            are stated as below
            - If extract_detected_objects = False or at its default value and output_type = 'file' or
                at its default value, you must parse in the 'output_image_path' as a string to the path you want
                the detected image to be saved. Then the function will return:
                1. an array of dictionaries, with each dictionary corresponding to the objects
                    detected in the image. Each dictionary contains the following property:
                    * name (string)
                    * percentage_probability (float)
                    * box_points (list of x1,y1,x2 and y2 coordinates)
            - If extract_detected_objects = False or at its default value and output_type = 'array' ,
              Then the function will return:
                1. a numpy array of the detected image
                2. an array of dictionaries, with each dictionary corresponding to the objects
                    detected in the image. Each dictionary contains the following property:
                    * name (string)
                    * percentage_probability (float)
                    * box_points (list of x1,y1,x2 and y2 coordinates)
            - If extract_detected_objects = True and output_type = 'file' or
                at its default value, you must parse in the 'output_image_path' as a string to the path you want
                the detected image to be saved. Then the function will return:
                1. an array of dictionaries, with each dictionary corresponding to the objects
                    detected in the image. Each dictionary contains the following property:
                    * name (string)
                    * percentage_probability (float)
                    * box_points (list of x1,y1,x2 and y2 coordinates)
                2. an array of string paths to the image of each object extracted from the image
            - If extract_detected_objects = True and output_type = 'array', the the function will return:
                1. a numpy array of the detected image
                2. an array of dictionaries, with each dictionary corresponding to the objects
                    detected in the image. Each dictionary contains the following property:
                    * name (string)
                    * percentage_probability (float)
                    * box_points (list of x1,y1,x2 and y2 coordinates)
                3. an array of numpy arrays of each object detected in the image
            :param input_image:
            :param output_image_path:
            :param input_type:
            :param output_type:
            :param extract_detected_objects:
            :param minimum_percentage_probability:
            :param display_percentage_probability:
            :param display_object_name:
            :param thread_safe:
            :return image_frame:
            :return output_objects_array:
            :return detected_objects_image_array:
        """

        if (self.__modelLoaded == False):
            raise ValueError("You must call the loadModel() function before making object detection.")
        elif (self.__modelLoaded == True):
            try:

                model_detections = list()
                detections = list()
                image_copy = None

                detected_objects_image_array = []
                min_probability = minimum_percentage_probability / 100

                if (input_type == "file"):
                    input_image = cv2.imread(input_image)
                elif (input_type == "array"):
                    input_image = np.array(input_image)

                detected_copy = input_image
                image_copy = input_image

                if (self.__modelType == "yolov3" or self.__modelType == "tinyyolov3"):

                    image_h, image_w, _ = detected_copy.shape
                    detected_copy = preprocess_input(detected_copy, self.__yolo_model_image_size)

                    model = self.__model_collection[0]
                    yolo_result = model.predict(detected_copy)

                    model_detections = retrieve_yolo_detections(yolo_result,
                            self.__yolo_anchors,
                            min_probability,
                            self.__nms_thresh,
                            self.__yolo_model_image_size,
                            (image_w, image_h),
                            self.numbers_to_names)
                            
                elif (self.__modelType == "retinanet"):
                    detected_copy = preprocess_image(detected_copy)
                    detected_copy, scale = resize_image(detected_copy)

                    model = self.__model_collection[0]
                    boxes, scores, labels = model.predict_on_batch(np.expand_dims(detected_copy, axis=0))

                    
                    boxes /= scale

                    for box, score, label in zip(boxes[0], scores[0], labels[0]):
                        # scores are sorted so we can break
                        if score < min_probability:
                            break

                        detection_dict = dict()
                        detection_dict["name"] = self.numbers_to_names[label]
                        detection_dict["percentage_probability"] = score * 100
                        detection_dict["box_points"] = box.astype(int).tolist()
                        model_detections.append(detection_dict)

                counting = 0
                objects_dir = output_image_path + "-objects"

                for detection in model_detections:
                    counting += 1
                    label = detection["name"]
                    percentage_probability = detection["percentage_probability"]
                    box_points = detection["box_points"]

                    if (custom_objects is not None):
                        if (custom_objects[label] != "valid"):
                            continue
                    
                    detections.append(detection)

                    if display_object_name == False:
                        label = None

                    if display_percentage_probability == False:
                        percentage_probability = None

                    
                    image_copy = draw_boxes(image_copy, 
                                    box_points,
                                    display_box,
                                    label, 
                                    percentage_probability, 
                                    self.__box_color)
                    
                    

                    if (extract_detected_objects == True):
                        splitted_copy = image_copy.copy()[box_points[1]:box_points[3],
                                        box_points[0]:box_points[2]]
                        if (output_type == "file"):
                            if (os.path.exists(objects_dir) == False):
                                os.mkdir(objects_dir)
                            splitted_image_path = os.path.join(objects_dir,
                                                                detection["name"] + "-" + str(
                                                                    counting) + ".jpg")
                            cv2.imwrite(splitted_image_path, splitted_copy)
                            detected_objects_image_array.append(splitted_image_path)
                        elif (output_type == "array"):
                            detected_objects_image_array.append(splitted_copy)

                image_copy = padded_fragment(image_copy)
                
                if (output_type == "file"):
                    cv2.imwrite(output_image_path, image_copy)

                if (extract_detected_objects == True):
                    if (output_type == "file"):
                        return detections, detected_objects_image_array
                    elif (output_type == "array"):
                        return image_copy, detections, detected_objects_image_array

                else:
                    if (output_type == "file"):
                        return detections
                    elif (output_type == "array"):
                        return image_copy, detections

            except:
                raise ValueError(
                    "Ensure you specified correct input image, input type, output type and/or output image path ")

    def CustomObjects(self, person=False, bicycle=False, car=False, motorcycle=False, airplane=False,
                      bus=False, train=False, truck=False, boat=False, traffic_light=False, fire_hydrant=False,
                      stop_sign=False,
                      parking_meter=False, bench=False, bird=False, cat=False, dog=False, horse=False, sheep=False,
                      cow=False, elephant=False, bear=False, zebra=False,
                      giraffe=False, backpack=False, umbrella=False, handbag=False, tie=False, suitcase=False,
                      frisbee=False, skis=False, snowboard=False,
                      sports_ball=False, kite=False, baseball_bat=False, baseball_glove=False, skateboard=False,
                      surfboard=False, tennis_racket=False,
                      bottle=False, wine_glass=False, cup=False, fork=False, knife=False, spoon=False, bowl=False,
                      banana=False, apple=False, sandwich=False, orange=False,
                      broccoli=False, carrot=False, hot_dog=False, pizza=False, donut=False, cake=False, chair=False,
                      couch=False, potted_plant=False, bed=False,
                      dining_table=False, toilet=False, tv=False, laptop=False, mouse=False, remote=False,
                      keyboard=False, cell_phone=False, microwave=False,
                      oven=False, toaster=False, sink=False, refrigerator=False, book=False, clock=False, vase=False,
                      scissors=False, teddy_bear=False, hair_dryer=False,
                      toothbrush=False):

        """
                         The 'CustomObjects()' function allows you to handpick the type of objects you want to detect
                         from an image. The objects are pre-initiated in the function variables and predefined as 'False',
                         which you can easily set to true for any number of objects available.  This function
                         returns a dictionary which must be parsed into the 'detectCustomObjectsFromImage()'. Detecting
                          custom objects only happens when you call the function 'detectCustomObjectsFromImage()'
                        * true_values_of_objects (array); Acceptable values are 'True' and False  for all object values present
                        :param boolean_values:
                        :return: custom_objects_dict
                """

        custom_objects_dict = {}
        input_values = [person, bicycle, car, motorcycle, airplane,
                        bus, train, truck, boat, traffic_light, fire_hydrant, stop_sign,
                        parking_meter, bench, bird, cat, dog, horse, sheep, cow, elephant, bear, zebra,
                        giraffe, backpack, umbrella, handbag, tie, suitcase, frisbee, skis, snowboard,
                        sports_ball, kite, baseball_bat, baseball_glove, skateboard, surfboard, tennis_racket,
                        bottle, wine_glass, cup, fork, knife, spoon, bowl, banana, apple, sandwich, orange,
                        broccoli, carrot, hot_dog, pizza, donut, cake, chair, couch, potted_plant, bed,
                        dining_table, toilet, tv, laptop, mouse, remote, keyboard, cell_phone, microwave,
                        oven, toaster, sink, refrigerator, book, clock, vase, scissors, teddy_bear, hair_dryer,
                        toothbrush]
        actual_labels = ["person", "bicycle", "car", "motorcycle", "airplane",
                         "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign",
                         "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear",
                         "zebra",
                         "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis",
                         "snowboard",
                         "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
                         "tennis racket",
                         "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
                         "orange",
                         "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
                         "bed",
                         "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
                         "microwave",
                         "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
                         "hair dryer",
                         "toothbrush"]

        for input_value, actual_label in zip(input_values, actual_labels):
            if (input_value == True):
                custom_objects_dict[actual_label] = "valid"
            else:
                custom_objects_dict[actual_label] = "invalid"

        return custom_objects_dict

    def detectCustomObjectsFromImage_Me(self, input_image="", output_image_path="", input_type="file", output_type="file",
                               extract_detected_objects=False, minimum_percentage_probability=50,
                               display_percentage_probability=True, display_object_name=True,
                               display_box=True, thread_safe=False, custom_objects=None):
        
        warnings.warn("'detectCustomObjectsFromImage()' function has been deprecated and will be removed in future versions of ImageAI. \n Kindly use 'detectObjectsFromImage()' ",
         DeprecationWarning, stacklevel=2)
        
        return self.detectObjectsFromImage_Me(input_image=input_image,
                                            output_image_path=output_image_path, 
                                            input_type=input_type, 
                                            output_type=output_type,
                                            extract_detected_objects=extract_detected_objects, 
                                            minimum_percentage_probability=minimum_percentage_probability,
                                            display_percentage_probability=display_percentage_probability, 
                                            display_object_name=display_object_name,
                                            display_box=display_box, 
                                            thread_safe=thread_safe, 
                                            custom_objects=custom_objects)
        

        
