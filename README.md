# meteor-realtime-object-tracking

This is a Computer Vision project aiming to create a real time object detector based on Retinanet (either YOLO) and Build a logical tracking system in the mean time. The implementation of this prototype has been based from transfer Learning with pretrained models on COCO Dataset.



## Goal 

This project aims to detect and track pedestrians (in motion or not) from a given videoshot in real time.

## Installation 

            pip install -r requirements.txt

## Data, Data processing & Model

As this project is based on a transfer learning, the pretrained weights have originally resulted from trainings over the famous COCO dataset. Moreover, in the context of this task, further traing were done on images collected personally. Those images are typical street shots, with pedestrians (In motion or not, from near or far and from a high view or not).

The training has been undertaken for the retinanet model as well as the YOLO model and both weights are available in [here]()

## Pipeline
      
<p align="center">
  <img src="https://github.com/IsmaelMekene/meteor-realtime-object-tracking/blob/main/data/pipeline.svg"/>
</p>


## Detection

For a given videoshot, detections are made each 5 frames. The first frame corresponding to the first detection also, will record the number of potential pedestrians and the record will be passed to the Scene Class. After the second detection, the process of tracking can technically begin.
For a j-th frame corresponding to a k-th detection, the system of tracking is used in pair with its (k-1)-th detection, technically corresponding to the (j-5)th frame.

 [x]the Pedestrians that have their bounding boxes matching `[k-th:matching:(k-1)-th]`, would keep their originel IDs.

## Inferences

`Original`             |  `Detected & Tracked`
:-------------------------:|:-------------------------:
![](https://github.com/IsmaelMekene/meteor-realtime-object-tracking/blob/main/data/testvid.gif)  |  ![](https://github.com/IsmaelMekene/meteor-realtime-object-tracking/blob/main/data/detection.gif)



