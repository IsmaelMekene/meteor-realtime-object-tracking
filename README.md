# meteor-realtime-object-tracking

This is a Computer Vision project aiming to create a real time object detector based on Retinanet (either YOLO) and Build a logical tracking system in the mean time. The implementation of this prototype has been based from transfer Learning with pretrained models on COCO Dataset.


## Pipeline
      
<p align="center">
  <img src="https://github.com/IsmaelMekene/meteor-realtime-object-tracking/blob/main/data/pipeline.svg"/>
</p>


## Goal 

This project aims to detect and track pedestrians (in motion or not) from a given videoshot in real time.

## Installation 

                                                `pip install -r requirements.txt`

## Inferences

`Original`             |  `Detected & Tracked`
:-------------------------:|:-------------------------:
![](https://github.com/IsmaelMekene/meteor-realtime-object-tracking/blob/main/data/testvid.gif)  |  ![](https://github.com/IsmaelMekene/meteor-realtime-object-tracking/blob/main/data/detection.gif)
