#  📹 meteor-realtime-object-tracking 📷

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

 - [x] The Pedestrians that have their bounding boxes matching `[k-th:matching:(k-1)-th]`, would keep their originel IDs, based on the tracking system of the Hungarian Algorithm.
 - [x] The pedestrians from the k-th detection that do not have any match in the (k-1)-th detection can either be new pedestrians entering into the camera's vision sight or even false detections. Therefore, a period of 3 consecutive detections will be observed before accepting them as new pedestrians.
 - [x] The pedestrians from the (k-1)-th detection that do not have any match in the k-th detection are the most difficult to deal with. These might pedestrians leaving the camera's vision sight, they might be pedestrians in occlusion (either with other pedestrians or any larger objects). 
 
 - [ ] In the case of the potential occlusion, the tracking system of the Hungarian Algorithm joined to the Kalman Filter would be used to predict the tracjectories of the concerned pedestrians and then predict their boxes positions after occlusion.
 - [ ] In the case of the pedestrians leaving the camera's vision sight, the cannot be automatically erase from memory, as there a possibility of them re-entering back in the vision. Therefore, a smarter technique is to give them a certain period of time and affecting them specific caracteristics in order to recognize them once there are back in the vision, or finally erase their IDs once the period of time is over.



## Visualization & Critics of the results

This is a sequence of a videoshot used in the aim of testing the prototype. As seen, this sequence relavely not dense in terms of number of pedestrians, therefore the risk of having occlusion is high reduced. 
`Original`             |  `Detected & Tracked`
:-------------------------:|:-------------------------:
![](https://github.com/IsmaelMekene/meteor-realtime-object-tracking/blob/main/data/testvid.gif)  |  ![](https://github.com/IsmaelMekene/meteor-realtime-object-tracking/blob/main/data/detection.gif)

However, the main difficulty of this project is still to deal with occlusion situation, as I could not find a better way to solve it (Not Yet!).





Please feel free to leave a comment in case you have a better trick, Thanks 😉!



