from detector.detector import ObjectDetection_Me
from tracker.classes import Pedestrian, Scene, KalmanTracker
from tracker.hungarian import hungarian

from google.colab.patches import cv2_imshow
import os
from tqdm import tqdm 
red = (0, 0, 255)
blue = (225,0,0)
green = (0,255,0)

execution_path = os.getcwd()



detector = ObjectDetection_Me()
detector.setModelTypeAsRetinaNet()
detector.setModelPath( os.path.join(execution_path , "/content/drive/MyDrive/resnet50_coco_best_v2.1.0.h5"))
detector.loadModel()

custom_objects = detector.CustomObjects(person=True)
width= 1080
height= 2069
video_path = '/content/drive/MyDrive/detected_test_chezmoi'+'.avi'
video = cv2.VideoWriter(video_path,cv2.VideoWriter_fourcc('M','J','P','G'),8,(width,height))


vidcap = cv2.VideoCapture('/content/drive/MyDrive/test_chezmoi.mp4')

for i in tqdm(range(441)):
  break

  success, image = vidcap.read()

  if success:

    #cv2_imshow(image)
    rotated = ndimage.rotate(image, -90)

    copy, detections = detector.detectCustomObjectsFromImage_Me(input_image=rotated, input_type="array", output_type="array", minimum_percentage_probability=80,display_percentage_probability=False,custom_objects=custom_objects)
    
    frame_width = int(copy.shape[1])
    #print('width=',frame_width)
    frame_height = int(copy.shape[0])
    #print('height=',frame_height)
    pad_frag_height = int((copy.shape[0]*50)/644)
    #break



  if i%5 ==0:

    if i ==0:

      #cv2_imshow(copy)
      pedestrian0 = []
      for i,det in enumerate(detections):
        bbox,label,confidence = detections[i]['box_points'],detections[i]['name'],detections[i]['percentage_probability']
        ped = Pedestrian(bbox,label,confidence)
        pedestrian0.append(ped)

      sc = Scene(pedestrian0)
      a,b = sc.count()
      print('First Frame: Nb_Pedestrians are', a,'and Unknown are',b)


    if i >0:
      #cv2_imshow(copy)
      pedestrians2 = []
      for i,det in enumerate(detections):
        bbox,label,confidence = detections[i]['box_points'],detections[i]['name'],detections[i]['percentage_probability']
        ped = Pedestrian(bbox,label,confidence)
        pedestrians2.append(ped)

      matched, unmatched_det, unmatched_trk = hungarian(pedestrian0,pedestrians2,iou_thrd=0.3)
      
      for pairs in matched:
        pairs[0].update_ped(pairs[1].bbox)

      new_pedestrian2 = np.concatenate((matched[:,0],unmatched_det),axis=0)

      for tg in matched[:,0]:
        draw_tracks(copy, tg, red)

      for th in unmatched_det:
        draw_tracks(copy,th,blue)

      for ti in unmatched_trk:
        draw_tracks(copy,ti,green)

      

      sc.update(new_pedestrian2,unmatched_trk)

      pedestrian0 = new_pedestrian2.tolist()

      nb_ped, unkn = sc.count()

      frame_width = int(copy.shape[1])
      frame_height = int(copy.shape[0])
      pad_frag_height = int((copy.shape[0]*50)/644)

      final_det = write_each_second(copy,pad_frag_height,nb_ped,unkn)

      video.write(final_det)
      
video.release()

