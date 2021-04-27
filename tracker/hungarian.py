from sklearn.utils.linear_assignment_ import linear_assignment

def hungarian (trackers, detections, iou_thrd = 0.3):
    '''
    From current list of trackers and new detections, output matched detections,
    unmatchted trackers, unmatched detections.
    '''    
    
    IOU_mat= np.zeros((len(trackers),len(detections)),dtype=np.float32)
    for t,trk in enumerate(trackers):
        #trk = convert_to_cv2bbox(trk) 
        for d,det in enumerate(detections):
         #   det = convert_to_cv2bbox(det)
            IOU_mat[t,d] = box_iou2(trk.bbox,det.bbox) 
    
    # Produces matches       
    # Solve the maximizing the sum of IOU assignment problem using the
    # Hungarian algorithm (also known as Munkres algorithm)
    
    matched_idx = linear_assignment(-IOU_mat)        

    unmatched_trackers, unmatched_detections = [], []
    for t,trk in enumerate(trackers):
        if(t not in matched_idx[:,0]):
            unmatched_trackers.append(trk)

    for d, det in enumerate(detections):
        if(d not in matched_idx[:,1]):
            unmatched_detections.append(det)

    matches = []
   
    # For creating trackers we consider any detection with an 
    # overlap less than iou_thrd to signifiy the existence of 
    # an untracked object
    
    for m in matched_idx:
        if (IOU_mat[m[0],m[1]]<iou_thrd):
            unmatched_trackers.append(trackers[m[0]])
            unmatched_detections.append(detections[m[1]])
        else:
            matches.append([trackers[m[0]],detections[m[1]]])
    
    if (len(matches)==0):
        matches = np.empty((0,2),dtype=int)
    else:
        matches = np.array(matches)
    
    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)       
    
