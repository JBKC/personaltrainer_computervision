'''
Utilises the separate pose_tracking_module file
Analyse a set of squats to output: 
(1) number of reps completed 
(2) max knee angle achieved at base of squat for each rep
(3) draw a hip trace for each rep
'''

import cv2
import numpy as np
import time
import copy
import pose_tracking_module as pm

cap = cv2.VideoCapture('your_video_file.mov')

pTime = 0
# create an object of the 'poseDetector' class called 'detector'
detector = pm.poseDetector()

# rep tracking initial conditions
dir = 0                     # direction (0 = down, 1 = up)
count = 0                   # number of reps till this point
max_angle = 0               # running measure of the max angle to this point per rep
threshold = 10              # minimum degree of limb that counts as a completed rep

depth_angles = []           # historical tracking of each rep's max angle
new_rep = True              # tracker to reset node trace at the start of each rep


while True:
    success, img = cap.read()
    if not success:
        print(f'Reps completed: {int(count)}')
        print(depth_angles)
        break

    detector.findPose(img,False)         
    lmList = detector.findPosition(img,False)

    # if landmarks are detected in the image:
    if len(lmList) != 0:
        angle = detector.findAngle(img, 24, 26, 28, False)      # get angle of limb
        per = np.interp(angle,(10,120),(0,100))                 # determine percent completed of (full) rep

        max_angle = max(max_angle, angle)                       # find the maximum angle until this point
        max_angle = float('{:.2f}'.format(max_angle))           

    # initial conditions at the top of the squat
    if angle < threshold and dir == 1:
        count += 0.5                                        # second half of rep achieved (rep completed)
        # save stats to output image
        img_save = img.copy()                               
        detector.traceNodeFree(img_save, 24, new_rep)       
        cv2.putText(img_save, f'Max angle: {max_angle}', (10, 70),
            cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)      
        cv2.imwrite(f"rep_{int(count)}.jpg", img_save)      

        dir = 0                                             # rest direction
        max_angle = 0                                       # reset the max angle for the new rep
        new_rep = False                                     # reset the rep tracker
        detector.traceNodeFree(img, 24, new_rep)            # clear the prior rep's node trace

    if angle >= threshold and dir == 0:                     
        new_rep = True

    # establish the base of the squat
    if angle > threshold and max_angle - angle > 2 and dir == 0:    # 2 degrees less max angle implies that max angle has been achieved
        count += 0.5
        dir = 1
        depth_angles.append(max_angle)                     

    if new_rep:
        detector.traceNodeFree(img, 24, new_rep)            # trace the hip position over the course of the rep
        detector.findAngle(img, 24, 26, 28, True)           # plots the angle in realtime

    # frames per second counter
    cTime = time.perf_counter()        
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,(0, 255, 0), 3)                         # FPS display
    cv2.putText(img, str(int(count)), (img.shape[1] - 100, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)      # rep counter display

    cv2.imshow("Image", img)
    cv2.waitKey(1)
