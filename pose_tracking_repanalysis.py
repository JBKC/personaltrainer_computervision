'''
Analyse a set of squats: (1) count reps (2) calculate max angle (3) draw trace image of hip track
'''

import cv2
import numpy as np
import time
import copy
import pose_tracking_module as pm

cap = cv2.VideoCapture('SprintVids/Muna Squat2.mov')

pTime = 0
# create an object of the 'poseDetector' class called 'detector'
detector = pm.poseDetector()

# rep tracking initial conditions
dir = 0                     # direction (0 = down, 1 = up)
count = 0                   # number of reps till this point
max_angle = 0               # running measure of the max angle to this point per rep
threshold = 10              # minimum degree of limb that counts as a completed rep (at the top)

depth_angles = []           # historical tracking of each rep's max angle
new_rep = True              # tracker to reset node trace at the start of each rep (if True, will continually draw)


# run the code as explained in pose_tracking_main
while True:
    success, img = cap.read()
    if not success:
        print(f'Reps completed: {int(count)}')
        print(depth_angles)
        break

    detector.findPose(img,False)          # set to False so it only draws the points called out in the module
    lmList = detector.findPosition(img,False)

    # if landmarks are detected in the image:
    if len(lmList) != 0:
        angle = detector.findAngle(img, 24, 26, 28, False)       # get angle of limb, but don't plot
        per = np.interp(angle,(10,120),(0,100))                 # determine percent completed of (full) rep

        max_angle = max(max_angle, angle)                       # find the maximum angle until this point
        max_angle = float('{:.2f}'.format(max_angle))           # convert to 2dp

    # initial conditions at the top of the squat
    if angle < threshold and dir == 1:
        count += 0.5                                        # second half of rep achieved (rep completed)
        img_save = img.copy()                               # make a copy of the image for saving
        detector.traceNodeFree(img_save, 24, new_rep)       # ensure node tracking is plotted on the image to save
        cv2.putText(img_save, f'Max angle: {max_angle}', (10, 70),
            cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)      # print the max angle on the image
        cv2.imwrite(f"rep_{int(count)}.jpg", img_save)      # save static image of the completed rep with trace

        dir = 0                                             # reset direction so it knows we're on a new rep
        max_angle = 0                                       # reset the max angle for the new rep
        new_rep = False                                     # reset the rep tracker
        detector.traceNodeFree(img, 24, new_rep)            # clear the prior rep's node trace

    if angle >= threshold and dir == 0:                     # once passed rep threshold, new rep is triggered
        new_rep = True

    # establish the base of the squat
    if angle > threshold and max_angle - angle > 2 and dir == 0:    # if angle drops by 2 degrees below max angle, know that we are on the way up and the max angle has been reached
        count += 0.5
        dir = 1
        depth_angles.append(max_angle)                      # append the max angle achieved

    if new_rep:
        detector.traceNodeFree(img, 24, new_rep)            # keep drawing the node whenever new_rep is in effect
        detector.findAngle(img, 24, 26, 28, True)           # keep plotting the angle as new_rep is true also

    cTime = time.perf_counter()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,(0, 255, 0), 3)         # FPS print
    cv2.putText(img, str(int(count)), (img.shape[1] - 100, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)       # rep counter

    cv2.imshow("Image", img)
    cv2.waitKey(1)