import cv2
import mediapipe as mp
import time
import math
import numpy as np

class poseDetector():

    def __init__(self,mode=False,complexity=1,smooth_lm=True,enable_seg=False,smooth_seg=True,detection_con=0.5,tracking_con=0.5):
        # variables taken from the Pose() module, and hence need to be repeated when Pose() is called below as well

        # create new variables of the object that are a repeat of the terms above
        self.mode = mode
        self.complexity = complexity
        self.smooth_lm = smooth_lm
        self.enable_seg = enable_seg
        self.smooth_seg = smooth_seg
        self.detection_con = detection_con
        self.tracking_con = tracking_con

        self.landmark_history = []          # used for pose landmarks on the body
        self.fixed_point_history = []       # used for fixed reference point


        # define the core functionalities of the code
        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        # create object pose (instance of the class) - hence the 'mpPose.Pose'
        self.pose = self.mpPose.Pose(self.mode, self.complexity, self.smooth_lm, self.enable_seg, self.smooth_seg, self.detection_con, self.tracking_con)

    def findPose(self,img,draw=True):
        '''
        Method to find and draw the landmarks on the input image
        '''

        # same code as in pose_tracking_main
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)                # process the converted image with mediapipe

        # if self.results.pose_landmarks:             # if pose can be detected in the image, draw the landmarks
        if draw:                                # i.e. if draw=True as defined in the function call
            self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
        return img

    def findPoseBoxed(self,img,x,y,w,h,draw=True):
        '''
        Method to find and draw the landmarks GIVEN A SPECIFIC BOUNDING BOX (e.g. selecting a specific person from a group)
        '''

        try:
            box_img = img[y:y+h, x:x+w]                         # conduct pose search only within selected bounding box
            imgRGB = cv2.cvtColor(box_img, cv2.COLOR_BGR2RGB)
            self.results = self.pose.process(imgRGB)                # find the landmarks that lie within the bounding box

            # only relevant if you've specified draw=True in executing code
            # if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)

        except Exception:
            pass

        return img
        # if you return box_img, it will just show the cropped rectangle instead of the whole image with rectangle inside

    def findPositionBoxed(self,img,x,y,w,h,draw=True):
        '''
        Method to find the coordinates of the landmarks within the box
        '''

        # landmark coordinates are currently using the box as reference - need to get them to image reference (i.e. 'shrink' them down)

        self.lmList = []         # list of landmarks

        if self.results.pose_landmarks:
            # same code as in pose_tracking_main
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h_img, w_img, _ = img.shape           # _ = channels (unused)
                # cx, cy = int(lm.x*w_img), int(lm.y*h_img) would give the dimensions of the full image (*img dimensions to scale it)
                # hence * w/w_img to scale to box
                # finally, +x to translate to box edges
                cx, cy = int(lm.x*w + x), int(lm.y*h +y)

                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (0, 255, 0), cv2.FILLED)

        return self.lmList

    def findPosition(self, img, draw=True):
        '''
        Method to find the coordinates of the landmarks
        '''

        self.lmList = []         # list of landmarks
        if self.results.pose_landmarks:
            # same code as in pose_tracking_main
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h,w,c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (0, 255, 0), cv2.FILLED)
        return self.lmList

    def findAngle(self, img, p1, p2, p3, draw=True):
        '''
        Method to find angle of a limb between 3 nodes (p1,p2,p3) and draw the limb in question
        '''

        try:
            # select the landmarks from the rows of lmList - format = lmLList[node no.][id,cx,cy]
            x1, y1 = self.lmList[p1][1:]
            x2, y2 = self.lmList[p2][1:]
            x3, y3 = self.lmList[p3][1:]

            # calculate the angle
            angle = math.degrees(math.atan2(y3-y2,x3-x2) - math.atan2(y1-y2,x1-x2))
            angle = angle % 360
            # convert into more logical format (0 = straight)
            angle = np.interp(angle,(180,360),(0,180))

            if draw:
                cv2.circle(img, (x1, y1), 10, (255, 0, 0), cv2.FILLED)
                cv2.circle(img, (x2, y2), 10, (255, 0, 0), cv2.FILLED)
                cv2.circle(img, (x3, y3), 10, (255, 0, 0), cv2.FILLED)
                cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 3)
                cv2.line(img, (x2, y2), (x3, y3), (255, 255, 255), 3)
                # placement of the angle text
                cv2.putText(img,str(int(angle)),(x2+50,y2),cv2.FONT_HERSHEY_PLAIN,3,
                        (255,255,255),3)

        except Exception:
            pass

        return angle

    def traceNodeFree(self, img, node, new_rep, draw=True):
        '''
        Method to track historical position of node over time, relative to camera's frame of reference
        '''

        try:
            # create new list tracking historical points of specific node
            # this is so all prior nodes can be plotted
            # BUT if it's a new rep, need to make sure node tracking is reset so you get a fresh trace for each rep

            if not new_rep:         # ie if new_rep is false
                self.landmark_history = []

            self.landmark_history.append(self.lmList[node])

            # self.landmark_history = self.landmark_history[-5:]

            if draw:
                # this for loop ensures all previous frames until now are plotted, not just the current frames. Hence creating a trace.
                for frame in self.landmark_history:
                    x, y = frame[1:]
                    # print(x,y)
                    cv2.circle(img, (x, y), 5, (0, 255, 0), cv2.FILLED)

        except Exception:
            pass

        return img

    def traceNodeFixed(self, img, node, fixed, draw=True):
        '''
        Method to track historical position of node over time, relative to object in the image
        '''

        ### 'TRY' error handling here.
        try:
            # Ensure lmList[node] is not None before appending to landmark_history
            if self.lmList[node] is not None:
                # NOTE fixed should not be a 'self' variable as it will change frame by frame. shouldn't initialize across the instance
                self.landmark_history.append(self.lmList[node])  # append landmark details of body node (format = id, x, y)
                self.fixed_point_history.append(fixed)  # append fixed reference node coordinates (format = x,y)
                # print(self.landmark_history)
                # print(len(self.landmark_history))

                if draw and len(self.landmark_history) > 1:
                    ### ERROR - you had a loop here:
                    #   "for i in range(1, len(self.landmark_history))"
                    # creating a double loop with the j loop below. this caused each node to be replotted and shifted back for each node, hence the frame wasn't refreshing each time
                    # LESSON = reason from first principles instead of letting ChatGPT give you the template

                    fixed_x, fixed_y = self.fixed_point_history[-1]  # fixed's current position in camera frame
                    landmark_x, landmark_y = self.landmark_history[-1][1:]  # landmark's current position in camera frame (removing first element, id)

                    cv2.circle(img, (landmark_x, landmark_y), 5, (255, 255, 255), cv2.FILLED)
                    for j in range(2, len(self.landmark_history)):
                        fixed_x_prev, fixed_y_prev = self.fixed_point_history[-j][:]  # fixed landmark's previous position in camera frame
                        shift_x = fixed_x - fixed_x_prev
                        shift_y = fixed_y - fixed_y_prev
                        landmark_x_prev, landmark_y_prev = self.landmark_history[-j][1:]  # landmark previous position in camera frame

                        landmark_x_shifted = landmark_x_prev + shift_x
                        landmark_y_shifted = landmark_y_prev + shift_y

                        # Plot relative positions (relative to the fixed background point)
                        cv2.circle(img, (landmark_x_shifted, landmark_y_shifted), 5, (255, 255, 255), cv2.FILLED)

        except Exception:
            pass

        return img


def main():
    '''
    Can be thought of as the 'wrapper' of the code that controls + executes the overall task.
    you can actually delete this because it belongs in the external file
    '''

    # cap = cv2.VideoCapture('SprintVids/JB_181223.MOV')
    # cap = cv2.VideoCapture('SprintVids/JB_021123.MP4')
    cap = cv2.VideoCapture('SprintVids/broadjump.mov')

    pTime = 0

    # create an object of the 'poseDetector' class called 'detector'
    detector = poseDetector()

    # run the code as explained in pose_tracking_main
    while True:
        success, img = cap.read()
        if not success:
            break

        img = detector.findPose(img)
        lmList = detector.findPosition(img)
        if len(lmList) != 0:
            detector.findAngle(img, 24, 26, 28)  # starting with right leg

        cTime = time.perf_counter()
        fps = 1/(cTime-pTime)
        pTime = cTime

        cv2.putText(img,str(int(fps)),(10,70), cv2.FONT_HERSHEY_PLAIN,3,
                    (255,255,255),3)

        cv2.imshow("Image",img)
        # can play with the delay here to manipulate the framerate
        cv2.waitKey(1)

# finally, run the main function
# to understand the __name__ feature, see Notion notes
if __name__ == '__main__':
    main()