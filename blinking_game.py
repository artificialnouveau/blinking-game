import cv2
import dlib
import time
import imutils
import pygame
from imutils.video import VideoStream
from scipy.spatial import distance as dist

# constants for EAR
EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 3

# initialize dlib's face detector (HOG-based) and the facial landmarks predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# get the indexes of facial landmarks for the left and right eye, respectively
(L_START, L_END) = (42, 48)
(R_START, R_END) = (36, 42)

# pygame initialization for timer
pygame.init()
pygame.font.init()
font = pygame.font.SysFont('Arial', 30)

# function to calculate EAR given eye's landmarks
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# initialize the frame counter and a boolean used to indicate if the alarm is going off
frame_counter = 0

# start the video stream thread
print("[INFO] starting video stream thread...")
vs = VideoStream(src=0).start()
time.sleep(1.0)

start_time = pygame.time.get_ticks()
blink_flag = False
loser_flag = False

# loop over frames from the video stream
while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=700)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect faces in the grayscale frame
    rects = detector(gray, 0)
    
    for rect in rects:
        shape = predictor(gray, rect)
        shape = imutils.FacialLandmarksParse(shape)

        # calculate EAR for both eyes
        leftEye = shape[L_START:L_END]
        rightEye = shape[R_START:R_END]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)

        ear = (leftEAR + rightEAR) / 2.0
        
        if ear < EYE_AR_THRESH:
            frame_counter += 1
            if not blink_flag:
                loser_flag = True
                blink_flag = True
                cv2.imwrite('loser.png', frame)  # Save the frame as 'loser.png'
        else:
            if frame_counter >= EYE_AR_CONSEC_FRAMES:
                cv2.putText(frame, "Blink detected!", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            frame_counter = 0

        cv2.putText(frame, "Blinks: {}".format(frame_counter), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        if loser_flag:
            cv2.putText(frame, "LOSER", (rect.left(), rect.bottom()), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        else:
            cv2.putText(frame, "WINNER", (rect.left(), rect.bottom()), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            cv2.imwrite('winner.png', frame)  # Save the frame as 'winner.png'
    
    # render the timer
    time_passed = pygame.time.get_ticks() - start_time
    time_passed_seconds = time_passed / 1000
    timer_text = font.render("Timer: " + str(time_passed_seconds), True, (255, 255, 255))
    frame.blit(pygame.surfarray.make_surface(timer_text), (10, 650))

    # show the frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cv2.destroyAllWindows()
vs.stop()
