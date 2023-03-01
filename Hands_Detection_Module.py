import mediapipe as mp
import cv2
import time

mp_holistic = mp.solutions.holistic
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

currTime = 0
prevTime = 0


def mediapipe_detection(img, model):
    '''
    function by default detect hands by mediapipe ,
    it receives two parameters , an image and the model to process the image
    :return the image as it was and list with result of the process
    the results will be type mediapipe.solutions_base.SolutionsOutputs (class_)
    '''
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img.flags.writeable = False
    results = model.process(img)
    img.flags.writeable = True
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img, results


def drawing_utilities(image, results):
    '''
    Function receives two parameters , an image and a result of processing an image .
    first we check if opencv detecting a hands , then we iterate through those results
    and draw connections between the landmarks
    '''
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, handLms, mp_hands.HAND_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4))


def detect_fingers_positions(image, results):
    Landmark_dict = {}
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                h, w, c = image.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                Landmark_dict[id] = [cx, cy]
    return Landmark_dict


def fingersUp(lmList):
    tipIds = [4, 8, 12, 16, 20]
    fingers = []
    # Thump
    if lmList[tipIds[0]][0] < lmList[tipIds[0] - 1][0]:
        fingers.append(1)
    else:
        fingers.append(0)

    # 4 fingers
    for id in range(1, 5):
        if lmList[tipIds[id]][1] < lmList[tipIds[id] - 2][1]:
            fingers.append(1)
        else:
            fingers.append(0)
    return fingers

def Capture_Video():
    global currTime, prevTime
    cap = cv2.VideoCapture(0)
    with mp_hands.Hands() as hands:
        while cap.isOpened():
            success, frame = cap.read()
            image, results = mediapipe_detection(frame, hands)
            drawing_utilities(image, results)
            # tempDict = detect_fingers_positions(image, results)
            currTime = time.time()
            fps = 1 / (currTime - prevTime)
            prevTime = currTime
            cv2.putText(image, str(int(fps)), (10, 70), cv2.FONT_ITALIC, 3, (100, 255, 70), 3)
            cv2.imshow("Live feed", image)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
