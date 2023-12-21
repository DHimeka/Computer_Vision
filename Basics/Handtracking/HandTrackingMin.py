import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)

# Hand tracking module
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

# Frame rate
pTime = 0
cTime = 0

while True:
    success, img = cap.read()
    # Convert to RGB
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Process the image
    results = hands.process(imgRGB)
    # print(results.multi_hand_landmarks)

    # If there are hands in the image
    if results.multi_hand_landmarks:
        # For each hand
        for handLms in results.multi_hand_landmarks:
            # Draw the landmarks
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    # Calculate the frame rate
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    # Display the frame rate
    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                (255, 0, 255), 3)

    # Display the image
    cv2.imshow("Image", img)
    cv2.waitKey(1)
    