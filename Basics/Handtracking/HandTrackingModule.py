import cv2
import mediapipe as mp
import time

class handDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        
        # Hand tracking module
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(static_image_mode=self.mode,
                                        max_num_hands=self.maxHands,
                                        min_detection_confidence=self.detectionCon,
                                        min_tracking_confidence=self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        
        # Convert to RGB
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Process the image
        self.results = self.hands.process(imgRGB)
        # print(results.multi_hand_landmarks)
        
        # If there are hands in the image
        if self.results.multi_hand_landmarks:
            # For each hand
            for handLms in self.results.multi_hand_landmarks:
                # Draw the landmarks
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        
        return img
    
    def findPosition(self, img, handNo=0, draw=True):
        
        lmList = []
        
        # If there are hands in the image
        if self.results.multi_hand_landmarks:
            # Get the hand
            myHand = self.results.multi_hand_landmarks[handNo]
            
            # For each landmark
            for id, lm in enumerate(myHand.landmark):
                # Get the pixel position
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                
                # Add the landmark to the list
                lmList.append([id, cx, cy])
                
                # Draw a circle on the landmark
                if draw:
                    cv2.circle(img, (cx, cy), 5, (0, 0, 255), cv2.FILLED)
        
        return lmList
    
def main():
        
        # Frame rate
        pTime = 0
        cTime = 0
        
        # Video capture
        cap = cv2.VideoCapture(0)
        
        # Hand detector
        detector = handDetector()
        
        while True:
            success, img = cap.read()
            
            # Find the hands
            img = detector.findHands(img)
            # Find the position of the landmarks
            lmList = detector.findPosition(img)
            if len(lmList) != 0:
                print(lmList[4])
                cv2.circle(img, (lmList[4][1], lmList[4][2]), 5, (255, 0, 255), cv2.FILLED)
            
            # Calculate the frame rate
            cTime = time.time()
            fps = 1 / (cTime - pTime)
            pTime = cTime
            
            # Display the frame rate
            cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 5,
                        (255, 0, 255), 3)
            
            # Display the image
            cv2.imshow("Image", img)
            cv2.waitKey(1)

if __name__ == "__main__":
    main()  