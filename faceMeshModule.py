import cv2
import mediapipe as mp
import time


class FaceMeshDetector():

    def __init__(self, staticMode=False, maxFaces=2, minDetectionConf=0.5, minTrackConf=0.5,
                drawnThickness=1, drawnCircleRadius=2, drawnColor=(255,100,80)):

        self.staticMode = staticMode
        self.maxFaces = maxFaces
        self.minDetectionConf = minDetectionConf
        self.minTrackConf = minTrackConf
        self.drawnThickness = drawnThickness
        self.drawnCircleRadius = drawnCircleRadius
        self.drawnColor = drawnColor

        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(self.staticMode, self.maxFaces,
                                                self.minDetectionConf, self.minTrackConf)
        self.drawSpec = self.mpDraw.DrawingSpec(thickness = self.drawnThickness, 
                                                circle_radius = self.drawnCircleRadius,
                                                color = self.drawnColor)

    def findFaceMesh(self, img, draw=True):

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.faceMesh.process(imgRGB)

        faces=[]
        if results.multi_face_landmarks:
            for faceLms in results.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, faceLms, self.mpFaceMesh.FACE_CONNECTIONS, 
                                                self.drawSpec, self.drawSpec)
                face=[]
                for id, lm in enumerate(faceLms.landmark):
                    ih, iw, ic = img.shape
                    x, y = int(lm.x*iw), int(lm.y*ih)
                    # Ver posição dos landmarkers:
                    # cv2.putText(img, f'{id}', (x, y), cv2.FONT_HERSHEY_PLAIN, 0.5, (255, 0, 0), 1)
                    face.append([x, y])

                faces.append(face)

        return img, faces


    


def main():
    cap = cv2.VideoCapture(0) # 0 (WebCan) ou "videos/1.mp4" (endereço do vídeo)
    pTime = 0

    detector = FaceMeshDetector(drawnThickness=2)
    
    while True:
        success, img = cap.read()

        img, faces = detector.findFaceMesh(img)
        # if len(faces) != 0:
        #     print(len(faces))

        cTime = time.time()
        fps = 1/(cTime - pTime)
        pTime = cTime
        cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 3)

        cv2.imshow("Image", img)
        cv2.waitKey(1)



if __name__ == "__main__":
    main()