import tkinter as tk
from tkinter import filedialog
import cv2
from PIL import Image, ImageTk
import argparse
import numpy as np

class AgeGenderDetectorApp:
    def __init__(self, master, video_source=0):
        self.master = master
        master.title("Age and Gender Detector")

        self.video_source = video_source
        self.video = cv2.VideoCapture(self.video_source)

        self.canvas = tk.Canvas(master, width=self.video.get(cv2.CAP_PROP_FRAME_WIDTH), height=self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.canvas.pack()

        self.padding = 20
        self.faceProto = "./model/opencv_face_detector.pbtxt"
        self.faceModel = "./model/opencv_face_detector_uint8.pb"
        self.ageProto = "./model/age_deploy.prototxt"
        self.ageModel = "./model/age_net.caffemodel"
        self.genderProto = "./model/gender_deploy.prototxt"
        self.genderModel = "./model/gender_net.caffemodel"

        self.MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
        self.ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
        self.genderList = ['Male','Female']

        self.faceNet = cv2.dnn.readNet(self.faceModel, self.faceProto)
        self.ageNet = cv2.dnn.readNet(self.ageModel, self.ageProto)
        self.genderNet = cv2.dnn.readNet(self.genderModel, self.genderProto)

        self.update()

    def update(self):
        ret, frame = self.video.read()

        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result_img, faceBoxes = self.highlightFace(frame_rgb)
            self.displayAgeGender(result_img, faceBoxes)
            self.photo = ImageTk.PhotoImage(image=Image.fromarray(result_img))
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)

        self.master.after(10, self.update)

    def __del__(self):
        if self.video.isOpened():
            self.video.release()

    def highlightFace(self, frame, conf_threshold=0.7):
        frameOpencvDnn = frame.copy()
        frameHeight = frameOpencvDnn.shape[0]
        frameWidth = frameOpencvDnn.shape[1]
        blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

        self.faceNet.setInput(blob)
        detections = self.faceNet.forward()
        faceBoxes = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > conf_threshold:
                x1 = int(detections[0, 0, i, 3] * frameWidth)
                y1 = int(detections[0, 0, i, 4] * frameHeight)
                x2 = int(detections[0, 0, i, 5] * frameWidth)
                y2 = int(detections[0, 0, i, 6] * frameHeight)
                faceBoxes.append([x1, y1, x2, y2])
                cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight / 150)), 8)
        
        return frameOpencvDnn, faceBoxes

    def displayAgeGender(self, frame, faceBoxes):
        for faceBox in faceBoxes:
            face = frame[max(0, faceBox[1] - self.padding): min(faceBox[3] + self.padding, frame.shape[0] - 1),
                         max(0, faceBox[0] - self.padding): min(faceBox[2] + self.padding, frame.shape[1] - 1)]

            blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), self.MODEL_MEAN_VALUES, swapRB=False)
            self.genderNet.setInput(blob)
            genderPreds = self.genderNet.forward()
            gender = self.genderList[genderPreds[0].argmax()]
            print(f'Gender: {gender}')

            self.ageNet.setInput(blob)
            agePreds = self.ageNet.forward()
            age = self.ageList[agePreds[0].argmax()]
            print(f'Age: {age[1:-1]} years')

            cv2.putText(frame, f'{gender}, {age}', (faceBox[0], faceBox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

def main():
    root = tk.Tk()
    app = AgeGenderDetectorApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
