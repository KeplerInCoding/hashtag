from flask import Flask, render_template, Response
import cv2
import argparse

app = Flask(__name__,template_folder='template',static_folder='static')

parser=argparse.ArgumentParser()
parser.add_argument('--image')

args, unknown = parser.parse_known_args()

faceProto="./model/opencv_face_detector.pbtxt"
faceModel="./model/opencv_face_detector_uint8.pb"
ageProto="./model/age_deploy.prototxt"
ageModel="./model/age_net.caffemodel"
genderProto="./model/gender_deploy.prototxt"
genderModel="./model/gender_net.caffemodel"

MODEL_MEAN_VALUES=(78.4263377603, 87.7689143744, 114.895847746)
genderList=['Male','Female']
ageList=['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']

padding=20
video=cv2.VideoCapture(args.image if args.image else 0)

faceNet=cv2.dnn.readNet(faceModel,faceProto)
ageNet=cv2.dnn.readNet(ageModel,ageProto)
genderNet=cv2.dnn.readNet(genderModel,genderProto)

cam = cv2.VideoCapture(0)

def gen_frame():
    while True:
        success, frame = cam.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

        yield(b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' +frame + b'\r\n')

def video_capture():  
    while True:
       
        success, frame = cam.read()  
        if not success:
            break
        else:

            frameOpencvDnn=frame.copy()
            frameHeight=frameOpencvDnn.shape[0]
            frameWidth=frameOpencvDnn.shape[1]
            blob=cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

            faceNet.setInput(blob)
            detections=faceNet.forward()
            faceBoxes=[]
            for i in range(detections.shape[2]):
                confidence=detections[0,0,i,2]
                if confidence>0.7:
                    x=int(detections[0,0,i,3]*frameWidth)
                    y=int(detections[0,0,i,4]*frameHeight)
                    xx=int(detections[0,0,i,5]*frameWidth)
                    yy=int(detections[0,0,i,6]*frameHeight)
                    faceBoxes.append([x,y,xx,yy])
                    cv2.rectangle(frameOpencvDnn, (x,y), (xx,yy), (0,255,0), int(round(frameHeight/150)), 8)
            
            if not faceBoxes:
                print("No face detected")

            for faceBox in faceBoxes:
                face=frame[max(0,faceBox[1]-padding):
                        min(faceBox[3]+padding,frame.shape[0]-1),max(0,faceBox[0]-padding)
                        :min(faceBox[2]+padding, frame.shape[1]-1)]

                blob=cv2.dnn.blobFromImage(face, 1.0, (227,227), MODEL_MEAN_VALUES, swapRB=False)

                genderNet.setInput(blob)
                genderPreds=genderNet.forward()
                gender=genderList[genderPreds[0].argmax()]
                print(f'Gender: {gender}')

                ageNet.setInput(blob)
                agePreds=ageNet.forward()
                age=ageList[agePreds[0].argmax()]
                print(f'Age: {age[1:-1]} years')

                cv2.putText(frameOpencvDnn, f'{gender}, {age}', (faceBox[0], faceBox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2, cv2.LINE_AA)

            ret, buffer = cv2.imencode('.jpg', frameOpencvDnn)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(video_capture(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video')
def video():
    # return Response(gen_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')
    return render_template('vid.html')

# if __name__ == '__main__':
#     app.run(debug=True)