# #hashtag: Age-and-Gender-Detection

<h2>Objective :</h2>
<p>To build a gender and age detector extension tool that can approximately guess the gender and age of the person (face) in a picture or through webcam.</p>

<h2>About the Project :</h2>
<p>In this Project, we had used Deep Learning to accurately identify the gender and age of a person from a single image of a face.We used the models trained by <a href="https://talhassner.github.io/home/projects/Adience/Adience-data.html">Tal Hassner and Gil Levi</a>. The predicted gender may be one of ‘Male’ and ‘Female’, and the predicted age may be one of the following ranges- (0 – 2), (4 – 6), (8 – 12), (15 – 20), (25 – 32), (38 – 43), (48 – 53), (60 – 100) (8 nodes in the final softmax layer). It is very difficult to accurately guess an exact age from a single image because of factors like makeup, lighting, obstructions, and facial expressions. And so, we further made an extension which can be used with various systems to provide additional features</p>

<h2>Dataset :</h2>
<p>For this project, we had used the Adience dataset; the dataset is available in the public domain and you can find it <a href="https://www.kaggle.com/ttungl/adience-benchmark-gender-and-age-classification">here</a>. This dataset serves as a benchmark for face photos and is inclusive of various real-world imaging conditions like noise, lighting, pose, and appearance. The images have been collected from Flickr albums and distributed under the Creative Commons (CC) license. It has a total of 26,580 photos of 2,284 subjects in eight age ranges (as mentioned above) and is about 1GB in size. The models we used had been trained on this dataset.</p>

<h2>Additional Python Libraries Required :</h2>
<ul>
  <li>OpenCV</li>
  
       pip install opencv-python
</ul>
<ul>
 <li>argparse</li>
  
       pip install argparse
</ul>

<h2>The contents of this Project :</h2>
<ul>
  <li>opencv_face_detector.pbtxt</li>
  <li>opencv_face_detector_uint8.pb</li>
  <li>age_deploy.prototxt</li>
  <li>age_net.caffemodel</li>
  <li>gender_deploy.prototxt</li>
  <li>gender_net.caffemodel</li>
  <li>a few pictures to try the project on</li>
  <li>gad.py</li>
 </ul>
 <p>For face detection, we have a .pb file- this is a protobuf file (protocol buffer); it holds the graph definition and the trained weights of the model. We can use this to run the trained model. And while a .pb file holds the protobuf in binary format, one with the .pbtxt extension holds it in text format. These are TensorFlow files. For age and gender, the .prototxt files describe the network configuration and the .caffemodel file defines the internal states of the parameters of the layers.</p>
 
# Working:

<h2>Examples :</h2>
<p><b>NOTE:- Reference images are from Google,if you have any query or problem we can remove them.</b></p>

    >python gad.py --image Detecting age and gender girl1.png
    Gender: Female
    Age: 25-32 years
    
<img src="Example/Detecting age and gender girl1.png">

    >python gad.py --image Detecting age and gender girl2.png
    Gender: Female
    Age: 8-12 years
    
<img src="Example/Detecting age and gender girl2.png">

    >python gad.py --image Detecting age and gender kid1.png
    Gender: Male
    Age: 4-6 years    
    
<img src="Example/Detecting age and gender kid1.png">

    >python gad.py --image Detecting age and gender kid2.png
    Gender: Female
    Age: 4-6 years  
    
<img src="Example/Detecting age and gender kid2.png">

    >python gad.py --image Detecting age and gender man1.png
    Gender: Male
    Age: 38-43 years
    
<img src="Example/Detecting age and gender man1.png">

    >python gad.py --image Detecting age and gender man2.png
    Gender: Male
    Age: 25-32 years
    
<img src="Example/Detecting age and gender man2.png">

    >python gad.py --image Detecting age and gender woman1.png
    Gender: Female
    Age: 38-43 years
    
<img src="Example/Detecting age and gender woman1.png">

 # PREVIEW:

  ### Website:

https://github.com/aksshatgovind/Age-and-Gender-Detection/assets/105073216/b4084157-2a70-4154-a842-ca567e0c4946

  ### Prototype:

https://github.com/aksshatgovind/Age-and-Gender-Detection/assets/105073216/e15cc2b3-52da-48b3-ad95-f587786317ca

 # EXTENSION:
 
 <p><b>NOTE:- Our website which will provide link to download the extension, is still in progress. We'll keep updating to the contents of this Repository, and thank you for your time.</b></p>
 
