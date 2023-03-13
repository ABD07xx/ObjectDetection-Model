#Library imports
import streamlit as st
import numpy as np
import cv2
from PIL import Image


#Pre-Trained Models
net = cv2.dnn.readNetFromCaffe("MobileNetSSD_deploy.prototxt.txt",'MobileNetSSD_deploy.caffemodel')

#Name of Classes
classes = ['background', 'aeroplane', 'bicycle',  'bird','boat',
            'bottle','bus','car','cat','chair','cow','diningtable',
           'dog','horse','motorbike','person','pottedplant',
            'sheep','sofa','train','tvmonitor']

#Giving random colors to class
colors = np.random.uniform(0,255,size=(len(classes)))

#Setting Title of App
st.title("Object Detection using MobileNet")
st.markdown("Upload an image: ")

#Uploading image
up_image = st.file_uploader(" ", type=['jpg','png','jpeg'])
image = Image.open(up_image)
image = np.array(image)
#Submit Button
submit = st.button('Predict')
#On predict button click
if submit:
    if up_image is not None:
        #Size of image 
        (h, w) = image.shape[:2]
        blob   = cv2.dnn.blobFromImage(cv2.resize(image,(300,300)),0.007843,(300,300),127.5)
        
        #Pass the blob to the network and find the results
        #predictions
        net.setInput(blob)
        detections = net.forward()
        
        #Iterating over the detections
        for i in np.arange(0,detections.shape[2]):
            #Extract the probability of predictions
            prob_detection = detections[0 , 0 , i , 2]
            #Filtering out the weak predictions (threshold here = .60)
            if prob_detection > .60:
                #Find the class acc. to the index
                #After the class find the boundaries for the image(x-y) co-ordinate
                indx = int(detections[0,0,i,1])
                box  = detections[0,0,i,3:7]*np.array([w,h,w,h])
                (startX,startY,endX,endY) = box.astype('int')
                #Display
                label = "{}: {:.2f}%".format(classes[indx],prob_detection*100)
                print("[INFO] {}".format(label))
                cv2.rectangle(image,(startX,startY),(endX,endY),colors[indx],2)
                y = startY - 15 if startY -15 > 15 else startY+15
                cv2.putText(image,label,(startX,y),cv2.FONT_HERSHEY_SIMPLEX,0.6,colors[indx],2)

                st.image(image)
        