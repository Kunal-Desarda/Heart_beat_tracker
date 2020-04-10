import os
import cv2
import numpy as np
import faceRecognition as fr
import requests
from tkinter import *
import tkfontchooser
name = {0 : "Priyanka",1 : "Kangana",3 : "kunal"}
root=Tk()
url = 'http://192.168.43.1:8080/shot.jpg'
def Train_data():
    fr.labels_for_training_data('C:/Users/dell1/Downloads/FaceRecognition-master/trainingImages')

def Video_live():
   face_recognizer = cv2.face.LBPHFaceRecognizer_create()
   face_recognizer.read('C:/Users/dell1/Downloads/FaceRecognition-master/trainingData.yml')#Load saved training data

   while True:
      img_resp = requests.get(url)
      img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
      test_img = cv2.imdecode(img_arr,0)
      resized_img=test_img
      faces_detected,gray_img=fr.faceDetection(test_img)
      ##cv2.waitKey(10)
      for (x,y,w,h) in faces_detected:
          cv2.rectangle(test_img,(x,y),(x+w,y+h),(0,255,0,),thickness=5)
          resized_img=cv2.resize(test_img,(1000,700))


      for face in faces_detected:
         (x,y,w,h)=face
         roi_gray=gray_img[y:y+w, x:x+h]
         label,confidence=face_recognizer.predict(roi_gray)#predicting the label of given image
         print("confidence:",confidence)
         print("label:",label)
         fr.draw_rect(test_img,face)
         predicted_name=name[label]

         if (confidence < 39):
             mylabel=Label(root,"="+str(confidence))
             mylabel.pack()
            fr.put_text(test_img, predicted_name, x, y)
            resized_img=test_img
            cv2.imshow('fAce recognized',resized_img)
         else:
             sleep(10):
             cv2.putText(test_img,"Unrecognized",x,y)
             cv2.imshow("unrecognized",test_img,)

     ##cv2.destroyAllWindows


face = StringVar()
face.set(name)
root.geometry("1370x900")
root.title("Ültimate projects")
root.configure(background="powder blue")
lblface = Label(root, text="Faces in Record", font=("arial", 40, 'bold'), padx=2, pady=2, bg="cadet blue")
lblface.grid(row=0, column=0)
txtface = Entry(root, font=("arial", 40, "bold"), bd=2, fg="black", textvariable=face, width=50, justify=LEFT).grid(
        row=1, column=0)

btn1 = Button(root, text="Start", font=("Times", 26, "bold"), height=1, width=5, bg=("gainsboro"), command=Video_live,
                 justify=CENTER)
btn1.grid(row=2, column=0)
btn2 = Button(root, text="End", font=("Times", 26, "bold"), height=1, width=5, bg=("gainsboro"), command=cv2.destroyAllWindows(),
                 justify=CENTER)
btn2.grid(row=3, column=0)
root.mainloop()


