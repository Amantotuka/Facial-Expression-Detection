import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import cv2
v=cv2.VideoCapture(0)
#r,i=v.read()
fd=cv2.CascadeClassifier(r'haarcascade_frontalface_default.xml')
model=load_model(r'C:\Users\hp\Downloads\_mini_XCEPTION.106-0.65.hdf5',compile=False)
#j=cv2.cvtColor(i,cv2.COLOR(i,cv2.COLOR_BGR2GRAY)
#f=fd.detectMultiScale(j,1.1,5)
exp=['angry','disgust','fear','happy','sad','surprised','neutral']

while(1):
    r,i=v.read()
    j=cv2.cvtColor(i,cv2.COLOR_BGR2GRAY)
    f=fd.detectMultiScale(j,1.1,5)               
    if(len(f)>0):
        [x,y,w,h]=f[0]
        cv2.rectangle(i,(x,y),(x+w,y+h),(0,255,0),2,5)
        roi=j[y:y+h,x:x+h]           
        roi=cv2.resize(roi,(48,48))
        roi=img_to_array(roi)
        roi=np.expand_dims(roi,axis=0)
        p=list(model.predict(roi)[0])
        # print(p)           
        #print(exp[p.index(max(p))])
        result=exp[p.index(max(p))]
        font=cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(i,result,(x,y),font,1,(200,0,0),3,cv2.LINE_AA)
    cv2.imshow('img',i)
    cv2.imwrite('result.jpg',i)
    k=cv2.waitKey(5)
    if(k==ord('q')):
      cv2.destroyAllWindows()
      break
               
        
