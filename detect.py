import cv2

face = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

img = cv2.imread("twoface.jpg")
img_grey = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

faces = face.detectMultiScale(img_grey,scaleFactor = 1.2, minNeighbors = 5)
print(type(faces))
print(faces)

for x,y,w,h in faces:
    img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),3)

re = cv2.resize(img,(800,500))
cv2.imshow("detected",re)
cv2.waitKey(0)
cv2.destroyAllWindows()