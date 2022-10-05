import cv2

img=cv2.imread('image.png')

classNames=[]
classFile='coco.names'
with open(classFile,'rt') as f:
    classNames=f.read().rstrip('\n').split('\n')

# print(classNames)

configPath=''
weightsPath=''

net=cv2.dnn_DetectionModel(weightsPath,configPath)
# pre trained face detection trained model
net.setInputSize(320,320)
net.setInputScale(1.0/127.5)
net.setInputMean((127.5,127.5,127.5))
net.setInputSwapRB(True)

classIds,confs,bbox=net.detect(img,confThreshold=0.5)
print(classIds,bbox)

for classId,confidence,box in zip(classIds.flatten(),confs.flatten(),bbox):
    cv2.rectangle(img,box,color=(0,0,255),thickness=3)  # 3 is for thickness
    cv2.putText(img,classNames[classId-1].upper(),(box[0]+10,box[1]+30),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2),

cv2.imshow("Output",img)
cv2.waitKey(0)