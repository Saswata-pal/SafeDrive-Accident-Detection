import cv2    
import time
cpt = 0
maxFrames = 80

count=0
cap=cv2.VideoCapture(r'C:\Users\Dell\Downloads\videoplayback (online-video-cutter.com) (1).mp4')
while cpt < maxFrames:
    ret, frame = cap.read()
    if not ret:
        break
    count += 1
    if count % 3 != 0:
        continue
    frame=cv2.resize(frame,(1080,500))
    cv2.imshow("test window", frame) # show image in window
    cv2.imwrite(r"C:\Users\Dell\Downloads\yolov8-vehicle-crash-detection-main\yolov8-vehicle-crash-detection-main\images\ca_%d.jpg" %cpt, frame)
    time.sleep(0.01)
    cpt += 1
    if cv2.waitKey(5)&0xFF==27:
        break
cap.release()   
cv2.destroyAllWindows()
