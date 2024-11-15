# from Hackthon.MahaMetro.models.official.projects.triviaqa.evaluation import f1_score
import cv2
from ultralytics import YOLO
import numpy as np
import torch
import math
# from sklearn import f1_score
# Load a model
model=YOLO('yolov8line.pt')
# model.to('cuda')

#Function to load names file
def load_class_names(file_path):
    with open(file_path, 'r') as f:
        class_names = f.read().splitlines()
    return class_names

def in_vision(x,y):
    result=cv2.pointPolygonTest(np.array(area,np.int32),((x,y)),False)
    if result>=0:
        return True
    else:
        return False

def euclidian(x1,y1,x2,y2):
    d = int(math.sqrt((x1 - x2)**2 + (y1 - y2)**2)*100)
    return d
#Function to check for yellowline and platform edge crossing
def is_crossing_line(x0,y0,x1,y1,x2,y2):                
    #(x0,y0):bottom point of the line
    #(x1,y1):top point of the line
    #(x2,y2):third point to be checked
    val=((x1 - x0)*(y2 - y0)) - ((x2 - x0)*(y1 - y0))
    if val<=0:
        return True
    return False  # Line crossing detected for this person

# Example usage
class_names = load_class_names(r'coco.names')  # Change the file path accordingly

video_path=r"Video\Edge_Crossing_4_Cam1_0.avi"
cam = cv2.VideoCapture(video_path)

# Open webcam
font = cv2.FONT_HERSHEY_PLAIN
# cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
# cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

ret, frame = cam.read()
results=model(frame)

for output in results:
        parameter=output.boxes
        for box in parameter:
            class_detect=box.cls[0]
            class_detect=int(class_detect)
            class_detect=class_names[class_detect]
            confidence=box.conf[0]
            center_x,center_y,w,h=box.xywh[0]
            center_x,center_y,w,h=int(center_x),int(center_y),int(w),int(h)
            if confidence>0.5 and class_detect == "yellow line":
                line_center_x=center_x
            elif confidence>0.5 and class_detect == "edge":
                edge_center_x=center_x
print(line_center_x,edge_center_x)

#Checking the camera angle
if line_center_x<edge_center_x:
    direction="L"
else:
    direction="R"
for output in results:
        parameter=output.boxes
        for box in parameter:
            class_detect=box.cls[0]
            class_detect=int(class_detect)
            class_detect=class_names[class_detect]
            confidence=box.conf[0]
            center_x,center_y,w,h=box.xywh[0]
            center_x,center_y,w,h=int(center_x),int(center_y),int(w),int(h)
            x = center_x - w // 2  # Top-left x-coordinate (center minus half width)
            y = center_y - h // 2  
            if direction=="L":
                if confidence>0.5 and class_detect == "yellow line":
                    line_x_bottom=x+w
                    line_y_bottom=y+h
                    line_x_top=x
                    line_y_top=y
                if confidence>0.5 and class_detect == "edge":
                    edge_x_bottom=x+w
                    edge_y_bottom=y+h
                    edge_x_top=x
                    edge_y_top=y
            elif direction=="R":
                if confidence>0.5 and class_detect == "yellow line":
                    line_x_bottom=x
                    line_y_bottom=y+h
                    line_x_top=x+w
                    line_y_top=y
                if confidence>0.5 and class_detect == "edge":
                    edge_x_bottom=x
                    edge_y_bottom=y+h
                    edge_x_top=x+w
                    edge_y_top=y

model=YOLO(r"CenterTrack_detect.pt")
results=model(frame)
for output in results:
        parameter=output.boxes
        for box in parameter:
            class_detect=box.cls[0]
            class_detect=int(class_detect)
            class_detect=class_names[class_detect]
            confidence=box.conf[0]
            center_x,center_y,w,h=box.xywh[0]
            center_x,center_y,w,h=int(center_x),int(center_y),int(w),int(h)
            x = center_x - w // 2  # Top-left x-coordinate
            y = center_y - h // 2  # Top-left y-coordinate
            if direction=='R':
                if confidence>0.5 and class_detect == "track center":
                    track_center_x_bottom=x
                    track_center_y_bottom=y+h
                    track_center_x_top=x+w
                    track_center_y_top=y
                    area=[(track_center_x_bottom,track_center_y_bottom),(track_center_x_top,track_center_y_top),(track_center_x_top,0),(1920,0),(1920,1080),(track_center_x_bottom,1080),(track_center_x_bottom,track_center_y_bottom)]
            elif direction=='L':
                if confidence>0.5 and class_detect == "track center":
                    track_center_x_bottom=x+w
                    track_center_y_bottom=y+h
                    track_center_x_top=x
                    track_center_y_top=y
                    area=[(track_center_x_bottom,track_center_y_bottom),(track_center_x_top,track_center_y_top),(track_center_x_top,0),(0,0),(0,1080),(track_center_x_bottom,1080),(track_center_x_bottom,track_center_y_bottom)]


model=YOLO('yolov8m.pt')

while True:
    ret, frame = cam.read()
    results=model(frame)
    
    for output in results:
        parameter=output.boxes
        for box in parameter:
            center_x,center_y,w,h=box.xywh[0]
            center_x,center_y,w,h=int(center_x),int(center_y),int(w),int(h)
            x = center_x - w // 2  # Top-left x-coordinate
            y = center_y - h // 2  # Top-left y-coordinate
            confidence=box.conf[0]
            class_detect=box.cls[0]
            class_detect=int(class_detect)
            class_detect=class_names[class_detect]
            if direction=="R":
                if confidence>0.5 and class_detect == "train" and in_vision(x,y+h):
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0,225), 2)
                    cv2.circle(frame,(x,y+h),10,(225,0,0),2)
                    break
                else:
                    if confidence>0.7 and class_detect == "person" and in_vision(center_x,y+h) and is_crossing_line(edge_x_bottom,edge_y_bottom,edge_x_top,edge_y_top,x+w,y+h):
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0,225), 2)
                        cv2.putText(frame, "!!!ON THE TRACK!!!", (x, y-10), font, 1, (0, 0,225), 2)
                    elif confidence>0.7 and class_detect == "person" and in_vision(center_x,y+h) and is_crossing_line(edge_x_bottom,edge_y_bottom,edge_x_top,edge_y_top,center_x+w//2,center_y):
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (225, 0,225), 2)
                        cv2.putText(frame, "Leaning", (x, y-10), font, 1, (225, 0,225), 2)
                    elif confidence>0.7 and class_detect == "person" and in_vision(center_x,y+h) and is_crossing_line(line_x_bottom,line_y_bottom,line_x_top,line_y_top,center_x,y+h):  # Check crossing line for this person only
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 225, 225), 2)
                        cv2.putText(frame, "Line Crossing!", (x, y-10), font, 1, (0, 225, 225), 2)
                    elif confidence>0.7 and class_detect == "person":
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (225,0,0), 2)
                        cv2.putText(frame, class_detect, (x, y - 5), font, 1, (225,0,0), 2)
            elif direction=="L":
                if confidence>0.5 and class_detect == "train" and in_vision(x+w,y+h):
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0,225), 2)
                    cv2.circle(frame,(x+w,y+h),10,(225,0,0),2)
                    break
                else:
                    if confidence>0.7 and class_detect == "person" and in_vision(center_x,y+h) and is_crossing_line(edge_x_top,edge_y_top,edge_x_bottom,edge_y_bottom,x,y+h):
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0,225), 2)
                        cv2.putText(frame, "!!!ON THE TRACK!!!", (x, y-10), font, 1, (0, 0,225), 2)
                    elif confidence>0.7 and class_detect == "person" and in_vision(center_x,y+h) and is_crossing_line(edge_x_top,edge_y_top,edge_x_bottom,edge_y_bottom,center_x-w//2,center_y):
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (225, 0,225), 2)
                        cv2.putText(frame, "Leaning", (x, y-10), font, 1, (225, 0,225), 2)
                    elif confidence>0.7 and class_detect == "person" and in_vision(center_x,y+h) and is_crossing_line(line_x_top,line_y_top,line_x_bottom,line_y_bottom,center_x,y+h):  # Check crossing line for this person only
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 225, 225), 2)
                        cv2.putText(frame, "Line Crossing!", (x, y-10), font, 1, (0, 225, 225), 2)
                    elif confidence>0.7 and class_detect == "person":
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (225,0,0), 2)
                        cv2.putText(frame, class_detect, (x, y - 5), font, 1, (225,0,0), 2)

    cv2.line(frame, (line_x_bottom,line_y_bottom), (line_x_top,line_y_top), (0,225,225), 3)
    cv2.line(frame, (edge_x_bottom,edge_y_bottom), (edge_x_top,edge_y_top), (0,225,225), 3)
    cv2.polylines(frame,[np.array(area,np.int32)],True,(0,0,225),2)

    # Display output
    cv2.namedWindow("Video Analysis", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Video Analysis", 1920, 1080) 
    cv2.imshow("Video Analysis", frame)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
cam.release()
cv2.destroyAllWindows()