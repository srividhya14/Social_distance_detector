import cv2
import numpy as np
import conf
import math
import random
import os
import time
import csv
import mysql.connector
import requests
from datetime import datetime
from datetime import timedelta
from itertools import combinations
index=1

def get_output_layers(net):
    
    layer_names = net.getLayerNames()
    
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers

def is_close(p1, p2):
    """
    # 1. Calculate Euclidean Distance between two points
    :param:
    p1, p2 = two points for calculating Euclidean Distance
    :return:
    dst = Euclidean Distance between two 2d points
    """
    dst = math.sqrt(p1**2 + p2**2)
    return dst 


def convertBack(x, y, w, h): 
    """
    # 2. Converts center coordinates to rectangle coordinates     
    :param:
    x, y = midpoint of bbox
    w, h = width, height of the bbox
    
    :return:
    xmin, ymin, xmax, ymax
    """
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax


def find_diff(t1, t2):
    s1 = (t1.hour * 60 + t1.minute) * 60 + t1.second
    s2 = (t2.hour * 60 + t2.minute) * 60 + t2.second

    return abs(s1 - s2)
    

 
def cvBoxes(detections, img):
    if len(detections) > 0:
        class_ids = []
        confidences = []
        boxes = []# At least 1 detection in the image and check detection presence in a frame  
        centroid_dict = dict()                                           # Function creates a dictionary and calls it centroid_dic        								# We inialize a variable called ObjectId and set it to 0
        for out in detections:
            for detection in out:
                scores=detection[5:]
                class_id=np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5 and class_id == 0:
                    x = int(detection[0] * Width)
                    y = int(detection[1] * Height)
                    w = int(detection[2] * Width)
                    h = int(detection[3] * Height)
                    xx = x - w / 2
                    yy = y - h / 2
                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([xx,yy,w,h])

        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
        id=0
        print(len(indices))
        for i in indices:
            i=i[0]
            box=boxes[i]
            xx=box[0]
            yy=box[1]
            w=box[2]
            h=box[3]
            x=xx + w / 2
            y=yy + h / 2
            xmin,ymin,xmax,ymax=convertBack(float(x),float(y),float(w),float(h))
            centroid_dict[id]=(int(x),int(y),xmin,ymin,xmax,ymax)
            id+=1
                
        
        # 3. Check which person bounding box are close to each other

        red_zone_list = [] # List containing which Object id is in under threshold distance condition. 
        red_line_list = []
        for (id1, p1), (id2, p2) in combinations(centroid_dict.items(), 2): # Get all the combinations of close detections, #List of multiple items - id1 1, points 2, 1,3
            dx, dy = p1[0] - p2[0], p1[1] - p2[1]  	# Check the difference between centroid x: 0, y :1
            distance = is_close(dx, dy) 			# Calculates the Euclidean distance
            if distance < 35.0:						# Set our social distance threshold - If they meet this condition then..
                if id1 not in red_zone_list:
                    red_zone_list.append(id1)       #  Add Id to a list
                    red_line_list.append(p1[0:2])   #  Add points to the list
                if id2 not in red_zone_list:
                    red_zone_list.append(id2)		# Same for the second id 
                    red_line_list.append(p2[0:2])
        
        for idx, box in centroid_dict.items():  # dict (1(key):red(value), 2 blue)  idx - key  box - value
            if idx in red_zone_list:   # if id is in red zone list
                cv2.rectangle(img, (box[2], box[3]), (box[4], box[5]), (0, 0, 255), 2) # Create Red bounding boxes  #starting point, ending point size of 2
            else:
                cv2.rectangle(img, (box[2], box[3]), (box[4], box[5]), (0, 255, 0), 2) # Create Green bounding boxes


    	# 3. Display risk analytics and risk indicators
    
        text = "No of at-risk people: %s" % str(len(red_zone_list))
        if len(centroid_dict)==0:
            total=1
        else:
            total=len(centroid_dict)# Count People at Risk
            
        risk_factor=round((len(red_zone_list)/total), 2)
        location = (10,25)												# Set the location of the displayed text
        cv2.putText(img, text, location, cv2.FONT_HERSHEY_SIMPLEX, 1, (246,86,86), 2, cv2.LINE_AA)  # Display Text
        global index
        global curr
        global nxt
        
        for check in range(0, len(red_line_list)-1):					# Draw line between nearby bboxes iterate through redlist items
            start_point = red_line_list[check] 
            end_point = red_line_list[check+1]
            check_line_x = abs(end_point[0] - start_point[0])   		# Calculate the line coordinates for x  
            check_line_y = abs(end_point[1] - start_point[1])			# Calculate the line coordinates for y
            if (check_line_x < 35) and (check_line_y < 25):				# If both are We check that the lines are below our threshold distance.
                cv2.line(img, start_point, end_point, (255, 0, 0), 2)   # Only above the threshold lines are displayed. 
        #=================================================================#
        #bolt alarm

        print(risk_factor)
        if risk_factor >= 0.4 and alarmstate == 0:
            response=mybolt.digitalWrite('0','HIGH')
            print(response)
        elif risk_factor < 0.4 and alarmstate == 1:
            response=mybolt.digitalWrite('0','LOW')
            print(response)
        
        #storing data in the database
        curr=datetime.now()
        if(find_diff(nxt, curr) == 5):
            index +=1
            now = datetime.now()
            dt_string = now.strftime("%Y-%m-%d %H:%M:%S")
            nxt = curr + timedelta(seconds = 5)
            mydb = mysql.connector.connect(
                host="localhost",
                user=conf.USERNAME,
                password=conf.PASSWORD,
                database=conf.DATABASE)
            mycursor = mydb.cursor()
            sql = "INSERT INTO timerf (sno,time,rf) VALUES (%s,%s,%s)"
            val = (index,dt_string,risk_factor)
            mycursor.execute(sql, val)
            mydb.commit()
            print(mycursor.rowcount, "record inserted.")
        
        #sending sms to concerned authorities
        print(risk_factor)
        global rft_now
        global rft_before
        global rf_fl
        
        if(risk_factor > 0.3):
            rft_now =datetime.now()
            if((rft_now > rft_before + timedelta(minutes = 30)) or rf_fl == 0):
                my_data = { 
                # Your default Sender ID 
                'sender_id': 'FSTSMS',  
                
                # Put your message here! 
                'message': conf.MESSAGE,  
                
                'language': 'english', 
                'route': 'p', 
                
                # You can send sms to multiple numbers 
                # separated by comma. 
                'numbers': conf.PHONE_NUMBER    
                }
                
                # create a dictionary 
                headers = { 
                    'authorization': conf.API_KEY, 
                    'Content-Type': "application/x-www-form-urlencoded", 
                    'Cache-Control': "no-cache"
                }
                
                response = requests.request("POST", 
                                            url, 
                                            data = my_data, 
                                            headers = headers)

                print(response.text)
                rf_fl = 1
                rft_before = rft_now

    return img


rft_before = datetime.now()
curr = datetime.now()
rf_fl = 0
nxt = curr + timedelta(seconds =5)        
url = conf.URL
classesFile='coco.names'
modelConfiguration='yolov3.cfg'
modelWeights='yolov3.weights'
scale = 0.00392

classes = None
with open(classesFile, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# generate different colors for different classes 
COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

# read pre-trained model and config file
net = cv2.dnn.readNet(modelWeights, modelConfiguration)

# initialization
conf_threshold = 0.5
nms_threshold = 0.4

cap = cv2.VideoCapture('test.mp4')

    
while True:
    success,image=cap.read()
    image=cv2.resize(image,(416,416))

    Width = image.shape[1]
    Height = image.shape[0]
    # read class names from text file
    # create input blob
    blob = cv2.dnn.blobFromImage(image, scale, (416,416), (0,0,0), True, crop=False)
    # set input blob for the network
    net.setInput(blob)
    # run inference through the network
    # and gather predictions from output layers
    outs = net.forward(get_output_layers(net))
    image=cvBoxes(outs, image)
    cv2.imshow('Demo', image)
    # wait until any key is pressed
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

# release resources
cap.release()
cv2.destroyAllWindows()
    


