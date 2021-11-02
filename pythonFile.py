import eel
import re
import cv2
import numpy as np
import dlib
import time
import math

 #pretrained car classifier 
car_tracker_file = "car_detector.xml"

#Create car classifier classifier
car_tracker = cv2.CascadeClassifier(car_tracker_file)

#pretrained pedestrian classifier
pedestrian_tracker_file = "haarcascade_fullbody.xml"

#create pedestrian classifier
pedestrian_tracker = cv2.CascadeClassifier(pedestrian_tracker_file)

# Set web files folder
eel.init('web')


@eel.expose  # Expose this function to Javascript
def cardClicked(x):
    print(x)

def carImage(x):
    #the image
    img_file =  x

    counter = 0

    #create opencv image
    img = cv2.imread(img_file)

    #Convert the image to grayscale(needed for haar cascade)
    black_n_white = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


    #Detect cars
    cars = car_tracker.detectMultiScale(black_n_white)
    
    #Draw rectangles around the cars
    for(x, y, w, h) in cars:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        counter += 1

    cv2.putText(img , "VEHICLE COUNTER : " + str(counter)  , (450 , 70) , cv2.FONT_HERSHEY_SIMPLEX , 2 , (0,100,100) , 5)
    #Display the image
    cv2.imshow("Car Image" , img)

    #Dont autoclose(Wait in code and listen for a keypress)
    cv2.waitKey()

def carVideo(x):
    #the video 
    video = cv2.VideoCapture(x)

    #Run untill car stops or crashes or something
    while True:

        #Read the current frame
        (read_successful, frame) = video.read()

        if read_successful:
            #Converting the frame to grayscale 
            grayscaled_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        else:
            break
        

        #Detect cars
        cars = car_tracker.detectMultiScale(grayscaled_frame)

        #Draw recatngles around the cars
        for(x, y, w, h) in cars:
            cv2.rectangle(frame, (x+1, y+2), (x+w, y+h), (255, 0, 0), 2)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        #Display the image
        cv2.imshow("Car Video" , frame)

        #Dont autoclose(Wait in code and listen for a keypress)
        key = cv2.waitKey(1)

        #Stop if Q key is pressed
        if key == 81 or key == 113 : 
            break
    
    cv2.destroyAllWindows() 
    #Release VideoCapture object
    video.release()


def pedImage(x):
    #the image
    counter = 0 
    img_file =  x

    #create opencv image
    img = cv2.imread(img_file)

    #Convert the image to grayscale(needed for haar cascade)
    black_n_white = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


    #Detect cars
    pedestrians = pedestrian_tracker.detectMultiScale(black_n_white)
    
    #Draw rectangles around the pedestrians
    for(x, y, w, h) in pedestrians:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 255), 2)
        counter += 1

    cv2.putText(img , "PEDESTRIAN COUNTER : " + str(counter)  , (450 , 70) , cv2.FONT_HERSHEY_SIMPLEX , 2 , (0,100,100) , 5) 
    #Display the image
    cv2.imshow("Pedestrian Image" , img)

    #Dont autoclose(Wait in code and listen for a keypress)
    cv2.waitKey()

def pedVideo(x):
     #the video 
    video = cv2.VideoCapture(x)

    #Run untill pedestrian stops or crashes or something
    while True:

        #Read the current frame
        (read_successful, frame) = video.read()

        if read_successful:
            #Converting the frame to grayscale 
            grayscaled_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        else:
            break
        

        #Detect pedestrians
        pedestrians = pedestrian_tracker.detectMultiScale(grayscaled_frame)

        #Draw recatngles around the cars
        for(x, y, w, h) in pedestrians:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
        
        #Display the image
        cv2.imshow("Pedestrian Video" , frame)

        #Dont autoclose(Wait in code and listen for a keypress)
        key = cv2.waitKey(1)

        #Stop if Q key is pressed
        if key == 81 or key == 113 : 
            break

    cv2.destroyAllWindows()         
    #Release VideoCapture object
    video.release()

def bothVideo(x):
     #the video 
    video = cv2.VideoCapture(x)

    #Run untill car stops or crashes or something
    while True:

        #Read the current frame
        (read_successful, frame) = video.read()

        if read_successful:
            #Converting the frame to grayscale 
            grayscaled_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        else:
            break
        

        #Detect cars and pedestrians
        cars = car_tracker.detectMultiScale(grayscaled_frame)
        pedestrians = pedestrian_tracker.detectMultiScale(grayscaled_frame)

        #Draw rectangles around the cars
        for(x, y, w, h) in cars:
            cv2.rectangle(frame, (x+1, y+2), (x+w, y+h), (255, 0, 0), 2)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        #Draw rectangles around the pedestrians
        for(x, y, w, h) in pedestrians: #different colour
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)

        #Display the image
        cv2.imshow("Both Video" , frame)

        #Dont autoclose(Wait in code and listen for a keypress)
        key = cv2.waitKey(1)

        #Stop if Q key is pressed
        if key == 81 or key == 113 : 
            break

    #Release VideoCapture object
    cv2.destroyAllWindows() 
    video.release()

def bothImage(x):
     #the image
    img_file =  x

    counter = 0
    #create opencv image
    img = cv2.imread(img_file)

    #Convert the image to grayscale(needed for haar cascade)
    black_n_white = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


    #Detect pedestrians
    pedestrians = pedestrian_tracker.detectMultiScale(black_n_white)
    
    #Detect cars
    cars = car_tracker.detectMultiScale(black_n_white)
    
    #Draw recatngles around the pedestrians
    for(x, y, w, h) in pedestrians:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 255), 2)

    #Draw rectangles around the cars
    for(x, y, w, h) in cars:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        counter += 1

    cv2.putText(img , "PEDESTRIAN AND CAR COUNTER : " + str(counter)  , (450 , 70) , cv2.FONT_HERSHEY_SIMPLEX , 2 , (0,100,100) , 5) 
    #Display the image
    cv2.imshow("Pedestrian Image" , img)

    #Dont autoclose(Wait in code and listen for a keypress)
    cv2.waitKey()

def center_handling(x,y,w,h):
    x1 = int(w/2)
    y1 = int(h/2)
    cx = x + x1
    cy = y + y1
    return cx,cy

def videoCarCounter(x):
    #Capture the video
    video = cv2.VideoCapture(x) 

    detect = []

    #declare minimum weight and height of rectangles
    min_width_rectangle = 80 
    min_height_rectangle = 80

    offset = 5 #Allowable error between pixel 
    counter = 0

    count_line_position = 500
    #Initialize Substructor
    #algo to subtract the background from vehicles as only vehicles are to be detected
    algo = cv2.createBackgroundSubtractorMOG2()
    
    while True:
        (read_successful,frame) = video.read() 
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #Convert to gray scale
        blur = cv2.GaussianBlur(grey,(3,3),5) 

        #Apply algorithm on each frame
        img_sub = algo.apply(blur)

        #structure of the algorithm will be specified on specific point i.e pixel neighbourhood gives it a shape and tries to detect it
        dilat = cv2.dilate(img_sub , np.ones((5,5)))

        #pass the kernel return a structuring elmeent of the specified size and shape for morphological operations
        kernel = cv2.getStructuringElement (cv2.MORPH_ELLIPSE,(5,5))

        #we will have multichannel images it performs the function of giving shape to the vehicles
        dilatada = cv2.morphologyEx(dilat, cv2.MORPH_CLOSE , kernel)
        dilatada = cv2.morphologyEx(dilatada, cv2.MORPH_CLOSE , kernel)
        ( counterShape  , h )= cv2.findContours(dilatada, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        #draw a line
        cv2.line(frame , (25,count_line_position) , (1600 , count_line_position) , (0,200,0), 3)

        #draw rectangles 
        for (i,c) in enumerate(counterShape):
            (x,y,w,h) = cv2.boundingRect(c) 
            ctr_validate = (w >= min_width_rectangle ) and (h >= min_height_rectangle)
            if not ctr_validate :
                continue

            #cv2.rectangle(frame , (x,y) , (x + w , y + h) , (255 , 0 , 0) , 2)

            center = center_handling(x,y,w,h)
            detect.append(center)

            cv2.circle(frame , center , 4 , (0,0,255) , -1)
        
            for (x , y) in detect :
                if y < (count_line_position + offset) and y > (count_line_position - offset) :
                    counter += 1
                cv2.line(frame , (25,count_line_position) , (1600 , count_line_position) , (50,100,50), 3)
                detect.remove((x,y))

        cv2.putText(frame , "VEHICLE COUNTER : " + str(counter)  , (450 , 70) , cv2.FONT_HERSHEY_SIMPLEX , 2 , (0,100,100) , 5)
        #cv2.imshow('Detector' , dilatada)
        cv2.imshow('Video Orignal' , frame)

        #Dont autoclose(Wait in code and listen for a keypress)
        key = cv2.waitKey(1)

        #Stop if Q key is pressed
        if key == 81 or key == 113 : 
            break

    #Release VideoCapture object
    cv2.destroyAllWindows() 
    video.release()  

def estimateSpeed(location1, location2):
    d_pixels = math.sqrt(math.pow(location2[0] - location1[0], 2) + math.pow(location2[1] - location1[1], 2))
    # ppm = location2[2] / carWidht
    ppm = 8.8
    d_meters = d_pixels / ppm
    fps = 18
    speed = d_meters * fps * 3.6
    return speed

def carSpeed(x) : 
    carCascade = cv2.CascadeClassifier('car_detector.xml')
    video = cv2.VideoCapture(x)

    WIDTH = 1280
    HEIGHT = 720

    rectangleColor = (0, 255, 0)
    frameCounter = 0
    currentCarID = 0
    fps = 0

    carTracker = {}
    carNumbers = {}
    carLocation1 = {}
    carLocation2 = {}
    speed = [None] * 1000

    out = cv2.VideoWriter('outNew.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 10, (WIDTH, HEIGHT))

    while True:
        start_time = time.time()
        rc, image = video.read()
        if type(image) == type(None):
            break

        image = cv2.resize(image, (WIDTH, HEIGHT))
        resultImage = image.copy()

        frameCounter = frameCounter + 1
        carIDtoDelete = []

        for carID in carTracker.keys():
            trackingQuality = carTracker[carID].update(image)

            if trackingQuality < 7:
                carIDtoDelete.append(carID)

        
        for carID in carIDtoDelete:
            print("Removing carID " + str(carID) + ' from list of trackers. ')
            print("Removing carID " + str(carID) + ' previous location. ')
            print("Removing carID " + str(carID) + ' current location. ')
            carTracker.pop(carID, None)
            carLocation1.pop(carID, None)
            carLocation2.pop(carID, None)

        
        if not (frameCounter % 10):
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            cars = carCascade.detectMultiScale(gray, 1.1, 13, 18, (24, 24))

            for (_x, _y, _w, _h) in cars:
                x = int(_x)
                y = int(_y)
                w = int(_w)
                h = int(_h)

                x_bar = x + 0.5 * w
                y_bar = y + 0.5 * h

                matchCarID = None

                for carID in carTracker.keys():
                    trackedPosition = carTracker[carID].get_position()

                    t_x = int(trackedPosition.left())
                    t_y = int(trackedPosition.top())
                    t_w = int(trackedPosition.width())
                    t_h = int(trackedPosition.height())

                    t_x_bar = t_x + 0.5 * t_w
                    t_y_bar = t_y + 0.5 * t_h

                    if ((t_x <= x_bar <= (t_x + t_w)) and (t_y <= y_bar <= (t_y + t_h)) and (x <= t_x_bar <= (x + w)) and (y <= t_y_bar <= (y + h))):
                        matchCarID = carID

                if matchCarID is None:
                    print(' Creating new tracker' + str(currentCarID))

                    tracker = dlib.correlation_tracker()
                    tracker.start_track(image, dlib.rectangle(x, y, x + w, y + h))

                    carTracker[currentCarID] = tracker
                    carLocation1[currentCarID] = [x, y, w, h]

                    currentCarID = currentCarID + 1

        for carID in carTracker.keys():
            trackedPosition = carTracker[carID].get_position()

            t_x = int(trackedPosition.left())
            t_y = int(trackedPosition.top())
            t_w = int(trackedPosition.width())
            t_h = int(trackedPosition.height())

            cv2.rectangle(resultImage, (t_x, t_y), (t_x + t_w, t_y + t_h), rectangleColor, 4)

            carLocation2[carID] = [t_x, t_y, t_w, t_h]

        end_time = time.time()

        if not (end_time == start_time):
            fps = 1.0/(end_time - start_time)

        for i in carLocation1.keys():
            if frameCounter % 1 == 0:
                [x1, y1, w1, h1] = carLocation1[i]
                [x2, y2, w2, h2] = carLocation2[i]

                carLocation1[i] = [x2, y2, w2, h2]

                if [x1, y1, w1, h1] != [x2, y2, w2, h2]:
                    if (speed[i] == None or speed[i] == 0) and y1 >= 275 and y1 <= 285:
                        speed[i] = estimateSpeed([x1, y1, w1, h1], [x1, y2, w2, h2])

                    if speed[i] != None and y1 >= 180:
                        cv2.putText(resultImage, str(int(speed[i])) + "km/h", (int(x1 + w1/2), int(y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 100) ,2)

        cv2.imshow('result', resultImage)

        out.write(resultImage)

        if cv2.waitKey(1) == 27:
            break

    video.release()  
    cv2.destroyAllWindows()
    out.release()
    

@eel.expose  # Expose this function to Javascript
def elementClicked(x):
    # print("Value of x " + x)
    if(re.search("^car" , x)) :
        carImage(x)
    elif(re.search("^Carvideo" , x)) :
        carVideo(x)
    elif(re.search("^ped" , x)) :
        pedImage(x)
    elif(re.search("^PedVIDEO" , x)) :
        pedVideo(x)
    elif(re.search("^bothCarPedvideo" , x)) :
        bothVideo(x)
    elif(re.search("^bothCarPed" , x)) :
        bothImage(x)
    elif(re.search("^CarSpeedvideo" , x)) :
        carSpeed(x)
    elif(re.search("^CarCountervideo" , x)) :
        videoCarCounter(x)

eel.start("index.html", size=(300, 200))  # Start
