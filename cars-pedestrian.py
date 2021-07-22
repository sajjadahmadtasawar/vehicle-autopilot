import cv2

#demo video
video = cv2.VideoCapture('pedestrian.mp4')

#trained data for cars and pedestrian
car_trained_data = cv2.CascadeClassifier('cars.xml')
pedestrian_trained_data = cv2.CascadeClassifier('pedestrian.xml')


while True:
    (success,frame) = video.read()

    #gray scaled video
    grayScaled_video = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    #video coordinates
    car_coordinates = car_trained_data.detectMultiScale(grayScaled_video)
    pedestrian_coordinates = pedestrian_trained_data.detectMultiScale(grayScaled_video)

    #drawing rectangles around cars
    for (x,y,w,h) in car_coordinates:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,266,0),2)
    
    #drawing rectangles around cars
    for (x,y,w,h) in pedestrian_coordinates:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,266),2)

    #opening the video    
    cv2.imshow('auto pilot detector',frame)
    key = cv2.waitKey(1)

    if key==81 or key==113:
        break

video.release()    