import cv2
import numpy as np

cap =cv2.VideoCapture(0) #Initializing the camera

while True: #Infinite loop for capturing the frames
    ret, img = cap.read()

    z = img.reshape((-1,3)) #Reshaping the image to be fed to the kmeans data

    z = np.float32(z) #Converting to float

    #Define Criteria, number of clusters(k), and apply Kmeans
    criteria = (cv2.TERM_CRITERIA_EPS * cv2.TERM_CRITERIA_MAX_ITER, 3, 0.9)

    k = 10  #Number of clusters to be defined

    ret, label, center = cv2.kmeans(z, k, None, criteria, 1, cv2.KMEANS_RANDOM_CENTERS)

    #Converting back into unit8, and making original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((img.shape))


    #Display the window
    cv2.imshow("Cartoonized", res2)

    #Closing the window when "q" is pressed
    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()