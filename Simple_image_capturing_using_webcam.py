import cv2

cam_port = 0
cam = cv2.VideoCapture(cam_port) 
result, image = cam.read() 

if result: 

	# showing result, it take frame name and image 
	# output 
	cv2.imshow("Picture", image) 

	# saving image in local storage 
	cv2.imwrite("data.jpg", image) 

	# If keyboard interrupt occurs, destroy image 
	# window 
	cv2.waitKey(0) 
	cv2.destroyWindow("Picture")
 
else: 
	print("No image detected. Please! try again")  