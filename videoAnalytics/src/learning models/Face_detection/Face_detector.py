# created by Navaneeth D on 9/05/2020
# last updated by Navaneeth D(navaneethsdk@gmail.com) on 9/05/2020
from matplotlib import pyplot
from mtcnn.mtcnn import MTCNN
#requirements before running . you should have MTCNN installed using
#pip install mtcnn


#test the functions by uncommenting following lines
# filename = 'fl.jpg'
# get_face_bounding_data(filename)



#thisfunction can used to obtain the facedata from the file namespecified
def get_face_bounding_data(filename):
	updated_faces = []
	# load image from file
	pixels = pyplot.imread(filename)
	# create the detector, using default weights
	detector = MTCNN()
	# detect faces in the image
	faces = detector.detect_faces(pixels)
	for face in faces:
		if face['confidence'] > 0.9666:
	    	updated_faces.append(face)
	return face_data(filename, updated_faces)





#returns a list of faces detected
def face_data(filename, result_list):
	faces=[]
	# load the image
	data = pyplot.imread(filename)
	# plot each face as a subplot
	for i in range(len(result_list)):
		# get coordinates
	x1, y1, width, height = result_list[i]['box']
	x2, y2 = x1 + width, y1 + height
	faces.append(data[y1:y2, x1:x2])
	return faces


	# display faces on the original image
	# draw_faces(filename, updated_faces)


#this fn can be used to draw the faces (optional)
def draw_faces(filename, result_list):
	# load the image
	data = pyplot.imread(filename)
	# plot each face as a subplot
	for i in range(len(result_list)):
		# get coordinates
		x1, y1, width, height = result_list[i]['box']
		x2, y2 = x1 + width, y1 + height
		# define subplot
		pyplot.subplot(1, len(result_list), i+1)
		pyplot.axis('off')
		# plot face
		pyplot.imshow(data[y1:y2, x1:x2])
	# show the plot
	pyplot.show()
