# Face Detection algorithm implemented on a folder of images using face landmark estimation
# dlib should be pre-installed
# Install face_recognition using command : pip3 install face_recognition
# For further details refer "github.com/ageitgey/face_recognition"

import cv2
import subprocess
import json

def detect_faces(folder_name) :	# The folder is assumed to be in the same directory as the code
		
	folder_name = "\\ ".join(folder_name.split())	# Replace all spaces in folder with "\ "
	cmd = "face_detection ./"+folder_name+"/"
	output = subprocess.getoutput(cmd)	# Stores the output of face_detection command to output

	face_cordinates = output.split('\n')
	# Details of each face is printed on each line of output
	# face_cordinates is thus a list with details of each face as elements

	with open("Face Coordinates.json","a+") as out :
		for face in face_cordinates :
			picnamestart = face.rindex('/') + 1	  # after the last '/', the name of the image and face coordinates are obtained
			
			# Slicing the output to receive image name and each face coordinate
			coordinatesstart = face.index(',') + 1
			picname = face[picnamestart:coordinatesstart - 1]
			coordinates = face[coordinatesstart : ]
			
			next_coordinate = coordinates.index(',')
			y1 = int(coordinates[:next_coordinate])
			coordinates = coordinates[next_coordinate + 1 : ]
			next_coordinate = coordinates.index(',')
			x1 = int(coordinates[:next_coordinate])
			coordinates = coordinates[next_coordinate + 1 : ]
			next_coordinate = coordinates.index(',')
			y2 = int(coordinates[:next_coordinate])
			coordinates = coordinates[next_coordinate + 1 : ]
			x2 = int(coordinates)
					
			val = {
				"Photo" : picname,
				"Top Left Coordinate" : {"x" : x2,"y" : y1},
				"Top Right Coordinate" : {"x" : x1,"y" : y1},
				"Bottom Left Coordinate" : {"x" : x2,"y" : y2},
				"Bottom Right Coordinate" : {"x" : x1,"y" : y2},
				"Centre" : {"x" : (x1+x2)//2, "y" : (y1+y2)//2}
				}
				
			# Photo name, coordinates of each face and their centres and stored to json file
			json_object = json.dumps(val, indent = 4)
			out.write(json_object)	# Write into json file

folder = "Cam1 Photos"
detect_faces(folder)
