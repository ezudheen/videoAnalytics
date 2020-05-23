# Created by Ajay Ravindran, last updated on 23rd May 
# Saves the coordinates of all faces found in images at a given directory to a json file

from Facenetmodel import extract_face_return_faces_coordinates
import os
import json

# This is the function to be invoked to find face coordinates in the images in a given folder
# To be invoked in the presentation layer to receive face coordinates
def find_face(folder_dir) :
	with open("Face Coordinates.json","a+") as out :	# Face Coordinates of faces in each image at the given directory are stored in Face Coordinates.json file
		for filename in os.listdir(folder_dir) :
			faces = extract_face_return_faces_coordinates(os.path.join(image_dir,filename))
			for face in faces :
				x1,y1,width,height = faces['box']
				x1, y1 = abs(x1), abs(y1)
				x2, y2 = x1 + width, y1 + height
				centre_x = (x1+x2) // 2
				centre_y = (y1+y2) // 2
				val = {
					"Photo" : filename,
					"Top Left Coordinate" : {"x" : x2,"y" : y1},
					"Top Right Coordinate" : {"x" : x1,"y" : y1},
					"Bottom Left Coordinate" : {"x" : x2,"y" : y2},
					"Bottom Right Coordinate" : {"x" : x1,"y" : y2},
					"Centre" : {"x" : centre_x, "y" : centre_y}
				}
				
			json_object = json.dumps(val, indent = 4)
			out.write(json_object)
			
folder_dir = '/home/ajay/DataLayer/SplitImages/Cam1 Photos'
find_face(folder_dir)
