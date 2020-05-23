
def local_coordinates(vid_name,videoDetails):
 # -> use the videofile to obtain the sample image json created from the video
 #    using video_to_frames and find_face function from data layer
 # -> use the json to obtain the time , cameraID (by parsing the photo name) and face cordinates
 # -> use these coordinates of face and their corresponding src_image to get the real 3d coordinates
 # 	and find the face id from the learning models layer 
 # -> returns json of the following form:

 # 		{
 # 			"cameraID":sample cameraID,
 # 			"personID":sample person id,
 # 			"estimated_pos":{
 # 				"x":sample x value,
 # 				"y":sample y value
 # 			},
 # 			"time":sample time
 # 		}