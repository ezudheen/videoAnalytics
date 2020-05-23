# Splits a given video into frames of given interval

import time
import cv2
import json
import os

def video_to_frames(vid_name,videoDetails) :
	# videoDetails is a list containing the specific Camera Name, Month, Date, start hour, start minute and start second of the video
	
	cam, mnth, day, hr, mins, sec = videoDetails
	
	cur_dir = os.getcwd()
	folder_name = cam+" Photos"		# Frames from the video are stored to folder named "<Camera Name> Photos"
	if not os.path.isdir(folder_name) :
		os.makedirs(folder_name)		# Create folder if it doesn't already exist
	
	split_interval = 2		# Time Interval in seconds when each each frame should be saved
	
	vid_cap = cv2.VideoCapture(vid_name)
	fps = vid_cap.get(cv2.CAP_PROP_FPS)		# Frames per second in the given Video
	fps_int = round(fps)
	frame_cnt = 0

	with open("Pic Details.json","a+") as out :
		while(vid_cap.isOpened()):	
			frame_cnt += 1
			ret, frame = vid_cap.read()
			if ret == False:
				break
				
			if frame_cnt % (fps_int * split_interval) == 0 :	# only considers frames at the given interval
				time_elapsed = int(frame_cnt / fps)
				cur_sec = sec + (time_elapsed % 60)
				extra_min = time_elapsed // 60
				if cur_sec >= 60 :
					cur_sec -= 60
					extra_mins += 1
				cur_mins = mins + (extra_min % 60)
				extra_hr = extra_min // 60
				if cur_mins >= 60 :
					cur_mins -= 60
					extra_hr += 1
				cur_hr = hr + (extra_hr % 24)
				extra_day = extra_hr // 24
				if cur_hr >= 24 :
					cur_hr -= 24
					extra_day += 1
				cur_day = day + extra_day
				
				image_name = cam+"-"+mnth+str(cur_day)+" "+str(cur_hr)+":"+str(cur_mins)+":"+str(cur_sec)+".jpg"
				cv2.imwrite(cur_dir+"/"+folder_name+"/"+image_name,frame)
				
				# File Name stored in json file contains Camera Name and the exact time the photo was captured in the camera
				val = {					
					"File Name" : cam+"-"+mnth+str(cur_day)+" "+str(cur_hr)+":"+str(cur_mins)+":"+str(cur_sec)+".jpg", 
					"Camera" : cam,
					"Date" : mnth+" "+str(cur_day),
					"Time" : str(cur_hr)+":"+str(cur_mins)+":"+str(cur_sec)
				}
				
				json_object = json.dumps(val, indent = 4)
				out.write(json_object)

	vid_cap.release()

video_location = '/home/ajay/Data Layer/SplitImages/samplevid.mp4'
videoDetails = ["Cam1","May",15,11,45,0]
video_to_frames(video_location, videoDetails)
