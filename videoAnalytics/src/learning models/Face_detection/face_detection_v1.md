The model used in the repository https://github.com/ageitgey/face_recognition has been used.

Make sure dlib has already been pre-installed, with python bindings

Install the module using the statement  
`pip3 install face_recognition`

Faces can be detected  using the command `face_detection`  
All the images are stored in a folder.  

Face Detection can be implemented by running the statement  
`face_detection  ./folder_with_pictures/`  

The statement prints one line as output for each face that was detected. The coordinates of the bounding box of each face is printed as output. The pixel of the top, right, bottom and left corner is displayed respectively.
