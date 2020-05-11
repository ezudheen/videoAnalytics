The model used in the repository https://github.com/ageitgey/face_recognition has been used.

Make sure dlib has already been pre-installed, with python bindings.

Install the module using the statement  
`pip3 install face_recognition`

Faces can be recognised by running the command `face_recognition`

Two folders with images are required.  
One folder contains images of known people, with each image named as the person in the image. This folder of images is used to train the model. Make sure each image in this folder contains only one face.
The other folder contains images of unknown people.  

Face recognition can be implemented by running the statement  
`$ face_recognition ./folder_of_pictures_of_people_i_know/ ./folder_of_unknown_pictures/`  

There's one line in the output for each face. The data is comma-separated with the filename and the name of the person found.
An unknown_person is a face in the image that didn't match anyone in your folder of known people.
