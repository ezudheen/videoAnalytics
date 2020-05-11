#created by Navaneeth D
import cv2
import os
import random
from shutil import copyfile
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
#converts videos into omages and use them for traing the model to recognize the face
#caution : all the directory should be used carefully

formats = ['.jpeg','.jpg','.png']
def getFrame(sec,directory,plabel,vidcap,count):
  vidcap.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
  hasFrames,image = vidcap.read()
  if hasFrames:
      cv2.imwrite(plabel+"_"+str(count)+".jpg", image)   # save frame as JPG file
  return hasFrames


def videoTOimage(video, directory , personlabel):
  os.chdir(directory)
  vidcap = cv2.VideoCapture(video)
  sec = 0
  frameRate = 0.5 #//it will capture image in each 0.5 second
  count=1
  success = getFrame(sec,directory,personlabel,vidcap,count)
  while success:
    count = count + 1
    sec = sec + frameRate
    sec = round(sec, 2)
    success = getFrame(sec,directory,personlabel,vidcap,count)
  print(count)
  print(len(os.listdir(directory)))

def split_data(SOURCE, TRAINING, TESTING, SPLIT_SIZE):
  lis = os.listdir(SOURCE)
  print(len(lis))
  lis_=[]
  for img in lis:
    if img[img.find('.'):] in formats:
      if(os.path.getsize(SOURCE)!=0):
        lis_.append(img)
  random.sample(lis_, len(lis_))
  thp = int(SPLIT_SIZE*len(lis_)) 
  training = lis_[0:thp]
  test = lis_[thp:]
  for img in training :
      copyfile(SOURCE+img, TRAINING+img)   
  for img in test :
      copyfile(SOURCE+img, TESTING+img)


train_dir = os.path.join( base_dir, 'train/')
validation_dir = os.path.join( base_dir, 'validation/')
train_P1_dir = os.path.join(train_dir, 'P1/') # Directory with our training p1 pictures
train_P2_dir = os.path.join(train_dir, 'P2/') # Directory with our training p2 pictures
validation_P1_dir = os.path.join(validation_dir, 'P1/') # Directory with our validation p1 pictures
validation_P2_dir = os.path.join(validation_dir, 'P2/')# Directory with our validation p2 pictures


os.mkdir(base_dir)
os.mkdir(train_dir)
os.mkdir(validation_dir)
os.mkdir(train_P1_dir)
os.mkdir(train_P2_dir)
os.mkdir(validation_P1_dir)
os.mkdir(validation_P2_dir)
os.mkdir('/tmp/datalakeP1')
os.mkdir('/tmp/datalakeP2')

directory = ['/tmp/datalakeP1/','/tmp/datalakeP2/']



people_videos = {'P1':'video1.mp4' ,'P2':'video2.mp4'}
count =0
print(people_videos.items())
for people in people_videos.items():
  print(people[1], directory[count] , people[0])
  videoTOimage(people[1], directory[count] , people[0])
  count +=1


split_data(directory[0],train_P1_dir,validation_P1_dir,0.7)
split_data(directory[1],train_P2_dir,validation_P2_dir,0.7)

#use the given Link to use the inception v3 model
#!wget --no-check-certificate https://storage.googleapis.com/mledu-datasets/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5 -O /tmp/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5

from tensorflow.keras.applications.inception_v3 import InceptionV3
local_weights_file = '/tmp/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'
pre_trained_model = InceptionV3(input_shape = (150, 150, 3), 
                                include_top = False, 
                                weights = None)

pre_trained_model.load_weights(local_weights_file)

for layer in pre_trained_model.layers:
  layer.trainable = False

# pre_trained_model.summary()

last_layer = pre_trained_model.get_layer('mixed7')
print('last layer output shape: ', last_layer.output_shape)
last_output = last_layer.output

from tensorflow.keras.optimizers import RMSprop

# Flatten the output layer to 1 dimension
x = layers.Flatten()(last_output)
# Add a fully connected layer with 1,024 hidden units and ReLU activation
x = layers.Dense(1024, activation='relu')(x)
# Add a dropout rate of 0.2
x = layers.Dropout(0.2)(x)                  
# Add a final softmax layer for classification
x = layers.Dense (2, activation='softmax')(x)     
    

model = Model( pre_trained_model.input, x) 

model.compile(loss = 'categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

model.summary()


import matplotlib.pyplot as plt
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend(loc=0)
plt.figure()


plt.show()