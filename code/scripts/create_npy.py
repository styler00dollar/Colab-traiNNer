import cv2
import glob
import numpy as np
import os
data_root = "/home/bro/Desktop/input/"
dest_dir = "/home/bro/Desktop/npy/"

upper_folders = glob.glob(data_root + "/*/")

samples = upper_folders

count = 0
for f in samples:
  imgpaths = [f + '/frame1.jpg', f + '/frame2.jpg', f + '/frame3.jpg']

  img1 = cv2.imread(imgpaths[0])
  img2 = cv2.imread(imgpaths[1])
  img3 = cv2.imread(imgpaths[2])
  img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
  img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
  img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)

  img1 = cv2.resize(img1, (1280, 720))
  img2 = cv2.resize(img2, (1280, 720))
  img3 = cv2.resize(img3, (1280, 720))

  img1 = np.swapaxes(img1,0,2).swapaxes(1,2)/255
  img2 = np.swapaxes(img2,0,2).swapaxes(1,2)/255
  img3 = np.swapaxes(img3,0,2).swapaxes(1,2)/255


  combined = np.stack([img1, img3, img2], axis=0)
  combined = combined.astype(np.float32)
  np.save(os.path.join(dest_dir, str(count)+".npy"), combined)
  count += 1