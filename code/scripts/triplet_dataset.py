#@markdown Folder path with all the videos
global  outputPath
from PIL import Image
from tqdm import tqdm
#@markdown Path to generate the Dataset
outputPath = "/x/" #@param{type:"string"}
temp = "/x/"
rootdir = '/x/'

#@markdown The bigger the value, the more alike the frames need to be to be considered a triplet.
psnr = 10 #@param{type:"number"}
#@markdown resize to 448x256
resize = True #@param{type:"boolean"}
import random
#Original dataset information:
#Vimeo-90K triplets dataset contains 91701 triplets extracted from 15k video clips.
#Each triplet is a short RGB video sequence that consists of 3 frames with fixed resolution 448x256
chance = 0.1 #@param{type:"slider", min:0.01, max:1, step:0.01}


def calculate_psnr(img1, img2, max_value=255):
    """"Calculating peak signal-to-noise ratio (PSNR) between two images."""
    i1 = PIL.Image.open(img1).convert('RGB')
    i2 = PIL.Image.open(img2).convert('RGB')

    mse = np.mean((np.array(i1, dtype=np.float32) - np.array(i2, dtype=np.float32)) ** 2)
    if mse == 0:
        return 100
    return 20 * np.log10(max_value / (np.sqrt(mse)))


def IsDiffScenes(img1, img2, diff=psnr):
  return calculate_psnr(img1, img2) <= diff


import os
import shutil
from shutil import copyfile
import time
import PIL
import numpy as np
import glob

if not os.path.isdir(outputPath):
  os.makedirs(outputPath)

if not os.path.isdir(os.path.join(outputPath, "sequences")):
  os.makedirs(os.path.join(outputPath, "sequences"))

counter = 1

files = glob.glob(rootdir + '/**/*.mkv', recursive=True)
files_webm = glob.glob(rootdir + '/**/*.webm', recursive=True)
files_mp4 = glob.glob(rootdir + '/**/*.mp4', recursive=True)
files.extend(files_webm)
files.extend(files_mp4)

for video in tqdm(files):
  try:
    shutil.rmtree(temp)
    os.mkdir(temp)
  except:
    print("xd")
  path = video

  rescale = ""
  if resize:
    rescale="scale=1280:720,"
  os.system(f'ffmpeg -hide_banner -loglevel error -i "{path}" -vf "{rescale}mpdecimate=hi=128*12:lo=320*1.00:max=16" -vsync 0 -qscale:v 1 -pix_fmt rgb24  "{temp}%10d.jpg"')

  root = os.path.join(outputPath, "sequences", str(counter).zfill(3))
  if not os.path.isdir(root):
    os.makedirs(root)


  triIndex = 1
  images = sorted(os.listdir(temp))
  for i in range(1, len(images)) :
    if  random.random()>chance:
        im1 = temp + images[i-2]
        im2 =  temp + images[i-1]
        im3 =  temp + images[i]
        if IsDiffScenes(im1, im2, psnr) == False and IsDiffScenes(im2, im3, psnr) == False:
          #print("Triplet Found")
          triPath = os.path.join(root, str(triIndex).zfill(5))
          if not os.path.isdir(triPath):
            os.makedirs(triPath)
          copyfile(im1, os.path.join(triPath, "im1.jpg"))
          copyfile(im2, os.path.join(triPath, "im2.jpg"))
          copyfile(im3, os.path.join(triPath, "im3.jpg"))
          triIndex +=1

  counter += 1

testingPercentage = 0.1 #@param{type:"slider", min:0.1, max:0.9, step:0.1}
database=outputPath
import random

if os.path.isfile(os.path.join(database, "tri_testlist.txt")):
  os.remove(os.path.join(database, "tri_testlist.txt"))
if os.path.isfile(os.path.join(database, "tri_trainlist.txt")):
  os.remove(os.path.join(database, "tri_trainlist.txt"))

from glob import glob
folders = glob(os.path.join(databas, "sequences/*/"))

paths = []

for i in range(0, len(folders)):
  triplets = glob(folders[i] + "/*/")
  for ii in range(0, len(triplets)):
    innerF = os.path.basename((os.path.dirname(triplets[ii])))
    outF = os.path.basename((os.path.dirname(folders[i])))
    paths.append(outF + "/" + innerF)

def partitionRankings(rawRatings, testPercent):
    howManyNumbers = int(round(testPercent*len(rawRatings)))
    shuffled = rawRatings[:]
    random.shuffle(shuffled)
    return shuffled[howManyNumbers:], shuffled[:howManyNumbers]

training, test = partitionRankings(paths, testingPercentage)


trainTxt = os.path.join(database, "tri_trainlist.txt")
trainFile = open(trainTxt, 'w')

testTxt = os.path.join(database, "tri_testlist.txt")

testFile = open(testTxt, 'w')

for txt in training:
  trainFile.write(txt + "\n")
trainFile.close()

for txt in test:
  testFile.write(txt + "\n")
testFile.close()

print(training)
print(test)
