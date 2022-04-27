# pip install face-alignment
# pip install matplotlib --upgrade

import face_alignment
from skimage import io
import numpy as np
import glob
from tqdm import tqdm
import os
import shutil

fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)

unchecked_input_path = "/content/validation"  # @param {type:"string"}
checked_output_path = "/content/validation"  # @param {type:"string"}
failed_output_path = "/content/validation"  # @param {type:"string"}
landmark_output_path = "/content/landmarks"  # @param {type:"string"}

if not os.path.exists(unchecked_input_path):
    os.makedirs(unchecked_input_path)
if not os.path.exists(checked_output_path):
    os.makedirs(checked_output_path)
if not os.path.exists(failed_output_path):
    os.makedirs(failed_output_path)
if not os.path.exists(landmark_output_path):
    os.makedirs(landmark_output_path)

files = glob.glob(unchecked_input_path + "/**/*.png", recursive=True)
files_jpg = glob.glob(unchecked_input_path + "/**/*.jpg", recursive=True)
files.extend(files_jpg)
err_files = []

for f in tqdm(files):
    input = io.imread(f)
    preds = fa.get_landmarks(input)
    # print(preds)
    if preds is not None:
        np.savetxt(
            os.path.join(landmark_output_path, os.path.basename(f) + ".txt"),
            preds[0],
            delimiter=" ",
            fmt="%1.3f",
        )  # X is an array
        shutil.move(f, os.path.join(checked_output_path, os.path.basename(f)))
    else:
        shutil.move(f, os.path.join(failed_output_path, os.path.basename(f)))
