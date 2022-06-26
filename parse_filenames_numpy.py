# prerequisites:
# Have a shortcut on your google drive of the share folder with the dataset images
# called train_val

from google.colab import drive
import os
drive.mount('/content/drive')
folder = '/content/drive/MyDrive/Colab Notebooks/data/train_val/'

import os
import numpy as np

imgs = os.listdir(folder)
age = np.zeros(20)
gender = np.zeros(20)

for i in range(num_of_samples):
  # age is 1 digit
  if imgs[i].index('_') == 1:
    age[i] = imgs[i][0:1]
    gender[i] = imgs[i][2]
  # age is 2 digit
  if imgs[i].index('_') == 2:
    age[i] = imgs[i][0:2]
    gender[i] = imgs[i][3]
  # age is 3 digit
  if imgs[i].index('_') == 3:
    age[i] = imgs[i][0:3]
    gender[i] = imgs[i][4]
  print("for file %s, age is %d and gender is %d" % (imgs[i],age[i], gender[i]))
