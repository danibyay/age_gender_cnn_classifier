# prerequisites:
# Have a shortcut on your google drive of the share folder with the dataset images
# called train_val

from google.colab import drive
import os
import pandas as pd
import numpy as np

drive.mount('/content/drive')
folder = '/content/drive/MyDrive/Colab Notebooks/data/train_val/'
imgs = os.listdir(folder)

columns = ['Filename', 'Age', 'Gender']
num_of_labels = len(columns) - 1
num_of_samples = len(imgs)
age_gender_data = np.zeros((num_of_samples,num_of_labels))

# find substrings according to the index where the first underscore is
# before first underscore is age
# after first underscore is gender
for i in range(num_of_samples):
  for a in range(3):
    age_digits = a + 1
    gender_index = age_digits + 1
    if imgs[i].index('_') == (age_digits):
      age_gender_data[i][0] = imgs[i][0:age_digits]
      age_gender_data[i][1] = imgs[i][gender_index]


df = pd.DataFrame(data = np.column_stack((imgs[:num_of_samples],age_gender_data)),
                  columns = columns)
