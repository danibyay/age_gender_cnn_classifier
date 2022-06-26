import os
import numpy as np
from google.colab import drive

drive.mount('/content/drive')
folder = '/content/drive/MyDrive/Colab Notebooks/data/train_val/'
imgs = os.listdir(folder)
age_viz = np.zeros(20)
gender_viz = np.zeros(20)

for i in range(20):
  for a in range(3):
    age_digits = a + 1
    gender_index = age_digits + 1
    if imgs[i].index('_') == (age_digits):
      age_viz[i] = imgs[i][0:age_digits]
      gender_viz[i] = imgs[i][gender_index]


import matplotlib.pyplot as plt
from matplotlib.image import imread
gender_dict = {0: "man", 1: "woman"}
plt.figure(figsize=(10,10))
for i in range(20):
    plt.subplot(4,5,i+1)
    my_title = "%s %d" % (gender_dict[gender_viz[i]],age_viz[i])
    plt.gca().set_title(my_title)
    filename = folder + imgs[i]
    if not os.path.exists(filename):
      print ('No such file:'+filename)
    image = imread(filename)
    plt.imshow(image)

plt.show()
