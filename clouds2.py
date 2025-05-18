import os
import sys
import math
import random as rd
import cv2
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import filedialog, Label, Button, Text, Scrollbar

# Initialize lists and counters
path = r'C:\Users\V MAYUKHA\Desktop\images'
files = []
meansl = []
m = []
M = []
c = 0

# Collect image files from the directory
for r, d, f in os.walk(path):
    for file in f:
        if file.lower().endswith(('.jpg', '.png', '.jpeg')):
            files.append(os.path.join(r, file))
            c += 1

# Process each image to compute mean values
for input_file in files:
    f_image = cv2.imread(input_file)
    if f_image is None:
        print(f"Failed to read {input_file}")
        continue
    h, w, bpp = f_image.shape

    sum1 = 0
    z = 0
    for py in range(h):
        for px in range(w):
            if f_image[py][px][0] != 0:  # Assuming 'n' was meant to be 'f_image'
                sum1 += f_image[py][px][0]
                z += 1

    if z == 0:
        mean = 0
    else:
        mean = sum1 / z

    meansl.append(mean)
    M.append(max(meansl))
    m.append(min(meansl))

def InitializeMeans(meansl, k, m_vals, M_vals):
    f = 1  # Number of features
    means = [[0 for _ in range(f)] for _ in range(k)]
    for item in means:
        for j in range(len(item)):
            item[j] = rd.uniform(m_vals[j] + 1, M_vals[j] - 1)
    return means

def EuclideanDistance(x, y):
    S = 0
    for i in range(len(x)):
        S += math.pow(x[i] - y[i], 2)
    return math.sqrt(S)

def UpdateMean(n, mean, item):
    for i in range(len(mean)):
        mean[i] = round((mean[i] * (n - 1) + item[i]) / float(n), 3)
    return mean

def Classify(means, item):
    minimum = sys.maxsize
    index = -1
    for i, mean in enumerate(means):
        dis = EuclideanDistance(item, mean)
        if dis < minimum:
            minimum = dis
            index = i
    return index

def CalculateMeans(k, items, maxIterations=10):
    cMin = [min(items)]
    cMax = [max(items)]
    means = InitializeMeans(meansl, k, cMin, cMax)

    clusterSizes = [0 for _ in range(len(means))]
    belongsTo = [0 for _ in range(len(meansl))]

    for _ in range(maxIterations):
        noChange = True
        for i, item in enumerate(items):
            index = Classify(means, [item])  # Assuming single feature
            clusterSizes[index] += 1
            means[index] = UpdateMean(clusterSizes[index], means[index], [item])

            if index != belongsTo[i]:
                noChange = False
                belongsTo[i] = index

        if noChange:
            break

    return means

def FindClusters(means, meansl):
    clusters = [[] for _ in range(len(means))]
    for item in meansl:
        index = Classify(means, [item])  # Assuming single feature
        clusters[index].append(item)
    return clusters

# Perform clustering
k = 4  # Number of clusters
means = CalculateMeans(k, meansl)
means.sort(key=lambda x: x[0])  # Sort based on the first feature
clusters = FindClusters(means, meansl)

# Tkinter GUI Setup
def uploadImage():
    file_path = filedialog.askopenfilename()
    if not file_path:
        return
    m_image = cv2.imread(file_path)
    if m_image is None:
        print("Failed to read the image.")
        return

    h, w, bpp = m_image.shape
    red = []
    blue = []
    green = []

    for py in range(h):
        for px in range(w):
            red.append(m_image[py][px][0])
            blue.append(m_image[py][px][1])
            green.append(m_image[py][px][2])

    red_max = max(red)
    blue_max = max(blue)
    green_max = max(green)

    red_min = min(red)
    blue_min = min(blue)
    green_min = min(green)

    sum_red = sum(red)
    c_pixels = len(red)
    mean_red = sum_red / c_pixels if c_pixels != 0 else 0

    # Modify the image based on mean
    for py in range(h):
        for px in range(w):
            if m_image[py][px][0] > mean_red:
                m_image[py][px] = [255, 255, 255]  # Set to white
            else:
                m_image[py][px][0] = 0  # Set blue channel to 0

    # Calculate new mean after modification
    sum1 = 0
    z = 0
    for py in range(h):
        for px in range(w):
            if m_image[py][px][0] != 0:
                sum1 += m_image[py][px][0]
                z += 1

    mean = sum1 / z if z != 0 else 0

    # Determine weather condition based on mean and clusters
    v = 0
    for i in range(len(means)):
        if mean > means[i][0]:
            v = i
    if v < len(means) - 1:
        o1 = means[v + 1][0] - mean
        o2 = mean - means[v][0]
        if o1 < o2:
            v += 1

    # Print weather condition
    print('Current weather condition is:')
    if v in [0, 1]:
        print('SUNNY DAY')
    elif v == 2:
        print('CLOUDY DAY')
    elif v == 3:
        print('RAINY DAY')
    else:
        print('UNKNOWN CONDITION')

    # Optionally display the modified image
    cv2.imshow('Processed Image', m_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Initialize Tkinter main window
main = tk.Tk()
main.title("Detecting Clouds and Predicting Their Movement from INSAT Imagery")
main.geometry("1300x800")
main.config(bg='snow3')

# Add Title Label
font_title = ('Times', 16, 'bold')
title = Label(main, text='Detecting Clouds and Predicting Their Movement from CLOUD Imagery', bg='light cyan', fg='pale violet red', font=font_title, height=3, width=120)
title.place(x=0, y=5)

# Add Upload Button
font_button = ('Times', 14, 'bold')
uploadButton = Button(main, text="Upload CLOUD Image", command=uploadImage, font=font_button)
uploadButton.place(x=50, y=100)

# Add Path Label
pathlabel = Label(main, bg='light cyan', fg='pale violet red', font=font_button)
pathlabel.place(x=460, y=100)

# Add Text Widget with Scrollbar
font_text = ('Times', 12, 'bold')
text = Text(main, height=20, width=150, font=font_text)
scroll = Scrollbar(main, command=text.yview)
text.configure(yscrollcommand=scroll.set)
text.place(x=10, y=250)
scroll.place(x=10 + 150*7, y=250, height=20*20)  # Adjust scrollbar position

main.mainloop()
