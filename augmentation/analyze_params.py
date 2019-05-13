import sys
import numpy as np
import scipy.io
import os
import matplotlib.pyplot as plt

pitch = []
yaw = []
roll = []

directories = sys.argv[1:]

num_images = 0

for directory in directories:
    for filename in os.listdir(directory):
        if filename.endswith(".mat"):

            full_filename = os.path.join(directory, filename)
            mat = scipy.io.loadmat(full_filename)

            num_images += 1

            pitch.append(180 * mat['Pose_Para'][0][0] / 3.1416)
            yaw.append(180 * mat['Pose_Para'][0][1] / 3.1416)
            roll.append(180 * mat['Pose_Para'][0][2] / 3.1416)

pitch = np.array(pitch)
# print("pitch")
# print(np.percentile(pitch, 5))
# print(min(pitch))
# print(np.percentile(pitch, 95))
# print(max(pitch))
pitch = pitch[np.abs(pitch) < 100]
plt.hist(pitch, 50)
plt.show()

yaw = np.array(yaw)
# print("yaw")
# print(np.percentile(yaw, 5))
# print(min(yaw))
# print(np.percentile(yaw, 95))
# print(max(yaw))
yaw = yaw[np.abs(yaw) < 100]
plt.hist(yaw, 50)
plt.show()

roll = np.array(roll)
# print("roll")
# print(np.percentile(roll, 5))
# print(min(roll))
# print(np.percentile(roll, 95))
# print(max(roll))
roll = roll[np.abs(roll) < 100]
plt.hist(roll, 50)
plt.show()
