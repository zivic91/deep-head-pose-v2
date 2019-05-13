import cv2
from math import cos, sin, acos, asin
import numpy as np
import scipy.io
from scipy.io import savemat
from scipy import ndimage
import os

# images to be rotated (it's assumed that accompaying .mat files are located in the same directory)
INPUT_DIR_PATH = 'LFPW/'
OUTPUT_DIR_PATH = 'LFPW_rotated_scaled/'
ANGLES = list(range(-20, 30, 10))
ANGLES.remove(0)

# save mat file for given yaw, pitch and roll
def save_ypr(yaw, pitch, roll, pt2d, mat_file_name):

    yaw = yaw * np.pi / 180
    pitch = pitch * np.pi / 180
    roll = roll * np.pi / 180

    mat = dict()
    mat['Pose_Para'] = [[pitch, yaw, roll]]
    mat['pt2d'] = pt2d

    savemat(mat_file_name, mat)


# same as draw axis instead it returns computed vectors
def create_vectors(yaw, pitch, roll):

    pitch = pitch * np.pi / 180
    yaw = -(yaw * np.pi / 180)
    roll = roll * np.pi / 180

    # X-Axis pointing to right. drawn in red
    x1 = cos(yaw) * cos(roll)
    y1 = cos(pitch) * sin(roll) + cos(roll) * sin(pitch) * sin(yaw)

    # Y-Axis | drawn in green
    #        v
    x2 = -cos(yaw) * sin(roll)
    y2 = cos(pitch) * cos(roll) - sin(pitch) * sin(yaw) * sin(roll)

    # Z-Axis (out of the screen) drawn in blue
    x3 = sin(yaw)
    y3 = -cos(yaw) * sin(pitch)

    return x1, y1, x2, y2, x3, y3


# takes computed vectors (x1, y1), (x2, y2) and (x3, y3) and returns yaw, pitch and roll
def inverse(x1, y1, x2, y2, x3, y3):
    yaw = asin(x3)
    pitch = asin(y3 / (- cos(yaw)))
    roll = asin(x2 / (- cos(yaw)))

    yaw = - yaw * 180 / np.pi
    pitch = pitch * 180 / np.pi
    roll = roll * 180 / np.pi

    return yaw, pitch, roll


# single point rotation in 2-dimensional Euclidian space
def rotate_point(x, y, angle):
    # convert to radians
    angle = angle * np.pi / 180

    new_x = cos(angle) * x - sin(angle) * y
    new_y = sin(angle) * x + cos(angle) * y
    return new_x, new_y


# transform yaw, pitch and roll
def rotate_ypr(yaw, pitch, roll, angle):
    # get vectors
    x1, y1, x2, y2, x3, y3 = create_vectors(yaw, pitch, roll)

    # rotate vectors
    x1, y1 = rotate_point(x1, y1, angle)
    x2, y2 = rotate_point(x2, y2, angle)
    x3, y3 = rotate_point(x3, y3, angle)

    # get rotated yaw, pitch and roll
    return inverse(x1, y1, x2, y2, x3, y3)

# transform pt2d
def rotate_pt2d(pt2d, angle):

    pt2d_new = np.copy(pt2d)
    s = 1 / (cos(abs(angle * np.pi / 180.)) + sin(abs(angle * np.pi / 180.)))

    for pt in range(pt2d_new.shape[1]):
        # non scaled new coordinates
        x, y = rotate_point(pt2d_new[0, pt], pt2d_new[1, pt], angle)
        # scale coordinates
        x = 225.0 + (x - 225.) * s
        y = 225.0 + (y - 225.) * s
        pt2d_new[0, pt], pt2d_new[1, pt] = x, y

    # get rotated pt2d
    return pt2d_new

if __name__ == '__main__':

    if not os.path.exists(OUTPUT_DIR_PATH):
        os.mkdir(OUTPUT_DIR_PATH)

    all_files = os.listdir(INPUT_DIR_PATH)
    counter = 0
    for mat_f in all_files:
        if mat_f.endswith('.mat') and np.random.random_sample() < 0.05:

            base_name = mat_f.split('.')[0]

            mat = scipy.io.loadmat(os.path.join(INPUT_DIR_PATH, mat_f))
            pitch = 180 * mat['Pose_Para'][0][0] / 3.1416
            yaw = 180 * mat['Pose_Para'][0][1] / 3.1416
            roll = 180 * mat['Pose_Para'][0][2] / 3.1416

            img = cv2.imread(os.path.join(INPUT_DIR_PATH, base_name + ".jpg"))

            for angle in ANGLES:
                # rotate image, RPY and landmarks
                img_rotated = ndimage.rotate(img, -angle)
                r_yaw, r_pitch, r_roll = rotate_ypr(yaw, pitch, roll, angle)

                pt2d = mat['pt2d']
                pt2d_rotated = rotate_pt2d(pt2d, angle)

                # save image and .mat file
                cv2.imwrite(os.path.join(OUTPUT_DIR_PATH, base_name + "_rotated_%s.jpg" % str(angle)), img_rotated)
                save_ypr(r_yaw, r_pitch, r_roll, pt2d, os.path.join(OUTPUT_DIR_PATH, base_name + "_rotated_%s.mat" % str(angle)))

            counter += 1
            if counter % 100 == 0:
                print(str(counter) + " images processed so far")
