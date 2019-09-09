import scipy.io
import numpy as np
import time
from matplotlib import pyplot as plt

from estimate_rot import estimate_rot
from quat_utils import *

DATASET = [1]
if __name__=="__main__":
    np.seterr(all='warn')
    for i in DATASET:
        a = time.time()
        roll, pitch, yaw = estimate_rot(i)
        print("Time taken to execute this code is: ")
        print(time.time() - a)

        vicon_dict = scipy.io.loadmat('vicon/viconRot'+str(i)+'.mat')
        vicon_data = np.array(vicon_dict['rots'], dtype=np.float64)
        vicon_timestamp = np.array(vicon_dict['ts'], dtype=np.float64)
        imu_dict = scipy.io.loadmat('imu/imuRaw'+str(i)+'.mat')
        imu_timestamp = np.array(imu_dict['ts'], dtype=np.float64)

        roll_gt = []
        pitch_gt = []
        yaw_gt = []
        for i in range(vicon_timestamp.shape[1]):
            R = vicon_data[:, :, i]
            r, p, y = rotmat2rpy(R)
            roll_gt.append(r)
            pitch_gt.append(p)
            yaw_gt.append(y)

        plt.figure(1)
        plt.subplot(311)
        plt.plot(roll, 'b')
        plt.plot(roll_gt, 'r')
        plt.ylabel('Roll')
        plt.subplot(312)
        plt.plot(pitch, 'b')
        plt.plot(pitch_gt, 'r')
        plt.subplot(313)
        plt.plot(yaw, 'b')
        plt.plot(yaw_gt, 'r')
        plt.ylabel('Yaw')
        plt.show()
