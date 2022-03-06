import numpy as np
from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation as R
import attr
import pandas as pd
import matplotlib.pyplot as plt

@attr.s
class Gyro:
    orientation: Quaternion = attr.ib(default=Quaternion(1, 0, 0, 0))

    def update(self, gyro_data: np.ndarray, timestep: float):
        self.orientation.integrate(gyro_data, timestep)

    def as_euler(self):
        r = R.from_quat(np.concatenate((self.orientation.elements[1:],
                                        [self.orientation.elements[0]])))
        return r.as_euler('xyz', degrees=True)


if __name__ == "__main__":
    gyro = Gyro()
    gyro_readings = pd.read_csv(
        '../../GNSS-INS_Logger/Calibr_IMU_Log/SM-G950U_IMU_20220223171411-gyro.txt')
    gyro_readings.drop_duplicates(subset=['UNIX Epoch timestamp-utc'], inplace=True)
    gyro_readings = gyro_readings[(gyro_readings['UNIX Epoch timestamp-utc'] > 1645654586738)
                                  & (gyro_readings['UNIX Epoch timestamp-utc'] < 1645654586738 + 250000)]
    gyro_readings['timestep'] = gyro_readings['UNIX Epoch timestamp-utc'].diff()
    gyro_readings['timestep'] /= 1000.0
    times = np.zeros(len(gyro_readings)+1)
    angles = np.zeros((len(gyro_readings)+1, 3))
    initial_time = gyro_readings.iloc[0]['UNIX Epoch timestamp-utc']
    times[0] = 0
    angles[0] = gyro.as_euler()
    print(gyro_readings)
    for i, (index, row) in enumerate(gyro_readings.iterrows()):
        if i == 0: continue
        gyro_data = np.array([row['x-axis'], row['y-axis'], row['z-axis']])
        timestep = row['timestep']
        gyro.update(gyro_data, timestep)
        angles[i+1] = gyro.as_euler()
        times[i+1] = (row['UNIX Epoch timestamp-utc'] - initial_time) / 1000.0

    plt.figure(figsize=(15, 10))
    plt.plot(times, angles[:, 0], label='x')
    plt.plot(times, angles[:, 1], label='y')
    plt.plot(times, angles[:, 2], label='z')
    plt.xlabel('Time (s)')
    plt.ylabel('Angle (deg)')
    plt.legend()
    plt.show()