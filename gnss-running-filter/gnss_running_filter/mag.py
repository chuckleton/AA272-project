import numpy as np
from pyquaternion import Quaternion
from magnetic_field_calculator import MagneticFieldCalculator
from scipy.spatial.transform import Rotation as R
import attr
import pandas as pd
import matplotlib.pyplot as plt

def get_rot_axis_angle(a, b):
    a_cross_b = np.cross(a, b)
    n = a_cross_b / np.linalg.norm(a_cross_b)
    theta = np.arctan2(np.linalg.norm(a_cross_b), np.dot(a, b))
    rotation = R.from_rotvec(n * theta)
    return rotation

@attr.s
class Mag:
    orientation: Quaternion = attr.ib(default=Quaternion(1, 0, 0, 0))

    def __attrs_post_init__(self):
        self.up_vec = np.array([0, 0, 1])
        self.mag_field_calculator = MagneticFieldCalculator()
        self.mag_field_orientation = self.get_mag_field_unit_vector(
            37.44611, 122.1592, 29, '2022-3-04')
        print('Mag Field Unit Vector [ENU]: ', self.mag_field_orientation)
        self.mag_field_rotation = get_rot_axis_angle(
            self.mag_field_orientation, self.up_vec)
        print('Mag Field Rotation: ', self.mag_field_rotation.as_euler('xyz', degrees=True))

    def get_mag_field_unit_vector(self, lat, long, alt, date):
        mag_field = self.get_mag_field(lat, long, alt, date)
        return mag_field / np.linalg.norm(mag_field)

    def get_mag_field(self, lat, long, alt, date):
        result = self.mag_field_calculator.calculate(
            latitude=lat,
            longitude=long,
            altitude=alt,
            date=date
        )
        field_value = result['field-value']
        north_intensity = field_value['north-intensity']['value']
        east_intensity = field_value['east-intensity']['value']
        vertical_intensity = -field_value['vertical-intensity']['value']
        field_ENU = np.array([east_intensity, north_intensity, vertical_intensity])
        return field_ENU

    def update(self, mag_data: np.ndarray):
        mag_data_unit = mag_data / np.linalg.norm(mag_data)
        print(f'Mag Data Unit: {mag_data_unit}')
        self.orientation = get_rot_axis_angle(
            self.mag_field_orientation, mag_data_unit)
        print('Mag to ENU rotation: ', self.orientation.as_euler('xyz', degrees=True))
        quat = self.orientation.as_quat()
        self.orientation = Quaternion(np.concatenate(([quat[-1]], quat[:-1])))
        return self.orientation

    def as_euler(self):
        r = R.from_quat(np.concatenate((self.orientation.elements[1:],
                                        [self.orientation.elements[0]])))
        return r.as_euler('xyz', degrees=True)

    def get_angle(self):
        return self.orientation.degrees

    def get_axis(self):
        return self.orientation.axis

if __name__ == "__main__":
    mag = Mag()
    mag_readings = pd.read_csv(
        '../../GNSS-INS_Logger/Calibr_IMU_Log/SM-G950U_IMU_20220223171411-mag.txt')
    mag_readings.drop_duplicates(
        subset=['UNIX Epoch timestamp-utc'], inplace=True)
    start_time = 1645655025617
    duration = 5000
    mag_readings = mag_readings[(mag_readings['UNIX Epoch timestamp-utc'] > start_time)
                                & (mag_readings['UNIX Epoch timestamp-utc'] < start_time + duration)]
    times = np.zeros(len(mag_readings)+1)
    angles = np.zeros((len(mag_readings)+1, 3))
    axes = np.zeros((len(mag_readings)+1, 3))
    angle_mags = np.zeros(len(mag_readings)+1)
    initial_time = mag_readings.iloc[0]['UNIX Epoch timestamp-utc']
    times[0] = 0
    angles[0] = mag.as_euler()
    axes[0] = mag.get_axis()
    angle_mags[0] = mag.get_angle()
    print(mag_readings)
    for i, (index, row) in enumerate(mag_readings.iterrows()):
        if i == 0: continue
        mag_data = np.array([row['x-axis'], row['y-axis'], row['z-axis']])
        mag.update(mag_data)
        angles[i+1] = mag.as_euler()
        axes[i+1] = mag.get_axis()
        angle_mags[i+1] = mag.get_angle()
        times[i+1] = (row['UNIX Epoch timestamp-utc'] - initial_time) / 1000.0

    plt.figure(figsize=(15, 10))
    plt.plot(times, angles[:, 0], label='x')
    plt.plot(times, angles[:, 1], label='y')
    plt.plot(times, angles[:, 2], label='z')
    plt.xlabel('Time (s)')
    plt.ylabel('Angle (deg)')
    plt.legend()

    # plt.figure(figsize=(15, 10))
    # plt.plot(times, axes[:, 0], label='x')
    # plt.plot(times, axes[:, 1], label='y')
    # plt.plot(times, axes[:, 2], label='z')
    # plt.xlabel('Time (s)')
    # plt.ylabel('Angle (deg)')
    # plt.legend()

    # plt.figure(figsize=(15, 10))
    # plt.plot(times, angle_mags, label='Angle Magnitude')
    # plt.xlabel('Time (s)')
    # plt.ylabel('Angle (deg)')
    # plt.legend()
    plt.show()
