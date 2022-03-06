from gnss_running_filter.fusion import Fusion
import pandas as pd
import numpy as np
from pyquaternion import Quaternion
from gnss_running_filter.gyro import Gyro
from gnss_running_filter.mag import Mag
from gnss_running_filter.accel import Accel
import matplotlib.pyplot as plt

fuse = Fusion(timediff= lambda start, end: end-start)

fname = '../../GNSS-INS_Logger/Calibr_IMU_Log/SM-G950U_IMU_20220223171411'
fname = '../../GNSS-INS_Logger/Calibr_IMU_Log/SM-G950U_IMU_20220305015817'
# fname = '../../GNSS-INS_Logger/Calibr_IMU_Log/SM-G950U_IMU_20220305023351'
start_time = 1646463497156
duration = 40000
orientation_cal_time = 5
mag = Mag()
mag_readings = pd.read_csv(fname + '-mag.txt')
mag_readings.drop_duplicates(
    subset=['UNIX Epoch timestamp-utc'], inplace=True)
mag_readings = mag_readings[(mag_readings['UNIX Epoch timestamp-utc'] > start_time)
                            & (mag_readings['UNIX Epoch timestamp-utc'] < start_time + duration)]

mag_reading_num = 0
def stopfunc():
    global mag_reading_num
    return mag_reading_num > 1000

def get_mag():
    global mag_reading_num
    for _ in range(1000):
        row = mag_readings.iloc[mag_reading_num]
        mag_data = [row['x-axis'], row['y-axis'], row['z-axis']]
        mag_reading_num += 1
        yield mag_data

fuse.calibrate(get_mag, stopfunc)

row = mag_readings.iloc[0]
mag_data = np.array([row['x-axis'], row['y-axis'], row['z-axis']])
# mag_data = np.array([-0.07943064,  0.56345383, - 0.82232022])
mag.update(mag_data)
initial_orientation = mag.orientation
print('initial orientation: ', mag.as_euler(),
      mag.get_angle(), mag.get_axis())

gyro = Gyro(orientation=initial_orientation)
accel = Accel(gyro=gyro)
gyro_readings = pd.read_csv(fname + '-gyro.txt')
accel_readings = pd.read_csv(fname + '-accel.txt')
gyro_readings.drop_duplicates(
    subset=['UNIX Epoch timestamp-utc'], inplace=True)
accel_readings.drop_duplicates(
    subset=['UNIX Epoch timestamp-utc'], inplace=True)
gyro_readings = gyro_readings[(gyro_readings['UNIX Epoch timestamp-utc'] > start_time)
                              & (gyro_readings['UNIX Epoch timestamp-utc'] < start_time + duration)]
accel_readings = accel_readings[(accel_readings['UNIX Epoch timestamp-utc'] > start_time)
                                & (accel_readings['UNIX Epoch timestamp-utc'] < start_time + duration)]
gyro_readings['timestep'] = gyro_readings['UNIX Epoch timestamp-utc'].diff()
gyro_readings['timestep'] /= 1000.0
accel_readings['timestep'] = accel_readings['UNIX Epoch timestamp-utc'].diff()
accel_readings['timestep'] /= 1000.0
times = np.zeros(len(gyro_readings)+1)
angles = np.zeros((len(gyro_readings)+1, 3))
positions = np.zeros((len(gyro_readings)+1, 3))
velocities = np.zeros((len(gyro_readings)+1, 3))
initial_time = gyro_readings.iloc[0]['UNIX Epoch timestamp-utc']
times[0] = 0
angles[0] = gyro.as_euler()
positions[0] = accel.position[:, 0]
velocities[0] = accel.velocity[:, 0]
print(gyro_readings)
print(accel_readings)
fuse.q = gyro.orientation.elements
mag_ind = 1
last_time = start_time
for i, ((index, row), (aindex, arow)) in enumerate(zip(gyro_readings.iterrows(), accel_readings.iterrows())):
    if mag_ind < len(mag_readings):
        mag_time = mag_readings.iloc[mag_ind]['UNIX Epoch timestamp-utc']
    else:
        mag_time = row['UNIX Epoch timestamp-utc'] + 1
    if i == 0 or (mag_time >= row['UNIX Epoch timestamp-utc'] and mag_ind == 1):
        continue
    timestep = row['timestep']
    accel_data = np.array([arow['x-axis'], arow['y-axis'], arow['z-axis']])
    gyro_data = np.array([row['x-axis'], row['y-axis'], row['z-axis']])
    gyro_data_degrees = np.degrees(gyro_data)
    gyro.orientation = Quaternion(fuse.q)
    times[i+1] = (row['UNIX Epoch timestamp-utc'] - initial_time) / 1000.0
    if times[i+1] > orientation_cal_time:
        accel.update(accel_data, timestep)
    if mag_time < row['UNIX Epoch timestamp-utc']:
        mag_data = np.array([mag_readings.iloc[mag_ind]['x-axis'],
                             mag_readings.iloc[mag_ind]['y-axis'],
                             mag_readings.iloc[mag_ind]['z-axis']])
        ts = (row['UNIX Epoch timestamp-utc'] - last_time)/1000.0
        fuse.update(accel_data, gyro_data_degrees, mag_data, ts=ts)
        mag_ind += 1
        angles[i+1] = np.array([fuse.roll, fuse.pitch, fuse.heading])
        last_time = row['UNIX Epoch timestamp-utc']
    else:
        # fuse.update_nomag(accel_data, gyro_data_degrees, ts=timestep)
        angles[i+1] = angles[i]
    positions[i+1] = accel.position[:, 0]
    velocities[i+1] = accel.velocity[:, 0]

plots = [positions, velocities, angles]
plot_labels = ['Position [m]', 'Velocity [m/s]', 'Angle [deg]']
fig, axs = plt.subplots(1, len(plots), figsize=(20, 7))
for i, value in enumerate(plots):
    axs[i].plot(times, value[:, 0], label='x')
    axs[i].plot(times, value[:, 1], label='y')
    axs[i].plot(times, value[:, 2], label='z')
    axs[i].set_xlabel('Time (s)')
    axs[i].set_ylabel(plot_labels[i])
    axs[i].legend()

plt.figure(figsize=(10,10))
plt.axis('equal')
plt.plot(positions[:, 0], positions[:, 1], label='Position')
plt.show()
