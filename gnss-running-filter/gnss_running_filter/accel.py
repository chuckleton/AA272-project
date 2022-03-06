import numpy as np
from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation as R
import attr
import pandas as pd
import matplotlib.pyplot as plt
from gnss_running_filter.gyro import Gyro
from gnss_running_filter.mag import Mag


def generate_A(timestep: float, Cstop: int = 0):
    A = np.zeros((10,10))
    A[:3, :3] = np.eye(3)
    A[3:6, :3] = np.eye(3) * timestep
    A[3:6, 3:6] = np.eye(3) * Cstop
    return A

def generate_B(timestep: float):
    B = np.zeros((10,3))
    B[:3, :] = np.eye(3) * timestep**2 / 2
    B[3:6, :] = np.eye(3) * timestep
    return B

def generate_C(q):
    C = np.zeros((10,1))
    C[-4:] = q.elements

def lat_lon_2_xy(lat0, lon0, lat1, lon1):
    a = 6378136.6
    c = 6356751.9
    phi = np.pi / 2 - (lat0 + lat1) / 2.0 * np.pi / 180.0
    coeff = np.sqrt(1/((np.sin(phi)/a)**2 + (np.cos(phi)/c)**2))
    x = np.cos(phi)*coeff*(lon1-lon0) * np.pi / 180.0
    y = coeff*(lat1-lat0) * np.pi / 180.0
    return np.array([x,y]).reshape((2,1))

def generate_H():
    H = np.zeros((2,10))
    H[0,0] = 1
    H[1,1] = 1
    return H

def compute_Sigma_hat(A, Sigma, N):
    return A @ Sigma @ A.T + N

def compute_K(Sigma_hat, H, R):
    return Sigma_hat @ H.T @ np.linalg.inv(H @ Sigma_hat @ H.T + R)

def compute_Sigma(K, H, Sigma_hat, R):
    return (np.eye(H.shape[0]) - K @ H) @ Sigma_hat @ (np.eye(H.shape[0]) - K @ H).T + R

def compute_X_hat(X, a: np.ndarray, q: np.ndarray, timestep: float, Cstop: int = 0):
    A = generate_A(timestep, Cstop)
    B = generate_B(timestep)
    C = generate_C(q)
    return A @ X + B @ a + C

def compute_X(X_hat, K, m):
    H = generate_H()
    return X_hat + K @ (m - H @ X_hat)

def filter_GNSS(X, Sigma, R, a, q, m, timestep, Cstop=0):
    X_hat = compute_X_hat(X, a, q, timestep, Cstop)
    Sigma_hat = compute_Sigma_hat(generate_A(timestep, Cstop), Sigma, R)
    K = compute_K(Sigma_hat, generate_H(), R)
    X_filtered = compute_X(X_hat, K, m)
    return X_filtered, Sigma_hat, K

@attr.s
class Accel:
    position: np.ndarray = attr.ib(default=np.zeros((3, 1)))
    velocity: np.ndarray = attr.ib(default=np.zeros((3, 1)))
    gyro: Gyro = attr.ib(default=Gyro())
    mag: Mag = attr.ib(default=Mag())
    g_mag: float = attr.ib(default=9.81)
    g: float = attr.ib(default=np.array(
        [0, 0, -9.81]).reshape((3, 1)))

    def update(self, accel_data: np.ndarray, timestep: float):
        g_updated = self.compute_g(accel_data)
        g = np.array([0, 0, -9.81]).reshape((3, 1))  # gravity vector
        Cns = self.gyro.orientation.rotation_matrix
        a = (np.matmul(Cns, accel_data.reshape((3, 1))) - g)
        a_offset = np.array([-5, 4, -11]).reshape((3, 1))
        a += a_offset
        if g_updated:
            print('g updated', self.g)
        self.position = self.position + timestep * self.velocity + \
            (timestep**2) * 0.5 * a
        self.velocity = self.velocity + timestep * a
        return g_updated

    def compute_g(self, accel_data: np.ndarray):
        accel_mag = np.linalg.norm(accel_data)

        # Only compute g if we are in right acceleration range
        if accel_mag < 0.95*self.g_mag or accel_mag > 1.05*self.g_mag:
            return False
        g_direction = accel_data / accel_mag
        self.g = (g_direction * self.g_mag).reshape((3, 1))
        self.velocity = np.zeros_like(self.velocity)
        return True

    def compute_mag_orientation(self, mag_data: np.ndarray):
        nav_vecs = np.empty((2, 3))
        body_vecs = np.empty((2, 3))

        body_vecs[0] = mag_data/np.linalg.norm(mag_data)
        nav_vecs[0] = mag.mag_field_orientation
        body_vecs[1] = self.g[:, 0]/np.linalg.norm(self.g[:, 0])
        nav_vecs[1] = np.array([0, 0, -1])

        print('body_vecs: ', body_vecs)
        print('nav_vecs: ', nav_vecs)

        self.mag_orientation = R.align_vectors(nav_vecs, body_vecs)[0]
        # self.mag_orientation = R.align_vectors(nav_vecs[1].reshape((1,3)), body_vecs[1].reshape((1,3)))[0]
        quat = self.mag_orientation.as_quat()
        self.mag_orientation = Quaternion(
            np.concatenate(([quat[-1]], quat[:-1])))
        return self.mag_orientation


def quat_to_euler(quat):
    r = R.from_quat(np.concatenate((quat.elements[1:],
                                    [quat.elements[0]])))
    return r.as_euler('xyz', degrees=True)


if __name__ == "__main__":
    fname = '../../GNSS-INS_Logger/Calibr_IMU_Log/SM-G950U_IMU_20220223171411'
    # fname = '../../GNSS-INS_Logger/Calibr_IMU_Log/SM-G950U_IMU_20220305015817'
    # fname = '../../GNSS-INS_Logger/Calibr_IMU_Log/SM-G950U_IMU_20220305023351'
    start_time = 1645655093646 + 30000
    duration = 30000
    mag = Mag()
    mag_readings = pd.read_csv(fname + '-mag.txt')
    mag_readings.drop_duplicates(
        subset=['UNIX Epoch timestamp-utc'], inplace=True)
    mag_readings = mag_readings[(mag_readings['UNIX Epoch timestamp-utc'] > start_time)
                                & (mag_readings['UNIX Epoch timestamp-utc'] < start_time + duration)]
    row = mag_readings.iloc[0]
    mag_data = np.array([row['x-axis'], row['y-axis'], row['z-axis']])

    gyro = Gyro()
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
    accel_readings['magnitude'] = np.sqrt(
        accel_readings['x-axis']**2 + accel_readings['y-axis']**2 + accel_readings['z-axis']**2)
    g_mag = accel_readings['magnitude'].mean()
    print('g_mag: ', g_mag)
    accel_readings['z-axis'] *= -1
    accel_readings['x-axis'] += 0.0
    accel_readings['y-axis'] += -0.0
    accel_readings['z-axis'] += -0.0

    mag_readings['x-axis'] += 0
    mag_readings['y-axis'] += 0
    mag_readings['z-axis'] += 0

    gyro_readings['timestep'] = gyro_readings['UNIX Epoch timestamp-utc'].diff()
    gyro_readings['timestep'] /= 1000.0
    accel_readings['timestep'] = accel_readings['UNIX Epoch timestamp-utc'].diff()
    accel_readings['timestep'] /= 1000.0

    row = accel_readings.iloc[0]
    accel_data = np.array([row['x-axis'], row['y-axis'], row['z-axis']])
    accel.compute_g(accel_data)
    accel.compute_mag_orientation(mag_data)
    initial_orientation = accel.mag_orientation
    gyro.orientation = initial_orientation
    print('initial orientation: ', quat_to_euler(initial_orientation))

    times = np.zeros(len(gyro_readings)+1)
    angles = np.zeros((len(gyro_readings)+1, 3))
    quaternions = np.zeros((len(gyro_readings)+1, 4))
    positions = np.zeros((len(gyro_readings)+1, 3))
    velocities = np.zeros((len(gyro_readings)+1, 3))
    initial_time = gyro_readings.iloc[0]['UNIX Epoch timestamp-utc']
    times[0] = 0
    angles[0] = gyro.as_euler()
    quaternions[0] = gyro.orientation.elements
    positions[0] = accel.position[:, 0]
    velocities[0] = accel.velocity[:, 0]
    print(gyro_readings)
    print(accel_readings)
    mag_ind = 1
    last_mag_update = 0
    for i, ((index, row), (aindex, arow)) in enumerate(zip(gyro_readings.iterrows(), accel_readings.iterrows())):
        if mag_ind < len(mag_readings):
            mag_time = mag_readings.iloc[mag_ind]['UNIX Epoch timestamp-utc']
        else:
            mag_time = row['UNIX Epoch timestamp-utc'] + 1
        if i == 0 or (mag_time >= row['UNIX Epoch timestamp-utc'] and mag_ind == 1):
            continue
        timestep = row['timestep']
        atimestep = arow['timestep']
        accel_data = np.array([arow['x-axis'], arow['y-axis'], arow['z-axis']])
        g_updated = accel.update(accel_data, atimestep)
        gyro_data = np.array([row['x-axis'], row['y-axis'], row['z-axis']])
        gyro_data = np.matmul(gyro.orientation.rotation_matrix, gyro_data.reshape((3,1)))
        gyro.update(gyro_data.flatten(), timestep)
        if (mag_time < row['UNIX Epoch timestamp-utc'] and
            abs(mag_time - row['UNIX Epoch timestamp-utc']) < 10):
            mag_data = np.array([mag_readings.iloc[mag_ind]['x-axis'],
                                mag_readings.iloc[mag_ind]['y-axis'],
                                mag_readings.iloc[mag_ind]['z-axis']])
            mag_ind += 1
            if g_updated and row['UNIX Epoch timestamp-utc'] - last_mag_update > 1:
                accel.compute_mag_orientation(mag_data)
                initial_orientation = accel.mag_orientation
                gyro.orientation = initial_orientation
                last_mag_update = row['UNIX Epoch timestamp-utc']
        if gyro.orientation.elements[0] < 0:
            gyro.orientation = -gyro.orientation
        angles[i+1] = gyro.as_euler()
        quaternions[i+1] = gyro.orientation.elements
        positions[i+1] = accel.position[:, 0]
        velocities[i+1] = accel.velocity[:, 0]
        times[i+1] = (row['UNIX Epoch timestamp-utc'] - initial_time) / 1000.0

    plots = [positions, velocities, angles]
    plot_labels = ['Position [m]', 'Velocity [m/s]', 'Angle [deg]']
    fig, axs = plt.subplots(1, 3, figsize=(20, 7))
    for i, value in enumerate(plots):
        axs[i].plot(times, value[:, 0], label='x')
        axs[i].plot(times, value[:, 1], label='y')
        axs[i].plot(times, value[:, 2], label='z')
        axs[i].set_xlabel('Time (s)')
        axs[i].set_ylabel(plot_labels[i])
        axs[i].legend()

    plt.figure(figsize=(10, 10))
    plt.axis('equal')
    plt.plot(positions[:, 0], positions[:, 1], label='Position')

    plt.figure(figsize=(10, 10))
    plt.plot(times, quaternions[:, 0], label='w')
    plt.plot(times, quaternions[:, 1], label='x')
    plt.plot(times, quaternions[:, 2], label='y')
    plt.plot(times, quaternions[:, 3], label='z')
    plt.legend()
    plt.show()
