from audioop import avg
from readings import Readings, ReadingType
from gyro import Gyro
from accel import Accel
from mag import Mag
from imu import IMU
from tqdm import tqdm

import numpy as np

from matplotlib import cm, colors
import matplotlib.pyplot as plt

IMU_fpath = "C:\\Users\\dkolano\\OneDrive - Agile Space Industries\\Documents\\Homework\\Global Positioning Systems\\Project\\GNSS-INS_Logger\\Calibr_IMU_Log\\SM-G950U_IMU_20220223171411"
GPS_fpath = "C:\\Users\\dkolano\\OneDrive - Agile Space Industries\\Documents\\Homework\\Global Positioning Systems\\Project\\WLS_LLA\\WLS_LLA_track.csv"

start_time = 1645654491442+1000
duration = 15000000

def lat_lon_2_xy(lat0, lon0, lat1, lon1):
    a = 6378136.6
    c = 6356751.9
    phi = np.pi / 2 - (lat0 + lat1) / 2.0 * np.pi / 180.0
    coeff = np.sqrt(1/((np.sin(phi)/a)**2 + (np.cos(phi)/c)**2))
    x = np.cos(phi)*coeff*(lon1-lon0) * np.pi / 180.0
    y = coeff*(lat1-lat0) * np.pi / 180.0
    return np.array([x, y]).reshape((2, 1))


def generate_A(timestep: float, Cstop: int = 1):
    A = np.zeros((10, 10))
    A[:3, :3] = np.eye(3)
    A[:3, 3:6] = np.eye(3) * timestep
    A[3:6, 3:6] = np.eye(3) * Cstop
    return A


def generate_B(timestep: float):
    B = np.zeros((10, 3))
    B[:3, :] = np.eye(3) * timestep**2 / 2
    B[3:6, :] = np.eye(3) * timestep
    return B


def generate_C(q):
    C = np.zeros((10, 1))
    C[-4:] = q.elements.reshape((4, 1))
    return C


def generate_H():
    H = np.zeros((2, 10))
    H[0, 0] = 1
    H[1, 1] = 1
    return H


def compute_Sigma_hat(A, Sigma, N):
    return A @ Sigma @ A.T + N


def compute_K(Sigma_hat, H, R):
    return Sigma_hat @ H.T @ np.linalg.inv(H @ Sigma_hat @ H.T + R)


def compute_Sigma(K, H, Sigma_hat, R):
    A = (np.eye(H.shape[1]) - K @ H)
    return A @ Sigma_hat @ A.T + K @ R @ K.T


def compute_X_hat(X, a: np.ndarray, q: np.ndarray, timestep: float, Cstop: int = 1):
    A = generate_A(timestep, Cstop)
    B = generate_B(timestep)
    C = generate_C(q)
    return A @ X + B @ a + C


def compute_X(X_hat, K, m):
    H = generate_H()
    return X_hat + K @ (m - H @ X_hat)


def filter_GNSS(X, Sigma, R, N, a, q, m, timestep, Cstop: int = 1):
    A = generate_A(timestep, Cstop)
    X_hat = compute_X_hat(X, a, q, timestep, Cstop)
    Sigma_hat = compute_Sigma_hat(A, Sigma, N)

    if m is not None:
        H = generate_H()
        K = compute_K(Sigma_hat, H, R)
        X_filtered = compute_X(X_hat, K, m)
        Sigma = compute_Sigma(K, H, Sigma_hat, R)
    else:
        X_filtered = X_hat
        Sigma = Sigma_hat
        K = None
    return X_filtered, Sigma, K

def run_Kalman_filter(readings: Readings, imu: IMU):
    first_reading = readings.get_next_reading(advance_time=True)
    assert first_reading.type == ReadingType.ACCEL_GYRO_GPS or \
           first_reading.type == ReadingType.ACCEL_GYRO_MAG_GPS, \
           'First reading must be ACCEL_GYRO_GPS or ACCEL_GYRO_MAG_GPS'

    # Get initial orientation from magnetometer and accelerometer
    imu.compute_g(first_reading.accel, always_compute=True)
    initial_orientation = imu.get_mag_orientation(first_reading.mag)
    imu.orientation = initial_orientation

    # Get initial position from first GPS reading
    initial_pos = first_reading.gps
    lat0, lon0 = initial_pos[0], initial_pos[1]

    X0 = np.zeros((10, 1))
    X0[-4:] = initial_orientation.elements.reshape((4, 1))

    accel_mag_prev = np.zeros(500)

    Sigma0 = np.eye(10) * 4.0

    X = X0
    Sigma = Sigma0

    N = np.eye(10) * 0.0002
    R = np.eye(2) * 10.0
    g = np.array([0, 0, -9.81]).reshape((3, 1))  # gravity vector

    timestamps = np.zeros(readings.num_readings)
    positions = np.zeros((readings.num_readings, 3))
    all_positions = np.zeros((readings.num_readings, 3))
    velocities = np.zeros((readings.num_readings, 3))
    orientations = np.zeros((readings.num_readings, 4))
    max_freqs = np.zeros(readings.num_readings)

    wls_positions = np.zeros((readings.num_readings, 3))

    print(readings.num_readings)

    positions[0] = X[:3,0]
    all_positions[0] = X[:3, 0]
    velocities[0] = X[3:6,0]
    orientations[0] = X[-4:,0]
    max_freqs[0] = 0.0
    timestamps[0] = first_reading.timestamp
    avg_timestep = 1.0 / 500.0

    reading_ind = 1

    pbar = tqdm(total=readings.num_readings-1)
    pbar.set_description("Kalman Filtering Data...")
    while True:
        reading = readings.get_next_reading()
        if reading is None:
            break

        accel = reading.accel
        accel_mag_prev[1:] = accel_mag_prev[:-1]
        accel_mag_prev[0] = np.linalg.norm(accel)

        gyro = reading.gyro
        mag = reading.mag
        gps = reading.gps
        if gps is not None:
            gps = lat_lon_2_xy(lat0, lon0, gps[0], gps[1])
        timestep = reading.timestep
        avg_timestep = avg_timestep*0.95 + timestep*0.05

        use_mag = (reading.type == ReadingType.ACCEL_GYRO_MAG_GPS
                   or reading.type == ReadingType.ACCEL_GYRO_MAG)
        imu.update(accel, gyro, mag, timestep, use_mag=use_mag)

        q = imu.orientation

        Cns = q.rotation_matrix
        accel = (np.matmul(Cns, accel) - g)

        if reading_ind > len(accel_mag_prev):
            avg_accel = np.mean(np.sqrt(accel_mag_prev**2))
            w = np.fft.rfft(accel_mag_prev - np.mean(accel_mag_prev))
            freqs = np.fft.rfftfreq(len(accel_mag_prev), d=avg_timestep)
            w = w[freqs < 4.75]
            max_freq = freqs[np.argmax(np.abs(w))]
            max_freqs[reading_ind] = max_freq
        else:
            max_freqs[reading_ind] = 0.0

        X_new, Sigma_new, K = filter_GNSS(X, Sigma, R, N, accel, q, gps, timestep, Cstop=1)

        if gps is not None:
            imu.position = X_new[:3]
            imu.velocity = X_new[3:6]
            wls_positions[reading_ind] = np.array([gps[0,0], gps[1,0], 0.0], dtype=np.float32)
            timestamps[reading_ind] = reading.timestamp
            positions[reading_ind] = X_new[:3, 0]
            velocities[reading_ind] = X_new[3:6, 0]
            orientations[reading_ind] = X_new[-4:, 0]
        else:
            wls_positions[reading_ind] = wls_positions[reading_ind - 1]
            positions[reading_ind] = positions[reading_ind - 1]
            velocities[reading_ind] = velocities[reading_ind - 1]
            timestamps[reading_ind] = timestamps[reading_ind - 1]
            orientations[reading_ind] = orientations[reading_ind - 1]
        # positions[reading_ind] = X_new[:3, 0]
        all_positions[reading_ind] = X_new[:3, 0]
        velocities[reading_ind] = X_new[3:6, 0]
        # orientations[reading_ind] = X_new[-4:, 0]
        timestamps[reading_ind] = reading.timestamp

        reading_ind += 1
        pbar.update(1)
        X = X_new
        Sigma = Sigma_new

    pbar.close()
    return timestamps, positions, velocities, orientations, wls_positions, all_positions, max_freqs


def main():
    readings = Readings(IMU_fpath, GPS_fpath,
                        start_time=start_time, duration=duration)
    gyro = Gyro()
    mag = Mag()
    accel = Accel(gyro=gyro, mag=mag)
    imu = IMU(gyro=gyro, accel=accel)

    timestamps, positions, velocities, orientations, wls_positions, all_positions, max_freqs = run_Kalman_filter(
        readings, imu)
    timestamps -= timestamps[0]
    timestamps /= 1000

    # Generate colormap for time data
    c = cm.get_cmap('viridis', 12)
    cmap = c((timestamps) / (timestamps[-1]))
    norm = colors.Normalize(vmin=0, vmax=(timestamps[-1]))
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
    sm.set_array([])

    # Plot results
    plt.style.use('seaborn-darkgrid')

    # Plot position and velocity
    plt.figure(figsize=(10, 10))
    plt.scatter(positions[:,0], positions[:,1], marker='o', s=2, c=cmap, zorder=10, label='EKF')
    plt.plot(positions[:,0], positions[:,1], linewidth=1, c='k', zorder=0)
    plt.plot(positions[:,0], positions[:,1], linewidth=1, c='k', zorder=0)
    plt.scatter(wls_positions[:,0], wls_positions[:,1], marker='o', s=4, c='r', zorder=9, label='WLS')
    plt.axis('equal')
    plt.colorbar(sm,
                 label='Time From Start [s]',
                 orientation='horizontal',
                 shrink=0.65)
    plt.legend()
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.title('Filtered Position')

    plt.figure(figsize=(10, 10))
    plt.plot(timestamps, velocities[:,0], label='x')
    plt.plot(timestamps, velocities[:,1], label='y')
    plt.legend()

    # plt.figure(figsize=(10, 10))
    # plt.plot(timestamps, orientations[:, 0], label='w')
    # plt.plot(timestamps, orientations[:, 1], label='x')
    # plt.plot(timestamps, orientations[:, 2], label='y')
    # plt.plot(timestamps, orientations[:, 3], label='z')
    # plt.legend()

    plt.figure(figsize=(10, 10))
    plt.plot(timestamps, max_freqs, label='max_freq')

    plt.show()


if __name__ == "__main__":
    main()