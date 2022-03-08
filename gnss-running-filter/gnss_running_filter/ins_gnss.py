from readings import Readings, ReadingType
from gyro import Gyro
from accel import Accel
from mag import Mag
from imu import IMU
from tqdm import tqdm

import numpy as np
import pandas as pd

from matplotlib import cm, colors
import matplotlib.pyplot as plt

path = 'Lake'
write_csv = True

if path == 'Track':
    IMU_fpath = "C:\\Users\\dkolano\\OneDrive - Agile Space Industries\\Documents\\Homework\\Global Positioning Systems\\Project\\GNSS-INS_Logger\\Calibr_IMU_Log\\SM-G950U_IMU_20220223171411"
    GPS_fpath = "C:\\Users\\dkolano\\OneDrive - Agile Space Industries\\Documents\\Homework\\Global Positioning Systems\\Project\\WLS_LLA\\WLS_LLA_track_xy.csv"


if path == 'Lake':
    IMU_fpath = "C:\\Users\\dkolano\\OneDrive - Agile Space Industries\\Documents\\Homework\\Global Positioning Systems\\Project\\GNSS-INS_Logger\\Calibr_IMU_Log\\SM-G950U_IMU_20220224190043"
    GPS_fpath = "C:\\Users\\dkolano\\OneDrive - Agile Space Industries\\Documents\\Homework\\Global Positioning Systems\\Project\\WLS_LLA\\WLS_LLA_lake_xy.csv"


if path == 'Random':
    IMU_fpath = "C:\\Users\\dkolano\\OneDrive - Agile Space Industries\\Documents\\Homework\\Global Positioning Systems\\Project\\GNSS-INS_Logger\\Calibr_IMU_Log\\SM-G950U_IMU_20220223174001"
    GPS_fpath = "C:\\Users\\dkolano\\OneDrive - Agile Space Industries\\Documents\\Homework\\Global Positioning Systems\\Project\\WLS_LLA\\WLS_LLA_rand_path_xy.csv"

# Note you can just make duration huge and it will run the whole thing
# Equivalently you can set duration = None and it will also run until the end of the file

if path == 'Track':
    start_time = 1645654491442+1000
    duration = None

if path == 'Lake':
    start_time = 1645747242435+1000
    duration = None

if path == 'Random':
    start_time = 1645656032432+1000
    duration = None

PROCESS_NOISE_COEFFICIENT_unused = 0.00025
GPS_MEASUREMENT_NOISE_COEFFICIENT = 10.0
FREQ_MEASUREMENT_NOISE_COEFFICIENT_unused = 1.5

def lat_lon_2_xy(lat0: float, lon0: float, lat1: float, lon1: float):
    """Convert latitude and longitude to x and y

    Args:
        lat0 (float): Reference latitude
        lon0 (float): Reference longitude
        lat1 (float): Current latitude
        lon1 (float): Current longitude

    Returns:
        np.ndarray: (2, 1) array of x and y [m]
    """
    a = 6378136.6
    c = 6356751.9
    phi = np.pi / 2 - (lat0 + lat1) / 2.0 * np.pi / 180.0
    coeff = np.sqrt(1/((np.sin(phi)/a)**2 + (np.cos(phi)/c)**2))
    x = np.cos(phi)*coeff*(lon1-lon0) * np.pi / 180.0
    y = coeff*(lat1-lat0) * np.pi / 180.0
    return np.array([x, y]).reshape((2, 1))

def xy_2_lat_lon(x: float, y: float, lat0: float, lon0: float):
    """Convert x and y to latitude and longitude

    Args:
        x (float): Current x [m]
        y (float): Current y [m]
        lat0 (float): Reference latitude
        lon0 (float): Reference longitude

    Returns:
        np.ndarray: (2, 1) array of lat and lon
    """
    a = 6378136.6
    c = 6356751.9
    phi = np.pi / 2 - (2*lat0 + np.rad2deg(y / a)) * np.pi / 180.0
    coeff = np.sqrt(1/((np.sin(phi)/a)**2 + (np.cos(phi)/c)**2))
    lon = x * (180.0 / np.pi) / np.cos(phi) / coeff + lon0
    lat = y * (180.0 / np.pi) / coeff + lat0
    return np.array([lat, lon]).reshape((2, 1))


def generate_A(timestep: float, Cstop: int = 1):
    """Generate the state transition matrix

    Args:
        timestep (float): Time step [s]
        Cstop (int, optional): 0 velocity update? 1=moving, 0=stopped. Defaults to 1.

    Returns:
        np.ndarray: State transition matrix
    """
    A = np.zeros((10, 10))
    A[:3, :3] = np.eye(3)
    A[:3, 3:6] = np.eye(3) * timestep
    A[3:6, 3:6] = np.eye(3) * Cstop
    return A


def generate_B(timestep: float):
    """Acceleration matrix

    Args:
        timestep (float): Time step [s]

    Returns:
        np.ndarray: Acceleration matrix
    """
    B = np.zeros((10, 3))
    B[:3, :] = np.eye(3) * timestep**2 / 2
    B[3:6, :] = np.eye(3) * timestep
    return B


def generate_C(q):
    """Quaternion matrix

    Args:
        q (Quaternion): Current quaternion

    Returns:
        np.ndarray: Quaternion matrix
    """
    C = np.zeros((10, 1))
    C[-4:] = q.elements.reshape((4, 1))
    return C


def generate_H(X_hat: np.ndarray, freq_data: bool = True):
    """Measurement matrix

    Args:
        X_hat (np.ndarray): Current state estimate
        freq_data (bool, optional): Have frequency data?. Defaults to True.

    Returns:
        np.ndarray: Measurement matrix
    """
    v_x = X_hat[3, 0]
    v_y = X_hat[4, 0]
    v_mag = np.sqrt(v_x**2 + v_y**2)
    H = np.zeros((3, 10))
    H[0, 0] = 1
    H[1, 1] = 1
    H[2, 3] = v_x / v_mag
    H[2, 4] = v_y / v_mag
    if not freq_data:
        H = H[:2, :]
    return H


def compute_Sigma_hat(A: np.ndarray, Sigma: np.ndarray, N: np.ndarray):
    """Compute the predicted covariance matrix

    Args:
        A (np.ndarray): State transition matrix
        Sigma (np.ndarray): Previous covariance matrix
        N (np.ndarray): Process noise covariance matrix

    Returns:
        np.ndarray: Predicted covariance matrix
    """
    return A @ Sigma @ A.T + N


def compute_K(Sigma_hat: np.ndarray, H: np.ndarray, R: np.ndarray):
    """Compute Kalman gain

    Args:
        Sigma_hat (np.ndarray): Predicted state covariance
        H (np.ndarray): Measurement matrix
        R (np.ndarray): Measurement covariance

    Returns:
        np.ndarray: Kalman gain
    """
    return Sigma_hat @ H.T @ np.linalg.inv(H @ Sigma_hat @ H.T + R)


def compute_Sigma(K: np.ndarray, H: np.ndarray, Sigma_hat: np.ndarray, R: np.ndarray):
    """Compute new state covariance
       Uses Joseph form to prevent roundoff errors

    Args:
        K (np.ndarray): Kalman gain
        H (np.ndarray): Measurement Matrix
        Sigma_hat (np.ndarray): Predicted state covariance
        R (np.ndarray): Measurement covariance

    Returns:
        np.ndarray: New state covariance
    """
    A = (np.eye(H.shape[1]) - K @ H)
    return A @ Sigma_hat @ A.T + K @ R @ K.T


def compute_X_hat(X: np.ndarray, a: np.ndarray, q, timestep: float, Cstop: int = 1):
    """Compute next predicted state

    Args:
        X (np.ndarray): Current state
        a (np.ndarray): Acceleration vector
        q (np.ndarray): Current quaternion
        timestep (float): Time step [s]
        Cstop (int, optional): Currently moving? 1=yes, 0=no. Defaults to 1.

    Returns:
        np.ndarray: Next predicted state
    """
    A = generate_A(timestep, Cstop)
    B = generate_B(timestep)
    C = generate_C(q)
    return A @ X + B @ a + C


def compute_X(X_hat: np.ndarray, H: np.ndarray, K: np.ndarray, m: np.ndarray):
    """Compute next filtered state

    Args:
        X_hat (np.ndarray): Next predicted state
        H (np.ndarray): Measurement matrix
        K (np.ndarray): Kalman gain matrix
        m (np.ndarray): Measurements

    Returns:
        np.ndarray: Next filtered state
    """
    return X_hat + K @ (m - H @ X_hat)


def filter_GNSS(X: np.ndarray, Sigma: np.ndarray, R: np.ndarray,
                N: np.ndarray, a: np.ndarray, q, gps: np.ndarray,
                v_freq: float, timestep: float, Cstop: int = 1):
    """Run EKF with GNSS and frequency data

    Args:
        X (np.ndarray): Current state
        Sigma (np.ndarray): Current covariance
        R (np.ndarray): Measurement noise covariance
        N (np.ndarray): Process noise covariance
        a (np.ndarray): Acceleration vector
        q (Quaternion): Current quaternion
        gps (np.ndarray): GPS measurements [m, m]
        v_freq (float): Velocity magnitude from frequency data [m/s]
        timestep (float): Time step [s]
        Cstop (int, optional): Currently moving? 1=yes, 0=no. Defaults to 1.

    Returns:
        tuple: Next filtered state, Next state covariance, Kalman gain
    """

    # Predict
    A = generate_A(timestep, Cstop)
    X_hat = compute_X_hat(X, a, q, timestep, Cstop)
    Sigma_hat = compute_Sigma_hat(A, Sigma, N)

    # If we have GPS data, update.
    # Else, return predicted state and covariance
    if gps is not None:
        # If we have frequency data, use it to predict velocity
        # Else just use GPS data
        freq_data = v_freq is not None
        if freq_data:
            m = np.array([gps[0,0], gps[1,0], v_freq]).reshape((3, 1))
        else:
            m = gps
            R = R[:2, :2]

        H = generate_H(X_hat, freq_data=freq_data)
        K = compute_K(Sigma_hat, H, R)
        X_filtered = compute_X(X_hat, H, K, m)
        Sigma = compute_Sigma(K, H, Sigma_hat, R)
    else:
        X_filtered = X_hat
        Sigma = Sigma_hat
        K = None
    return X_filtered, Sigma, K


def run_Kalman_filter(readings: Readings, imu: IMU, PROCESS_NOISE_COEFFICIENT: float, FREQ_MEASUREMENT_NOISE_COEFFICIENT: float):
    """Run EKF on the given readings and IMU data.

    Args:
        readings (Readings): IMU and GPS readings.
        imu (IMU): IMU object

    Returns:
        tuple: see return for now, should be a dict or something
    """

    # Get the first reading for initializing
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
    print('Initial Lat Lon: ', lat0, lon0)

    X0 = np.zeros((10, 1))
    X0[:3] = initial_pos
    X0[-4:] = initial_orientation.elements.reshape((4, 1))

    accel_mag_prev = np.zeros(800)

    Sigma0 = np.eye(10) * 4.0

    X = X0
    Sigma = Sigma0

    N = np.eye(10) * PROCESS_NOISE_COEFFICIENT
    R = np.eye(3) * GPS_MEASUREMENT_NOISE_COEFFICIENT
    R[2, 2] = FREQ_MEASUREMENT_NOISE_COEFFICIENT

    g = np.array([0, 0, -9.81]).reshape((3, 1))  # gravity vector

    # Record a bunch of data, maybe should be a class or dict?
    timestamps = np.zeros(readings.num_readings)
    gps_timestamps = np.zeros(len(readings.gps))
    positions = np.zeros((readings.num_readings, 3))
    gps_positions = np.zeros((len(readings.gps), 2), dtype=np.float64)
    coords = np.zeros((len(readings.gps), 2), dtype=np.float64)
    all_positions = np.zeros((readings.num_readings, 3))
    velocities = np.zeros((readings.num_readings, 3))
    orientations = np.zeros((readings.num_readings, 4))
    v_freqs = np.zeros(readings.num_readings)
    wls_positions = np.zeros((readings.num_readings, 3))
    accel_mags = np.zeros(readings.num_readings)
    gyro_readings = np.zeros((readings.num_readings, 3))

    # Initialize all of the saved data
    positions[0] = X[:3,0]
    gps_positions[0] = X[:2,0]
    coords[0] = xy_2_lat_lon(positions[0][0], positions[0][1], lat0, lon0)[:, 0]
    all_positions[0] = X[:3, 0]
    velocities[0] = X[3:6,0]
    orientations[0] = X[-4:,0]
    v_freqs[0] = 0.0
    timestamps[0] = first_reading.timestamp
    gps_timestamps[0] = first_reading.timestamp
    accel_mags[0] = np.linalg.norm(first_reading.accel)
    gyro_readings[0] = first_reading.gyro[:, 0]

    avg_timestep = 1.0 / 500.0

    reading_ind = 1
    gps_ind = 1

    # Use tqdm to show progress
    pbar = tqdm(total=readings.num_readings-1)
    pbar.set_description("Kalman Filtering Data...")

    # Run until out of data
    while True:
        reading = readings.get_next_reading()
        if reading is None:
            break

        # Get current accel data
        accel = reading.accel
        accel_mag_prev[1:] = accel_mag_prev[:-1]
        accel_mag_prev[0] = np.linalg.norm(accel)

        # Get current gyro, mag, and gps data
        gyro = reading.gyro
        mag = reading.mag
        gps = reading.gps

        # Convert gps to x, y from lat and long
        if gps is not None:
            # gps = lat_lon_2_xy(lat0, lon0, gps[0], gps[1])
            gps = np.array([gps[0], gps[1]]).reshape((2, 1))

        # Get timestep
        timestep = reading.timestep
        avg_timestep = avg_timestep*0.95 + timestep*0.05

        # Check if we have mag data and update our orientation
        use_mag = (reading.type == ReadingType.ACCEL_GYRO_MAG_GPS
                   or reading.type == ReadingType.ACCEL_GYRO_MAG)
        imu.update(accel, gyro, mag, timestep, use_mag=use_mag)

        q = imu.orientation

        # Get our direction cosine matrix body -> world, apply to accel data
        Cns = q.rotation_matrix
        accel = (np.matmul(Cns, accel) - g)

        # Compute velocity using accelerometer frequency data
        # Only if average acceleration high enough to say we are running
        v_freq = None
        v_freqs[reading_ind] = 0.0
        if reading_ind > len(accel_mag_prev):
            avg_accel = np.mean(np.sqrt(accel_mag_prev**2))
            w = np.fft.rfft(accel_mag_prev - np.mean(accel_mag_prev))
            freqs = np.fft.rfftfreq(len(accel_mag_prev), d=avg_timestep)
            w = w[freqs < 4.75]
            max_freq = freqs[np.argmax(np.abs(w))]
            if avg_accel > 12.0:
                v_freq = 0.155*(2*max_freq)+2.6
                v_freqs[reading_ind] = v_freq

        X_new, Sigma_new, K = filter_GNSS(X, Sigma, R, N, accel, q, gps, v_freq, timestep, Cstop=1)

        if gps is not None:
            imu.position = X_new[:3]
            imu.velocity = X_new[3:6]
            wls_positions[reading_ind] = np.array([gps[0,0], gps[1,0], 0.0], dtype=np.float32)
            timestamps[reading_ind] = reading.timestamp
            positions[reading_ind] = X_new[:3, 0]
            velocities[reading_ind] = X_new[3:6, 0]
            orientations[reading_ind] = X_new[-4:, 0]
            gps_timestamps[gps_ind] = reading.timestamp
            coords[gps_ind] = xy_2_lat_lon(
                positions[reading_ind][0], positions[reading_ind][1], lat0, lon0)[:, 0]
            gps_positions[gps_ind] = X[:2, 0]
            gps_ind += 1
        else:
            wls_positions[reading_ind] = wls_positions[reading_ind - 1]
            positions[reading_ind] = positions[reading_ind - 1]
            velocities[reading_ind] = velocities[reading_ind - 1]
            timestamps[reading_ind] = timestamps[reading_ind - 1]
            orientations[reading_ind] = orientations[reading_ind - 1]
        # positions[reading_ind] = X_new[:3, 0]
        all_positions[reading_ind] = X_new[:3, 0]
        velocities[reading_ind] = X_new[3:6, 0]
        orientations[reading_ind] = X_new[-4:, 0]
        timestamps[reading_ind] = reading.timestamp
        accel_mags[reading_ind] = np.linalg.norm(reading.accel)
        gyro_readings[reading_ind] = reading.gyro[:, 0]

        reading_ind += 1
        pbar.update(1)
        X = X_new
        Sigma = Sigma_new

    pbar.close()
    return timestamps, positions, gps_positions, velocities, orientations, wls_positions, all_positions, v_freqs, gps_timestamps, coords, accel_mags, gyro_readings


def main():
    PROCESS_NOISE_COEFFICIENTs = np.linspace(0.00005, 0.00075, num=4)
    FREQ_MEASUREMENT_NOISE_COEFFICIENTs = [1.5, 10000.0]

    for PROCESS_NOISE_COEFFICIENT in PROCESS_NOISE_COEFFICIENTs:
        for FREQ_MEASUREMENT_NOISE_COEFFICIENT in FREQ_MEASUREMENT_NOISE_COEFFICIENTs:
            # Load the data
            readings = Readings(IMU_fpath, GPS_fpath,
                                start_time=start_time, duration=duration)

            # Initialize IMU objects
            gyro = Gyro()
            mag = Mag()
            accel = Accel(gyro=gyro, mag=mag)
            imu = IMU(gyro=gyro, accel=accel)

            # Run EKF
            timestamps, positions, gps_positions, velocities, orientations, wls_positions, all_positions, v_freqs, gps_timestamps, coords, accel_mags, gyro_readings = run_Kalman_filter(
                readings, imu, PROCESS_NOISE_COEFFICIENT, FREQ_MEASUREMENT_NOISE_COEFFICIENT)

            # Timestamps from ms to s, remove offsets
            timestamps -= timestamps[0]
            timestamps /= 1000

            # Save data
            # df = pd.DataFrame(gps_timestamps, columns=['UNIX Epoch timestamp-utc'])
            # df = pd.concat([df, pd.DataFrame(coords)], axis=1)
            # df.columns = ['UNIX Epoch timestamp-utc', 'lat', 'lon']
            # if write_csv:
            #     df.to_csv(GPS_fpath.replace(
            #         '.csv',
            #         f'_filtered_{PROCESS_NOISE_COEFFICIENT}_{GPS_MEASUREMENT_NOISE_COEFFICIENT}_{FREQ_MEASUREMENT_NOISE_COEFFICIENT}.csv'),
            #         index=False,
            #         header=True)

            # Save data
            df = pd.DataFrame(gps_timestamps, columns=['UNIX Epoch timestamp-utc'])
            df = pd.concat([df, pd.DataFrame(gps_positions)], axis=1)
            df.columns = ['UNIX Epoch timestamp-utc', 'x', 'y']
            if write_csv:
                df.to_csv(GPS_fpath.replace(
                    '.csv',
                    f'_filtered_{PROCESS_NOISE_COEFFICIENT}_{GPS_MEASUREMENT_NOISE_COEFFICIENT}_{FREQ_MEASUREMENT_NOISE_COEFFICIENT}.csv'),
                    index=False,
                    header=True)

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
    plt.scatter(positions[:,0], positions[:,1], marker='o', s=2, c=cmap, zorder=10, label='Inertial Navigation')
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
    plt.title('Pure Inertial Navigation vs GPS Weighted-Least-Squares')

    # Plot velocities
    plt.figure(figsize=(17, 10))
    plt.title('Inertial Navigation With Cadence: Velocity Magnitude vs Time', fontsize=18)
    # plt.plot(timestamps, velocities[:,0], label='x')
    # plt.plot(timestamps, velocities[:,1], label='y')
    plt.plot(timestamps, np.linalg.norm(velocities[:,:2], axis=1), label='Velocity Magnitude')
    # plt.plot(timestamps, 26.8224*np.ones_like(timestamps),
    #          'k--', label='1 Minute Mile Pace', linewidth=2)
    plt.plot([27.0, 27.0], [0, 5], 'g-.', label='Start of Run', linewidth=2)
    plt.xlabel('Time From Start [s]', fontsize=14)
    plt.ylabel('Velocity [m/s]', fontsize=14)
    plt.legend(fontsize=14)

    plt.figure(figsize=(10, 10))
    plt.title('Orientation Solution vs Time', fontsize=18)
    plt.plot(timestamps, orientations[:, 0], label='w')
    plt.plot(timestamps, orientations[:, 1], label='x')
    plt.plot(timestamps, orientations[:, 2], label='y')
    plt.plot(timestamps, orientations[:, 3], label='z')
    plt.legend(fontsize=14)
    plt.xlabel('Time From Start [s]', fontsize=14)
    plt.ylabel('Quaternion Component []', fontsize=14)

    # Plot Accelerometer Magnitude
    plt.figure(figsize=(21, 7))
    plt.title('Accelerometer Reading Magnitude vs Time While Running', fontsize=18)
    plt.plot(timestamps, accel_mags, label='Accelerometer Reading Magnitude')
    plt.plot(timestamps, 9.81*np.ones_like(timestamps), 'k--', label='g = 9.81 $m/s^2$')
    plt.ylim(bottom=0)
    plt.legend(fontsize=14)
    plt.xlabel('Time From Start [s]', fontsize = 14)
    plt.ylabel('Acceleration [$m/s^2$]', fontsize=14)

    # Plot rotation rates
    plt.figure(figsize=(17, 10))
    plt.title('Gyroscope Readings vs Time While Running', fontsize=18)
    plt.plot(timestamps, (180/np.pi)*gyro_readings[:, 0],
             label='X-Axis')
    plt.plot(timestamps, (180/np.pi)*gyro_readings[:, 1],
             label='Y-Axis')
    plt.plot(timestamps, (180/np.pi)*gyro_readings[:, 2],
             label='Z-Axis')
    plt.legend(fontsize=14)
    plt.xlabel('Time From Start [s]', fontsize = 14)
    plt.ylabel('Rotation Rate [$\circ$/s]', fontsize=14)

    plt.figure(figsize=(10, 10))
    plt.plot(timestamps, v_freqs, label='v_freq')

    plt.show()


if __name__ == "__main__":
    main()