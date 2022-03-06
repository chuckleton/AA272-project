from imu import IMU


def run_Kalman_filter(imu_fname: str, gps_fname: str, start_time: int, duration: int):
    imu_readings = get_imu_readings(imu_fname, start_time, duration)
    gps_readings = get_sensor_readings(gps_fname, start_time, duration)

    gyro = Gyro()
    accel = Accel(gyro=gyro)

if __name__ == "__main__":
    imu = IMU()