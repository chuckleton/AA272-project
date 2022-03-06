from readings import Readings, ReadingType
from imu import IMU

IMU_fpath = "C:\\Users\\dkolano\\OneDrive - Agile Space Industries\\Documents\\Homework\\Global Positioning Systems\\Project\\GNSS-INS_Logger\\Calibr_IMU_Log\\SM-G950U_IMU_20220223171411"
GPS_fpath = "C:\\Users\\dkolano\\OneDrive - Agile Space Industries\\Documents\\Homework\\Global Positioning Systems\\Project\\WLS_LLA\\WLS_LLA_track.csv"

start_time = 1645654491442
duration = 5000

def run_Kalman_filter(imu_fname: str, gps_fname: str, start_time: int, duration: int):
    imu_readings = get_imu_readings(imu_fname, start_time, duration)
    gps_readings = get_sensor_readings(gps_fname, start_time, duration)

    gyro = Gyro()
    accel = Accel(gyro=gyro)

if __name__ == "__main__":
    readings = Readings(IMU_fpath, GPS_fpath, start_time=start_time, duration=duration)
    for _ in range(100):
        reading = readings.get_next_reading()
        print(reading)