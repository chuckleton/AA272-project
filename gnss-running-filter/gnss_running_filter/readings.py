import pandas as pd
import numpy as np
import attr

@attr.s
class Reading:
    ACCEL_GYRO = 0
    ACCEL_GYRO_MAG = 1
    ACCEL_GYRO_GPS = 2
    ACCEL_GYRO_MAG_GPS = 3
    time: int = attr.ib(default=0)
    accel: np.ndarray = attr.ib(default=np.zeros((3, 1)))
    gyro: np.ndarray = attr.ib(default=np.zeros((3, 1)))
    mag: np.ndarray = attr.ib(default=np.zeros((3, 1)))
    gps: np.ndarray = attr.ib(default=np.zeros((3, 1)))
    type: int = attr.ib(default=ACCEL_GYRO)


@attr.s
class Readings:
    imu_fname: str = attr.ib(default=None)
    gps_fname: str = attr.ib(default=None)
    start_time: int = attr.ib(default=None)
    gps_start_time: bool = attr.ib(default=True)
    duration: int = attr.ib(default=None)

    imu: pd.DataFrame = attr.ib(default=pd.DataFrame())
    gps: pd.DataFrame = attr.ib(default=pd.DataFrame())

    def __attrs_post_init__(self):
        if self.imu_fname is not None and self.gps_fname is not None:
            if self.start_time is None:
                if self.gps_start_time:
                    test_df = pd.read_csv(self.gps_fname, nrows=1)
                else:
                    test_df = pd.read_csv(self.imu_fname, nrows=1)
                self.start_time = test_df.iloc[0]['UNIX Epoch timestamp-utc']-1
            if self.duration is None:
                self.duration = 1000000
            self.gps = get_sensor_readings(
                self.gps_fname, self.start_time, self.duration)

            # We want only data starting after we have GPS signal
            self.start_time = self.gps.iloc[0]['UNIX Epoch timestamp-utc']-1
            self.imu = get_imu_readings(self.imu_fname, self.start_time, self.duration)
            self.time = self.start_time
            self.imu_loc = 0
            self.gps_loc = 0
            self.mag_loc = 0

    def get_next_reading_time(self):
        if self.time - self.start_time >= self.duration:
            return None
        next_reading_time = self.imu['accel'].iloc[self.imu_loc]['UNIX Epoch timestamp-utc']
        return next_reading_time

    def check_gps_data(self, time: int):
        next_gps_time = self.gps.iloc[self.gps_loc]['UNIX Epoch timestamp-utc']
        return next_gps_time <= time

    def check_mag_data(self, time: int):
        next_mag_time = self.imu['mag'].iloc[self.mag_loc]['UNIX Epoch timestamp-utc']
        return next_mag_time <= time

    def get_imu_reading(self, sensor, index):
        reading = self.imu[sensor].iloc[index]
        reading = np.array([reading['x-axis'],
                            reading['y-axis'],
                            reading['z-axis']]).reshape((3, 1))
        return reading

    def get_next_reading(self):
        if self.time - self.start_time >= self.duration:
            return None

        next_reading_time = self.get_next_reading_time()
        has_gps_data = self.check_gps_data(next_reading_time)
        has_mag_data = self.check_mag_data(next_reading_time)

        gyro_reading = self.get_imu_reading('gyro', self.imu_loc)
        accel_reading = self.get_imu_reading('accel', self.imu_loc)
        self.imu_loc += 1

        type = Reading.ACCEL_GYRO

        if has_gps_data:
            gps_reading = self.gps.iloc[self.gps_loc]
            gps_reading = np.array([gps_reading['lat'],
                                    gps_reading['lon'],
                                    gps_reading['alt']]).reshape((3, 1))
            self.gps_loc += 1
            type = Reading.ACCEL_GYRO_GPS
            if has_mag_data:
                type = Reading.ACCEL_GYRO_MAG_GPS
            else:
                type = Reading.ACCEL_GYRO_GPS
            self.gps_loc += 1

        if has_mag_data:
            mag_reading = self.get_imu_reading('mag', self.mag_loc)
            if not has_gps_data:
                type = Reading.ACCEL_GYRO_MAG
            self.mag_loc += 1

        reading = Reading(
            time=next_reading_time,
            accel=accel_reading,
            gyro=gyro_reading,
            mag=mag_reading,
            gps=gps_reading,
            type=type
        )
        return reading





def get_sensor_readings(fname: str, start_time: int, duration: int):
    readings = pd.read_csv(fname)
    readings.drop_duplicates(subset=['UNIX Epoch timestamp-utc'], inplace=True)
    readings = readings[(readings['UNIX Epoch timestamp-utc'] > start_time)
                        & (readings['UNIX Epoch timestamp-utc'] < start_time + duration)]
    readings['timestep'] = readings['UNIX Epoch timestamp-utc'].diff()
    readings['timestep'] /= 1000.0
    return readings

def get_imu_readings(imu_fname: str, start_time: int, duration: int):
    mag_readings = get_sensor_readings(
        imu_fname + '-mag.txt', start_time, duration)
    accel_readings = get_sensor_readings(
        imu_fname + '-accel.txt', start_time, duration)
    gyro_readings = get_sensor_readings(
        imu_fname + '-gyro.txt', start_time, duration)

    accel_readings['magnitude'] = np.sqrt(
        accel_readings['x-axis']**2 + accel_readings['y-axis']**2 + accel_readings['z-axis']**2)
    # g_mag = accel_readings['magnitude'].mean()
    # print('g_mag: ', g_mag)

    accel_readings['z-axis'] *= -1

    # Accel offsets
    accel_readings['x-axis'] += 0.0
    accel_readings['y-axis'] += -0.0
    accel_readings['z-axis'] += -0.0

    # Magnetometer Offsets
    mag_readings['x-axis'] += 0
    mag_readings['y-axis'] += 0
    mag_readings['z-axis'] += 0

    return {'accel': accel_readings,
            'gyro': gyro_readings,
            'mag': mag_readings}
