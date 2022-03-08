import pandas as pd
import numpy as np
import attr
from enum import Enum

class ReadingType(Enum):
    ACCEL_GYRO = 0
    ACCEL_GYRO_MAG = 1
    ACCEL_GYRO_GPS = 2
    ACCEL_GYRO_MAG_GPS = 3

@attr.s
class Reading:
    """A single set of sensor readings at a given timestamp

       type - The components present in the reading as a ReadingType enum
    """
    timestamp: int = attr.ib(default=0)
    timestep: float = attr.ib(default=0)
    accel: np.ndarray = attr.ib(default=np.zeros((3, 1)))
    gyro: np.ndarray = attr.ib(default=np.zeros((3, 1)))
    mag: np.ndarray = attr.ib(default=np.zeros((3, 1)))
    gps: np.ndarray = attr.ib(default=np.zeros((3, 1)))
    type: int = attr.ib(default=ReadingType.ACCEL_GYRO)


@attr.s
class Readings:
    """Class for holding and accessing all the sensor readings
       for a given time period.
    """
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

            # We want only data starting after we have GPS signal, but need enough to get mag data
            # before we start for initializing orientation.
            # This is a really dumb way of doing it, but it works.
            mag_reading_delay = 25
            self.start_time = self.gps.iloc[0]['UNIX Epoch timestamp-utc'] - \
                mag_reading_delay
            self.imu = get_imu_readings(
                self.imu_fname, self.start_time, mag_reading_delay)
            nearest_mag_time = self.imu['mag'].iloc[0]['UNIX Epoch timestamp-utc']-1
            self.imu = get_imu_readings(
                self.imu_fname, nearest_mag_time, mag_reading_delay)
            self.last_mag_reading = self.get_imu_reading('mag', 0)
            self.start_time += mag_reading_delay - 1
            self.imu = get_imu_readings(
                self.imu_fname, self.start_time, self.duration)

            self.time = self.start_time
            self.imu_loc = 0
            self.gps_loc = 0
            self.mag_loc = 0
            self.num_readings = len(self.imu['accel'])

    def get_next_reading_time(self):
        """Get the next timestamp with accel and gyro data

        Returns:
            int: next timestamp with accel and gyro data
        """
        next_reading_time = self.imu['accel'].iloc[self.imu_loc]['UNIX Epoch timestamp-utc']
        return next_reading_time

    def check_gps_data(self, time: int):
        """Check if there is new GPS data available at the given time

        Args:
            time (int): timestamp to check for GPS data

        Returns:
            bool: new GPS data available
        """
        if self.gps_loc >= len(self.gps):
            return False
        next_gps_time = self.gps.iloc[self.gps_loc]['UNIX Epoch timestamp-utc']
        return next_gps_time <= time

    def check_mag_data(self, time: int):
        """Check if there is new Magnetometer data available at the given time

        Args:
            time (int): timestamp to check for mag data

        Returns:
            bool: new mag data available
        """
        if self.mag_loc >= len(self.imu['mag']):
            return False
        next_mag_time = self.imu['mag'].iloc[self.mag_loc]['UNIX Epoch timestamp-utc']
        return next_mag_time <= time

    def get_imu_reading(self, sensor: str, index: int):
        """Get a single IMU reading

        Args:
            sensor (str): The sensor to get the reading from ['accel', 'gyro', 'mag']
            index (int): Dataframe index of the reading

        Returns:
            np.ndarray: (3,1) numpy array of the reading
        """
        reading = self.imu[sensor].iloc[index]
        reading = np.array([reading['x-axis'],
                            reading['y-axis'],
                            reading['z-axis']]).reshape((3, 1))
        return reading

    def get_next_reading(self, advance_time: bool = True):
        """Get the next available reading

        Args:
            advance_time (bool): Move to next timestep after reading

        Returns:
            Reading: next available reading
        """
        if self.imu_loc >= self.num_readings:
            return None
        if self.imu_loc >= len(self.imu['gyro']):
            return None

        self.time = self.get_next_reading_time()
        has_gps_data = self.check_gps_data(self.time)
        has_mag_data = self.check_mag_data(self.time)

        gyro_reading = self.get_imu_reading('gyro', self.imu_loc)
        accel_reading = self.get_imu_reading('accel', self.imu_loc)
        timestep = self.imu['accel'].iloc[self.imu_loc]['timestep']
        if advance_time:
            self.imu_loc += 1

        type = ReadingType.ACCEL_GYRO

        gps_reading = None
        if has_gps_data:
            gps_reading = self.gps.iloc[self.gps_loc]
            gps_reading = np.array([gps_reading['x'],
                                    gps_reading['y'],
                                    gps_reading['z']]).reshape((3, 1))
            type = ReadingType.ACCEL_GYRO_GPS
            if has_mag_data:
                type = ReadingType.ACCEL_GYRO_MAG_GPS
            if advance_time:
                self.gps_loc += 1

        mag_reading = self.last_mag_reading
        if has_mag_data:
            mag_reading = self.get_imu_reading('mag', self.mag_loc)
            self.last_mag_reading = mag_reading
            if not has_gps_data:
                type = ReadingType.ACCEL_GYRO_MAG
            if advance_time:
                self.mag_loc += 1

        reading = Reading(
            timestamp=self.time,
            timestep=timestep,
            accel=accel_reading,
            gyro=gyro_reading,
            mag=mag_reading,
            gps=gps_reading,
            type=type
        )
        return reading





def get_sensor_readings(fname: str, start_time: int, duration: int):
    """Get a dataframe of sensor readings from a file

    Args:
        fname (str): filename/path
        start_time (int): UTC (ms) time of first reading
        duration (int): UTC (ms) time duration to read

    Returns:
        pd.DataFrame: sensor readings dataframe
    """
    readings = pd.read_csv(fname)
    readings.drop_duplicates(subset=['UNIX Epoch timestamp-utc'], inplace=True)
    readings = readings[(readings['UNIX Epoch timestamp-utc'] > start_time)
                        & (readings['UNIX Epoch timestamp-utc'] < start_time + duration)]
    readings['timestep'] = readings['UNIX Epoch timestamp-utc'].diff()
    readings['timestep'] /= 1000.0
    return readings

def get_imu_readings(imu_fname: str, start_time: int, duration: int):
    """Get the IMU readings from their files

    Args:
        imu_fname (str): Base filepath/name of IMU files
        start_time (int): UTC (ms) time of first reading
        duration (int): UTC (ms) time duration to read

    Returns:
        dict: dictionary of IMU readings as pd.DataFrames
                {'accel', 'gyro', 'mag'}
    """
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

    # accel_readings['z-axis'] *= -1
    # mag_readings['z-axis'] *= -1

    # Accel offsets
    accel_readings['x-axis'] += 0.0
    accel_readings['y-axis'] += 2.3
    accel_readings['z-axis'] += -0.0

    # Magnetometer Offsets
    mag_readings['x-axis'] += 0.0
    mag_readings['y-axis'] += 0.0
    mag_readings['z-axis'] -= 0.0

    return {'accel': accel_readings,
            'gyro': gyro_readings,
            'mag': mag_readings}
