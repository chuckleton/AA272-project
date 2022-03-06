import numpy as np
from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation as R
import attr
from gnss_running_filter.gyro import Gyro
from gnss_running_filter.accel import Accel


@attr.s
class IMU:
    gyro: Gyro = attr.ib(default=Gyro)
    accel: Accel = attr.ib(default=Accel(gyro))

    def get_mag_orientation(self, mag_data: np.ndarray):
        return self.accel.get_mag_orientation(mag_data)