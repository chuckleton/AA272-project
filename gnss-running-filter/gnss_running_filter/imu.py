import numpy as np
from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation as R
import attr
from gnss_running_filter.gyro import Gyro
from gnss_running_filter.accel import Accel

def R_to_quat(R):
    q = R.as_quat()
    return Quaternion(q[3], q[0], q[1], q[2])
@attr.s
class IMU:
    gyro: Gyro = attr.ib(default=Gyro)
    accel: Accel = attr.ib(default=Accel(gyro))

    def compute_g(self, accel_data: np.ndarray, **kwargs):
        return self.accel.compute_g(accel_data, **kwargs)

    def get_mag_orientation(self, mag_data: np.ndarray):
        return self.accel.compute_mag_orientation(mag_data)

    def update_orientation_gyro(self, gyro_data: np.ndarray):
        self.gyro.update(gyro_data)

    def update_orientation_mag(self, mag_data: np.ndarray):
        self.orientation = self.get_mag_orientation(mag_data)

    def update(self, accel_data: np.ndarray, gyro_data: np.ndarray,
               mag_data: np.ndarray, timestep: float, use_mag: bool = False):
        new_g = self.compute_g(accel_data, always_compute=True)
        self.gyro.update(gyro_data, timestep)
        if use_mag and new_g:
            self.update_orientation_mag(mag_data)

    @property
    def orientation(self):
        return self.gyro.orientation

    @orientation.setter
    def orientation(self, orientation, scalar_first=True):
        """Set the orientation of the IMU using a quaternion.

        Args:
            orientation (Quaternion, R, np.ndarray): Quaternion,
                Rotation matrix, or numpy array of orientation
            scalar_first (bool, optional): (w, x, y, z)=True,
                (x, y, z, w)=False. Defaults to True.
        """
        if isinstance(orientation, Quaternion):
            self.gyro.orientation = orientation
        elif isinstance(orientation, R):
            self.gyro.orientation = R_to_quat(orientation)
        elif isinstance(orientation, np.ndarray):
            if scalar_first:
                self.gyro.orientation = Quaternion(orientation)
            else:
                self.gyro.orientation = Quaternion(orientation[3],
                                                   orientation[0],
                                                   orientation[1],
                                                   orientation[2])
        else:
            raise TypeError(
                'orientation must be a Quaternion, scipy.spatial.transform.Rotation, or numpy array')

    @property
    def velocity(self):
        return self.accel.velocity

    @velocity.setter
    def velocity(self, velocity: np.ndarray):
        """Set the velocity of the IMU.

        Args:
            velocity (np.ndarray): velocity in m/s
        """
        assert isinstance(velocity, np.ndarray), 'velocity must be a numpy array'
        assert len(velocity) == 3, 'velocity must be a 3-vector'
        self.accel.position = velocity.reshape((3, 1))

    @property
    def position(self):
        return self.accel.position

    @position.setter
    def position(self, position: np.ndarray):
        """Set the position of the IMU.

        Args:
            position (np.ndarray): position in m
        """
        assert isinstance(position, np.ndarray), 'position must be a numpy array'
        assert len(position) == 3, 'position must be a 3-vector'
        self.accel.position = position.reshape((3, 1))