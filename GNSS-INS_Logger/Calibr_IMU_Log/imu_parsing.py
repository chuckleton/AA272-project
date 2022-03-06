import pandas as pd
import numpy as np

# Replace fname here with the name of the file you want to parse
fname = 'SM-G950U_IMU_20220305023351'


if __name__ == '__main__':
    names = ['UNIX Epoch timestamp-utc', 'timestamp-gps', 'sensor Flag', 'x-axis', 'y-axis', 'z-axis']
    dtypes = {names[0]: np.int64, names[1]: np.int64, names[2]: np.int8, names[3]: np.float64, names[4]: np.float64, names[5]: np.float64}
    db = pd.read_csv(fname+'.txt', sep=',',
                    header=None, names=names, skiprows=6, na_values='N/A', skipinitialspace=True)

    accel_db = db[db['sensor Flag'] == 1]
    gyro_db = db[db['sensor Flag'] == 2]
    mag_db = db[db['sensor Flag'] == 3]

    accel_db.to_csv(fname+'-accel.txt', index=False)
    gyro_db.to_csv(fname+'-gyro.txt', index=False)
    mag_db.to_csv(fname+'-mag.txt', index=False)
