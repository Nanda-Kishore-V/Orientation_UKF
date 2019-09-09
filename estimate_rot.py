import scipy.io

from quat_utils import *


def prediction(state, control, P, Q):
    qu = vec2quat(np.atleast_2d(control).T)

    S = np.linalg.cholesky(P + Q)
    n = P.shape[0]
    S = S * np.sqrt(0.5 * n)
    W = np.array(np.hstack([S, -S]))

    noise_quaternions = vec2quat(W)
    X = quat_multiply(noise_quaternions.T, state)

    Y = quat_multiply(X, qu.T)

    next_state, error = quat_average(Y.T, state)

    next_cov = np.dot(error, error.T) / 12.0

    return next_state, next_cov, Y, error


def estimate_rot(data_num=1):
    # Data from dataset
    imu = scipy.io.loadmat("imu/imuRaw" + str(data_num) + ".mat")
    imu_vals = np.array(imu['vals'], dtype=np.float64)
    imu_vals = imu_vals.T
    imu_ts = imu['ts']
    imu_ts = imu_ts.T

    Vref = 3300

    acc_x = -np.array(imu_vals[:, 0])
    acc_y = -np.array(imu_vals[:, 1])
    acc_z = np.array(imu_vals[:, 2])
    acc = np.array([acc_x, acc_y, acc_z]).T

    acc_sensitivity = 330.0
    acc_scale_factor = Vref / 1023.0 / acc_sensitivity
    acc_bias = acc[0, :] - (np.array([0, 0, 1]) / acc_scale_factor)
    acc_val = (acc - acc_bias) * acc_scale_factor

    gyro_x = np.array(imu_vals[:, 4])
    gyro_y = np.array(imu_vals[:, 5])
    gyro_z = np.array(imu_vals[:, 3])
    gyro = np.array([gyro_x, gyro_y, gyro_z]).T

    if data_num in [1, 2, 3, 4]:
        gyro_bias = np.array([373.73906112, 375.59007972, 377.03554384])
        gyro_sensitivity = 3.3
    elif data_num == 5:
        gyro_bias = np.array([373.73906112, 375.59007972, 367.53554384])
        gyro_sensitivity = 3.2
    else:
        gyro_bias = np.array([373.73906112, 375.59007972, 364.53554384])
        gyro_sensitivity = 3.2

    gyro_scale_factor = Vref / 1023 / gyro_sensitivity
    gyro_val = (gyro - gyro_bias) * gyro_scale_factor * (np.pi / 180)

    state = np.array([1, 0, 0, 0])

    # Constants
    if data_num in [1, 2, 3, 4]:
        P = 0.00000000001 * np.identity(3)  # 0.00001
        Q = np.array([[0.003, 0, 0],
                      [0, 0.003, 0],
                      [0, 0, 0.003]])
        R = np.array([[0.005, 0, 0],
                      [0, 0.005, 0],
                      [0, 0, 0.005]])
    else:
        P = 0.000000001 * np.identity(3)  # 0.00001
        Q = np.array([[0.03, 0, 0],
                      [0, 0.03, 0],
                      [0, 0, 0.03]])
        R = np.array([[0.05, 0, 0],
                      [0, 0.05, 0],
                      [0, 0, 0.05]])

    # Storage spaces
    roll = []
    pitch = []
    yaw = []

    # UKF loop
    prev_timestep = imu_ts[0] - 0.01
    for i in range(imu_ts.shape[0]):
        control = gyro_val[i] * (imu_ts[i] - prev_timestep)
        prev_timestep = imu_ts[i]

        state, P, sigma_points, error = prediction(state, control, P, Q)

        Z = quat_multiply(quat_inverse(sigma_points), np.array([0, 0, 0, 1]))
        Z = quat_multiply(Z, sigma_points)
        Z = Z[:, 1:]

        z_est = np.mean(Z, axis=0)

        Z_error = (Z - z_est).T
        Pzz = np.dot(Z_error, Z_error.T) / 12.0
        Pvv = Pzz + R
        Pxz = np.dot(error, Z_error.T) / 12.0

        K = np.dot(Pxz, np.linalg.inv(Pvv))

        nu = np.transpose(acc_val[i] - z_est)
        Knu = vec2quat(np.atleast_2d(np.dot(K, nu)).T)
        state = quat_multiply(Knu.T, state).reshape(4, )
        P = P - np.dot(np.dot(K, Pvv), np.transpose(K))

        if data_num in [1, 2, 3, 4]:
            r, p, y = rotmat2rpy(quat2rotmat(state))
            roll.append(r)
            pitch.append(p)
            yaw.append(y)
        else:
            r, p, y = quat2rpy(state)
            roll.append(r)
            pitch.append(p)
            yaw.append(y)

    roll = np.array(roll)
    pitch = np.array(pitch)
    yaw = np.array(yaw)

    return roll, pitch, yaw
