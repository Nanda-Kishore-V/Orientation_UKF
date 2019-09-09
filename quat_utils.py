import numpy as np


def quat_conjugate(q):
    q_internal = np.transpose(np.atleast_2d(q.copy()))
    q_internal[1:, :] = -q_internal[1:, :]
    return np.transpose(q_internal)


def quat_normalize(q):
    q_internal = np.transpose(np.atleast_2d(q.copy()))
    return np.transpose(q_internal / np.linalg.norm(q_internal, axis=0))


def quat_inverse(q):
    q_conj = np.transpose(np.atleast_2d(quat_conjugate(q)))
    return np.transpose(q_conj / np.sum(np.square(q_conj), axis=0))


def quat_multiply_left(q1, q2):
    """
    Multiplies one quaternion with multiple quaternions.
    q1 - shape: (4,)
    q2 - shape: (4,m) m-number of quaternions
    return q - shape (4,m)
    """
    q1_internal = np.transpose(np.atleast_2d(q1.copy()))
    q2_internal = np.transpose(np.atleast_2d(q2.copy()))
    q = np.zeros_like(q2_internal)
    q[0, :] = q1_internal[0] * q2_internal[0, :] - np.sum(q1_internal[1:] * q2_internal[1:, :], axis=0)
    cross_pdt_matrix = np.array([[0, -q1_internal[3], q1_internal[2]], [q1_internal[3], 0, -q1_internal[1]],
                                 [-q1_internal[2], q1_internal[1], 0]])
    q[1:, :] = q1_internal[0] * q2_internal[1:, :] + q2_internal[0, :] * q1_internal[1:] + np.dot(cross_pdt_matrix,
                                                                                                  q2_internal[1:, :])
    return np.transpose(q)


def quat_multiply_right(q1, q2):
    """
    Multiplies multiple quaternions with one.
    q1_internal - shape: (4,m) m-number of quaternions
    q2_internal - shape: (4,)
    return q - shape (4,m)
    """
    q1_internal = np.transpose(np.atleast_2d(q1.copy()))
    q2_internal = np.transpose(np.atleast_2d(q2.copy()))
    q = np.zeros_like(q1_internal)
    q[0, :] = q1_internal[0, :] * q2_internal[0] - np.sum(q1_internal[1:, :] * q2_internal[1:], axis=0)
    cross_pdt_matrix = np.array([[0, q2_internal[3], -q2_internal[2]], [-q2_internal[3], 0, q2_internal[1]],
                                 [q2_internal[2], -q2_internal[1], 0]])
    q[1:, :] = q1_internal[0, :] * q2_internal[1:] + q2_internal[0] * q1_internal[1:, :] + np.dot(cross_pdt_matrix,
                                                                                                  q1_internal[1:, :])
    return np.transpose(q)


def quat_multiply(q1, q2):
    """
    Multiplies multiple quaternions with multiple one to one correspondence.
    q1_internal - shape: (4,m) m-number of quaternions
    q2_internal - shape: (4,m) m-number of quaternions
    return q - shape (4,m)
    """
    q1_internal = np.transpose(np.atleast_2d(q1.copy()))
    q2_internal = np.transpose(np.atleast_2d(q2.copy()))
    m1, n1 = q1_internal.shape
    m2, n2 = q2_internal.shape
    q = np.zeros((max(m1, m2), max(n1, n2)))
    q[0, :] = q1_internal[0, :] * q2_internal[0, :] - np.sum(q1_internal[1:, :] * q2_internal[1:, :], axis=0)
    q[1:, :] = q1_internal[0, :] * q2_internal[1:, :] + q2_internal[0, :] * q1_internal[1:, :] + np.cross(
        q1_internal[1:, :], q2_internal[1:, :], axis=0)
    return np.transpose(q)


def quat_print(q, number=0):
    q_internal = np.transpose(np.atleast_2d(q.copy()))
    string = ""
    for i in range(q_internal.shape[1]):
        string += "Quaterion {}.{}: \t".format(number, i + 1)
        string += "{:0.2f} {:+0.2f}i {:+0.2f}j {:+0.2f}k\n".format(q_internal[0, i], q_internal[1, i], q_internal[2, i],
                                                                   q_internal[3, i])
    print(string)


def is_unit_quat(q):
    if np.isclose(np.linalg.norm(q), 1):
        return True
    return False


def quat2rotmat(q):
    w = q[0]
    x = q[1]
    y = q[2]
    z = q[3]
    R = np.array([[1 - 2 * y ** 2 - 2 * z ** 2, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
                  [2 * x * y + 2 * z * w, 1 - 2 * x ** 2 - 2 * z ** 2, 2 * y * z - 2 * x * w],
                  [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x ** 2 - 2 * y ** 2]])
    return R


def quat2rpy(q):
    q0 = q[0]
    q1 = q[1]
    q2 = q[2]
    q3 = q[3]
    r = np.arctan2(2 * q2 * q3 + 2 * q0 * q1, q3 ** 2 - q2 ** 2 - q1 ** 2 + q0 ** 2)
    p = -np.arcsin(2 * q1 * q3 - 2 * q0 * q2)
    y = np.arctan2(2 * q1 * q2 + 2 * q0 * q3, q1 ** 2 + q0 ** 2 - q3 ** 2 - q2 ** 2)
    return r, p, y


def vec2quat(w, del_t=1):
    w_internal = np.atleast_2d(w.copy())
    alpha = np.linalg.norm(w_internal, axis=0)
    e = np.divide(w_internal, alpha, out=np.zeros_like(w_internal), where=alpha != 0)
    alpha = alpha * del_t
    W = np.zeros((4, w_internal.shape[1]))
    W[0, :] = np.cos(alpha / 2)
    W[1:, :] = e * np.sin(alpha / 2)
    return W


def quat2vec(qs):
    q = quat_normalize(qs.T).T
    thetas = 2 * np.arccos(q[0, :])
    vectors = q[1:, :]
    unit_vectors = np.divide(vectors, np.linalg.norm(vectors, axis=0), out=np.zeros_like(vectors), where=vectors != 0)
    return thetas * unit_vectors


def quat_mean(qs):
    mean_q = qs[:, 0].copy()
    iterations = 0
    while 1:
        error_quats = quat_multiply(qs.T, quat_inverse(mean_q)).T
        error_vecs = quat2vec(error_quats)
        mean_error = np.sum(error_vecs, axis=1) / qs.shape[1]
        if np.linalg.norm(mean_error) < 0.001 or iterations > 1000:
            break
        error_quat = vec2quat(np.atleast_2d(mean_error).T).reshape(4, )
        mean_q = quat_multiply(error_quat, mean_q).reshape(4, )
        iterations += 1
    return mean_q, error_vecs


def quat_average(qs, qt):
    mean_q = qt
    iterations = 0
    while (1):
        error_quats = quat_multiply(qs.T, quat_inverse(mean_q)).T
        error_vecs = quat2vec(error_quats)
        mean_error = np.sum(error_vecs, axis=1) / qs.shape[1]
        if np.linalg.norm(mean_error) < 0.01 or iterations > 100:
            break
        error_quat = vec2quat(np.atleast_2d(mean_error).T).reshape(4, )
        mean_q = quat_multiply(error_quat, mean_q).reshape(4, )
        iterations += 1
    return mean_q, error_vecs


def quat2rotmat(q):
    q = quat_normalize(q)
    qhat = np.zeros([3, 3])
    qhat[0, 1] = -q[:, 3]
    qhat[0, 2] = q[:, 2]
    qhat[1, 2] = -q[:, 1]
    qhat[1, 0] = q[:, 3]
    qhat[2, 0] = -q[:, 2]
    qhat[2, 1] = q[:, 1]

    R = np.eye(3) + 2 * np.dot(qhat, qhat) + 2 * np.array(q[:, 0]) * qhat
    return R


def rotmat2rpy(r):
    norm = np.sqrt(r[2, 1] ** 2 + r[2, 2] ** 2)
    roll_x = np.arctan(r[2, 1] / r[2, 2])
    pitch_y = np.arctan(-r[2, 0] / norm)
    yaw_z = np.arctan(r[1, 0] / r[0, 0])
    return roll_x, pitch_y, yaw_z
