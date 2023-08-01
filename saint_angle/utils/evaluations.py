import tensorflow as tf
from keras import backend
from keras.layers import Lambda
import numpy as np

# Performance Evaluation Metrics Description:-
#     mse -> Mean Squared Error (MSE)
#     mae -> Mean Absolute Error (MAE)
#     sae -> Sum of Absolute Error (SAE) = MAE * total_residue_count

def mean_mse(y_true, y_predicted):
    mask = 1 - backend.cast(backend.equal(y_true[:, :, 0], -500), dtype="float32")
    count = backend.sum(mask)

    y_true_phi_sine, y_true_phi_cosine = y_true[:, :, 0] * mask, y_true[:, :, 1] * mask
    y_true_psi_sine, y_true_psi_cosine = y_true[:, :, 2] * mask, y_true[:, :, 3] * mask
    y_pred_phi_sine, y_pred_phi_cosine = y_predicted[:, :, 0] * mask, y_predicted[:, :, 1] * mask
    y_pred_psi_sine, y_pred_psi_cosine = y_predicted[:, :, 2] * mask, y_predicted[:, :, 3] * mask

    phi_diff_sine, phi_diff_cosine = backend.abs(y_true_phi_sine - y_pred_phi_sine), backend.abs(y_true_phi_cosine - y_pred_phi_cosine)
    psi_diff_sine, psi_diff_cosine = backend.abs(y_true_psi_sine - y_pred_psi_sine), backend.abs(y_true_psi_cosine - y_pred_psi_cosine)
    phi_mse_sine, phi_mse_cosine = backend.sum(backend.square(phi_diff_sine)) / count, backend.sum(backend.square(phi_diff_cosine)) / count
    psi_mse_sine, psi_mse_cosine = backend.sum(backend.square(psi_diff_sine)) / count, backend.sum(backend.square(psi_diff_cosine)) / count

    mean_mse = 0.25 * (phi_mse_sine + phi_mse_cosine + psi_mse_sine + psi_mse_cosine)
    return mean_mse

def mean_mae(y_true, y_predicted):
    y_true_phi_angle = tf.atan2(y_true[:, :, 0], y_true[:, :, 1]) * 180 / np.pi
    y_pred_phi_angle = tf.atan2(y_predicted[:, :, 0], y_predicted[:, :, 1]) * 180 / np.pi
    y_true_psi_angle = tf.atan2(y_true[:, :, 2], y_true[:, :, 3]) * 180 / np.pi
    y_pred_psi_angle = tf.atan2(y_predicted[:, :, 2], y_predicted[:, :, 3]) * 180 / np.pi

    mask = 1 - backend.cast(backend.equal(y_true[:, :, 0], -500), dtype="float32")
    count = backend.sum(mask)

    phi_diff, psi_diff = backend.abs(y_true_phi_angle - y_pred_phi_angle), backend.abs(y_true_psi_angle - y_pred_psi_angle)
    phi_diff_rev, psi_diff_rev = Lambda(lambda x: 360 - x)(phi_diff), Lambda(lambda x: 360 - x)(psi_diff)

    phi_mask = backend.cast(backend.greater(phi_diff[:, :], 180), dtype="float32")
    phi_mask_rev = 1 - phi_mask
    psi_mask = backend.cast(backend.greater(psi_diff[:, :], 180), dtype="float32")
    psi_mask_rev = 1 - psi_mask

    phi_error, psi_error = phi_diff * phi_mask_rev + phi_diff_rev * phi_mask, psi_diff * psi_mask_rev + psi_diff_rev * psi_mask
    phi_mae, psi_mae = backend.sum(phi_error * mask) / count, backend.sum(psi_error * mask) / count

    mean_mae = 0.5 * (phi_mae + psi_mae)
    return mean_mae

def phi_mae(y_true, y_predicted):
    y_true_phi_angle = tf.atan2(y_true[:, :, 0], y_true[:, :, 1]) * 180 / np.pi
    y_pred_phi_angle = tf.atan2(y_predicted[:, :, 0], y_predicted[:, :, 1]) * 180 / np.pi

    mask = 1 - backend.cast(backend.equal(y_true[:, :, 0], -500), dtype="float32")
    count = backend.sum(mask)

    phi_diff = backend.abs(y_true_phi_angle - y_pred_phi_angle)
    phi_diff_rev = Lambda(lambda x: 360 - x)(phi_diff)

    phi_mask = backend.cast(backend.greater(phi_diff[:, :], 180), dtype="float32")
    phi_mask_rev = 1 - phi_mask

    phi_error = phi_diff * phi_mask_rev + phi_diff_rev * phi_mask
    phi_mae = backend.sum(phi_error * mask) / count
    return phi_mae

def psi_mae(y_true, y_predicted):
    y_true_psi_angle = tf.atan2(y_true[:, :, 2], y_true[:, :, 3]) * 180 / np.pi
    y_pred_psi_angle = tf.atan2(y_predicted[:, :, 2], y_predicted[:, :, 3]) * 180 / np.pi
    
    mask = 1 - backend.cast(backend.equal(y_true[:, :, 0], -500), dtype="float32")
    count = backend.sum(mask)
    
    psi_diff = backend.abs(y_true_psi_angle - y_pred_psi_angle)
    psi_diff_rev = Lambda(lambda x: 360 - x)(psi_diff)
    
    psi_mask = backend.cast(backend.greater(psi_diff[:, :], 180), dtype="float32")
    psi_mask_rev = 1 - psi_mask

    psi_error = psi_diff * psi_mask_rev + psi_diff_rev * psi_mask
    psi_mae = backend.sum(psi_error * mask) / count
    return psi_mae

def mean_sae(y_true, y_predicted):
    y_true_phi_angle = tf.atan2(y_true[:, :, 0], y_true[:, :, 1]) * 180 / np.pi
    y_pred_phi_angle = tf.atan2(y_predicted[:, :, 0], y_predicted[:, :, 1]) * 180 / np.pi
    y_true_psi_angle = tf.atan2(y_true[:, :, 2], y_true[:, :, 3]) * 180 / np.pi
    y_pred_psi_angle = tf.atan2(y_predicted[:, :, 2], y_predicted[:, :, 3]) * 180 / np.pi

    mask = 1 - backend.cast(backend.equal(y_true[:, :, 0], -500), dtype="float32")

    phi_diff, psi_diff = backend.abs(y_true_phi_angle - y_pred_phi_angle), backend.abs(y_true_psi_angle - y_pred_psi_angle)
    phi_diff_rev, psi_diff_rev = Lambda(lambda x: 360 - x)(phi_diff), Lambda(lambda x: 360 - x)(psi_diff)

    phi_mask = backend.cast(backend.greater(phi_diff[:, :], 180), dtype="float32")
    phi_mask_rev = 1 - phi_mask
    psi_mask = backend.cast(backend.greater(psi_diff[:, :], 180), dtype="float32")
    psi_mask_rev = 1 - psi_mask

    phi_error, psi_error = phi_diff * phi_mask_rev + phi_diff_rev * phi_mask, psi_diff * psi_mask_rev + psi_diff_rev * psi_mask
    phi_sae, psi_sae = backend.sum(phi_error * mask), backend.sum(psi_error * mask)
    
    mean_sae = 0.5 * (phi_sae + psi_sae)
    return mean_sae

def phi_sae(y_true, y_predicted):
    y_true_phi_angle = tf.atan2(y_true[:, :, 0], y_true[:, :, 1]) * 180 / np.pi
    y_pred_phi_angle = tf.atan2(y_predicted[:, :, 0], y_predicted[:, :, 1]) * 180 / np.pi
    
    mask = 1 - backend.cast(backend.equal(y_true[:, :, 0], -500), dtype="float32")
    
    phi_diff = backend.abs(y_true_phi_angle - y_pred_phi_angle)
    phi_diff_rev = Lambda(lambda x: 360 - x)(phi_diff)

    phi_mask = backend.cast(backend.greater(phi_diff[:, :], 180), dtype="float32")
    phi_mask_rev = 1 - phi_mask

    phi_error = phi_diff * phi_mask_rev + phi_diff_rev * phi_mask
    phi_sae = backend.sum(phi_error * mask)
    return phi_sae

def psi_sae(y_true, y_predicted):
    y_true_psi_angle = tf.atan2(y_true[:, :, 2], y_true[:, :, 3]) * 180 / np.pi
    y_pred_psi_angle = tf.atan2(y_predicted[:, :, 2], y_predicted[:, :, 3]) * 180 / np.pi

    mask = 1 - backend.cast(backend.equal(y_true[:, :, 0], -500), dtype="float32")

    psi_diff = backend.abs(y_true_psi_angle - y_pred_psi_angle)
    psi_diff_rev = Lambda(lambda x: 360 - x)(psi_diff)

    psi_mask = backend.cast(backend.greater(psi_diff[:, :], 180), dtype="float32")
    psi_mask_rev = 1 - psi_mask

    psi_error = psi_diff * psi_mask_rev + psi_diff_rev * psi_mask
    psi_sae = backend.sum(psi_error * mask)
    return psi_sae