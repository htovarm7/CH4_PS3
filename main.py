## BIOL2430 - Modeling Motor Control
# Coding Assignment #2

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, freqz
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import pdist

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def derivative(x, fs):
    """Numerical derivative using central differences.
    Central differences give O(dt^2) accuracy vs O(dt) for forward/backward."""
    dt = 1.0 / fs
    dx = np.zeros_like(x)
    dx[1:-1] = (x[2:] - x[:-2]) / (2 * dt)
    dx[0] = (x[1] - x[0]) / dt
    dx[-1] = (x[-1] - x[-2]) / dt
    return dx


def inverse_kinematics(x, y, l1, l2):
    """2-link planar arm inverse kinematics (elbow-up solution).
    x, y: end-effector position relative to shoulder.
    Returns theta1 (shoulder) and theta2 (elbow) in radians."""
    r2 = x**2 + y**2
    cos_theta2 = (r2 - l1**2 - l2**2) / (2 * l1 * l2)
    cos_theta2 = np.clip(cos_theta2, -1, 1)
    theta2 = np.arccos(cos_theta2)  # elbow-up
    k1 = l1 + l2 * np.cos(theta2)
    k2 = l2 * np.sin(theta2)
    theta1 = np.arctan2(y, x) - np.arctan2(k2, k1)
    return theta1, theta2


# =============================================================================
# SECTION 1: DATA PARSING AND CONDITIONING
# =============================================================================

Fs = 120        # Sampling frequency (Hz)
CutOff = 5      # Low-pass cutoff frequency (Hz)
trial_range = range(26, 61)

# --- 1a. Load all trials ---
TrajData = {}

for TNum in trial_range:
    filename = f"data/Traj-{TNum}.txt"
    rows = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            line = line.replace(',', ' ')
            vals = line.split()
            rows.append([float(v) for v in vals])
    data = np.array(rows)

    time   = data[:, 0].astype(int)
    s2x    = data[:, 4]
    s2y    = data[:, 5]
    s2z    = data[:, 6]
    tx     = data[:, 7]
    ty     = data[:, 8]
    Goflag = data[:, 16].astype(int)

    # Find the index where the target disappears (ty goes to ~0)
    indexY = np.where(np.abs(ty) < 0.001)[0]
    if len(indexY) > 0 and indexY[0] > 0 and ty[indexY[0]] != ty[indexY[0] - 1]:
        EndIndex = indexY[0] - 1
    else:
        EndIndex = indexY[0] if len(indexY) > 0 else len(ty) - 1

    TrajData[TNum] = {
        'TargetDisappearIdx': EndIndex,
        'Time': time,
        'Sen2X_raw': s2x.copy(),
        'Sen2Y_raw': s2y.copy(),
        'Sen2X': s2x.copy(),
        'Sen2Y': s2y.copy(),
        'Sen2Z': s2z,
        'Tx': tx,
        'Ty': ty,
        'GoFlag': Goflag,
    }


# --- 1b. Filter design and justification ---
# JUSTIFICATION:
#   - A 2nd-order Butterworth low-pass filter at 5 Hz is used.
#   - Butterworth is chosen for its maximally-flat magnitude response in the
#     passband, meaning it does not distort the amplitude of low-frequency
#     movement signals.
#   - 5 Hz cutoff: voluntary arm reaching movements have dominant frequency
#     content below 2-3 Hz (typical upper limb bandwidth). A 5 Hz cutoff
#     preserves all movement-related content while rejecting sensor noise,
#     tremor (8-12 Hz), and electromagnetic interference.
#   - 2nd order provides -40 dB/decade rolloff — sufficient attenuation
#     without excessive phase distortion or ringing.
#   - filtfilt (zero-phase) is used to apply the filter forward and backward,
#     eliminating phase lag. This doubles the effective order to 4th.
#   - The Nyquist frequency is 60 Hz (Fs/2), so the normalized cutoff is
#     5/60 ≈ 0.083, well within the stable filter design range.

b, a = butter(2, CutOff / (Fs / 2), btype='low')

# --- Filter frequency response plot ---
w, h = freqz(b, a, worN=2048, fs=Fs)

fig_filt, ax_filt = plt.subplots(2, 1, figsize=(8, 5))
fig_filt.suptitle('Butterworth Filter Characteristics (2nd order, 5 Hz cutoff)')

ax_filt[0].plot(w, 20 * np.log10(np.abs(h)), 'b')
ax_filt[0].axvline(CutOff, color='r', linestyle='--', label=f'Cutoff = {CutOff} Hz')
ax_filt[0].set_ylabel('Magnitude (dB)')
ax_filt[0].set_xlim([0, 30])
ax_filt[0].set_ylim([-60, 5])
ax_filt[0].legend()
ax_filt[0].grid(True, alpha=0.3)

angles = np.unwrap(np.angle(h))
ax_filt[1].plot(w, np.degrees(angles), 'b')
ax_filt[1].axvline(CutOff, color='r', linestyle='--')
ax_filt[1].set_ylabel('Phase (degrees)')
ax_filt[1].set_xlabel('Frequency (Hz)')
ax_filt[1].set_xlim([0, 30])
ax_filt[1].grid(True, alpha=0.3)
ax_filt[1].set_title('Note: filtfilt produces zero net phase shift')

fig_filt.tight_layout()

# --- Apply filter ---
for TNum in trial_range:
    TrajData[TNum]['Sen2X'] = filtfilt(b, a, TrajData[TNum]['Sen2X'])
    TrajData[TNum]['Sen2Y'] = filtfilt(b, a, TrajData[TNum]['Sen2Y'])

# --- Show raw vs filtered for one representative trial ---
fig_rf, axes_rf = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
fig_rf.suptitle('Raw vs Filtered Signal — Trial 27')
TNum_demo = 27
t_demo = TrajData[TNum_demo]['Time']
axes_rf[0].plot(t_demo, TrajData[TNum_demo]['Sen2X_raw'], 'r', alpha=0.5, label='Raw')
axes_rf[0].plot(t_demo, TrajData[TNum_demo]['Sen2X'], 'b', label='Filtered')
axes_rf[0].set_ylabel('X position (cm)')
axes_rf[0].legend()
axes_rf[0].grid(True, alpha=0.3)
axes_rf[1].plot(t_demo, TrajData[TNum_demo]['Sen2Y_raw'], 'r', alpha=0.5, label='Raw')
axes_rf[1].plot(t_demo, TrajData[TNum_demo]['Sen2Y'], 'b', label='Filtered')
axes_rf[1].set_ylabel('Y position (cm)')
axes_rf[1].set_xlabel('Time (ms)')
axes_rf[1].legend()
axes_rf[1].grid(True, alpha=0.3)
fig_rf.tight_layout()


# --- 1c. Statistical assessment of filtering error ---
# Compute the residual (raw - filtered) as a proxy for removed noise.
# If the filter is well-chosen, this residual should be zero-mean noise.

residuals_x = []
residuals_y = []
rmse_x_list = []
rmse_y_list = []
snr_x_list = []
snr_y_list = []

for TNum in trial_range:
    res_x = TrajData[TNum]['Sen2X_raw'] - TrajData[TNum]['Sen2X']
    res_y = TrajData[TNum]['Sen2Y_raw'] - TrajData[TNum]['Sen2Y']
    residuals_x.append(res_x)
    residuals_y.append(res_y)

    rmse_x = np.sqrt(np.mean(res_x**2))
    rmse_y = np.sqrt(np.mean(res_y**2))
    rmse_x_list.append(rmse_x)
    rmse_y_list.append(rmse_y)

    # SNR: signal power / noise power
    sig_power_x = np.var(TrajData[TNum]['Sen2X'])
    sig_power_y = np.var(TrajData[TNum]['Sen2Y'])
    snr_x = 10 * np.log10(sig_power_x / np.var(res_x)) if np.var(res_x) > 0 else np.inf
    snr_y = 10 * np.log10(sig_power_y / np.var(res_y)) if np.var(res_y) > 0 else np.inf
    snr_x_list.append(snr_x)
    snr_y_list.append(snr_y)

print("=" * 65)
print("FILTERING ERROR STATISTICS")
print("=" * 65)
print(f"{'Metric':<30} {'X':>15} {'Y':>15}")
print("-" * 65)
print(f"{'Mean RMSE (cm)':<30} {np.mean(rmse_x_list):>15.4f} {np.mean(rmse_y_list):>15.4f}")
print(f"{'Std RMSE (cm)':<30} {np.std(rmse_x_list):>15.4f} {np.std(rmse_y_list):>15.4f}")
print(f"{'Mean SNR (dB)':<30} {np.mean(snr_x_list):>15.1f} {np.mean(snr_y_list):>15.1f}")
print(f"{'Min SNR (dB)':<30} {np.min(snr_x_list):>15.1f} {np.min(snr_y_list):>15.1f}")
print("=" * 65)

fig_err, axes_err = plt.subplots(2, 2, figsize=(12, 7))
fig_err.suptitle('Filtering Error Assessment')

# RMSE across trials
trial_nums = list(trial_range)
axes_err[0, 0].bar(trial_nums, rmse_x_list, color='steelblue', alpha=0.7)
axes_err[0, 0].set_ylabel('RMSE X (cm)')
axes_err[0, 0].set_title('RMSE of removed noise — X')
axes_err[0, 0].axhline(np.mean(rmse_x_list), color='r', linestyle='--', label=f'Mean={np.mean(rmse_x_list):.3f}')
axes_err[0, 0].legend(fontsize=8)

axes_err[0, 1].bar(trial_nums, rmse_y_list, color='coral', alpha=0.7)
axes_err[0, 1].set_ylabel('RMSE Y (cm)')
axes_err[0, 1].set_title('RMSE of removed noise — Y')
axes_err[0, 1].axhline(np.mean(rmse_y_list), color='r', linestyle='--', label=f'Mean={np.mean(rmse_y_list):.3f}')
axes_err[0, 1].legend(fontsize=8)

# Histogram of residuals (pooled)
all_res_x = np.concatenate(residuals_x)
all_res_y = np.concatenate(residuals_y)
axes_err[1, 0].hist(all_res_x, bins=60, color='steelblue', alpha=0.7, density=True)
axes_err[1, 0].set_xlabel('Residual X (cm)')
axes_err[1, 0].set_ylabel('Density')
axes_err[1, 0].set_title(f'Residual distribution (mean={np.mean(all_res_x):.4f}, std={np.std(all_res_x):.4f})')

axes_err[1, 1].hist(all_res_y, bins=60, color='coral', alpha=0.7, density=True)
axes_err[1, 1].set_xlabel('Residual Y (cm)')
axes_err[1, 1].set_ylabel('Density')
axes_err[1, 1].set_title(f'Residual distribution (mean={np.mean(all_res_y):.4f}, std={np.std(all_res_y):.4f})')

fig_err.tight_layout()

Fs = 120
deltaT = 1 / Fs
g = 9.81  # gravity (m/s^2)

H = 1.76   # Average male height (m)
M = 80.7   # Average weight (kg)

# Anthropometrics (2-link arm)
l1 = 0.32                  # Upper arm length (m)
l2 = 0.29 + 0.19           # Forearm + hand length (m)
m1 = 2.23                  # Upper arm mass (kg)
m2 = 1.90                  # Forearm + hand mass (kg)
pCOM1 = 0.564 * l1         # Upper arm COM from shoulder
pCOM2 = 0.682 * l2         # Forearm+hand COM from elbow
rg1 = 0.322 * l1           # Radius of gyration — upper arm
rg2 = 0.468 * l2           # Radius of gyration — forearm+hand

# Moments of inertia about COM
I1 = 1/12 * m1 * (3 * rg1**2 + l1**2) + m1 * pCOM1**2
I2 = 1/12 * m2 * (3 * rg2**2 + l2**2) + m2 * pCOM2**2

# Coordinate shifts (shoulder as origin)
shift_x = -0.21
shift_y = 0.10

for TNum in trial_range:
    # Convert to meters and shift to shoulder origin
    ex = TrajData[TNum]['Sen2X'] / 100.0 + shift_x  # end-effector x (m)
    ey = TrajData[TNum]['Sen2Y'] / 100.0 + shift_y  # end-effector y (m)

    # Inverse kinematics
    theta1, theta2 = inverse_kinematics(ex, ey, l1, l2)

    # Angular velocities and accelerations
    dtheta1 = derivative(theta1, Fs)
    dtheta2 = derivative(theta2, Fs)
    ddtheta1 = derivative(dtheta1, Fs)
    ddtheta2 = derivative(dtheta2, Fs)

    # --- Torque computation (Lagrangian dynamics) ---
    c2 = np.cos(theta2)
    s2 = np.sin(theta2)
    c1 = np.cos(theta1)
    s12 = np.sin(theta1 + theta2)

    # Inertia matrix elements
    M11 = I1 + I2 + m1 * pCOM1**2 + m2 * (l1**2 + pCOM2**2 + 2 * l1 * pCOM2 * c2)
    M12 = I2 + m2 * (pCOM2**2 + l1 * pCOM2 * c2)
    M22 = I2 + m2 * pCOM2**2

    # Coriolis / centrifugal terms
    h = -m2 * l1 * pCOM2 * s2
    C1 = h * dtheta2 * (2 * dtheta1 + dtheta2)
    C2 = h * (-dtheta1**2)

    # Gravity terms
    G1 = (m1 * pCOM1 + m2 * l1) * g * c1 + m2 * pCOM2 * g * np.cos(theta1 + theta2)
    G2 = m2 * pCOM2 * g * np.cos(theta1 + theta2)

    # Inertial torque components (M * ddtheta)
    inertial_tau1 = M11 * ddtheta1 + M12 * ddtheta2
    inertial_tau2 = M12 * ddtheta1 + M22 * ddtheta2

    # Total torques
    tau1 = inertial_tau1 + C1 + G1
    tau2 = inertial_tau2 + C2 + G2

    # Cartesian velocity
    vx = derivative(TrajData[TNum]['Sen2X'], Fs)
    vy = derivative(TrajData[TNum]['Sen2Y'], Fs)
    speed = np.sqrt(vx**2 + vy**2)

    # Path length (cm)
    dx_path = np.diff(TrajData[TNum]['Sen2X'])
    dy_path = np.diff(TrajData[TNum]['Sen2Y'])
    path_length = np.sum(np.sqrt(dx_path**2 + dy_path**2))

    # Target distance (cm)
    x0, y0 = TrajData[TNum]['Sen2X'][0], TrajData[TNum]['Sen2Y'][0]
    tx_val, ty_val = TrajData[TNum]['Tx'][0], TrajData[TNum]['Ty'][0]
    target_dist = np.sqrt((tx_val - x0)**2 + (ty_val - y0)**2)

    # Store results
    TrajData[TNum]['theta1'] = theta1
    TrajData[TNum]['theta2'] = theta2
    TrajData[TNum]['dtheta1'] = dtheta1
    TrajData[TNum]['dtheta2'] = dtheta2
    TrajData[TNum]['ddtheta1'] = ddtheta1
    TrajData[TNum]['ddtheta2'] = ddtheta2
    TrajData[TNum]['tau1'] = tau1
    TrajData[TNum]['tau2'] = tau2
    TrajData[TNum]['inertial_tau1'] = inertial_tau1
    TrajData[TNum]['inertial_tau2'] = inertial_tau2
    TrajData[TNum]['coriolis_tau1'] = C1
    TrajData[TNum]['coriolis_tau2'] = C2
    TrajData[TNum]['gravity_tau1'] = G1
    TrajData[TNum]['gravity_tau2'] = G2
    TrajData[TNum]['speed'] = speed
    TrajData[TNum]['path_length'] = path_length
    TrajData[TNum]['target_dist'] = target_dist


# --- Plot: Targets and all trajectories ---
targetx = np.array([TrajData[TNum]['Tx'][0] for TNum in trial_range])
targety = np.array([TrajData[TNum]['Ty'][0] for TNum in trial_range])

plt.figure(figsize=(8, 8))
for TNum in trial_range:
    plt.plot(TrajData[TNum]['Sen2X'], TrajData[TNum]['Sen2Y'], 'r', alpha=0.4)
plt.plot(targetx, targety, 'ko', markersize=8, label='Targets')
plt.title('Targets and Filtered Trajectories')
plt.xlabel('X (cm)')
plt.ylabel('Y (cm)')
plt.axis('equal')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()


# --- Plot: Joint angles for all trials ---
fig3, axes3 = plt.subplots(2, 2, figsize=(12, 8))
fig3.suptitle('Joint Angles — All Trajectories')
for TNum in trial_range:
    t = TrajData[TNum]['Time']
    axes3[0, 0].plot(t, np.degrees(TrajData[TNum]['theta1']), alpha=0.4)
    axes3[0, 1].plot(t, np.degrees(TrajData[TNum]['theta2']), alpha=0.4)
    axes3[1, 0].plot(t, np.degrees(TrajData[TNum]['dtheta1']), alpha=0.4)
    axes3[1, 1].plot(t, np.degrees(TrajData[TNum]['dtheta2']), alpha=0.4)
axes3[0, 0].set_ylabel('Shoulder angle θ₁ (deg)')
axes3[0, 1].set_ylabel('Elbow angle θ₂ (deg)')
axes3[1, 0].set_ylabel('Shoulder vel dθ₁/dt (deg/s)')
axes3[1, 0].set_xlabel('Time (ms)')
axes3[1, 1].set_ylabel('Elbow vel dθ₂/dt (deg/s)')
axes3[1, 1].set_xlabel('Time (ms)')
for ax in axes3.flat:
    ax.grid(True, alpha=0.3)
fig3.tight_layout()


# --- Plot: Angular acceleration for all trials ---
fig_acc, axes_acc = plt.subplots(1, 2, figsize=(12, 4))
fig_acc.suptitle('Joint Angular Accelerations — All Trajectories')
for TNum in trial_range:
    t = TrajData[TNum]['Time']
    axes_acc[0].plot(t, np.degrees(TrajData[TNum]['ddtheta1']), alpha=0.4)
    axes_acc[1].plot(t, np.degrees(TrajData[TNum]['ddtheta2']), alpha=0.4)
axes_acc[0].set_ylabel('Shoulder accel (deg/s²)')
axes_acc[0].set_xlabel('Time (ms)')
axes_acc[1].set_ylabel('Elbow accel (deg/s²)')
axes_acc[1].set_xlabel('Time (ms)')
for ax in axes_acc:
    ax.grid(True, alpha=0.3)
fig_acc.tight_layout()


# --- Plot: Joint torques for all trials ---
fig4, axes4 = plt.subplots(2, 1, figsize=(10, 6))
fig4.suptitle('Joint Torques — All Trajectories')
for TNum in trial_range:
    t = TrajData[TNum]['Time']
    axes4[0].plot(t, TrajData[TNum]['tau1'], alpha=0.4)
    axes4[1].plot(t, TrajData[TNum]['tau2'], alpha=0.4)
axes4[0].set_ylabel('Shoulder torque τ₁ (N·m)')
axes4[1].set_ylabel('Elbow torque τ₂ (N·m)')
axes4[1].set_xlabel('Time (ms)')
for ax in axes4:
    ax.grid(True, alpha=0.3)
fig4.tight_layout()


# --- Plot: Single trial detail (#27) ---
TNum = 27
t = TrajData[TNum]['Time']

fig5, axes5 = plt.subplots(3, 2, figsize=(12, 10))
fig5.suptitle(f'Trajectory #{TNum} — Angles, Velocities & Torques')

axes5[0, 0].plot(t, np.degrees(TrajData[TNum]['theta1']))
axes5[0, 0].set_ylabel('Shoulder angle (deg)')
axes5[0, 1].plot(t, np.degrees(TrajData[TNum]['theta2']))
axes5[0, 1].set_ylabel('Elbow angle (deg)')

axes5[1, 0].plot(t, np.degrees(TrajData[TNum]['dtheta1']))
axes5[1, 0].set_ylabel('Shoulder vel (deg/s)')
axes5[1, 1].plot(t, np.degrees(TrajData[TNum]['dtheta2']))
axes5[1, 1].set_ylabel('Elbow vel (deg/s)')

axes5[2, 0].plot(t, TrajData[TNum]['tau1'])
axes5[2, 0].set_ylabel('Shoulder torque (N·m)')
axes5[2, 0].set_xlabel('Time (ms)')
axes5[2, 1].plot(t, TrajData[TNum]['tau2'])
axes5[2, 1].set_ylabel('Elbow torque (N·m)')
axes5[2, 1].set_xlabel('Time (ms)')

for ax in axes5.flat:
    ax.grid(True, alpha=0.3)
fig5.tight_layout()


# =============================================================================
# SECTION 3: TRIAL ANALYSIS — ERROR, NORMALIZATION, TORQUE COMPARISON
# =============================================================================

# --- 3a. Movement onset/offset detection and trial duration ---
# Use a velocity threshold to define movement onset and offset.
# This normalizes trials to the actual movement window.
VEL_THRESHOLD = 2.0  # cm/s — threshold for movement onset/offset

trial_stats = {}
for TNum in trial_range:
    speed = TrajData[TNum]['speed']
    t = TrajData[TNum]['Time']

    # Find first sample where speed exceeds threshold
    above = np.where(speed > VEL_THRESHOLD)[0]
    if len(above) > 0:
        onset = above[0]
        offset = above[-1]
    else:
        onset = 0
        offset = len(speed) - 1

    duration_ms = t[offset] - t[onset]
    peak_speed = np.max(speed)

    # End-point error: distance from final position to target
    x_end = TrajData[TNum]['Sen2X'][offset]
    y_end = TrajData[TNum]['Sen2Y'][offset]
    tx_val = TrajData[TNum]['Tx'][0]
    ty_val = TrajData[TNum]['Ty'][0]
    endpoint_error = np.sqrt((x_end - tx_val)**2 + (y_end - ty_val)**2)

    trial_stats[TNum] = {
        'onset': onset,
        'offset': offset,
        'duration_ms': duration_ms,
        'peak_speed': peak_speed,
        'endpoint_error': endpoint_error,
        'path_length': TrajData[TNum]['path_length'],
        'target_dist': TrajData[TNum]['target_dist'],
    }

# Print summary
print("\n" + "=" * 80)
print("TRIAL SUMMARY STATISTICS")
print("=" * 80)
durations = [trial_stats[t]['duration_ms'] for t in trial_range]
peak_speeds = [trial_stats[t]['peak_speed'] for t in trial_range]
errors = [trial_stats[t]['endpoint_error'] for t in trial_range]
path_lengths = [trial_stats[t]['path_length'] for t in trial_range]
target_dists = [trial_stats[t]['target_dist'] for t in trial_range]

print(f"{'Metric':<30} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
print("-" * 80)
print(f"{'Duration (ms)':<30} {np.mean(durations):>10.1f} {np.std(durations):>10.1f} "
      f"{np.min(durations):>10.1f} {np.max(durations):>10.1f}")
print(f"{'Peak speed (cm/s)':<30} {np.mean(peak_speeds):>10.1f} {np.std(peak_speeds):>10.1f} "
      f"{np.min(peak_speeds):>10.1f} {np.max(peak_speeds):>10.1f}")
print(f"{'Endpoint error (cm)':<30} {np.mean(errors):>10.2f} {np.std(errors):>10.2f} "
      f"{np.min(errors):>10.2f} {np.max(errors):>10.2f}")
print(f"{'Path length (cm)':<30} {np.mean(path_lengths):>10.2f} {np.std(path_lengths):>10.2f} "
      f"{np.min(path_lengths):>10.2f} {np.max(path_lengths):>10.2f}")
print(f"{'Target distance (cm)':<30} {np.mean(target_dists):>10.2f} {np.std(target_dists):>10.2f} "
      f"{np.min(target_dists):>10.2f} {np.max(target_dists):>10.2f}")
print("=" * 80)


# --- 3b. Data error across all trials ---
# Check for trends in endpoint error vs trial number, speed, duration
fig_trend, axes_trend = plt.subplots(2, 2, figsize=(12, 8))
fig_trend.suptitle('Error Trends Across Trials')

trial_nums = list(trial_range)

# Error vs trial number (learning/fatigue trend?)
axes_trend[0, 0].scatter(trial_nums, errors, c='steelblue', alpha=0.7)
z = np.polyfit(trial_nums, errors, 1)
p = np.poly1d(z)
axes_trend[0, 0].plot(trial_nums, p(trial_nums), 'r--',
                       label=f'slope={z[0]:.4f} cm/trial')
axes_trend[0, 0].set_xlabel('Trial number')
axes_trend[0, 0].set_ylabel('Endpoint error (cm)')
axes_trend[0, 0].set_title('Error vs Trial Order (learning/fatigue?)')
axes_trend[0, 0].legend(fontsize=8)
axes_trend[0, 0].grid(True, alpha=0.3)

# Error vs peak speed (speed-accuracy tradeoff?)
axes_trend[0, 1].scatter(peak_speeds, errors, c='coral', alpha=0.7)
z2 = np.polyfit(peak_speeds, errors, 1)
p2 = np.poly1d(z2)
xs = np.linspace(min(peak_speeds), max(peak_speeds), 50)
axes_trend[0, 1].plot(xs, p2(xs), 'r--', label=f'slope={z2[0]:.4f}')
axes_trend[0, 1].set_xlabel('Peak speed (cm/s)')
axes_trend[0, 1].set_ylabel('Endpoint error (cm)')
axes_trend[0, 1].set_title("Error vs Speed (Fitts' law?)")
axes_trend[0, 1].legend(fontsize=8)
axes_trend[0, 1].grid(True, alpha=0.3)

# Error vs duration
axes_trend[1, 0].scatter(durations, errors, c='green', alpha=0.7)
axes_trend[1, 0].set_xlabel('Movement duration (ms)')
axes_trend[1, 0].set_ylabel('Endpoint error (cm)')
axes_trend[1, 0].set_title('Error vs Duration')
axes_trend[1, 0].grid(True, alpha=0.3)

# Duration vs trial number (speed trend?)
axes_trend[1, 1].scatter(trial_nums, durations, c='purple', alpha=0.7)
z3 = np.polyfit(trial_nums, durations, 1)
p3 = np.poly1d(z3)
axes_trend[1, 1].plot(trial_nums, p3(trial_nums), 'r--',
                       label=f'slope={z3[0]:.1f} ms/trial')
axes_trend[1, 1].set_xlabel('Trial number')
axes_trend[1, 1].set_ylabel('Duration (ms)')
axes_trend[1, 1].set_title('Duration vs Trial Order')
axes_trend[1, 1].legend(fontsize=8)
axes_trend[1, 1].grid(True, alpha=0.3)

fig_trend.tight_layout()


# --- 3c. Normalized velocity vs normalized time ---
# We compare multiple normalization strategies and justify which is most meaningful.
#
# JUSTIFICATION OF NORMALIZATION APPROACHES:
#   1. No normalization: raw speed vs time — hard to compare across trials
#      with different durations.
#   2. Time normalization only (speed vs % movement time): aligns temporal
#      phases but velocity magnitudes differ due to distance/speed differences.
#   3. Speed / peak speed: normalizes shape to [0,1], good for comparing the
#      temporal profile (bell-shaped?) but loses magnitude information.
#   4. Speed / (path_length / duration): normalizes by mean speed — values
#      around 1.0 indicate average speed, peaks show how much faster than
#      mean. This preserves relative speed modulation.
#   5. Speed / target distance: normalizes by task demand (how far to reach).
#      More meaningful biomechanically — it relates kinematics to the spatial
#      goal. Faster reaches to closer targets and slower to far ones would
#      converge if movement scales with distance.
#
# The most meaningful normalization is speed/peak_speed vs normalized time,
# because it reveals the invariant velocity profile shape (bell-shaped for
# minimum-jerk, asymmetric for other control strategies) independent of
# movement amplitude and duration. This is the key prediction of motor
# control models (e.g., minimum-jerk, minimum-torque-change).

N_INTERP = 200  # number of interpolation points for time normalization

fig_norm, axes_norm = plt.subplots(2, 2, figsize=(14, 10))
fig_norm.suptitle('Normalized Velocity Profiles — All Trials Superimposed')

for TNum in trial_range:
    ons = trial_stats[TNum]['onset']
    off = trial_stats[TNum]['offset']
    if off <= ons:
        continue

    speed_seg = TrajData[TNum]['speed'][ons:off+1]
    t_seg = TrajData[TNum]['Time'][ons:off+1].astype(float)

    # Normalized time [0, 1]
    t_norm = (t_seg - t_seg[0]) / (t_seg[-1] - t_seg[0])
    t_interp = np.linspace(0, 1, N_INTERP)
    speed_interp = np.interp(t_interp, t_norm, speed_seg)

    pk = trial_stats[TNum]['peak_speed']
    pl = trial_stats[TNum]['path_length']
    dur_s = trial_stats[TNum]['duration_ms'] / 1000.0
    td = trial_stats[TNum]['target_dist']
    mean_speed = pl / dur_s if dur_s > 0 else 1.0

    # (a) Raw speed vs normalized time
    axes_norm[0, 0].plot(t_interp, speed_interp, alpha=0.4)

    # (b) Speed / peak_speed vs normalized time
    axes_norm[0, 1].plot(t_interp, speed_interp / pk if pk > 0 else speed_interp, alpha=0.4)

    # (c) Speed / mean_speed vs normalized time
    axes_norm[1, 0].plot(t_interp, speed_interp / mean_speed, alpha=0.4)

    # (d) Speed / target_distance vs normalized time
    axes_norm[1, 1].plot(t_interp, speed_interp / td if td > 0 else speed_interp, alpha=0.4)

axes_norm[0, 0].set_ylabel('Speed (cm/s)')
axes_norm[0, 0].set_title('(a) Raw speed vs norm. time')
axes_norm[0, 1].set_ylabel('Speed / Peak speed')
axes_norm[0, 1].set_title('(b) Speed / peak speed — profile shape (RECOMMENDED)')
axes_norm[1, 0].set_ylabel('Speed / Mean speed')
axes_norm[1, 0].set_title('(c) Speed / mean speed — relative modulation')
axes_norm[1, 1].set_ylabel('Speed / Target distance (1/s)')
axes_norm[1, 1].set_title('(d) Speed / target distance — task-normalized')

for ax in axes_norm.flat:
    ax.set_xlabel('Normalized time (0–1)')
    ax.grid(True, alpha=0.3)
fig_norm.tight_layout()


# --- 3d. Torque comparison across trials ---
# Decompose total torque into its components: inertial, Coriolis, gravity.
# This reveals what dominates the torque and how it varies across trials.

# Interpolate torque components to normalized time for comparison
torque_features = []  # for clustering

fig_tcomp, axes_tcomp = plt.subplots(3, 2, figsize=(14, 12))
fig_tcomp.suptitle('Torque Components — All Trials (Normalized Time)')

for TNum in trial_range:
    ons = trial_stats[TNum]['onset']
    off = trial_stats[TNum]['offset']
    if off <= ons:
        continue

    t_seg = TrajData[TNum]['Time'][ons:off+1].astype(float)
    t_norm = (t_seg - t_seg[0]) / (t_seg[-1] - t_seg[0])
    t_interp = np.linspace(0, 1, N_INTERP)

    # Interpolate all torque components
    tau1_i = np.interp(t_interp, t_norm, TrajData[TNum]['tau1'][ons:off+1])
    tau2_i = np.interp(t_interp, t_norm, TrajData[TNum]['tau2'][ons:off+1])
    iner1_i = np.interp(t_interp, t_norm, TrajData[TNum]['inertial_tau1'][ons:off+1])
    iner2_i = np.interp(t_interp, t_norm, TrajData[TNum]['inertial_tau2'][ons:off+1])
    cor1_i = np.interp(t_interp, t_norm, TrajData[TNum]['coriolis_tau1'][ons:off+1])
    cor2_i = np.interp(t_interp, t_norm, TrajData[TNum]['coriolis_tau2'][ons:off+1])
    grav1_i = np.interp(t_interp, t_norm, TrajData[TNum]['gravity_tau1'][ons:off+1])
    grav2_i = np.interp(t_interp, t_norm, TrajData[TNum]['gravity_tau2'][ons:off+1])

    axes_tcomp[0, 0].plot(t_interp, tau1_i, alpha=0.3)
    axes_tcomp[0, 1].plot(t_interp, tau2_i, alpha=0.3)
    axes_tcomp[1, 0].plot(t_interp, iner1_i, alpha=0.3)
    axes_tcomp[1, 1].plot(t_interp, iner2_i, alpha=0.3)
    axes_tcomp[2, 0].plot(t_interp, grav1_i, alpha=0.3, color='green')
    axes_tcomp[2, 1].plot(t_interp, grav2_i, alpha=0.3, color='green')

    # Build feature vector for clustering:
    # Use RMS of each torque component as summary statistics
    feature = [
        np.sqrt(np.mean(tau1_i**2)),
        np.sqrt(np.mean(tau2_i**2)),
        np.sqrt(np.mean(iner1_i**2)),
        np.sqrt(np.mean(iner2_i**2)),
        np.sqrt(np.mean(cor1_i**2)),
        np.sqrt(np.mean(cor2_i**2)),
        np.sqrt(np.mean(grav1_i**2)),
        np.sqrt(np.mean(grav2_i**2)),
        trial_stats[TNum]['peak_speed'],
        trial_stats[TNum]['target_dist'],
    ]
    torque_features.append(feature)

axes_tcomp[0, 0].set_ylabel('Total τ₁ (N·m)')
axes_tcomp[0, 0].set_title('Shoulder — Total Torque')
axes_tcomp[0, 1].set_ylabel('Total τ₂ (N·m)')
axes_tcomp[0, 1].set_title('Elbow — Total Torque')
axes_tcomp[1, 0].set_ylabel('Inertial τ₁ (N·m)')
axes_tcomp[1, 0].set_title('Shoulder — Inertial Component')
axes_tcomp[1, 1].set_ylabel('Inertial τ₂ (N·m)')
axes_tcomp[1, 1].set_title('Elbow — Inertial Component')
axes_tcomp[2, 0].set_ylabel('Gravity τ₁ (N·m)')
axes_tcomp[2, 0].set_title('Shoulder — Gravity Component')
axes_tcomp[2, 1].set_ylabel('Gravity τ₂ (N·m)')
axes_tcomp[2, 1].set_title('Elbow — Gravity Component')
for ax in axes_tcomp.flat:
    ax.set_xlabel('Normalized time')
    ax.grid(True, alpha=0.3)
fig_tcomp.tight_layout()


# Coriolis component (separate because typically much smaller)
fig_cor, axes_cor = plt.subplots(1, 2, figsize=(12, 4))
fig_cor.suptitle('Coriolis/Centrifugal Torque Components')
for TNum in trial_range:
    ons = trial_stats[TNum]['onset']
    off = trial_stats[TNum]['offset']
    if off <= ons:
        continue
    t_seg = TrajData[TNum]['Time'][ons:off+1].astype(float)
    t_norm = (t_seg - t_seg[0]) / (t_seg[-1] - t_seg[0])
    t_interp = np.linspace(0, 1, N_INTERP)
    cor1_i = np.interp(t_interp, t_norm, TrajData[TNum]['coriolis_tau1'][ons:off+1])
    cor2_i = np.interp(t_interp, t_norm, TrajData[TNum]['coriolis_tau2'][ons:off+1])
    axes_cor[0].plot(t_interp, cor1_i, alpha=0.3)
    axes_cor[1].plot(t_interp, cor2_i, alpha=0.3)
axes_cor[0].set_ylabel('Coriolis τ₁ (N·m)')
axes_cor[0].set_xlabel('Normalized time')
axes_cor[1].set_ylabel('Coriolis τ₂ (N·m)')
axes_cor[1].set_xlabel('Normalized time')
for ax in axes_cor:
    ax.grid(True, alpha=0.3)
fig_cor.tight_layout()


# --- 3e. Torque-based clustering of trials ---
# Can we group trials by their torque behavior?
# Using hierarchical clustering on the torque feature vectors.
#
# INTERPRETATION:
#   - If trials cluster into distinct groups, it suggests qualitatively
#     different movement strategies (e.g., fast/ballistic vs slow/guided).
#   - If they form a continuum, torque demands scale smoothly with
#     movement parameters (distance, direction, speed).

torque_features = np.array(torque_features)
# Normalize features to zero mean, unit variance
feat_mean = torque_features.mean(axis=0)
feat_std = torque_features.std(axis=0)
feat_std[feat_std == 0] = 1
feat_norm = (torque_features - feat_mean) / feat_std

# Hierarchical clustering
dist_matrix = pdist(feat_norm, metric='euclidean')
Z = linkage(dist_matrix, method='ward')

fig_dendro, ax_dendro = plt.subplots(1, 1, figsize=(14, 5))
dendrogram(Z, labels=[str(t) for t in trial_range], ax=ax_dendro,
           leaf_rotation=90, leaf_font_size=8)
ax_dendro.set_title('Hierarchical Clustering of Trials by Torque Features (Ward linkage)')
ax_dendro.set_xlabel('Trial number')
ax_dendro.set_ylabel('Distance')
fig_dendro.tight_layout()

# Cut into 3 clusters and color-code trajectories
n_clusters = 3
cluster_labels = fcluster(Z, n_clusters, criterion='maxclust')
colors = ['tab:blue', 'tab:orange', 'tab:green']
cluster_names = [f'Group {i+1}' for i in range(n_clusters)]

fig_clust, axes_clust = plt.subplots(1, 3, figsize=(18, 5))
fig_clust.suptitle(f'Trials Grouped by Torque Profile ({n_clusters} clusters)')

# Trajectory paths colored by cluster
for idx, TNum in enumerate(trial_range):
    c = colors[cluster_labels[idx] - 1]
    axes_clust[0].plot(TrajData[TNum]['Sen2X'], TrajData[TNum]['Sen2Y'],
                       color=c, alpha=0.5)
axes_clust[0].plot(targetx, targety, 'kx', markersize=8)
axes_clust[0].set_title('Trajectories colored by cluster')
axes_clust[0].set_xlabel('X (cm)')
axes_clust[0].set_ylabel('Y (cm)')
axes_clust[0].axis('equal')
axes_clust[0].grid(True, alpha=0.3)

# Box plot of peak speed by cluster
cluster_speeds = [[] for _ in range(n_clusters)]
cluster_dists = [[] for _ in range(n_clusters)]
for idx, TNum in enumerate(trial_range):
    ci = cluster_labels[idx] - 1
    cluster_speeds[ci].append(trial_stats[TNum]['peak_speed'])
    cluster_dists[ci].append(trial_stats[TNum]['target_dist'])

bp1 = axes_clust[1].boxplot(cluster_speeds, labels=cluster_names, patch_artist=True)
for patch, color in zip(bp1['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.5)
axes_clust[1].set_ylabel('Peak speed (cm/s)')
axes_clust[1].set_title('Peak speed by cluster')
axes_clust[1].grid(True, alpha=0.3)

bp2 = axes_clust[2].boxplot(cluster_dists, labels=cluster_names, patch_artist=True)
for patch, color in zip(bp2['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.5)
axes_clust[2].set_ylabel('Target distance (cm)')
axes_clust[2].set_title('Target distance by cluster')
axes_clust[2].grid(True, alpha=0.3)

fig_clust.tight_layout()

# --- Print cluster memberships ---
print("\n" + "=" * 65)
print(f"CLUSTER ASSIGNMENTS ({n_clusters} groups by torque profile)")
print("=" * 65)
for ci in range(n_clusters):
    members = [TNum for idx, TNum in enumerate(trial_range)
               if cluster_labels[idx] == ci + 1]
    print(f"\n{cluster_names[ci]} ({len(members)} trials): {members}")
    speeds = [trial_stats[t]['peak_speed'] for t in members]
    dists = [trial_stats[t]['target_dist'] for t in members]
    print(f"  Peak speed: {np.mean(speeds):.1f} ± {np.std(speeds):.1f} cm/s")
    print(f"  Target dist: {np.mean(dists):.1f} ± {np.std(dists):.1f} cm")
print("=" * 65)


# --- 3f. Torque component dominance analysis ---
# For each trial, show relative contribution of inertial, Coriolis, gravity
fig_dom, axes_dom = plt.subplots(1, 2, figsize=(14, 5))
fig_dom.suptitle('Relative Torque Component Contribution (RMS)')

inertial_frac_1 = []
coriolis_frac_1 = []
gravity_frac_1 = []
inertial_frac_2 = []
coriolis_frac_2 = []
gravity_frac_2 = []

for TNum in trial_range:
    ons = trial_stats[TNum]['onset']
    off = trial_stats[TNum]['offset']
    if off <= ons:
        inertial_frac_1.append(0); coriolis_frac_1.append(0); gravity_frac_1.append(0)
        inertial_frac_2.append(0); coriolis_frac_2.append(0); gravity_frac_2.append(0)
        continue

    seg = slice(ons, off+1)
    rms_i1 = np.sqrt(np.mean(TrajData[TNum]['inertial_tau1'][seg]**2))
    rms_c1 = np.sqrt(np.mean(TrajData[TNum]['coriolis_tau1'][seg]**2))
    rms_g1 = np.sqrt(np.mean(TrajData[TNum]['gravity_tau1'][seg]**2))
    total1 = rms_i1 + rms_c1 + rms_g1
    if total1 > 0:
        inertial_frac_1.append(rms_i1 / total1)
        coriolis_frac_1.append(rms_c1 / total1)
        gravity_frac_1.append(rms_g1 / total1)
    else:
        inertial_frac_1.append(0); coriolis_frac_1.append(0); gravity_frac_1.append(0)

    rms_i2 = np.sqrt(np.mean(TrajData[TNum]['inertial_tau2'][seg]**2))
    rms_c2 = np.sqrt(np.mean(TrajData[TNum]['coriolis_tau2'][seg]**2))
    rms_g2 = np.sqrt(np.mean(TrajData[TNum]['gravity_tau2'][seg]**2))
    total2 = rms_i2 + rms_c2 + rms_g2
    if total2 > 0:
        inertial_frac_2.append(rms_i2 / total2)
        coriolis_frac_2.append(rms_c2 / total2)
        gravity_frac_2.append(rms_g2 / total2)
    else:
        inertial_frac_2.append(0); coriolis_frac_2.append(0); gravity_frac_2.append(0)

x_pos = np.arange(len(trial_nums))
w = 0.8

axes_dom[0].bar(x_pos, gravity_frac_1, w, label='Gravity', color='green', alpha=0.7)
axes_dom[0].bar(x_pos, inertial_frac_1, w, bottom=gravity_frac_1,
                label='Inertial', color='steelblue', alpha=0.7)
axes_dom[0].bar(x_pos, coriolis_frac_1, w,
                bottom=np.array(gravity_frac_1) + np.array(inertial_frac_1),
                label='Coriolis', color='coral', alpha=0.7)
axes_dom[0].set_xticks(x_pos)
axes_dom[0].set_xticklabels(trial_nums, rotation=90, fontsize=7)
axes_dom[0].set_ylabel('Fraction of RMS torque')
axes_dom[0].set_title('Shoulder (τ₁) — Component Dominance')
axes_dom[0].legend(fontsize=8)

axes_dom[1].bar(x_pos, gravity_frac_2, w, label='Gravity', color='green', alpha=0.7)
axes_dom[1].bar(x_pos, inertial_frac_2, w, bottom=gravity_frac_2,
                label='Inertial', color='steelblue', alpha=0.7)
axes_dom[1].bar(x_pos, coriolis_frac_2, w,
                bottom=np.array(gravity_frac_2) + np.array(inertial_frac_2),
                label='Coriolis', color='coral', alpha=0.7)
axes_dom[1].set_xticks(x_pos)
axes_dom[1].set_xticklabels(trial_nums, rotation=90, fontsize=7)
axes_dom[1].set_ylabel('Fraction of RMS torque')
axes_dom[1].set_title('Elbow (τ₂) — Component Dominance')
axes_dom[1].legend(fontsize=8)

fig_dom.tight_layout()


plt.show()
