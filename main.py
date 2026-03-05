# Coding Assignment #2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, freqz
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import pdist

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
    theta2 = np.arccos(cos_theta2)  # elbow-up
    k1 = l1 + l2 * np.cos(theta2)
    k2 = l2 * np.sin(theta2)
    theta1 = np.arctan2(y, x) - np.arctan2(k2, k1)
    return theta1, theta2

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
    s2x    = data[:, 4] / 100.0  # cm → m
    s2y    = data[:, 5] / 100.0
    s2z    = data[:, 6] / 100.0
    tx     = data[:, 7] / 100.0
    ty     = data[:, 8] / 100.0
    Goflag = data[:, 16].astype(int)

    # Find the index where the target disappears (ty goes to ~0)
    indexY = np.where(np.abs(ty) < 0.00001)[0]
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



b, a = butter(2, CutOff / (Fs / 2), btype='low')

# --- Apply filter ---
for TNum in trial_range:
    TrajData[TNum]['Sen2X'] = filtfilt(b, a, TrajData[TNum]['Sen2X'])
    TrajData[TNum]['Sen2Y'] = filtfilt(b, a, TrajData[TNum]['Sen2Y'])

# --- Compute filtering error statistics ---
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
print(f"{'Mean RMSE (m)':<30} {np.mean(rmse_x_list):>15.6f} {np.mean(rmse_y_list):>15.6f}")
print(f"{'Std RMSE (m)':<30} {np.std(rmse_x_list):>15.6f} {np.std(rmse_y_list):>15.6f}")
print(f"{'Mean SNR (dB)':<30} {np.mean(snr_x_list):>15.1f} {np.mean(snr_y_list):>15.1f}")
print(f"{'Min SNR (dB)':<30} {np.min(snr_x_list):>15.1f} {np.min(snr_y_list):>15.1f}")
print("=" * 65)

fig1, ax1 = plt.subplots(3, 2, figsize=(13, 10))
fig1.suptitle('Figure 1: Data Conditioning — Filter Design & Error Assessment', fontsize=13)

# [0,0] Filter magnitude response
w, h = freqz(b, a, worN=2048, fs=Fs)
ax1[0, 0].plot(w, 20 * np.log10(np.abs(h)), 'b')
ax1[0, 0].axvline(CutOff, color='r', linestyle='--', label=f'Cutoff = {CutOff} Hz')
ax1[0, 0].set_ylabel('Magnitude (dB)')
ax1[0, 0].set_xlim([0, 30]); ax1[0, 0].set_ylim([-60, 5])
ax1[0, 0].legend(fontsize=8)
ax1[0, 0].set_title('Butterworth 2nd-order response')
ax1[0, 0].grid(True, alpha=0.3)

# [0,1] Filter phase response
angles = np.unwrap(np.angle(h))
ax1[0, 1].plot(w, np.degrees(angles), 'b')
ax1[0, 1].axvline(CutOff, color='r', linestyle='--')
ax1[0, 1].set_ylabel('Phase (degrees)')
ax1[0, 1].set_xlabel('Frequency (Hz)')
ax1[0, 1].set_xlim([0, 30])
ax1[0, 1].set_title('Phase (filtfilt → zero net shift)')
ax1[0, 1].grid(True, alpha=0.3)

# [1,0] Raw vs filtered X — Trial 27
TNum_demo = 27
t_demo = TrajData[TNum_demo]['Time']
ax1[1, 0].plot(t_demo, TrajData[TNum_demo]['Sen2X_raw'], 'r', alpha=0.5, label='Raw')
ax1[1, 0].plot(t_demo, TrajData[TNum_demo]['Sen2X'], 'b', label='Filtered')
ax1[1, 0].set_ylabel('X position (m)')
ax1[1, 0].set_title('Raw vs Filtered — Trial 27, X')
ax1[1, 0].legend(fontsize=8)
ax1[1, 0].grid(True, alpha=0.3)

# [1,1] Raw vs filtered Y — Trial 27
ax1[1, 1].plot(t_demo, TrajData[TNum_demo]['Sen2Y_raw'], 'r', alpha=0.5, label='Raw')
ax1[1, 1].plot(t_demo, TrajData[TNum_demo]['Sen2Y'], 'b', label='Filtered')
ax1[1, 1].set_ylabel('Y position (m)')
ax1[1, 1].set_xlabel('Time (ms)')
ax1[1, 1].set_title('Raw vs Filtered — Trial 27, Y')
ax1[1, 1].legend(fontsize=8)
ax1[1, 1].grid(True, alpha=0.3)

# [2,0] RMSE per trial (X and Y combined)
trial_nums = list(trial_range)
ax1[2, 0].bar(np.array(trial_nums) - 0.2, rmse_x_list, 0.4, color='steelblue', alpha=0.7, label='X')
ax1[2, 0].bar(np.array(trial_nums) + 0.2, rmse_y_list, 0.4, color='coral', alpha=0.7, label='Y')
ax1[2, 0].axhline(np.mean(rmse_x_list), color='steelblue', linestyle='--', alpha=0.5)
ax1[2, 0].axhline(np.mean(rmse_y_list), color='coral', linestyle='--', alpha=0.5)
ax1[2, 0].set_ylabel('RMSE (m)')
ax1[2, 0].set_xlabel('Trial number')
ax1[2, 0].set_title('RMSE of removed noise per trial')
ax1[2, 0].legend(fontsize=8)
ax1[2, 0].grid(True, alpha=0.3)

# [2,1] Residual histograms (pooled)
all_res_x = np.concatenate(residuals_x)
all_res_y = np.concatenate(residuals_y)
ax1[2, 1].hist(all_res_x, bins=60, color='steelblue', alpha=0.6, density=True, label=f'X (σ={np.std(all_res_x):.3f})')
ax1[2, 1].hist(all_res_y, bins=60, color='coral', alpha=0.6, density=True, label=f'Y (σ={np.std(all_res_y):.3f})')
ax1[2, 1].set_xlabel('Residual (m)')
ax1[2, 1].set_ylabel('Density')
ax1[2, 1].set_title('Pooled residual distributions (≈ Gaussian → noise)')
ax1[2, 1].legend(fontsize=8)
ax1[2, 1].grid(True, alpha=0.3)

fig1.tight_layout()

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
    ex = TrajData[TNum]['Sen2X'] + shift_x
    ey = TrajData[TNum]['Sen2Y'] + shift_y

    theta1, theta2 = inverse_kinematics(ex, ey, l1, l2)

    dtheta1 = derivative(theta1, Fs)
    dtheta2 = derivative(theta2, Fs)
    ddtheta1 = derivative(dtheta1, Fs)
    ddtheta2 = derivative(dtheta2, Fs)

    c2 = np.cos(theta2)
    s2 = np.sin(theta2)
    c1 = np.cos(theta1)

    M11 = I1 + I2 + m1 * pCOM1**2 + m2 * (l1**2 + pCOM2**2 + 2 * l1 * pCOM2 * c2)
    M12 = I2 + m2 * (pCOM2**2 + l1 * pCOM2 * c2)
    M22 = I2 + m2 * pCOM2**2

    h_val = -m2 * l1 * pCOM2 * s2
    C1 = h_val * dtheta2 * (2 * dtheta1 + dtheta2)
    C2 = h_val * (-dtheta1**2)

    G1 = (m1 * pCOM1 + m2 * l1) * g * c1 + m2 * pCOM2 * g * np.cos(theta1 + theta2)
    G2 = m2 * pCOM2 * g * np.cos(theta1 + theta2)

    inertial_tau1 = M11 * ddtheta1 + M12 * ddtheta2
    inertial_tau2 = M12 * ddtheta1 + M22 * ddtheta2
    tau1 = inertial_tau1 + C1 + G1
    tau2 = inertial_tau2 + C2 + G2

    vx = derivative(TrajData[TNum]['Sen2X'], Fs)
    vy = derivative(TrajData[TNum]['Sen2Y'], Fs)
    speed = np.sqrt(vx**2 + vy**2)

    dx_path = np.diff(TrajData[TNum]['Sen2X'])
    dy_path = np.diff(TrajData[TNum]['Sen2Y'])
    path_length = np.sum(np.sqrt(dx_path**2 + dy_path**2))

    x0, y0 = TrajData[TNum]['Sen2X'][0], TrajData[TNum]['Sen2Y'][0]
    tx_val, ty_val = TrajData[TNum]['Tx'][0], TrajData[TNum]['Ty'][0]
    target_dist = np.sqrt((tx_val - x0)**2 + (ty_val - y0)**2)

    TrajData[TNum].update({
        'theta1': theta1, 'theta2': theta2,
        'dtheta1': dtheta1, 'dtheta2': dtheta2,
        'ddtheta1': ddtheta1, 'ddtheta2': ddtheta2,
        'tau1': tau1, 'tau2': tau2,
        'inertial_tau1': inertial_tau1, 'inertial_tau2': inertial_tau2,
        'coriolis_tau1': C1, 'coriolis_tau2': C2,
        'gravity_tau1': G1, 'gravity_tau2': G2,
        'speed': speed, 'path_length': path_length, 'target_dist': target_dist,
    })

targetx = np.array([TrajData[TNum]['Tx'][0] for TNum in trial_range])
targety = np.array([TrajData[TNum]['Ty'][0] for TNum in trial_range])

fig2, ax2 = plt.subplots(4, 2, figsize=(14, 14))
fig2.suptitle('Figure 2: Kinematics — Trajectories, Joint Angles, Velocities & Accelerations', fontsize=13)

# [0,0] All trajectories + targets
for TNum in trial_range:
    ax2[0, 0].plot(TrajData[TNum]['Sen2X'], TrajData[TNum]['Sen2Y'], 'r', alpha=0.4)
ax2[0, 0].plot(targetx, targety, 'ko', markersize=6, label='Targets')
ax2[0, 0].set_xlabel('X (m)'); ax2[0, 0].set_ylabel('Y (m)')
ax2[0, 0].set_title('All filtered trajectories + targets')
ax2[0, 0].legend(fontsize=8)
ax2[0, 0].axis('equal')
ax2[0, 0].grid(True, alpha=0.3)

# [0,1] Single trial (#27) detail
TNum = 27
t27 = TrajData[TNum]['Time']
ax2[0, 1].plot(TrajData[TNum]['Sen2X'], TrajData[TNum]['Sen2Y'], 'b')
ax2[0, 1].plot(TrajData[TNum]['Tx'][0], TrajData[TNum]['Ty'][0], 'ro', markersize=8)
ax2[0, 1].set_xlabel('X (m)'); ax2[0, 1].set_ylabel('Y (m)')
ax2[0, 1].set_title('Trial #27 — path + target')
ax2[0, 1].axis('equal')
ax2[0, 1].grid(True, alpha=0.3)

# [1,0] θ₁ all trials    [1,1] θ₂ all trials
for TNum in trial_range:
    t = TrajData[TNum]['Time']
    ax2[1, 0].plot(t, np.degrees(TrajData[TNum]['theta1']), alpha=0.4)
    ax2[1, 1].plot(t, np.degrees(TrajData[TNum]['theta2']), alpha=0.4)
ax2[1, 0].set_ylabel('Shoulder θ₁ (deg)'); ax2[1, 0].set_title('Shoulder angle — all trials')
ax2[1, 1].set_ylabel('Elbow θ₂ (deg)'); ax2[1, 1].set_title('Elbow angle — all trials')

# [2,0] dθ₁/dt all trials    [2,1] dθ₂/dt all trials
for TNum in trial_range:
    t = TrajData[TNum]['Time']
    ax2[2, 0].plot(t, np.degrees(TrajData[TNum]['dtheta1']), alpha=0.4)
    ax2[2, 1].plot(t, np.degrees(TrajData[TNum]['dtheta2']), alpha=0.4)
ax2[2, 0].set_ylabel('dθ₁/dt (deg/s)'); ax2[2, 0].set_title('Shoulder angular velocity')
ax2[2, 1].set_ylabel('dθ₂/dt (deg/s)'); ax2[2, 1].set_title('Elbow angular velocity')

# [3,0] ddθ₁/dt² all trials    [3,1] ddθ₂/dt² all trials
for TNum in trial_range:
    t = TrajData[TNum]['Time']
    ax2[3, 0].plot(t, np.degrees(TrajData[TNum]['ddtheta1']), alpha=0.4)
    ax2[3, 1].plot(t, np.degrees(TrajData[TNum]['ddtheta2']), alpha=0.4)
ax2[3, 0].set_ylabel('ddθ₁/dt² (deg/s²)'); ax2[3, 0].set_title('Shoulder angular acceleration')
ax2[3, 0].set_xlabel('Time (ms)')
ax2[3, 1].set_ylabel('ddθ₂/dt² (deg/s²)'); ax2[3, 1].set_title('Elbow angular acceleration')
ax2[3, 1].set_xlabel('Time (ms)')

for a in ax2.flat:
    a.grid(True, alpha=0.3)
fig2.tight_layout()

VEL_THRESHOLD = 0.02  # m/s — threshold for movement onset/offset

trial_stats = {}
for TNum in trial_range:
    speed = TrajData[TNum]['speed']
    t = TrajData[TNum]['Time']

    above = np.where(speed > VEL_THRESHOLD)[0]
    if len(above) > 0:
        onset = above[0]
        offset = above[-1]
    else:
        onset = 0
        offset = len(speed) - 1

    duration_ms = t[offset] - t[onset]
    peak_speed = np.max(speed)

    x_end = TrajData[TNum]['Sen2X'][offset]
    y_end = TrajData[TNum]['Sen2Y'][offset]
    tx_val = TrajData[TNum]['Tx'][0]
    ty_val = TrajData[TNum]['Ty'][0]
    endpoint_error = np.sqrt((x_end - tx_val)**2 + (y_end - ty_val)**2)

    trial_stats[TNum] = {
        'onset': onset, 'offset': offset,
        'duration_ms': duration_ms, 'peak_speed': peak_speed,
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
print(f"{'Peak speed (m/s)':<30} {np.mean(peak_speeds):>10.3f} {np.std(peak_speeds):>10.3f} "
      f"{np.min(peak_speeds):>10.3f} {np.max(peak_speeds):>10.3f}")
print(f"{'Endpoint error (m)':<30} {np.mean(errors):>10.4f} {np.std(errors):>10.4f} "
      f"{np.min(errors):>10.4f} {np.max(errors):>10.4f}")
print(f"{'Path length (m)':<30} {np.mean(path_lengths):>10.4f} {np.std(path_lengths):>10.4f} "
      f"{np.min(path_lengths):>10.4f} {np.max(path_lengths):>10.4f}")
print(f"{'Target distance (m)':<30} {np.mean(target_dists):>10.4f} {np.std(target_dists):>10.4f} "
      f"{np.min(target_dists):>10.4f} {np.max(target_dists):>10.4f}")
print("=" * 80)


fig3, ax3 = plt.subplots(4, 2, figsize=(14, 16))
fig3.suptitle('Figure 3: Trial Analysis — Error Trends & Normalized Velocity Profiles', fontsize=13)

# --- Top half: Error trends (rows 0-1) ---

# [0,0] Error vs trial number
ax3[0, 0].scatter(trial_nums, errors, c='steelblue', alpha=0.7, s=30)
z = np.polyfit(trial_nums, errors, 1)
p = np.poly1d(z)
ax3[0, 0].plot(trial_nums, p(trial_nums), 'r--', label=f'slope={z[0]:.5f} m/trial')
ax3[0, 0].set_xlabel('Trial number'); ax3[0, 0].set_ylabel('Endpoint error (m)')
ax3[0, 0].set_title('Error vs Trial Order (learning/fatigue?)')
ax3[0, 0].legend(fontsize=8); ax3[0, 0].grid(True, alpha=0.3)

# [0,1] Error vs peak speed
ax3[0, 1].scatter(peak_speeds, errors, c='coral', alpha=0.7, s=30)
z2 = np.polyfit(peak_speeds, errors, 1)
p2 = np.poly1d(z2)
xs = np.linspace(min(peak_speeds), max(peak_speeds), 50)
ax3[0, 1].plot(xs, p2(xs), 'r--', label=f'slope={z2[0]:.4f}')
ax3[0, 1].set_xlabel('Peak speed (m/s)'); ax3[0, 1].set_ylabel('Endpoint error (m)')
ax3[0, 1].set_title("Error vs Speed (Fitts' law?)")
ax3[0, 1].legend(fontsize=8); ax3[0, 1].grid(True, alpha=0.3)

# [1,0] Error vs duration
ax3[1, 0].scatter(durations, errors, c='green', alpha=0.7, s=30)
ax3[1, 0].set_xlabel('Movement duration (ms)'); ax3[1, 0].set_ylabel('Endpoint error (m)')
ax3[1, 0].set_title('Error vs Duration'); ax3[1, 0].grid(True, alpha=0.3)

# [1,1] Duration vs trial number
ax3[1, 1].scatter(trial_nums, durations, c='purple', alpha=0.7, s=30)
z3 = np.polyfit(trial_nums, durations, 1)
p3 = np.poly1d(z3)
ax3[1, 1].plot(trial_nums, p3(trial_nums), 'r--', label=f'slope={z3[0]:.1f} ms/trial')
ax3[1, 1].set_xlabel('Trial number'); ax3[1, 1].set_ylabel('Duration (ms)')
ax3[1, 1].set_title('Duration vs Trial Order')
ax3[1, 1].legend(fontsize=8); ax3[1, 1].grid(True, alpha=0.3)

N_INTERP = 200

for TNum in trial_range:
    ons = trial_stats[TNum]['onset']
    off = trial_stats[TNum]['offset']
    if off <= ons:
        continue

    speed_seg = TrajData[TNum]['speed'][ons:off+1]
    t_seg = TrajData[TNum]['Time'][ons:off+1].astype(float)
    t_norm = (t_seg - t_seg[0]) / (t_seg[-1] - t_seg[0])
    t_interp = np.linspace(0, 1, N_INTERP)
    speed_interp = np.interp(t_interp, t_norm, speed_seg)

    pk = trial_stats[TNum]['peak_speed']
    pl = trial_stats[TNum]['path_length']
    dur_s = trial_stats[TNum]['duration_ms'] / 1000.0
    td = trial_stats[TNum]['target_dist']
    mean_speed = pl / dur_s if dur_s > 0 else 1.0

    ax3[2, 0].plot(t_interp, speed_interp, alpha=0.4)
    ax3[2, 1].plot(t_interp, speed_interp / pk if pk > 0 else speed_interp, alpha=0.4)
    ax3[3, 0].plot(t_interp, speed_interp / mean_speed, alpha=0.4)
    ax3[3, 1].plot(t_interp, speed_interp / td if td > 0 else speed_interp, alpha=0.4)

ax3[2, 0].set_ylabel('Speed (m/s)'); ax3[2, 0].set_title('(a) Raw speed vs norm. time')
ax3[2, 1].set_ylabel('Speed / Peak speed'); ax3[2, 1].set_title('(b) Speed / peak speed — shape (RECOMMENDED)')
ax3[3, 0].set_ylabel('Speed / Mean speed'); ax3[3, 0].set_title('(c) Speed / mean speed — relative modulation')
ax3[3, 0].set_xlabel('Normalized time (0–1)')
ax3[3, 1].set_ylabel('Speed / Target dist (1/s)'); ax3[3, 1].set_title('(d) Speed / target distance')
ax3[3, 1].set_xlabel('Normalized time (0–1)')

for a in ax3.flat:
    a.grid(True, alpha=0.3)
fig3.tight_layout()


torque_features = []

fig4, ax4 = plt.subplots(4, 2, figsize=(14, 14))
fig4.suptitle('Figure 4: Joint Torques — Total & Component Decomposition (Normalized Time)', fontsize=13)

for TNum in trial_range:
    ons = trial_stats[TNum]['onset']
    off = trial_stats[TNum]['offset']
    if off <= ons:
        continue

    t_seg = TrajData[TNum]['Time'][ons:off+1].astype(float)
    t_norm = (t_seg - t_seg[0]) / (t_seg[-1] - t_seg[0])
    t_interp = np.linspace(0, 1, N_INTERP)

    tau1_i = np.interp(t_interp, t_norm, TrajData[TNum]['tau1'][ons:off+1])
    tau2_i = np.interp(t_interp, t_norm, TrajData[TNum]['tau2'][ons:off+1])
    iner1_i = np.interp(t_interp, t_norm, TrajData[TNum]['inertial_tau1'][ons:off+1])
    iner2_i = np.interp(t_interp, t_norm, TrajData[TNum]['inertial_tau2'][ons:off+1])
    cor1_i = np.interp(t_interp, t_norm, TrajData[TNum]['coriolis_tau1'][ons:off+1])
    cor2_i = np.interp(t_interp, t_norm, TrajData[TNum]['coriolis_tau2'][ons:off+1])
    grav1_i = np.interp(t_interp, t_norm, TrajData[TNum]['gravity_tau1'][ons:off+1])
    grav2_i = np.interp(t_interp, t_norm, TrajData[TNum]['gravity_tau2'][ons:off+1])

    ax4[0, 0].plot(t_interp, tau1_i, alpha=0.3)
    ax4[0, 1].plot(t_interp, tau2_i, alpha=0.3)
    ax4[1, 0].plot(t_interp, iner1_i, alpha=0.3)
    ax4[1, 1].plot(t_interp, iner2_i, alpha=0.3)
    ax4[2, 0].plot(t_interp, grav1_i, alpha=0.3, color='green')
    ax4[2, 1].plot(t_interp, grav2_i, alpha=0.3, color='green')
    ax4[3, 0].plot(t_interp, cor1_i, alpha=0.3, color='coral')
    ax4[3, 1].plot(t_interp, cor2_i, alpha=0.3, color='coral')

    feature = [
        np.sqrt(np.mean(tau1_i**2)), np.sqrt(np.mean(tau2_i**2)),
        np.sqrt(np.mean(iner1_i**2)), np.sqrt(np.mean(iner2_i**2)),
        np.sqrt(np.mean(cor1_i**2)), np.sqrt(np.mean(cor2_i**2)),
        np.sqrt(np.mean(grav1_i**2)), np.sqrt(np.mean(grav2_i**2)),
        trial_stats[TNum]['peak_speed'], trial_stats[TNum]['target_dist'],
    ]
    torque_features.append(feature)

labels_left = ['Total τ₁ (N·m)', 'Inertial τ₁ (N·m)', 'Gravity τ₁ (N·m)', 'Coriolis τ₁ (N·m)']
labels_right = ['Total τ₂ (N·m)', 'Inertial τ₂ (N·m)', 'Gravity τ₂ (N·m)', 'Coriolis τ₂ (N·m)']
titles_left = ['Shoulder — Total Torque', 'Shoulder — Inertial', 'Shoulder — Gravity', 'Shoulder — Coriolis/Centrifugal']
titles_right = ['Elbow — Total Torque', 'Elbow — Inertial', 'Elbow — Gravity', 'Elbow — Coriolis/Centrifugal']

for i in range(4):
    ax4[i, 0].set_ylabel(labels_left[i]); ax4[i, 0].set_title(titles_left[i])
    ax4[i, 1].set_ylabel(labels_right[i]); ax4[i, 1].set_title(titles_right[i])
ax4[3, 0].set_xlabel('Normalized time')
ax4[3, 1].set_xlabel('Normalized time')
for a in ax4.flat:
    a.grid(True, alpha=0.3)
fig4.tight_layout()


torque_features = np.array(torque_features)
feat_mean = torque_features.mean(axis=0)
feat_std = torque_features.std(axis=0)
feat_std[feat_std == 0] = 1
feat_norm = (torque_features - feat_mean) / feat_std

dist_matrix = pdist(feat_norm, metric='euclidean')
Z = linkage(dist_matrix, method='ward')

n_clusters = 3
cluster_labels = fcluster(Z, n_clusters, criterion='maxclust')
colors = ['tab:blue', 'tab:orange', 'tab:green']
cluster_names = [f'Group {i+1}' for i in range(n_clusters)]

fig5 = plt.figure(figsize=(16, 14))
fig5.suptitle('Figure 5: Trial Clustering by Torque Profile & Component Dominance', fontsize=13)

# Use gridspec for flexible layout: dendrogram on top spanning full width
gs = fig5.add_gridspec(3, 3, hspace=0.35, wspace=0.3)

# [0, :] Dendrogram (full width)
ax_dendro = fig5.add_subplot(gs[0, :])
dendrogram(Z, labels=[str(t) for t in trial_range], ax=ax_dendro,
           leaf_rotation=90, leaf_font_size=8)
ax_dendro.set_title('Hierarchical Clustering of Trials by Torque Features (Ward linkage)')
ax_dendro.set_xlabel('Trial number'); ax_dendro.set_ylabel('Distance')

# [1, 0] Trajectories colored by cluster
ax_traj = fig5.add_subplot(gs[1, 0])
for idx, TNum in enumerate(trial_range):
    c = colors[cluster_labels[idx] - 1]
    ax_traj.plot(TrajData[TNum]['Sen2X'], TrajData[TNum]['Sen2Y'], color=c, alpha=0.5)
ax_traj.plot(targetx, targety, 'kx', markersize=8)
ax_traj.set_title('Trajectories by cluster')
ax_traj.set_xlabel('X (m)'); ax_traj.set_ylabel('Y (m)')
ax_traj.axis('equal'); ax_traj.grid(True, alpha=0.3)

# [1, 1] Peak speed by cluster
cluster_speeds = [[] for _ in range(n_clusters)]
cluster_dists = [[] for _ in range(n_clusters)]
for idx, TNum in enumerate(trial_range):
    ci = cluster_labels[idx] - 1
    cluster_speeds[ci].append(trial_stats[TNum]['peak_speed'])
    cluster_dists[ci].append(trial_stats[TNum]['target_dist'])

ax_spd = fig5.add_subplot(gs[1, 1])
bp1 = ax_spd.boxplot(cluster_speeds, labels=cluster_names, patch_artist=True)
for patch, color in zip(bp1['boxes'], colors):
    patch.set_facecolor(color); patch.set_alpha(0.5)
ax_spd.set_ylabel('Peak speed (m/s)'); ax_spd.set_title('Peak speed by cluster')
ax_spd.grid(True, alpha=0.3)

# [1, 2] Target distance by cluster
ax_dst = fig5.add_subplot(gs[1, 2])
bp2 = ax_dst.boxplot(cluster_dists, labels=cluster_names, patch_artist=True)
for patch, color in zip(bp2['boxes'], colors):
    patch.set_facecolor(color); patch.set_alpha(0.5)
ax_dst.set_ylabel('Target distance (m)'); ax_dst.set_title('Target distance by cluster')
ax_dst.grid(True, alpha=0.3)

# [2, 0:1] Shoulder torque dominance stacked bar
# [2, 1:2] Elbow torque dominance stacked bar

inertial_frac_1, coriolis_frac_1, gravity_frac_1 = [], [], []
inertial_frac_2, coriolis_frac_2, gravity_frac_2 = [], [], []

for TNum in trial_range:
    ons = trial_stats[TNum]['onset']
    off = trial_stats[TNum]['offset']
    if off <= ons:
        for lst in [inertial_frac_1, coriolis_frac_1, gravity_frac_1,
                    inertial_frac_2, coriolis_frac_2, gravity_frac_2]:
            lst.append(0)
        continue

    seg = slice(ons, off+1)
    for (data_key_i, data_key_c, data_key_g, i_lst, c_lst, g_lst) in [
        ('inertial_tau1', 'coriolis_tau1', 'gravity_tau1',
         inertial_frac_1, coriolis_frac_1, gravity_frac_1),
        ('inertial_tau2', 'coriolis_tau2', 'gravity_tau2',
         inertial_frac_2, coriolis_frac_2, gravity_frac_2),
    ]:
        rms_i = np.sqrt(np.mean(TrajData[TNum][data_key_i][seg]**2))
        rms_c = np.sqrt(np.mean(TrajData[TNum][data_key_c][seg]**2))
        rms_g = np.sqrt(np.mean(TrajData[TNum][data_key_g][seg]**2))
        total = rms_i + rms_c + rms_g
        if total > 0:
            i_lst.append(rms_i / total)
            c_lst.append(rms_c / total)
            g_lst.append(rms_g / total)
        else:
            i_lst.append(0); c_lst.append(0); g_lst.append(0)

x_pos = np.arange(len(trial_nums))
bar_w = 0.8

ax_dom1 = fig5.add_subplot(gs[2, 0:2])
ax_dom1.bar(x_pos, gravity_frac_1, bar_w, label='Gravity', color='green', alpha=0.7)
ax_dom1.bar(x_pos, inertial_frac_1, bar_w, bottom=gravity_frac_1,
            label='Inertial', color='steelblue', alpha=0.7)
ax_dom1.bar(x_pos, coriolis_frac_1, bar_w,
            bottom=np.array(gravity_frac_1) + np.array(inertial_frac_1),
            label='Coriolis', color='coral', alpha=0.7)
ax_dom1.set_xticks(x_pos); ax_dom1.set_xticklabels(trial_nums, rotation=90, fontsize=7)
ax_dom1.set_ylabel('Fraction of RMS torque')
ax_dom1.set_title('Shoulder (τ₁) — Component Dominance')
ax_dom1.legend(fontsize=8)

ax_dom2 = fig5.add_subplot(gs[2, 2])
ax_dom2.bar(x_pos, gravity_frac_2, bar_w, label='Gravity', color='green', alpha=0.7)
ax_dom2.bar(x_pos, inertial_frac_2, bar_w, bottom=gravity_frac_2,
            label='Inertial', color='steelblue', alpha=0.7)
ax_dom2.bar(x_pos, coriolis_frac_2, bar_w,
            bottom=np.array(gravity_frac_2) + np.array(inertial_frac_2),
            label='Coriolis', color='coral', alpha=0.7)
ax_dom2.set_xticks(x_pos); ax_dom2.set_xticklabels(trial_nums, rotation=90, fontsize=7)
ax_dom2.set_ylabel('Fraction of RMS torque')
ax_dom2.set_title('Elbow (τ₂) — Component Dominance')
ax_dom2.legend(fontsize=8)

fig5.tight_layout()

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
    print(f"  Peak speed: {np.mean(speeds):.3f} ± {np.std(speeds):.3f} m/s")
    print(f"  Target dist: {np.mean(dists):.4f} ± {np.std(dists):.4f} m")
print("=" * 65)


plt.show()
