import platform
print("Python " + platform.python_version())
import numpy as np
print("Numpy " + np.__version__)
import matplotlib
print("Matplotlib " + matplotlib.__version__)
import matplotlib.pyplot as plt
import support_files_drone_constraints as sfd_constraints
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation
from mpl_toolkits import mplot3d
from qpsolvers import *

# Create an object for the support functions.
support = sfd_constraints.SupportFilesDrone()
constants = support.constants

# Load the constant values needed in the main file
Ts = constants['Ts']
controlled_states = constants['controlled_states']  # number of outputs
innerDyn_length = constants['innerDyn_length']  # number of inner control loop iterations
sub_loop = constants['sub_loop']
sim_version = constants['sim_version']

if sim_version == 1:
    sim_version = 1
elif sim_version == 2:
    sim_version = 2
else:
    print("Please assign only 1 or 2 to the variable 'sim_version' in the input function")
    exit()

# Generate the reference signals
t = np.arange(0, 100 + Ts * innerDyn_length, Ts * innerDyn_length)  # time from 0 to 100 seconds
t_angles = np.arange(0, t[-1] + Ts, Ts)
t_ani = np.arange(0, t[-1] + Ts / sub_loop, Ts / sub_loop)
X_ref, X_dot_ref, X_dot_dot_ref, Y_ref, Y_dot_ref, Y_dot_dot_ref, Z_ref, Z_dot_ref, Z_dot_dot_ref, psi_ref = support.trajectory_generator(t)
plotl = len(t)  # Number of outer control loop iterations

# Load the initial state vector
ut = 0
vt = 0
wt = 0
pt = 0
qt = 0
rt = 0
xt = 0
yt = 0
zt = 3
phit = 0
thetat = 0
psit = psi_ref[0]

states = np.array([ut, vt, wt, pt, qt, rt, xt, yt, zt, phit, thetat, psit])
statesTotal = [states]  # Track all states
statesTotals = [states]
statesTotal_ani = [states[6:len(states)]]
ref_angles_total = np.array([[phit, thetat, psit]])
velocityXYZ_total = np.array([[0, 0, 0]])

# Get the minimum and maximum inputs
omega_min = constants['omega_min']
omega_max = constants['omega_max']

# Initial drone propeller states
omega1 = omega_min
omega2 = omega_min
omega3 = omega_min
omega4 = omega_min
omega_total = omega1 - omega2 + omega3 - omega4

ct = constants['ct']
cq = constants['cq']
l = constants['l']

# Plus configuration
U1 = ct * (omega1**2 + omega2**2 + omega3**2 + omega4**2)
U2 = ct * l * (omega2**2 - omega4**2)
U3 = ct * l * (omega3**2 - omega1**2)
U4 = cq * (-omega1**2 + omega2**2 - omega3**2 + omega4**2)

UTotal = np.array([[U1, U2, U3, U4]])
omegas_bundle = np.array([[omega1, omega2, omega3, omega4]])
UTotal_ani = UTotal
error_history = []

y_max = 0
y_min = 0

U1_min = ct * 4 * omega_min**2
U1_max = ct * 4 * omega_max**2
U2_min = ct * l * (omega_min**2 - omega_max**2)
U2_max = ct * l * (omega_max**2 - omega_min**2)
U3_min = ct * l * (omega_min**2 - omega_max**2)
U3_max = ct * l * (omega_max**2 - omega_min**2)
U4_min = cq * (-2 * omega_max**2 + 2 * omega_min**2)
U4_max = cq * (-2 * omega_min**2 + 2 * omega_max**2)

y_max = np.array([[U2_max], [U3_max], [U4_max]])
y_min = np.array([[U2_min], [U3_min], [U4_min]])

# Noise parameters
sigma_pos = 0.05  # Position sensor noise (m)

# # Kalman Filter parameters (for control inputs)
# Q_u = np.diag([0.1, 0.001, 0.001, 0.001])  # Process noise covariance
# # R_u = np.diag([sigma_act**2, sigma_act**2, sigma_act**2, sigma_act**2])  # Measurement noise covariance
# R_u = np.diag([0.01, 0.001, 0.001, 0.001])
# P_u = np.eye(4)
# u_hat = np.array([U1, U2, U3, U4])

# def add_noise(U):
#     noisy_U = U.copy()
#     noisy_U += np.random.normal(0, sigma_act, 4)  # U1, U2, U3, U4
#     return noisy_U

# EKF implementation
# def kalman_filter(y_u, u_hat, P_u):
#     # Prediction step
#     u_pred = u_hat
#     P_pred = P_u + Q_u
#     # Update step
#     H_u = np.eye(4)
#     y_pred = H_u @ u_pred
#     K_u = P_pred @ np.transpose(H_u) @ np.linalg.inv(H_u @ P_pred @ np.transpose(H_u) + R_u)
#     u_hat = u_pred + K_u @ (y_u - y_pred)
#     P_u = (np.eye(4) - K_u @ H_u) @ P_pred
#     return u_hat, P_u

# EKF Parameters
Q_pos = np.diag([0.001,0.001,0.0001])
R_pos = np.diag([0.05, 0.05, 0.5])
P_pos = np.diag([0.5, 0.5, 0.5])
x_hat = np.array([0, 0, 3])  # Initial [x, y, z]
Ts = support.constants['Ts']

def state_transition(x, u, v, w, phi, theta, psi, Ts):
    R = np.array([
        [np.cos(theta)*np.cos(psi), np.sin(phi)*np.sin(theta)*np.cos(psi) - np.cos(phi)*np.sin(psi), np.cos(phi)*np.sin(theta)*np.cos(psi) + np.sin(phi)*np.sin(psi)],
        [np.cos(theta)*np.sin(psi), np.sin(phi)*np.sin(theta)*np.sin(psi) + np.cos(phi)*np.cos(psi), np.cos(phi)*np.sin(theta)*np.sin(psi) - np.sin(phi)*np.cos(psi)],
        [-np.sin(theta), np.sin(phi)*np.cos(theta), np.cos(phi)*np.cos(theta)]
    ])
    vel_body = np.array([u, v, w])
    vel_inertial = R @ vel_body
    return x + Ts * vel_inertial

def ekf_predict(x_hat, P, u, v, w, phi, theta, psi, Ts):
    x_pred = state_transition(x_hat, u, v, w, phi, theta, psi, Ts)
    F = np.eye(3)  # Jacobian
    P_pred = P + Q_pos
    return x_pred, P_pred

def ekf_update(x_pred, P_pred, y):
    H = np.eye(3)
    y_pred = H @ x_pred
    S = H @ P_pred @ H.T + R_pos
    K = P_pred @ H.T @ np.linalg.inv(S)
    x_hat = x_pred + K @ (y - y_pred)
    P = (np.eye(3) - K @ H) @ P_pred
    return x_hat, P

########## Start the global controller #################################

for i_global in range(0, plotl - 1):

    # Implement the position controller
    phi_ref, theta_ref, U1 = support.pos_controller(
        X_ref[i_global + 1], X_dot_ref[i_global + 1], X_dot_dot_ref[i_global + 1],
        Y_ref[i_global + 1], Y_dot_ref[i_global + 1], Y_dot_dot_ref[i_global + 1],
        Z_ref[i_global + 1], Z_dot_ref[i_global + 1], Z_dot_dot_ref[i_global + 1],
        psi_ref[i_global + 1], states)

    Phi_ref = np.transpose([phi_ref * np.ones(innerDyn_length + 1)])
    Theta_ref = np.transpose([theta_ref * np.ones(innerDyn_length + 1)])

    # Check boundaries for U1
    if U1 < U1_min:
        U1 = U1_min
    if U1 > U1_max:
        U1 = U1_max

    Psi_ref = np.transpose([np.zeros(innerDyn_length + 1)])
    for yaw_step in range(0, innerDyn_length + 1):
        Psi_ref[yaw_step] = psi_ref[i_global] + (psi_ref[i_global + 1] - psi_ref[i_global]) / (Ts * innerDyn_length) * Ts * yaw_step

    temp_angles = np.concatenate((Phi_ref[1:len(Phi_ref)], Theta_ref[1:len(Theta_ref)], Psi_ref[1:len(Psi_ref)]), axis=1)
    ref_angles_total = np.concatenate((ref_angles_total, temp_angles), axis=0)

    # Build reference signal vector
    refSignals = np.zeros(len(Phi_ref) * controlled_states)
    k = 0
    for i in range(0, len(refSignals), controlled_states):
        refSignals[i] = Phi_ref[k]
        refSignals[i + 1] = Theta_ref[k]
        refSignals[i + 2] = Psi_ref[k]
        k = k + 1

    # Inner control loop
    hz = constants['hz']
    k = 0
    for i in range(0, innerDyn_length):
        # Generate discrete state space matrices
        Ad, Bd, Cd, Dd, x_dot, y_dot, z_dot, phi, phi_dot, theta, theta_dot, psi, psi_dot = support.LPV_cont_discrete(states, omega_total)
        x_dot = np.transpose([x_dot])
        y_dot = np.transpose([y_dot])
        z_dot = np.transpose([z_dot])
        temp_velocityXYZ = np.concatenate(([[x_dot], [y_dot], [z_dot]]), axis=1)
        velocityXYZ_total = np.concatenate((velocityXYZ_total, temp_velocityXYZ), axis=0)

        # Generate augmented state and reference
        x_aug_t = np.transpose([np.concatenate(([phi, phi_dot, theta, theta_dot, psi, psi_dot], [U2, U3, U4]), axis=0)])
        k = k + controlled_states
        if k + controlled_states * hz <= len(refSignals):
            r = refSignals[k:k + controlled_states * hz]
        else:
            r = refSignals[k:len(refSignals)]
            hz = hz - 1

        # MPC simplification
        Hdb, Fdbt, Cdb, Adc, C_cm_g, y_max_global, y_min_global = support.mpc_simplification(Ad, Bd, Cd, Dd, hz, y_max, y_min)
        ft = np.matmul(np.concatenate((np.transpose(x_aug_t)[0][0:len(x_aug_t)], r), axis=0), Fdbt)
        CC = np.matmul(C_cm_g, Cdb)
        G = np.concatenate((CC, -CC), axis=0)
        CAX = np.matmul(C_cm_g, Adc)
        CAX = np.matmul(CAX, x_aug_t)
        h1 = y_max_global - CAX
        h2 = -y_min_global + CAX
        h = np.concatenate((h1, h2), axis=0)
        ht = np.transpose(h)[0]

        du = solve_qp(Hdb, ft, G, ht, solver="cvxopt")

        # Update inputs
        U2 = U2 + du[0]
        U3 = U3 + du[1]
        U4 = U4 + du[2]

        # Add after calculating U2, U3, U4 but before using them
        # Define noise standard deviation - adjust this value based on your needs
        noise_std = 0.001

        # Add Gaussian noise while respecting bounds
        U2 = np.clip(U2 + np.random.normal(0, noise_std), U2_min, U2_max)
        U3 = np.clip(U3 + np.random.normal(0, noise_std), U3_min, U3_max)
        U4 = np.clip(U4 + np.random.normal(0, noise_std), U4_min, U4_max)

        # Add actuator noise
        U = np.array([U1, U2, U3, U4])
        # noisy_U = add_noise(U)
        #noisy_U = Check_Constraints_Control_Input()

        # u_hat, P_u = kalman_filter(noisy_U, u_hat, P_u)
        # u_hat[0] = np.clip(u_hat[0], U1_min, U1_max)
        # u_hat[1] = np.clip(u_hat[1], U2_min, U2_max)
        # u_hat[2] = np.clip(u_hat[2], U3_min, U3_max)
        # u_hat[3] = np.clip(u_hat[3], U4_min, U4_max)

        # U1, U2, U3, U4 = u_hat
        U1, U2, U3, U4 = U

        # Keep track of inputs
        UTotal = np.concatenate((UTotal, np.array([[U1, U2, U3, U4]])), axis=0)

        # Compute new omegas
        U1C = U1 / ct
        U2C = U2 / (ct * l)
        U3C = U3 / (ct * l)
        U4C = U4 / cq

        UC_vector = np.zeros((4, 1))
        UC_vector[0, 0] = U1C
        UC_vector[1, 0] = U2C
        UC_vector[2, 0] = U3C
        UC_vector[3, 0] = U4C

        omega_Matrix = np.zeros((4, 4))
        omega_Matrix[0, 0] = 1
        omega_Matrix[0, 1] = 1
        omega_Matrix[0, 2] = 1
        omega_Matrix[0, 3] = 1
        omega_Matrix[1, 0] = 0
        omega_Matrix[1, 1] = 1
        omega_Matrix[1, 2] = 0
        omega_Matrix[1, 3] = -1
        omega_Matrix[2, 0] = -1
        omega_Matrix[2, 1] = 0
        omega_Matrix[2, 2] = 1
        omega_Matrix[2, 3] = 0
        omega_Matrix[3, 0] = -1
        omega_Matrix[3, 1] = 1
        omega_Matrix[3, 2] = -1
        omega_Matrix[3, 3] = 1

        omega_Matrix_inverse = np.linalg.inv(omega_Matrix)
        omegas_vector = np.matmul(omega_Matrix_inverse, UC_vector)

        omega1P2 = omegas_vector[0, 0]
        omega2P2 = omegas_vector[1, 0]
        omega3P2 = omegas_vector[2, 0]
        omega4P2 = omegas_vector[3, 0]

        if omega1P2 <= 0 or omega2P2 <= 0 or omega3P2 <= 0 or omega4P2 <= 0:
            print("You can't take a square root of a negative number")
            print("The problem might be that the trajectory is too chaotic or it might have large discontinuous jumps")
            print("Try to make a smoother trajectory without discontinuous jumps")
            print("Other possible causes might be values for variables such as Ts, hz, innerDyn_length, px, py, pz")
            print("If problems occur, please download the files again, use the default settings and try to change values one by one.")
            exit()
        else:
            omega1 = np.sqrt(omega1P2)
            omega2 = np.sqrt(omega2P2)
            omega3 = np.sqrt(omega3P2)
            omega4 = np.sqrt(omega4P2)

        omegas_bundle = np.concatenate((omegas_bundle, np.array([[omega1, omega2, omega3, omega4]])), axis=0)

        # Compute new total omega
        omega_total = omega1 - omega2 + omega3 - omega4

        x_or = states[6]
        y_or = states[7]
        z_or = states[8]
        phi_or = states[9]
        theta_or = states[10]
        psi_or = states[11]

        # Compute new states with process noise (in open_loop_new_states)
        states, states_ani, U_ani = support.open_loop_new_states(states, omega_total, U1, U2, U3, U4, sigma_pos)

        x = states[6]
        y = states[7]
        z = states[8]

        x_pred, P_pred = ekf_predict(x_hat, P_pos, states[0], states[1], states[2], states[9], states[10], states[11], Ts)
        y_pos = np.array([states[6], states[7], states[8]])  # Noisy measurements
        x_hat, P_pos = ekf_update(x_pred, P_pred, y_pos)

        states[6] = x_hat[0]
        states[7] = x_hat[1]
        states[8] = x_hat[2]

        states_ani = support.ani_state(states[6], states[7], states[8], states[9], states[10], states[11], x_or, y_or, z_or, phi_or, theta_or, psi_or)

        statesTotal = np.concatenate((statesTotal, [states]), axis=0)
        statesTotal_ani = np.concatenate((statesTotal_ani, states_ani), axis=0)
        UTotal_ani = np.concatenate((UTotal_ani, U_ani), axis=0)
    statesTotals = np.concatenate((statesTotals, [states]), axis=0)

# The rest of the code (animation and plotting) remains unchanged
# ... (Copy the animation and plotting code from the original program)

# Animation setup
if max(Y_ref) >= max(X_ref):
    max_ref = max(Y_ref)
else:
    max_ref = max(X_ref)

if min(Y_ref) <= min(X_ref):
    min_ref = min(Y_ref)
else:
    min_ref = min(X_ref)

statesTotal_x = statesTotal_ani[:, 0]
statesTotal_y = statesTotal_ani[:, 1]
statesTotal_z = statesTotal_ani[:, 2]
statesTotal_phi = statesTotal_ani[:, 3]
statesTotal_theta = statesTotal_ani[:, 4]
statesTotal_psi = statesTotal_ani[:, 5]
UTotal_U1 = UTotal_ani[:, 0]
UTotal_U2 = UTotal_ani[:, 1]
UTotal_U3 = UTotal_ani[:, 2]
UTotal_U4 = UTotal_ani[:, 3]
frame_amount = int(len(statesTotal_x))
length_x = max_ref * 0.15
length_y = max_ref * 0.15

def update_plot(num):
    R_x = np.array([[1, 0, 0], [0, np.cos(statesTotal_phi[num]), -np.sin(statesTotal_phi[num])],
                    [0, np.sin(statesTotal_phi[num]), np.cos(statesTotal_phi[num])]])
    R_y = np.array([[np.cos(statesTotal_theta[num]), 0, np.sin(statesTotal_theta[num])], [0, 1, 0],
                    [-np.sin(statesTotal_theta[num]), 0, np.cos(statesTotal_theta[num])]])
    R_z = np.array([[np.cos(statesTotal_psi[num]), -np.sin(statesTotal_psi[num]), 0],
                    [np.sin(statesTotal_psi[num]), np.cos(statesTotal_psi[num]), 0], [0, 0, 1]])
    R_matrix = np.matmul(R_z, np.matmul(R_y, R_x))

    drone_pos_body_x = np.array([[length_x], [0], [0]])
    drone_pos_inertial_x = np.matmul(R_matrix, drone_pos_body_x)

    drone_pos_body_x_neg = np.array([[-length_x], [0], [0]])
    drone_pos_inertial_x_neg = np.matmul(R_matrix, drone_pos_body_x_neg)

    drone_pos_body_y = np.array([[0], [length_y], [0]])
    drone_pos_inertial_y = np.matmul(R_matrix, drone_pos_body_y)

    drone_pos_body_y_neg = np.array([[0], [-length_y], [0]])
    drone_pos_inertial_y_neg = np.matmul(R_matrix, drone_pos_body_y_neg)

    drone_body_x.set_xdata([statesTotal_x[num] + drone_pos_inertial_x_neg[0][0], statesTotal_x[num] + drone_pos_inertial_x[0][0]])
    drone_body_x.set_ydata([statesTotal_y[num] + drone_pos_inertial_x_neg[1][0], statesTotal_y[num] + drone_pos_inertial_x[1][0]])

    drone_body_y.set_xdata([statesTotal_x[num] + drone_pos_inertial_y_neg[0][0], statesTotal_x[num] + drone_pos_inertial_y[0][0]])
    drone_body_y.set_ydata([statesTotal_y[num] + drone_pos_inertial_y_neg[1][0], statesTotal_y[num] + drone_pos_inertial_y[1][0]])

    real_trajectory.set_xdata(statesTotal_x[0:num])
    real_trajectory.set_ydata(statesTotal_y[0:num])
    real_trajectory.set_3d_properties(statesTotal_z[0:num])

    drone_body_x.set_3d_properties([statesTotal_z[num] + drone_pos_inertial_x_neg[2][0], statesTotal_z[num] + drone_pos_inertial_x[2][0]])
    drone_body_y.set_3d_properties([statesTotal_z[num] + drone_pos_inertial_y_neg[2][0], statesTotal_z[num] + drone_pos_inertial_y[2][0]])

    if sim_version == 1:
        drone_body_phi.set_data([-length_y * 0.9 * 0.9, length_y * 0.9 * 0.9], [drone_pos_inertial_y_neg[2][0], drone_pos_inertial_y[2][0]])
        drone_body_theta.set_data([length_x * 0.9 * 0.9, -length_x * 0.9 * 0.9], [drone_pos_inertial_x[2][0], drone_pos_inertial_x_neg[2][0]])
        U1_function.set_data(t_ani[0:num], UTotal_U1[0:num])
        U2_function.set_data(t_ani[0:num], UTotal_U2[0:num])
        U3_function.set_data(t_ani[0:num], UTotal_U3[0:num])
        U4_function.set_data(t_ani[0:num], UTotal_U4[0:num])

        return drone_body_x, drone_body_y, real_trajectory, \
               drone_body_phi, drone_body_theta, U1_function, U2_function, U3_function, U4_function

    else:
        drone_position_x.set_data(t_ani[0:num], statesTotal_x[0:num])
        drone_position_y.set_data(t_ani[0:num], statesTotal_y[0:num])
        drone_position_z.set_data(t_ani[0:num], statesTotal_z[0:num])
        drone_orientation_phi.set_data(t_ani[0:num], statesTotal_phi[0:num])
        drone_orientation_theta.set_data(t_ani[0:num], statesTotal_theta[0:num])
        drone_orientation_psi.set_data(t_ani[0:num], statesTotal_psi[0:num])

        return drone_body_x, drone_body_y, real_trajectory, \
               drone_position_x, drone_position_y, drone_position_z, \
               drone_orientation_phi, drone_orientation_theta, drone_orientation_psi

# Set up figure properties
fig_x = 16
fig_y = 9
fig = plt.figure(figsize=(fig_x, fig_y), dpi=120, facecolor=(0.8, 0.8, 0.8))
n = 4
m = 3
gs = gridspec.GridSpec(n, m)

# Drone motion
ax0 = fig.add_subplot(gs[0:3, 0:2], projection='3d', facecolor=(0.9, 0.9, 0.9))
ax0.set_title(' ', fontsize=15)
ref_trajectory = ax0.plot(X_ref, Y_ref, Z_ref, 'b', linewidth=1, label='reference')
real_trajectory, = ax0.plot([], [], [], 'r', linewidth=1, label='trajectory')
drone_body_x, = ax0.plot([], [], [], 'r', linewidth=5, label='drone_x')
drone_body_y, = ax0.plot([], [], [], 'g', linewidth=5, label='drone_y')
ax0.set_xlim(min_ref, max_ref)
ax0.set_ylim(min_ref, max_ref)
ax0.set_zlim(0, max(Z_ref))
ax0.set_xlabel('X [m]')
ax0.set_ylabel('Y [m]')
ax0.set_zlabel('Z [m]')
ax0.legend(loc='upper left')

if sim_version == 1:
    ax1 = fig.add_subplot(gs[3, 0], facecolor=(0.9, 0.9, 0.9))
    drone_body_phi, = ax1.plot([], [], '--g', linewidth=2, label='drone_y (+: Z-up,Y-right,phi-CCW)')
    ax1.set_xlim(-length_y * 0.9, length_y * 0.9)
    ax1.set_ylim(-length_y * 1.1 * 0.01, length_y * 1.1 * 0.01)
    ax1.legend(loc='upper left', fontsize='small')
    plt.grid(True)

    ax2 = fig.add_subplot(gs[3, 1], facecolor=(0.9, 0.9, 0.9))
    drone_body_theta, = ax2.plot([], [], '--r', linewidth=2, label='drone_x (+: Z-up,X-left,theta-CCW)')
    ax2.set_xlim(length_x * 0.9, -length_x * 0.9)
    ax2.set_ylim(-length_x * 1.1 * 0.01, length_x * 1.1 * 0.01)
    ax2.legend(loc='upper left', fontsize='small')
    plt.grid(True)

    ax3 = fig.add_subplot(gs[0, 2], facecolor=(0.9, 0.9, 0.9))
    U1_function, = ax3.plot([], [], 'b', linewidth=1, label='Thrust (U1) [N]')
    plt.xlim(0, t_ani[-1])
    plt.ylim(np.min(UTotal_U1) - 0.01, np.max(UTotal_U1) + 0.01)
    plt.grid(True)
    plt.legend(loc='upper right', fontsize='small')

    ax4 = fig.add_subplot(gs[1, 2], facecolor=(0.9, 0.9, 0.9))
    U2_function, = ax4.plot([], [], 'b', linewidth=1, label='Roll (U2) [Nm]')
    plt.xlim(0, t_ani[-1])
    plt.ylim(np.min(UTotal_U2) - 0.01, np.max(UTotal_U2) + 0.01)
    plt.grid(True)
    plt.legend(loc='upper right', fontsize='small')

    ax5 = fig.add_subplot(gs[2, 2], facecolor=(0.9, 0.9, 0.9))
    U3_function, = ax5.plot([], [], 'b', linewidth=1, label='Pitch (U3) [Nm]')
    plt.xlim(0, t_ani[-1])
    plt.ylim(np.min(UTotal_U3) - 0.01, np.max(UTotal_U3) + 0.01)
    plt.grid(True)
    plt.legend(loc='upper right', fontsize='small')

    ax6 = fig.add_subplot(gs[3, 2], facecolor=(0.9, 0.9, 0.9))
    U4_function, = ax6.plot([], [], 'b', linewidth=1, label='Yaw (U4) [Nm]')
    plt.xlim(0, t_ani[-1])
    plt.ylim(np.min(UTotal_U4) - 0.01, np.max(UTotal_U4) + 0.01)
    plt.grid(True)
    plt.legend(loc='upper right', fontsize='small')
    plt.xlabel('t-time [s]', fontsize=15)
else:
    ax1 = fig.add_subplot(gs[3, 0], facecolor=(0.9, 0.9, 0.9))
    ax1.plot(t, X_ref, 'b', linewidth=1, label='X_ref [m]')
    drone_position_x, = ax1.plot([], [], 'r', linewidth=1, label='X [m]')
    ax1.set_xlim(0, t_ani[-1])
    ax1.set_ylim(np.min(statesTotal_x) - 0.01, np.max(statesTotal_x) + 0.01)
    ax1.legend(loc='lower right', fontsize='small')
    plt.grid(True)
    plt.xlabel('t-time [s]', fontsize=15)

    ax2 = fig.add_subplot(gs[3, 1], facecolor=(0.9, 0.9, 0.9))
    ax2.plot(t, Y_ref, 'b', linewidth=1, label='Y_ref [m]')
    drone_position_y, = ax2.plot([], [], 'r', linewidth=1, label='Y [m]')
    ax2.set_xlim(0, t_ani[-1])
    ax2.set_ylim(np.min(statesTotal_y) - 0.01, np.max(statesTotal_y) + 0.01)
    ax2.legend(loc='lower right', fontsize='small')
    plt.grid(True)
    plt.xlabel('t-time [s]', fontsize=15)

    ax3 = fig.add_subplot(gs[3, 2], facecolor=(0.9, 0.9, 0.9))
    ax3.plot(t, Z_ref, 'b', linewidth=1, label='Z_ref [m]')
    drone_position_z, = ax3.plot([], [], 'r', linewidth=1, label='Z [m]')
    plt.xlim(0, t_ani[-1])
    plt.ylim(np.min(statesTotal_z) - 0.01, np.max(statesTotal_z) + 0.01)
    plt.grid(True)
    plt.legend(loc='lower right', fontsize='small')
    plt.xlabel('t-time [s]', fontsize=15)

    ax4 = fig.add_subplot(gs[0, 2], facecolor=(0.9, 0.9, 0.9))
    ax4.plot(t_angles, ref_angles_total[:, 0], 'b', linewidth=1, label='Phi_ref [rad]')
    drone_orientation_phi, = ax4.plot([], [], 'r', linewidth=1, label='Phi [rad]')
    plt.xlim(0, t_ani[-1])
    plt.ylim(np.min(statesTotal_phi) - 0.01, np.max(statesTotal_phi) + 0.01)
    plt.grid(True)
    plt.legend(loc='lower right', fontsize='small')

    ax5 = fig.add_subplot(gs[1, 2], facecolor=(0.9, 0.9, 0.9))
    ax5.plot(t_angles, ref_angles_total[:, 1], 'b', linewidth=1, label='Theta_ref [rad]')
    drone_orientation_theta, = ax5.plot([], [], 'r', linewidth=1, label='Theta [rad]')
    plt.xlim(0, t_ani[-1])
    plt.ylim(np.min(statesTotal_theta) - 0.01, np.max(statesTotal_theta) + 0.01)
    plt.grid(True)
    plt.legend(loc='lower right', fontsize='small')

    ax6 = fig.add_subplot(gs[2, 2], facecolor=(0.9, 0.9, 0.9))
    ax6.plot(t_angles, ref_angles_total[:, 2], 'b', linewidth=1, label='Psi_ref [rad]')
    drone_orientation_psi, = ax6.plot([], [], 'r', linewidth=1, label='Psi [rad]')
    plt.xlim(0, t_ani[-1])
    plt.ylim(np.min(statesTotal_psi) - 0.01, np.max(statesTotal_psi) + 0.01)
    plt.grid(True)
    plt.legend(loc='lower right', fontsize='small')

    ax7 = fig.add_subplot(gs[3, 0], facecolor=(0.9, 0.9, 0.9))
    ax7.plot(t, X_ref-drone_position_x, 'b', linewidth=1, label='X_err [m]')
    ax7.set_xlim(0, t_ani[-1])
    ax7.set_ylim(np.min(statesTotal_x) - 0.01, np.max(statesTotal_x) + 0.01)
    ax7.legend(loc='lower right', fontsize='small')
    plt.grid(True)
    plt.xlabel('t-time [s]', fontsize=15)

drone_ani = animation.FuncAnimation(fig, update_plot, frames=frame_amount, interval=20, repeat=True, blit=True)
plt.show()

# Additional plots (unchanged)
no_plots = constants['no_plots']
if no_plots == 1:
    exit()
else:
    ax = plt.axes(projection='3d')
    ax.plot(X_ref, Y_ref, Z_ref, 'b', label='reference')
    ax.plot(statesTotal_x, statesTotal_y, statesTotal_z, 'r', label='trajectory')
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_zlabel('Z [m]')
    plt.savefig("WithKF.png", dpi=300, bbox_inches="tight")


    ax.legend()
    plt.show()

    plt.subplot(3, 1, 1)
    plt.plot(t, (X_ref - statesTotals[:,6]), 'r', linewidth=1, label='X_err')
    plt.plot(t, np.zeros(251), 'b', linewidth=1)
    plt.xlabel('t-time [s]', fontsize=15)
    plt.ylabel('X [m]', fontsize=15)
    plt.grid(True)

    plt.subplot(3, 1, 2)
    plt.plot(t, (Y_ref - statesTotals[:,7]), 'r', linewidth=1, label='Y_err')
    plt.plot(t, np.zeros(251), 'b', linewidth=1)
    plt.xlabel('t-time [s]', fontsize=15)
    plt.ylabel('Y [m]', fontsize=15)
    plt.grid(True)

    plt.subplot(3, 1, 3)
    plt.plot(t, (Z_ref - statesTotals[:,8]), 'r', linewidth=1, label='Z_err')
    plt.plot(t, np.zeros(251), 'b', linewidth=1)
    plt.xlabel('t-time [s]', fontsize=15)
    plt.ylabel('Z [m]', fontsize=15)
    plt.grid(True)

    plt.savefig("Err_With_KF", dpi=300, bbox_inches="tight")

    plt.show()

    plt.subplot(3, 1, 1)
    plt.plot(t, (np.abs(X_ref - statesTotals[:,6])/X_ref)*100, 'r', linewidth=1, label='X_err')
    plt.plot(t, np.zeros(251), 'b', linewidth=1)
    plt.xlabel('t-time [s]', fontsize=15)
    plt.ylabel('X [m]', fontsize=15)
    plt.grid(True)

    plt.subplot(3, 1, 2)
    plt.plot(t, (np.abs(Y_ref - statesTotals[:,7])/Y_ref)*100, 'r', linewidth=1, label='Y_err')
    plt.plot(t, np.zeros(251), 'b', linewidth=1)
    plt.xlabel('t-time [s]', fontsize=15)
    plt.ylabel('Y [m]', fontsize=15)
    plt.grid(True)

    plt.subplot(3, 1, 3)
    plt.plot(t, (np.abs(Z_ref - statesTotals[:,8])/Z_ref)*100, 'r', linewidth=1, label='Z_err')
    plt.plot(t, np.zeros(251), 'b', linewidth=1)
    plt.xlabel('t-time [s]', fontsize=15)
    plt.ylabel('Z [m]', fontsize=15)
    plt.grid(True)

    plt.savefig("%_Err_Default_RK5", dpi=300, bbox_inches="tight")

    plt.show()

    plt.subplot(2, 1, 1)
    plt.plot(t, X_ref, 'b', linewidth=1, label='X_ref')
    plt.plot(t_angles, statesTotal[:, 6], 'r', linewidth=1, label='X')
    plt.xlabel('t-time [s]', fontsize=15)
    plt.ylabel('X [m]', fontsize=15)
    plt.grid(True)
    plt.legend(loc='center right', fontsize='small')

    plt.subplot(2, 1, 2)
    plt.plot(t, X_dot_ref, 'b', linewidth=1, label='X_dot_ref')
    plt.plot(t, velocityXYZ_total[0:len(velocityXYZ_total):innerDyn_length, 0], 'r', linewidth=1, label='X_dot')
    plt.xlabel('t-time [s]', fontsize=15)
    plt.ylabel('X_dot [m/s]', fontsize=15)
    plt.grid(True)
    plt.legend(loc='center right', fontsize='small')
    plt.show()

    plt.subplot(2, 1, 1)
    plt.plot(t, Y_ref, 'b', linewidth=1, label='Y_ref')
    plt.plot(t_angles, statesTotal[:, 7], 'r', linewidth=1, label='Y')
    plt.xlabel('t-time [s]', fontsize=15)
    plt.ylabel('Y [m]', fontsize=15)
    plt.grid(True)
    plt.legend(loc='center right', fontsize='small')

    plt.subplot(2, 1, 2)
    plt.plot(t, Y_dot_ref, 'b', linewidth=1, label='Y_dot_ref')
    plt.plot(t, velocityXYZ_total[0:len(velocityXYZ_total):innerDyn_length, 1], 'r', linewidth=1, label='Y_dot')
    plt.xlabel('t-time [s]', fontsize=15)
    plt.ylabel('Y_dot [m/s]', fontsize=15)
    plt.grid(True)
    plt.legend(loc='center right', fontsize='small')
    plt.show()

    plt.subplot(2, 1, 1)
    plt.plot(t, Z_ref, 'b', linewidth=1, label='Z_ref')
    plt.plot(t_angles, statesTotal[:, 8], 'r', linewidth=1, label='Z')
    plt.xlabel('t-time [s]', fontsize=15)
    plt.ylabel('Z [m]', fontsize=15)
    plt.grid(True)
    plt.legend(loc='center right', fontsize='small')

    plt.subplot(2, 1, 2)
    plt.plot(t, Z_dot_ref, 'b', linewidth=1, label='Z_dot_ref')
    plt.plot(t, velocityXYZ_total[0:len(velocityXYZ_total):innerDyn_length, 2], 'r', linewidth=1, label='Z_dot')
    plt.xlabel('t-time [s]', fontsize=15)
    plt.ylabel('Z_dot [m/s]', fontsize=15)
    plt.grid(True)
    plt.legend(loc='center right', fontsize='small')
    plt.show()

    plt.subplot(3, 1, 1)
    plt.plot(t_angles, ref_angles_total[:, 0], 'b', linewidth=1, label='Phi_ref')
    plt.plot(t_angles, statesTotal[:, 9], 'r', linewidth=1, label='Phi')
    plt.xlabel('t-time [s]', fontsize=15)
    plt.ylabel('Phi [rad]', fontsize=15)
    plt.grid(True)
    plt.legend(loc='lower right', fontsize='small')

    plt.subplot(3, 1, 2)
    plt.plot(t_angles, ref_angles_total[:, 1], 'b', linewidth=1, label='Theta_ref')
    plt.plot(t_angles, statesTotal[:, 10], 'r', linewidth=1, label='Theta')
    plt.xlabel('t-time [s]', fontsize=15)
    plt.ylabel('Theta [rad]', fontsize=15)
    plt.grid(True)
    plt.legend(loc='lower right', fontsize='small')

    plt.subplot(3, 1, 3)
    plt.plot(t_angles, ref_angles_total[:, 2], 'b', linewidth=1, label='Psi_ref')
    plt.plot(t_angles, statesTotal[:, 11], 'r', linewidth=1, label='Psi')
    plt.xlabel('t-time [s]', fontsize=15)
    plt.ylabel('Psi [rad]', fontsize=15)
    plt.grid(True)
    plt.legend(loc='lower right', fontsize='small')
    plt.show()

    plt.subplot(4, 2, 1)
    plt.plot(t_angles, UTotal[0:len(UTotal), 0], 'b', linewidth=1, label='U1')
    plt.xlabel('t-time [s]', fontsize=15)
    plt.ylabel('U1 [N]', fontsize=15)
    plt.grid(True)
    plt.legend(loc='upper right', fontsize='small')

    plt.subplot(4, 2, 3)
    plt.plot(t_angles, UTotal[0:len(UTotal), 1], 'b', linewidth=1, label='U2')
    plt.xlabel('t-time [s]', fontsize=15)
    plt.ylabel('U2 [Nm]', fontsize=15)
    plt.grid(True)
    plt.legend(loc='upper right', fontsize='small')

    plt.subplot(4, 2, 5)
    plt.plot(t_angles, UTotal[0:len(UTotal), 2], 'b', linewidth=1, label='U3')
    plt.xlabel('t-time [s]', fontsize=15)
    plt.ylabel('U3 [Nm]', fontsize=15)
    plt.grid(True)
    plt.legend(loc='upper right', fontsize='small')

    plt.subplot(4, 2, 7)
    plt.plot(t_angles, UTotal[0:len(UTotal), 3], 'b', linewidth=1, label='U4')
    plt.xlabel('t-time [s]', fontsize=15)
    plt.ylabel('U4 [Nm]', fontsize=15)
    plt.grid(True)
    plt.legend(loc='upper right', fontsize='small')

    plt.subplot(4, 2, 2)
    plt.plot(t_angles, omegas_bundle[0:len(omegas_bundle), 0], 'b', linewidth=1, label='omega 1')
    plt.xlabel('t-time [s]', fontsize=15)
    plt.ylabel('omega 1 [rad/s]', fontsize=15)
    plt.grid(True)
    plt.legend(loc='upper right', fontsize='small')

    plt.subplot(4, 2, 4)
    plt.plot(t_angles, omegas_bundle[0:len(omegas_bundle), 1], 'b', linewidth=1, label='omega 2')
    plt.xlabel('t-time [s]', fontsize=15)
    plt.ylabel('omega 2 [rad/s]', fontsize=15)
    plt.grid(True)
    plt.legend(loc='upper right', fontsize='small')

    plt.subplot(4, 2, 6)
    plt.plot(t_angles, omegas_bundle[0:len(omegas_bundle), 2], 'b', linewidth=1, label='omega 3')
    plt.xlabel('t-time [s]', fontsize=15)
    plt.ylabel('omega 3 [rad/s]', fontsize=15)
    plt.grid(True)
    plt.legend(loc='upper right', fontsize='small')

    plt.subplot(4, 2, 8)
    plt.plot(t_angles, omegas_bundle[0:len(omegas_bundle), 3], 'b', linewidth=1, label='omega 4')
    plt.xlabel('t-time [s]', fontsize=15)
    plt.ylabel('omega 4 [rad/s]', fontsize=15)
    plt.grid(True)
    plt.legend(loc='upper right', fontsize='small')
    plt.show()
    # New plots: X_err, Y_err, Z_err in one figure
    # Interpolate reference positions to match t_angles
    X_ref_interp = np.interp(t_angles, t, X_ref)
    Y_ref_interp = np.interp(t_angles, t, Y_ref)
    Z_ref_interp = np.interp(t_angles, t, Z_ref)

    # Compute errors
    X_err = X_ref_interp - statesTotal[:, 6]
    Y_err = Y_ref_interp - statesTotal[:, 7]
    Z_err = Z_ref_interp - statesTotal[:, 8]


    plt.subplot(3, 1, 1)
    plt.plot(t_angles, X_err, 'b', linewidth=1, label='X_err')
    plt.xlabel('t-time [s]', fontsize=15)
    plt.ylabel('X_err [m]', fontsize=15)
    plt.grid(True)
    plt.legend(loc='upper right', fontsize='small')
    plt.title('Position Errors', fontsize=15)

    plt.subplot(3, 1, 2)
    plt.plot(t_angles, Y_err, 'b', linewidth=1, label='Y_err')
    plt.xlabel('t-time [s]', fontsize=15)
    plt.ylabel('Y_err [m]', fontsize=15)
    plt.grid(True)
    plt.legend(loc='upper right', fontsize='small')

    plt.subplot(3, 1, 3)
    plt.plot(t_angles, Z_err, 'b', linewidth=1, label='Z_err')
    plt.xlabel('t-time [s]', fontsize=15)
    plt.ylabel('Z_err [m]', fontsize=15)
    plt.grid(True)
    plt.legend(loc='upper right', fontsize='small')

    plt.tight_layout()
    plt.show()