import platform
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import support_files_drone_adaptive_constraints as sfd_constraints
from qpsolvers import *

# Create an object for the support functions.
support = sfd_constraints.SupportFilesDrone()
constants = support.constants

# Load the constant values needed in the main file
Ts = constants['Ts']
controlled_states = constants['controlled_states']                  # number of outputs
innerDyn_length = constants['innerDyn_length']                      # number of inner control loop iterations

sub_loop = constants['sub_loop']

# Generate the refence signals
t = np.arange(0,50 + Ts * innerDyn_length, Ts * innerDyn_length)   # time from 0 to 100 seconds, sample time (Ts=0.4 second)
t_angles = np.arange(0, t[-1] + Ts, Ts)
t_ani = np.arange(0, t[-1] + Ts/sub_loop, Ts/sub_loop)
X_ref, X_dot_ref, X_dot_dot_ref, Y_ref, Y_dot_ref, Y_dot_dot_ref, Z_ref, Z_dot_ref, Z_dot_dot_ref, psi_ref = support.trajectory_generator(t)

# Load the initial state vector
xt = 0
yt = 0
zt = 0
phit = 0
thetat = 0
psit = psi_ref[0]
ut = 0
vt = 0
wt = 0
pt = 0
qt = 0
rt = 0

states = np.array([xt, yt, zt, phit, thetat, psit, ut, vt, wt, pt, qt, rt])
states_mpc = np.array([xt, yt, zt, phit, thetat, psit, ut, vt, wt, pt, qt, rt])

statesTotal = [states] # It will keep track of all your states during the entire manoeuvre
statesTotal_mpc = [states_mpc]

statesTotals = [states]  # For error
statesTotals_mpc = [states_mpc]

statesTotal_ani = [states[0:len(states)-6]]
statesTotal_ani_mpc = [states_mpc[0:len(states_mpc)-6]]

# Assume that first Phi_ref, Theta_ref, Psi_ref are equal to the first phit, thetat, psit
ref_angles_total = np.array([[phit, thetat, psit]])
ref_angles_total_mpc = np.array([[phit, thetat, psit]])

velocityXYZ_total = np.array([[0, 0, 0]])
velocityXYZ_total_mpc = np.array([[0, 0, 0]])

# Get the minimum and maximum inputs
omega_min = constants['omega_min']
omega_max = constants['omega_max']

# Initial drone propeller states
omega1 = omega_min # rad/s at t=-Ts s (Ts seconds before NOW)
omega2 = omega_min # rad/s at t=-Ts s (Ts seconds before NOW)
omega3 = omega_min # rad/s at t=-Ts s (Ts seconds before NOW)
omega4 = omega_min # rad/s at t=-Ts s (Ts seconds before NOW)
omega_total = omega1 - omega2 + omega3 - omega4
omega_total_mpc = omega1 - omega2 + omega3 - omega4

ct = constants['ct']
cq = constants['cq']
l = constants['l']

# Plus configuration
U1 = ct * (omega1 ** 2 + omega2 ** 2 + omega3 ** 2 + omega4 ** 2)           # Input at t = -Ts s
U2 = ct * l * (omega2 ** 2 - omega4 ** 2)                                   # Input at t = -Ts s
U3 = ct * l * (omega3 ** 2 - omega1 ** 2)                                   # Input at t = -Ts s
U4 = cq * (-omega1 ** 2 + omega2 ** 2 - omega3 ** 2 + omega4 ** 2)          # Input at t = -Ts s

# Plus configuration
U1_mpc = ct * (omega1 ** 2 + omega2 ** 2 + omega3 ** 2 + omega4 ** 2)       # Input at t = -Ts s
U2_mpc = ct * l *(omega2 ** 2 - omega4 ** 2)                                # Input at t = -Ts s
U3_mpc = ct * l *(omega3 ** 2 - omega1 ** 2)                                # Input at t = -Ts s
U4_mpc = cq * (-omega1 ** 2 + omega2 ** 2 - omega3 ** 2 + omega4 ** 2)      # Input at t = -Ts s

UTotal = np.array([[U1, U2, U3, U4]])                                       # 4 inputs MPC
UTotal_mpc = np.array([[U1_mpc, U2_mpc, U3_mpc, U4_mpc]])                   # 4 inputs mpc
omegas_bundle = np.array([[omega1, omega2, omega3, omega4]])
omegas_bundle_mpc = np.array([[omega1, omega2, omega3, omega4]])
UTotal_ani = UTotal
UTotal_ani_mpc = UTotal_mpc

y_max = 0
y_min = 0

U1_min = ct * 4 * omega_min ** 2
U1_max = ct * 4 * omega_max ** 2

U2_min = ct * l * (omega_min ** 2 - omega_max ** 2)
U2_max = ct * l * (omega_max ** 2 - omega_min ** 2)

U3_min = ct * l * (omega_min ** 2 - omega_max ** 2)
U3_max = ct * l * (omega_max **2 - omega_min ** 2)

U4_min = cq * (-2 * omega_max ** 2 + 2* omega_min ** 2)
U4_max = cq * (-2 * omega_min ** 2 + 2* omega_max ** 2)

phi_min = -np.pi/6
phi_max = np.pi/6

phi_dot_min = -3
phi_dot_max = 3

psi_min = -np.pi*8
psi_max = np.pi*8

psi_dot_min = -3
psi_dot_max = 3

theta_min = -np.pi/6
theta_max = np.pi/6

theta_dot_min = -3
theta_dot_max = 3

y_max = np.array([[phi_max], [phi_dot_max], [theta_max], [theta_dot_max], [psi_max], [psi_dot_max], [U2_max], [U3_max], [U4_max]])
y_min = np.array([[phi_min], [phi_dot_min], [theta_min], [theta_dot_min], [psi_min], [psi_dot_min], [U2_min], [U3_min], [U4_min]])

# y_max = np.array([[U2_max], [U3_max], [U4_max]])
# y_min = np.array([[U2_min], [U3_min], [U4_min]])

# Noise variables
np.random.seed(42)                                                          # For reproducible results

# Add these noise parameters after the initial constants
noise_std = 0.002                                                       # Standard deviation of the Gaussian noise
noise_mean = 0                                                              # Mean of the Gaussian noise

########## Start the global controller #################################

for i_global in range(0, len(t) - 1):
    # Implement the position controller (state feedback linearization)
    # MPC Ref
    phi_ref, theta_ref, U1 = support.adaptive_pos_controller(X_ref[i_global + 1], X_dot_ref[i_global + 1],
                                                  X_dot_dot_ref[i_global + 1], Y_ref[i_global + 1], 
                                                  Y_dot_ref[i_global + 1], Y_dot_dot_ref[i_global + 1], 
                                                  Z_ref[i_global + 1], Z_dot_ref[i_global + 1], 
                                                  Z_dot_dot_ref[i_global + 1], psi_ref[i_global + 1], states)
    # mpc Ref
    phi_ref_mpc, theta_ref_mpc, U1_mpc = support.pos_controller(
        X_ref[i_global + 1], X_dot_ref[i_global + 1], X_dot_dot_ref[i_global + 1],
        Y_ref[i_global + 1], Y_dot_ref[i_global + 1], Y_dot_dot_ref[i_global + 1],
        Z_ref[i_global + 1], Z_dot_ref[i_global + 1], Z_dot_dot_ref[i_global + 1],
        psi_ref[i_global + 1], states_mpc
    )
    Phi_ref = np.transpose([phi_ref * np.ones(innerDyn_length + 1)])
    Theta_ref = np.transpose([theta_ref * np.ones(innerDyn_length + 1)])
    Phi_ref_mpc = np.transpose([phi_ref_mpc * np.ones(innerDyn_length + 1)])
    Theta_ref_mpc = np.transpose([theta_ref_mpc * np.ones(innerDyn_length + 1)])

    # Check the boundaries for U1
    if U1 < U1_min:
        U1 = U1_min
    if U1 > U1_max:
        U1 = U1_max
    if U1_mpc < U1_min:
        U1_mpc = U1_min
    if U1_mpc > U1_max:
        U1_mpc = U1_max

    Psi_ref = np.transpose([np.zeros(innerDyn_length + 1)])
    Psi_ref_mpc = np.transpose([np.zeros(innerDyn_length + 1)])

    for yaw_step in range(0, innerDyn_length + 1):
        Psi_ref[yaw_step] = psi_ref[i_global] + (psi_ref[i_global + 1] - psi_ref[i_global]) / (Ts * innerDyn_length) * Ts * yaw_step
        Psi_ref_mpc[yaw_step] = psi_ref[i_global] + (psi_ref[i_global + 1] - psi_ref[i_global]) / (Ts * innerDyn_length) * Ts * yaw_step

    temp_angles = np.concatenate((Phi_ref[1:len(Phi_ref)], Theta_ref[1:len(Theta_ref)], Psi_ref[1:len(Psi_ref)]), axis = 1)
    ref_angles_total = np.concatenate((ref_angles_total, temp_angles), axis = 0)

    temp_angles_mpc = np.concatenate((Phi_ref_mpc[1:len(Phi_ref_mpc)], Theta_ref_mpc[1:len(Theta_ref_mpc)], Psi_ref_mpc[1:len(Psi_ref_mpc)]), axis=1)
    ref_angles_total_mpc = np.concatenate((ref_angles_total_mpc, temp_angles_mpc), axis=0)
    # Create a reference vector
    refSignals = np.zeros(len(Phi_ref) * controlled_states)
    refSignals_mpc = np.zeros(len(Phi_ref_mpc) * controlled_states)

    # Build up the reference signal vector:
    # refSignal = [Phi_ref_0, Theta_ref_0, Psi_ref_0, Phi_ref_1, Theta_ref_2, Psi_ref_2, ... etc.]
    k = 0
    for i in range(0, len(refSignals), controlled_states):
        refSignals[i] = float(Phi_ref[k].item())
        refSignals[i + 1] = float(Theta_ref[k].item())
        refSignals[i + 2] = float(Psi_ref[k].item())
        refSignals_mpc[i] = float(Phi_ref_mpc[k].item())
        refSignals_mpc[i + 1] = float(Theta_ref_mpc[k].item())
        refSignals_mpc[i + 2] = float(Psi_ref_mpc[k].item())
        k = k + 1

    # Initiate the controller - simulation loops
    noise_switch = constants['noise_switch']
    hz = constants['hz'] # horizon period
    hz_mpc = constants['hz'] # horizon period
    k = 0 # for reading reference signals
    k_mpc = 0 # for reading reference signals mpc
    # statesTotal2 = np.concatenate((statesTotal2, [states]), axis = 0)
    for i in range(0, innerDyn_length):
        # Generate the discrete state space matrices
        Ad, Bd, Cd, Dd, x_dot, y_dot, z_dot, phi, phi_dot, theta, theta_dot, psi, psi_dot = support.LPV_cont_discrete(states, omega_total)
        x_dot = np.transpose([x_dot])
        y_dot = np.transpose([y_dot])
        z_dot = np.transpose([z_dot])
        temp_velocityXYZ = np.concatenate(([[x_dot], [y_dot], [z_dot]]), axis = 1)
        velocityXYZ_total = np.concatenate((velocityXYZ_total, temp_velocityXYZ), axis = 0)
        # Generate the augmented current state and the reference vector
        x_aug_t = np.transpose([np.concatenate(([phi, phi_dot, theta, theta_dot, psi, psi_dot], [U2, U3, U4]), axis = 0)])
        k = k + controlled_states
        if k + controlled_states * hz <= len(refSignals):
            r = refSignals[k:k + controlled_states * hz]
        else:
            r = refSignals[k:len(refSignals)]
            hz = hz - 1

        # Generate the compact simplification matrices for the cost function
        Hdb, Fdbt, Cdb, Adc, C_cm_g, y_max_global, y_min_global = support.mpc_simplification(Ad, Bd, Cd, Dd, hz, y_max, y_min)
        ft = np.matmul(np.concatenate((np.transpose(x_aug_t)[0][0:len(x_aug_t)], r), axis = 0), Fdbt)

        CC = np.matmul(C_cm_g, Cdb)
        G = np.concatenate((CC, -CC), axis = 0)
        CAX = np.matmul(C_cm_g, Adc)
        CAX = np.matmul(CAX, x_aug_t)
        h1 = y_max_global - CAX
        h2 = -y_min_global + CAX
        h = np.concatenate((h1, h2), axis = 0)
        ht = np.transpose(h)[0]

        du = solve_qp(Hdb, ft, G, ht, solver = "cvxopt")
        # Update the real inputs
        U2 = U2 + du[0]
        U3 = U3 + du[1]
        U4 = U4 + du[2]

        # Inside the global controller loop, after calculating U1-U4 for both controllers
        # For MPC (after U2, U3, U4 calculation):

        # Keep track of your inputs
        UTotal = np.concatenate((UTotal, np.array([[U1, U2, U3, U4]])), axis=0)
        ## End of MPC

        # Compute the new omegas based on the new U-s
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

        ##############################################################################################################################################
        # Generate the discrete state space matrices
        Ad_mpc, Bd_mpc, Cd_mpc, Dd_mpc, x_dot_mpc, y_dot_mpc, z_dot_mpc, phi_mpc, phi_dot_mpc, theta_mpc, theta_dot_mpc, psi_mpc, psi_dot_mpc = support.LPV_cont_discrete(states_mpc, omega_total_mpc)
        x_dot_mpc = np.transpose([x_dot_mpc])
        y_dot_mpc = np.transpose([y_dot_mpc])
        z_dot_mpc = np.transpose([z_dot_mpc])
        temp_velocityXYZ_mpc = np.concatenate(([[x_dot_mpc], [y_dot_mpc], [z_dot_mpc]]), axis = 1)
        velocityXYZ_total_mpc = np.concatenate((velocityXYZ_total_mpc, temp_velocityXYZ_mpc), axis = 0)
        # Generate the augmented current state and the reference vector
        x_aug_t_mpc = np.transpose([np.concatenate(([phi_mpc, phi_dot_mpc, theta_mpc, theta_dot_mpc, psi_mpc, psi_dot_mpc], [U2_mpc, U3_mpc, U4_mpc]), axis = 0)])
        k_mpc = k_mpc + controlled_states
        if k_mpc + controlled_states * hz_mpc <= len(refSignals_mpc):
            r = refSignals_mpc[k_mpc:k_mpc + controlled_states * hz_mpc]
        else:
            r = refSignals_mpc[k_mpc:len(refSignals)]
            hz_mpc = hz_mpc - 1

        # Generate the compact simplification matrices for the cost function
        Hdb_mpc, Fdbt_mpc, Cdb_mpc, Adc_mpc, C_cm_g_mpc, y_max_global_mpc, y_min_global_mpc = support.mpc_simplification(Ad_mpc, Bd_mpc, Cd_mpc, Dd_mpc, hz, y_max, y_min)
        ft_mpc = np.matmul(np.concatenate((np.transpose(x_aug_t_mpc)[0][0:len(x_aug_t_mpc)], r), axis = 0), Fdbt_mpc)

        CC_mpc = np.matmul(C_cm_g_mpc, Cdb_mpc)
        G_mpc = np.concatenate((CC_mpc, -CC_mpc), axis = 0)
        CAX_mpc = np.matmul(C_cm_g_mpc, Adc_mpc)
        CAX_mpc = np.matmul(CAX_mpc, x_aug_t_mpc)
        h1_mpc = y_max_global_mpc - CAX_mpc
        h2_mpc = -y_min_global_mpc + CAX_mpc
        h_mpc = np.concatenate((h1_mpc, h2_mpc), axis = 0)
        ht_mpc = np.transpose(h)[0]

        du_mpc = solve_qp(Hdb_mpc, ft_mpc, G_mpc, ht_mpc, solver = "cvxopt")
        # Update the real inputs
        U2_mpc = U2_mpc + du_mpc[0]
        U3_mpc = U3_mpc + du_mpc[1]
        U4_mpc = U4_mpc + du_mpc[2]

        # Keep track of inputs
        UTotal_mpc = np.concatenate((UTotal_mpc, np.array([[U1_mpc, U2_mpc, U3_mpc, U4_mpc]])), axis = 0)

        U1C_mpc = U1_mpc / ct
        U2C_mpc = U2_mpc / (ct * l)
        U3C_mpc = U3_mpc / (ct * l)
        U4C_mpc = U4_mpc / cq
        UC_mpc_vector = np.zeros((4, 1))
        UC_mpc_vector[0, 0] = U1C_mpc
        UC_mpc_vector[1, 0] = U2C_mpc
        UC_mpc_vector[2, 0] = U3C_mpc
        UC_mpc_vector[3, 0] = U4C_mpc

        omegas_vector_mpc = np.matmul(omega_Matrix_inverse, UC_mpc_vector)

        omega1P2_mpc = omegas_vector_mpc[0, 0]
        omega2P2_mpc = omegas_vector_mpc[1, 0]
        omega3P2_mpc = omegas_vector_mpc[2, 0]
        omega4P2_mpc = omegas_vector_mpc[3, 0]

        if omega1P2 <= 0 or omega2P2 <= 0 or omega3P2 <= 0 or omega4P2 <= 0 or omega1P2_mpc <= 0 or omega2P2_mpc <= 0 or omega3P2_mpc <= 0 or omega4P2_mpc <= 0:
            print("You can't take a square root of a negative number")
            print("The problem might be that the trajectory is too chaotic or it might have discontinuous jumps")
            print("Try to make a smoother trajectory without discontinuous jumps")
            print("Other possible causes might be values for variables such as Ts, hz, innerDyn_length, px, py, pz")
            print("If problems occur, please download the files again, use the default settings and try to change values one by one.")
            exit()
        else:
            omega1 = np.sqrt(omega1P2)
            omega2 = np.sqrt(omega2P2)
            omega3 = np.sqrt(omega3P2)
            omega4 = np.sqrt(omega4P2)
            omega1_mpc = np.sqrt(omega1P2_mpc)
            omega2_mpc = np.sqrt(omega2P2_mpc)
            omega3_mpc = np.sqrt(omega3P2_mpc)
            omega4_mpc = np.sqrt(omega4P2_mpc)

        # Apply constraints on omega
        omega1 = np.clip(omega1, omega_min, omega_max)
        omega2 = np.clip(omega2, omega_min, omega_max)
        omega3 = np.clip(omega3, omega_min, omega_max)
        omega4 = np.clip(omega4, omega_min, omega_max)
        omega1_mpc = np.clip(omega1_mpc, omega_min, omega_max)
        omega2_mpc = np.clip(omega2_mpc, omega_min, omega_max)
        omega3_mpc = np.clip(omega3_mpc, omega_min, omega_max)
        omega4_mpc = np.clip(omega4_mpc, omega_min, omega_max)

        omegas_bundle = np.concatenate((omegas_bundle, np.array([[omega1, omega2, omega3, omega4]])), axis = 0)
        omegas_bundle_mpc = np.concatenate((omegas_bundle_mpc, np.array([[omega1_mpc, omega2_mpc, omega3_mpc, omega4_mpc]])), axis = 0)

        # Compute the new total omega
        omega_total = omega1 - omega2 + omega3 - omega4
        # Compute new states in the open loop system (interval: Ts/10)
        # MPC
        states, states_ani, U_ani = support.open_loop_new_states(states, omega_total, U1, U2, U3, U4)
        statesTotal = np.concatenate((statesTotal, [states]), axis = 0)
        statesTotal_ani = np.concatenate((statesTotal_ani, states_ani), axis = 0)
        UTotal_ani = np.concatenate((UTotal_ani, U_ani), axis = 0)

        # mpc
        states_mpc, states_ani_mpc, U_ani_mpc = support.open_loop_new_states(states_mpc, omega_total_mpc, U1_mpc, U2_mpc, U3_mpc, U4_mpc)
        statesTotal_mpc = np.concatenate((statesTotal_mpc, [states_mpc]), axis = 0)
        statesTotal_ani_mpc = np.concatenate((statesTotal_ani_mpc, states_ani_mpc), axis = 0)
        UTotal_ani_mpc = np.concatenate((UTotal_ani_mpc, U_ani_mpc), axis = 0)

    statesTotals = np.concatenate((statesTotals, [states]), axis = 0)
    statesTotals_mpc = np.concatenate((statesTotals_mpc, [states_mpc]), axis = 0)

################################ ANIMATION LOOP ###############################
# MPC Animation
statesTotal_x=statesTotal_ani[:,0]
statesTotal_y=statesTotal_ani[:,1]
statesTotal_z=statesTotal_ani[:,2]
statesTotal_phi=statesTotal_ani[:,3]
statesTotal_theta=statesTotal_ani[:,4]
statesTotal_psi=statesTotal_ani[:,5]
UTotal_U1=UTotal_ani[:,0]
UTotal_U2=UTotal_ani[:,1]
UTotal_U3=UTotal_ani[:,2]
UTotal_U4=UTotal_ani[:,3]

# mpc Animation
statesTotal_x_mpc=statesTotal_ani_mpc[:,0]
statesTotal_y_mpc=statesTotal_ani_mpc[:,1]
statesTotal_z_mpc=statesTotal_ani_mpc[:,2]
statesTotal_phi_mpc=statesTotal_ani_mpc[:,3]
statesTotal_theta_mpc=statesTotal_ani_mpc[:,4]
statesTotal_psi_mpc=statesTotal_ani_mpc[:,5]
UTotal_U1_mpc=UTotal_ani_mpc[:,0]
UTotal_U2_mpc=UTotal_ani_mpc[:,1]
UTotal_U3_mpc=UTotal_ani_mpc[:,2]
UTotal_U4_mpc=UTotal_ani_mpc[:,3]

print(states[9], states_mpc[10], states[11])

# Plot the world
plt.figure(figsize = (20, 12))
plt.get_current_fig_manager().set_window_title('Trajectory tracking')
# Plot the 3D trajectory
ax = plt.subplot(111, projection = '3d')
#ax=plt.axes(projection='3d')
ax.plot(X_ref,Y_ref,Z_ref,'b',label='Reference')
ax.plot(statesTotal_x_mpc,statesTotal_y_mpc,statesTotal_z_mpc,'g',label='Feedback Linearization')
ax.plot(statesTotal_x,statesTotal_y,statesTotal_z,'r',label='Proposed Method')
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')
ax.legend(loc='upper right', fontsize="20")
plt.show()

# Plot X, Y, Z Error %
plt.figure().canvas.manager.set_window_title('Error Values')
plt.subplot(3, 1, 1)
plt.plot(t, (X_ref - statesTotals_mpc[:, 0]), 'g', linewidth=2, label='Feedback Linearization')
plt.plot(t, (X_ref - statesTotals[:, 0]), 'r', linewidth=2, label='Proposed Method')
plt.plot(t, np.zeros(126), 'b', linewidth=1)
plt.ylabel('X Error', fontsize=20)
plt.legend(loc='upper right', fontsize="20")
plt.grid(True)

plt.subplot(3, 1, 2)
plt.plot(t, (Y_ref - statesTotals_mpc[:, 1]), 'g', linewidth=2)
plt.plot(t, (Y_ref - statesTotals[:, 1]), 'r', linewidth=2)
plt.plot(t, np.zeros(126), 'b', linewidth=1)
plt.ylabel('Y Error', fontsize=20)
plt.grid(True)

plt.subplot(3, 1, 3)
plt.plot(t, (Z_ref - statesTotals_mpc[:, 2]), 'g', linewidth=2)
plt.plot(t, (Z_ref - statesTotals[:, 2]), 'r', linewidth=2)
plt.plot(t, np.zeros(126), 'b', linewidth=1)
plt.xlabel('t (s)', fontsize=20)
plt.ylabel('Z Error', fontsize=20)
plt.grid(True)
plt.show()

# Plot X, Y, Z Values
plt.figure().canvas.manager.set_window_title('X Values')
plt.subplot(1, 1, 1)
plt.plot(t, X_ref, 'b', linewidth=1, label='Reference')
plt.plot(t_angles, statesTotal_mpc[:, 0], 'g', linewidth=2, label='Feedback Linearization')
plt.plot(t_angles, statesTotal[:, 0], 'r', linewidth=2, label='Proposed Method')
plt.ylabel('X (m)', fontsize=20)
plt.grid(True)
plt.legend(loc='upper right', fontsize="20")
plt.show()

plt.figure().canvas.manager.set_window_title('Y Values')
plt.subplot(1, 1, 1)
plt.plot(t, Y_ref, 'b', linewidth=1, label='Reference')
plt.plot(t_angles, statesTotal_mpc[:, 1], 'g', linewidth=2, label='Feedback Linearization')
plt.plot(t_angles, statesTotal[:, 1], 'r', linewidth=2, label='Proposed Method')
plt.ylabel('Y (m)', fontsize=20)
plt.grid(True)
plt.legend(loc='upper right', fontsize="20")
plt.show()

plt.figure().canvas.manager.set_window_title('Z Values')
plt.subplot(1, 1, 1)
plt.plot(t, Z_ref, 'b', linewidth=1, label='Reference')
plt.plot(t_angles, statesTotal_mpc[:, 2], 'g', linewidth=2, label='Feedback Linearization')
plt.plot(t_angles, statesTotal[:, 2], 'r', linewidth=2, label='Proposed Method')
plt.xlabel('t (s)', fontsize=20)
plt.ylabel('Z (m)', fontsize=20)
plt.grid(True)
plt.legend(loc='upper right', fontsize="20")
plt.show()

# Plot Angles
plt.figure().canvas.manager.set_window_title('Angle Values')
plt.subplot(3, 2, 1)
plt.plot(t_angles, ref_angles_total[:, 0], 'b', linewidth=1, label='Phi_ref')
plt.plot(t_angles, statesTotal[:, 3], 'r', linewidth=1, label='Phi(Proposed Method)')
plt.xlabel('t (s)', fontsize=15)
plt.ylabel('Phi (rad)', fontsize=15)
plt.grid(True)
plt.legend(loc='upper right', fontsize="15")

plt.subplot(3, 2, 3)
plt.plot(t_angles, ref_angles_total[:, 1], 'b', linewidth=1, label='Theta_ref')
plt.plot(t_angles, statesTotal[:, 4], 'r', linewidth=1, label='Theta(Proposed Method)')
plt.xlabel('t (s)', fontsize=15)
plt.ylabel('Theta (rad)', fontsize=15)
plt.grid(True)
plt.legend(loc='upper right', fontsize="15")

plt.subplot(3, 2, 5)
plt.plot(t_angles, ref_angles_total[:, 2], 'b', linewidth=1, label='Psi_ref')
plt.plot(t_angles, statesTotal[:, 5], 'r', linewidth=1, label='Psi(Proposed Method)')
plt.xlabel('t (s)', fontsize=15)
plt.ylabel('Psi (rad)', fontsize=15)
plt.grid(True)
plt.legend(loc='upper right', fontsize="15")

# Plot Angles
plt.subplot(3, 2, 2)
plt.plot(t_angles, ref_angles_total[:, 0], 'b', linewidth=1, label='Phi_ref')
plt.plot(t_angles, statesTotal_mpc[:, 3], 'g', linewidth=1, label='Phi(mpc)')
plt.xlabel('t (s)', fontsize=15)
plt.ylabel('Phi (rad)', fontsize=15)
plt.grid(True)
plt.legend(loc='upper right', fontsize="15")

plt.subplot(3, 2, 4)
plt.plot(t_angles, ref_angles_total[:, 1], 'b', linewidth=1, label='Theta_ref')
plt.plot(t_angles, statesTotal_mpc[:, 4], 'g', linewidth=1, label='Theta(mpc)')
plt.xlabel('t (s)', fontsize=15)
plt.ylabel('Theta (rad)', fontsize=15)
plt.grid(True)
plt.legend(loc='upper right', fontsize="15")

plt.subplot(3, 2, 6)
plt.plot(t_angles, ref_angles_total[:, 2], 'b', linewidth=1, label='Psi_ref')
plt.plot(t_angles, statesTotal_mpc[:, 5], 'g', linewidth=1, label='Psi(mpc)')
plt.xlabel('t (s)', fontsize=15)
plt.ylabel('Psi (rad)', fontsize=15)
plt.grid(True)
plt.legend(loc='upper right', fontsize="15")
plt.show()

# Plot angles errors
plt.figure().canvas.manager.set_window_title('Angle Errors')
plt.subplot(3, 2, 2)
plt.plot(t, np.zeros(126), 'b', linewidth=1)
plt.plot(t, (phi_ref_mpc - statesTotals_mpc[:, 3]), 'g', linewidth=1, label='Phi(mpc) Error')
plt.xlabel('t (s)', fontsize=15)
plt.ylabel('Rad', fontsize=15)
plt.legend(loc='upper right', fontsize="15")
plt.grid(True)

plt.subplot(3, 2, 4)
plt.plot(t, np.zeros(126), 'b', linewidth=1)
plt.plot(t, (theta_ref_mpc - statesTotals_mpc[:, 4]), 'g', linewidth=1, label='Theta(mpc) Error ')
plt.xlabel('t (s)', fontsize=15)
plt.ylabel('Rad', fontsize=15)
plt.legend(loc='upper right', fontsize="15")
plt.grid(True)

plt.subplot(3, 2, 6)
plt.plot(t, np.zeros(126), 'b', linewidth=1)
plt.plot(t, (Psi_ref_mpc[0,:] - statesTotals_mpc[:, 5]), 'g', linewidth=1, label='Psi(mpc) Error ')
plt.xlabel('t (s)', fontsize=15)
plt.ylabel('Rad', fontsize=15)
plt.legend(loc='upper right', fontsize="15")
plt.grid(True)

plt.subplot(3, 2, 1)
plt.plot(t, (phi_ref - statesTotals[:, 3]), 'r', linewidth=1, label='Phi(Proposed Method) Error')
plt.plot(t, np.zeros(126), 'b', linewidth=1)
plt.xlabel('t (s)', fontsize=15)
plt.ylabel('Rad', fontsize=15)
plt.legend(loc='upper right', fontsize="15")
plt.grid(True)

plt.subplot(3, 2, 3)
plt.plot(t, (theta_ref - statesTotals[:, 4]), 'r', linewidth=1, label='Theta(Proposed Method) Error')
plt.plot(t, np.zeros(126), 'b', linewidth=1)
plt.xlabel('t (s)', fontsize=15)
plt.ylabel('Rad', fontsize=15)
plt.legend(loc='upper right', fontsize="15")
plt.grid(True)

plt.subplot(3, 2, 5)
plt.plot(t, (Psi_ref[0,:] - statesTotals[:, 5]), 'r', linewidth=1, label='Psi(Proposed Method) Error')
plt.plot(t, np.zeros(126), 'b', linewidth=1)
plt.xlabel('t (s)', fontsize=15)
plt.ylabel('Rad', fontsize=15)
plt.legend(loc='upper right', fontsize="15")
plt.grid(True)
plt.show()

# Plot Control Inputs
plt.figure().canvas.manager.set_window_title('Control Inputs')
plt.subplot(4, 1, 1)
plt.plot(t_angles, UTotal[0:len(UTotal), 0], 'r', linewidth=1, label='U1(Proposed Method)')
plt.plot(t_angles, UTotal_mpc[0:len(UTotal_mpc), 0], 'g', linewidth=1, label='U1(mpc)')
# plt.xlabel('t (s)', fontsize=15)
plt.ylabel('U1 (N)', fontsize=15)
plt.grid(True)
plt.legend(loc='upper right', fontsize="15")

plt.subplot(4, 1, 2)
plt.plot(t_angles, UTotal[0:len(UTotal), 1], 'r', linewidth=1, label='U2(Proposed Method)')
plt.plot(t_angles, UTotal_mpc[0:len(UTotal_mpc), 1], 'g', linewidth=1, label='U2(mpc)')
# plt.xlabel('t (s)', fontsize=15)
plt.ylabel('U2 (Nm)', fontsize=15)
plt.grid(True)
plt.legend(loc='upper right', fontsize="15")

plt.subplot(4, 1, 3)
plt.plot(t_angles, UTotal[0:len(UTotal), 2], 'r', linewidth=1, label='U3(Proposed Method)')
plt.plot(t_angles, UTotal_mpc[0:len(UTotal_mpc), 2], 'g', linewidth=1, label='U3(mpc)')
# plt.xlabel('t (s)', fontsize=15)
plt.ylabel('U3 (Nm)', fontsize=15)
plt.grid(True)
plt.legend(loc='upper right', fontsize="15")

plt.subplot(4, 1, 4)
plt.plot(t_angles, UTotal[0:len(UTotal), 3], 'r', linewidth=1, label='U4(Proposed Method)')
plt.plot(t_angles, UTotal_mpc[0:len(UTotal_mpc), 3], 'g', linewidth=1, label='U4(mpc)')
plt.xlabel('t (s)', fontsize=15)
plt.ylabel('U4 (Nm)', fontsize=15)
plt.grid(True)
plt.legend(loc='upper right', fontsize="15")
plt.show()

plt.figure().canvas.manager.set_window_title('Omegas')
plt.subplot(4, 1, 1)
plt.plot(t_angles, omegas_bundle[0:len(omegas_bundle), 0], 'r', linewidth=1, label='omega 1(Proposed Method)')
plt.plot(t_angles, omegas_bundle_mpc[0:len(omegas_bundle_mpc), 0], 'g', linewidth=1, label='omega 1(mpc)')
# plt.xlabel('t (s)', fontsize=15)
plt.ylabel('Omega 1 (rad/s)', fontsize=15)
plt.grid(True)
plt.legend(loc='upper right', fontsize="15")

plt.subplot(4, 1, 2)
plt.plot(t_angles, omegas_bundle[0:len(omegas_bundle), 1], 'r', linewidth=1, label='omega 2(Proposed Method)')
plt.plot(t_angles, omegas_bundle_mpc[0:len(omegas_bundle_mpc), 1], 'g', linewidth=1, label='omega 2(mpc)')
# plt.xlabel('t (s)', fontsize=15)
plt.ylabel('Omega 2 (rad/s)', fontsize=15)
plt.grid(True)
plt.legend(loc='upper right', fontsize="15")

plt.subplot(4, 1, 3)
plt.plot(t_angles, omegas_bundle[0:len(omegas_bundle), 2], 'r', linewidth=1, label='omega 3(Proposed Method)')
plt.plot(t_angles, omegas_bundle_mpc[0:len(omegas_bundle_mpc), 2], 'g', linewidth=1, label='omega 3(mpc)')
# plt.xlabel('t (s)', fontsize=15)
plt.ylabel('Omega 3 (rad/s)', fontsize=15)
plt.grid(True)
plt.legend(loc='upper right', fontsize="15")

plt.subplot(4, 1, 4)
plt.plot(t_angles, omegas_bundle[0:len(omegas_bundle), 3], 'r', linewidth=1, label='omega 4(Proposed Method)')
plt.plot(t_angles, omegas_bundle_mpc[0:len(omegas_bundle_mpc), 3], 'g', linewidth=1, label='omega 4(mpc)')
plt.xlabel('t (s)', fontsize=15)
plt.ylabel('Omega 4 (rad/s)', fontsize=15)
plt.grid(True)
plt.legend(loc='upper right', fontsize="15")
plt.show()