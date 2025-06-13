import numpy as np

class SupportFilesDrone:
    ''' The following functions interact with the main file'''

    def __init__(self):
        ''' Load the constants that do not change'''

        # Constants
        Ix = 0.0025  # kg*m^2
        Iy = 0.0025  # kg*m^2
        Iz = 0.004  # kg*m^2
        m = 0.5  # kg
        g = 9.81  # m/s^2
        Jtp = 1.25 * 10 ** (-6)  # N*m*s^2=kg*m^2
        Ts = 0.1 # s

        # Matrix weights for the cost function (They must be diagonal)
        Q=np.matrix('1 0 0;0 1 0;0 0 1') # weights for outputs (all samples, except the last one)
        S=np.matrix('1 0 0;0 1 0;0 0 1') # weights for the final horizon period outputs
        R=np.matrix('0.75 0 0;0 0.75 0;0 0 0.75') # weights for inputs

        ct = 7.6184*10**(-8)*(60/(2*np.pi))**2 # N*s^2
        cq = 2.6839*10**(-9)*(60/(2*np.pi))**2 # N*m*s^2
        l = 0.171 # m

        controlled_states=3 # Number of attitude outputs: Phi, Theta, Psi
        hz = 4 # horizon period

        innerDyn_length=4 # Number of inner control loop iterations

        r=2
        f=0.025
        height_i=5
        height_f=25

        sub_loop=5 # for animation purposes

        # Drag force:
        drag_switch=1 # Must be either 0 or 1 (0 - drag force OFF, 1 - drag force ON)

        # Wind noise
        wind_inertial = np.array([5,0,0])       # m/s

        # Drag force coefficients [-]:
        C_D_u = 1.5
        C_D_v=1.5
        C_D_w=2.0

        # Drag force cross-section area [m^2]
        A_u=2*l*0.01+0.05**2
        A_v=2*l*0.01+0.05**2
        A_w=2*2*l*0.01+0.05**2

        # Air density
        rho = 1.225 # [kg/m^3]

        # Noise switch:
        noise_switch=0
        # Must be either 0 or 1 (0 - no noise, 1 - Control Input noise ON - 2 - Wind noise ON)

        ##########################################################################################################
        eta_1 = 1.1
        eta_2 = 0.1
        lambda1 = 0.7
        lambda2 = 1.2
        k1 = 1
        k2 = 1

        Kpx = -1
        Kdx = -2
        Kpy = -1
        Kdy = -2
        Kpz = -1
        Kdz = -2

        # Trajectory
        trajectory = 10

        # Constraints
        omega_min=110*np.pi/3 # [rad/s]
        omega_max=860*np.pi/3 # [rad/s]

        C_cm=np.matrix('1 0 0 0 0 0 0 0 0;0 1 0 0 0 0 0 0 0;0 0 1 0 0 0 0 0 0;0 0 0 1 0 0 0 0 0;0 0 0 0 1 0 0 0 0;0 0 0 0 0 1 0 0 0;0 0 0 0 0 0 1 0 0;0 0 0 0 0 0 0 1 0;0 0 0 0 0 0 0 0 1') # constraint matrix for extracting desired outputs for constraints


        # C_cm=np.matrix('0 0 0 0 0 0 1 0 0;0 0 0 0 0 0 0 1 0;0 0 0 0 0 0 0 0 1') # constraint matrix for extracting desired outputs for constraints

        # This is good
        kp_att = np.array([0.05, 0.05, 0.05])
        ki_att = np.array([0.008, 0.008, 0.008])
        kd_att = np.array([0.03, 0.03, 0.03])

        # This should be bad
        # kp_att = np.array([0.02, 0.02, 0.02])
        # ki_att = np.array([0.008, 0.008, 0.008])
        # kd_att = np.array([0.04, 0.04, 0.04])

        integral_att = np.zeros(3)

        # Derivative filter (low-pass)
        alpha = 0.1  # Filter coefficient (0 < alpha <= 1)
        prev_error_att = np.zeros(3)
        filtered_derive_att = np.zeros(3)
        U1_min = ct * 4 * omega_min**2
        U1_max = ct * 4 * omega_max**2
        U2_min = ct * l * (omega_min**2 - omega_max**2)
        U2_max = ct * l * (omega_max**2 - omega_min**2)
        U3_min = ct * l * (omega_min**2 - omega_max**2)
        U3_max = ct * l * (omega_max**2 - omega_min**2)
        U4_min = cq *(-2 * omega_max**2 + 2 * omega_min**2)
        U4_max = cq *(-2 * omega_min**2 + 2 * omega_max**2)

        Kp = -1
        Kd = -2

        self.constants={'Ix':Ix,'Iy':Iy,'Iz':Iz,'m':m,'g':g,'Jtp':Jtp,'Ts':Ts,
                        'Q':Q,'S':S,'R':R,'ct':ct,'cq':cq,'l':l,
                        'controlled_states':controlled_states,'hz':hz,
                        'innerDyn_length':innerDyn_length,
                        'r':r,'f':f,'height_i':height_i,
                        'height_f':height_f, 'sub_loop':sub_loop,
                        'drag_switch':drag_switch,'C_D_u':C_D_u,'C_D_v':C_D_v,
                        'C_D_w':C_D_w,'A_u':A_u,'A_v':A_v,'A_w':A_w,'rho':rho,
                        'trajectory':trajectory, 'noise_switch': noise_switch,
                        'kp_att': kp_att, 'ki_att': ki_att, 'kd_att': kd_att,
                        'alpha': alpha, 'integral_att': integral_att, 'prev_error_att': prev_error_att,
                        'filtered_derive_att': filtered_derive_att, 'U1_min': U1_min, 'U1_max': U1_max,
                        'U2_min': U2_min, 'U2_max': U2_max, 'U3_min': U3_min, 'U3_max': U3_max,
                        'U4_min': U4_min, 'U4_max': U4_max,
                        'omega_min':omega_min,'omega_max':omega_max,
                        'C_cm':C_cm, 'wind_inertial': wind_inertial,
                        'Kp': Kp, 'Kd': Kd,
                        #########################################
                        'eta_1': eta_1, 'eta_2': eta_2, 'lambda1': lambda1, 'lambda2': lambda2,
                        'Kpx': Kpx, 'Kdx': Kdx, 'Kpy': Kpy, 'Kdy': Kdy, 'Kpz': Kpz, 'Kdz': Kdz,
                        'k1': k1, 'k2': k2
                        }

        return None

    def compute_adaptive_gains(self, errors_pos, errors_dot_pos):
        eta_1 = self.constants['eta_1']
        eta_2 = self.constants['eta_2']
        lambda1 = self.constants['lambda1']
        lambda2 = self.constants['lambda2']
        k1 = self.constants['k1']
        k2 = self.constants['k2']

        # Apply thresholding
        x1, x2, x3 = [e if abs(e) > eta_1 else eta_1 for e in errors_pos]
        y1, y2, y3 = [e if abs(e) > eta_2 else eta_2 for e in errors_dot_pos]

        # Compute adaptive gains
        Kp_hat = np.diag([k1 * abs(x) ** (lambda1 - 1) for x in [x1, x2, x3]])
        Kd_hat = np.diag([k2 * abs(y) ** (lambda2 - 1) for y in [y1, y2, y3]])

        return Kp_hat, Kd_hat

    def trajectory_generator(self,t):
        '''This method creates the trajectory for a drone to follow'''

        r=self.constants['r']
        f=self.constants['f']
        height_i=self.constants['height_i']
        height_f=self.constants['height_f']
        trajectory=self.constants['trajectory']
        d_height=height_f-height_i

        # Define the x, y, z dimensions for the drone trajectories
        alpha=2*np.pi*f*t

        if trajectory==1 or trajectory==2 or trajectory==3 or trajectory==4:
            # Trajectory 1
            x=r*np.cos(alpha)
            y=r*np.sin(alpha)
            # z=height_i+d_height/(t[-1])*t
            z = 2*np.ones(len(t))

            x_dot=-r*np.sin(alpha)*2*np.pi*f
            y_dot=r*np.cos(alpha)*2*np.pi*f
            # z_dot=d_height/(t[-1])*np.ones(len(t))
            z_dot = 0*np.ones(len(t))

            x_dot_dot=-r*np.cos(alpha)*(2*np.pi*f)**2
            y_dot_dot=-r*np.sin(alpha)*(2*np.pi*f)**2
            # z_dot_dot=0*np.ones(len(t))
            z_dot_dot = 0*np.ones(len(t))

            if trajectory==2:
                # Trajectory 2
                # Make sure you comment everything except Trajectory 1 and this bonus trajectory
                x[101:len(x)]=2*(t[101:len(t)]-t[100])/20+x[100]
                y[101:len(y)]=2*(t[101:len(t)]-t[100])/20+y[100]
                z[101:len(z)]=z[100]+d_height/t[-1]*(t[101:len(t)]-t[100])

                x_dot[101:len(x_dot)]=1/10*np.ones(len(t[101:len(t)]))
                y_dot[101:len(y_dot)]=1/10*np.ones(len(t[101:len(t)]))
                z_dot[101:len(z_dot*(t/20))]=d_height/(t[-1])*np.ones(len(t[101:len(t)]))

                x_dot_dot[101:len(x_dot_dot)]=0*np.ones(len(t[101:len(t)]))
                y_dot_dot[101:len(y_dot_dot)]=0*np.ones(len(t[101:len(t)]))
                z_dot_dot[101:len(z_dot_dot)]=0*np.ones(len(t[101:len(t)]))

            elif trajectory==3:
                # Trajectory 3
                # Make sure you comment everything except Trajectory 1 and this bonus trajectory
                x[101:len(x)]=2*(t[101:len(t)]-t[100])/20+x[100]
                y[101:len(y)]=2*(t[101:len(t)]-t[100])/20+y[100]
                z[101:len(z)]=z[100]+d_height/t[-1]*(t[101:len(t)]-t[100])**2

                x_dot[101:len(x_dot)]=1/10*np.ones(len(t[101:len(t)]))
                y_dot[101:len(y_dot)]=1/10*np.ones(len(t[101:len(t)]))
                z_dot[101:len(z_dot)]=2*d_height/(t[-1])*(t[101:len(t)]-t[100])

                x_dot_dot[101:len(x_dot_dot)]=0*np.ones(len(t[101:len(t)]))
                y_dot_dot[101:len(y_dot_dot)]=0*np.ones(len(t[101:len(t)]))
                z_dot_dot[101:len(z_dot_dot)]=2*d_height/t[-1]*np.ones(len(t[101:len(t)]))

            elif trajectory==4:
                # Trajectory 4
                # Make sure you comment everything except Trajectory 1 and this bonus trajectory
                x[101:len(x)]=2*(t[101:len(t)]-t[100])/20+x[100]
                y[101:len(y)]=2*(t[101:len(t)]-t[100])/20+y[100]
                z[101:len(z)]=z[100]+50*d_height/t[-1]*np.sin(0.1*(t[101:len(t)]-t[100]))

                x_dot[101:len(x_dot)]=1/10*np.ones(len(t[101:len(t)]))
                y_dot[101:len(y_dot)]=1/10*np.ones(len(t[101:len(t)]))
                z_dot[101:len(z_dot)]=5*d_height/t[-1]*np.cos(0.1*(t[101:len(t)]-t[100]))

                x_dot_dot[101:len(x_dot_dot)]=0*np.ones(len(t[101:len(t)]))
                y_dot_dot[101:len(y_dot_dot)]=0*np.ones(len(t[101:len(t)]))
                z_dot_dot[101:len(z_dot_dot)]=-0.5*d_height/t[-1]*np.sin(0.1*(t[101:len(t)]-t[100]))

        elif trajectory==5 or trajectory==6:
            if trajectory==5:
                power=1
            else:
                power=2

            if power == 1:
                # Trajectory 5
                r_2=r/15
                x=(r_2*t**power+r)*np.cos(alpha)
                y=(r_2*t**power+r)*np.sin(alpha)
                z=height_i+d_height/t[-1]*t

                x_dot=r_2*np.cos(alpha)-(r_2*t+r)*np.sin(alpha)*2*np.pi*f
                y_dot=r_2*np.sin(alpha)+(r_2*t+r)*np.cos(alpha)*2*np.pi*f
                z_dot=d_height/(t[-1])*np.ones(len(t))

                x_dot_dot=-r_2*np.sin(alpha)*4*np.pi*f-(r_2*t+r)*np.cos(alpha)*(2*np.pi*f)**2
                y_dot_dot=r_2*np.cos(alpha)*4*np.pi*f-(r_2*t+r)*np.sin(alpha)*(2*np.pi*f)**2
                z_dot_dot=0*np.ones(len(t))
            else:
                # Trajectory 6
                r_2=r/500
                x=(r_2*t**power+r)*np.cos(alpha)
                y=(r_2*t**power+r)*np.sin(alpha)
                z=height_i+d_height/t[-1]*t

                x_dot=power*r_2*t**(power-1)*np.cos(alpha)-(r_2*t**(power)+r)*np.sin(alpha)*2*np.pi*f
                y_dot=power*r_2*t**(power-1)*np.sin(alpha)+(r_2*t**(power)+r)*np.cos(alpha)*2*np.pi*f
                z_dot=d_height/(t[-1])*np.ones(len(t))

                x_dot_dot=(power*(power-1)*r_2*t**(power-2)*np.cos(alpha)-power*r_2*t**(power-1)*np.sin(alpha)*2*np.pi*f)-\
                            (power*r_2*t**(power-1)*np.sin(alpha)*2*np.pi*f+(r_2*t**power+r_2)*np.cos(alpha)*(2*np.pi*f)**2)
                y_dot_dot=(power*(power-1)*r_2*t**(power-2)*np.sin(alpha)+power*r_2*t**(power-1)*np.cos(alpha)*2*np.pi*f)+\
                            (power*r_2*t**(power-1)*np.cos(alpha)*2*np.pi*f-(r_2*t**power+r_2)*np.sin(alpha)*(2*np.pi*f)**2)
                z_dot_dot=0*np.ones(len(t))

        elif trajectory==7:
            # Trajectory 7
            x=2*t/20
            y=2*t/20
            # z=height_i+d_height/t[-1]*t
            z = 2*np.ones(len(t))

            x_dot=1/10*np.ones(len(t))
            y_dot=1/10*np.ones(len(t))
            # z_dot=d_height/(t[-1])*np.ones(len(t))
            z_dot = 0*np.ones(len(t))

            x_dot_dot=0*np.ones(len(t))
            y_dot_dot=0*np.ones(len(t))
            z_dot_dot=0*np.ones(len(t))

        elif trajectory==8:
            # Trajectory 8
            x=r/5*np.sin(alpha)+t/100
            y=t/100-1
            z=height_i+d_height/t[-1]*t

            x_dot=r/5*np.cos(alpha)*2*np.pi*f+1/100
            y_dot=1/100*np.ones(len(t))
            z_dot=d_height/(t[-1])*np.ones(len(t))

            x_dot_dot=-r/5*np.sin(alpha)*(2*np.pi*f)**2
            y_dot_dot=0*np.ones(len(t))
            z_dot_dot=0*np.ones(len(t))

        elif trajectory==9:
            # Trajectory 9
            wave_w=1
            x=r*np.cos(alpha)
            y=r*np.sin(alpha)
            z=height_i+7*d_height/t[-1]*np.sin(wave_w*t)

            x_dot=-r*np.sin(alpha)*2*np.pi*f
            y_dot=r*np.cos(alpha)*2*np.pi*f
            z_dot=7*d_height/(t[-1])*np.cos(wave_w*t)*wave_w

            x_dot_dot=-r*np.cos(alpha)*(2*np.pi*f)**2
            y_dot_dot=-r*np.sin(alpha)*(2*np.pi*f)**2
            z_dot_dot=-7*d_height/(t[-1])*np.sin(wave_w*t)*wave_w**2

        elif trajectory==10:
            # Figure 8 parameters
            a = 4.0  # Width of the figure-8
            b = 2.0  # Height of the figure-8
            period = 30.0  # Time to complete one full figure-8

            # Angular frequency
            w = 2 * np.pi / period

            # Position
            x = a * np.sin(w * t)  # X position
            y = b * np.sin(2 * w * t)  # Y position
            z = 1.0 * np.ones(len(t))  # Constant height

            # Velocity (derivatives)
            x_dot = a * w * np.cos(w * t)
            y_dot = 2 * b * w * np.cos(2 * w * t)
            z_dot = np.zeros(len(t))

            # Acceleration (second derivatives)
            x_dot_dot = -a * w ** 2 * np.sin(w * t)
            y_dot_dot = -4 * b * w ** 2 * np.sin(2 * w * t)
            z_dot_dot = np.zeros(len(t))

        elif trajectory==11:
            # Trajectory 11 - Hover at a point
            waypoints = np.array([[5, 5, 2]])  # Single hover point

            # Initialize arrays 
            x = np.ones(len(t)) * 9  # Constant x position
            y = np.ones(len(t)) * 9  # Constant y position
            z = np.ones(len(t)) * 9  # Constant z position
            x_dot = np.zeros(len(t))  # Zero velocity
            y_dot = np.zeros(len(t))
            z_dot = np.zeros(len(t))
            x_dot_dot = np.zeros(len(t))  # Zero acceleration
            y_dot_dot = np.zeros(len(t))
            z_dot_dot = np.zeros(len(t))
        else:
            exit()

        # Vector of x and y changes per sample time
        dx=x[1:len(x)]-x[0:len(x)-1]
        dy=y[1:len(y)]-y[0:len(y)-1]
        dz=z[1:len(z)]-z[0:len(z)-1]

        dx=np.append(np.array(dx[0]),dx)
        dy=np.append(np.array(dy[0]),dy)
        dz=np.append(np.array(dz[0]),dz)

        # Define the reference yaw angles
        psi=np.zeros(len(x))
        psiInt=psi
        psi[0]=np.arctan2(y[0],x[0])+np.pi/2
        psi[1:len(psi)]=np.arctan2(dy[1:len(dy)],dx[1:len(dx)])

        # We want the yaw angle to keep track the amount of rotations
        dpsi=psi[1:len(psi)]-psi[0:len(psi)-1]
        psiInt[0]=psi[0]
        for i in range(1,len(psiInt)):
            if dpsi[i-1]<-np.pi:
                psiInt[i]=psiInt[i-1]+(dpsi[i-1]+2*np.pi)
            elif dpsi[i-1]>np.pi:
                psiInt[i]=psiInt[i-1]+(dpsi[i-1]-2*np.pi)
            else:
                psiInt[i]=psiInt[i-1]+dpsi[i-1]

        return x, x_dot, x_dot_dot, y, y_dot, y_dot_dot, z, z_dot, z_dot_dot, psiInt

    def adaptive_pos_controller(self,X_ref,X_dot_ref,X_dot_dot_ref,Y_ref,Y_dot_ref,Y_dot_dot_ref,Z_ref,Z_dot_ref,Z_dot_dot_ref,Psi_ref,states):
        '''This function is a position controller - it computes the necessary U1 for the open loop system, and phi & theta angles for the MPC controller'''

        # Load the constants
        m=self.constants['m']
        g=self.constants['g']

        # Assign the states
        # States: [u,v,w,p,q,r,x,y,z,phi,theta,psi]
        x = states[0]
        y = states[1]
        z = states[2]
        phi = states[3]
        theta = states[4]
        psi = states[5]
        u = states[6]
        v = states[7]
        w = states[8]

        # Rotational matrix that relates u,v,w with x_dot,y_dot,z_dot
        R_x=np.array([[1, 0, 0],[0, np.cos(phi), -np.sin(phi)],[0, np.sin(phi), np.cos(phi)]])
        R_y=np.array([[np.cos(theta),0,np.sin(theta)],[0,1,0],[-np.sin(theta),0,np.cos(theta)]])
        R_z=np.array([[np.cos(psi),-np.sin(psi),0],[np.sin(psi),np.cos(psi),0],[0,0,1]])
        R_matrix=np.matmul(R_z,np.matmul(R_y,R_x))
        pos_vel_body=np.array([[u],[v],[w]])
        pos_vel_fixed=np.matmul(R_matrix,pos_vel_body)
        x_dot=pos_vel_fixed[0]
        y_dot=pos_vel_fixed[1]
        z_dot=pos_vel_fixed[2]

        # Compute the errors
        ex=X_ref-x
        ex_dot=X_dot_ref-x_dot
        ey=Y_ref-y
        ey_dot=Y_dot_ref-y_dot
        ez=Z_ref-z
        ez_dot=Z_dot_ref-z_dot

        Kp_hat, Kd_hat = self.compute_adaptive_gains([ex, ey, ez], [ex_dot[0], ey_dot[0], ez_dot[0]])

        Kpx = self.constants['Kpx'] * np.diag(Kp_hat)
        Kdx = self.constants['Kdx'] * np.diag(Kd_hat)
        Kpy = self.constants['Kpy'] * np.diag(Kp_hat)
        Kdy = self.constants['Kdy'] * np.diag(Kd_hat)
        Kpz = self.constants['Kpz'] * np.diag(Kp_hat)
        Kdz = self.constants['Kdz'] * np.diag(Kd_hat)

        # Compute the values vx, vy, vz for the position controller
        ux=Kpx*ex+Kdx*ex_dot
        uy=Kpy*ey+Kdy*ey_dot
        uz=Kpz*ez+Kdz*ez_dot

        vx=X_dot_dot_ref-ux[0]
        vy=Y_dot_dot_ref-uy[0]
        vz=Z_dot_dot_ref-uz[0]

        # Compute phi, theta, U1
        a=vx/(vz+g)
        b=vy/(vz+g)
        c=np.cos(Psi_ref)
        d=np.sin(Psi_ref)
        tan_theta=a*c+b*d
        Theta_ref=np.arctan(tan_theta)

        if Psi_ref>=0:
            Psi_ref_singularity=Psi_ref-np.floor(abs(Psi_ref)/(2*np.pi))*2*np.pi
        else:
            Psi_ref_singularity=Psi_ref+np.floor(abs(Psi_ref)/(2*np.pi))*2*np.pi

        if ((np.abs(Psi_ref_singularity)<np.pi/4 or np.abs(Psi_ref_singularity)>7*np.pi/4) or \
            (np.abs(Psi_ref_singularity)>3*np.pi/4 and np.abs(Psi_ref_singularity)<5*np.pi/4)):
            tan_phi=np.cos(Theta_ref)*(np.tan(Theta_ref)*d-b)/c
        else:
            tan_phi=np.cos(Theta_ref)*(a-np.tan(Theta_ref)*c)/d
        Phi_ref=np.arctan(tan_phi)
        U1=(vz+g)*m/(np.cos(Phi_ref)*np.cos(Theta_ref))
        print(Phi_ref,Theta_ref,U1)
        return Phi_ref, Theta_ref, U1

    def pos_controller(self,X_ref,X_dot_ref,X_dot_dot_ref,Y_ref,Y_dot_ref,Y_dot_dot_ref,Z_ref,Z_dot_ref,Z_dot_dot_ref,Psi_ref,states):
        '''This function is a position controller - it computes the necessary U1 for the open loop system, and phi & theta angles for the MPC controller'''

        # Load the constants
        m=self.constants['m']
        g=self.constants['g']
        kx1 = self.constants['Kp']
        kx2 = self.constants['Kd']
        ky1 = self.constants['Kp']
        ky2 = self.constants['Kd']
        kz1 = self.constants['Kp']
        kz2 = self.constants['Kd']

        # Assign the states
        # States: [u,v,w,p,q,r,x,y,z,phi,theta,psi]
        x = states[0]
        y = states[1]
        z = states[2]
        phi = states[3]
        theta = states[4]
        psi = states[5]
        u = states[6]
        v = states[7]
        w = states[8]

        # Rotational matrix that relates u,v,w with x_dot,y_dot,z_dot
        R_x=np.array([[1, 0, 0],[0, np.cos(phi), -np.sin(phi)],[0, np.sin(phi), np.cos(phi)]])
        R_y=np.array([[np.cos(theta),0,np.sin(theta)],[0,1,0],[-np.sin(theta),0,np.cos(theta)]])
        R_z=np.array([[np.cos(psi),-np.sin(psi),0],[np.sin(psi),np.cos(psi),0],[0,0,1]])
        R_matrix=np.matmul(R_z,np.matmul(R_y,R_x))
        pos_vel_body=np.array([[u],[v],[w]])
        pos_vel_fixed=np.matmul(R_matrix,pos_vel_body)
        x_dot=pos_vel_fixed[0]
        y_dot=pos_vel_fixed[1]
        z_dot=pos_vel_fixed[2]

        # Compute the errors
        ex=X_ref-x
        ex_dot=X_dot_ref-x_dot
        ey=Y_ref-y
        ey_dot=Y_dot_ref-y_dot
        ez=Z_ref-z
        ez_dot=Z_dot_ref-z_dot

        # Compute the values vx, vy, vz for the position controller
        ux=kx1*ex+kx2*ex_dot
        uy=ky1*ey+ky2*ey_dot
        uz=kz1*ez+kz2*ez_dot

        vx=X_dot_dot_ref-ux[0]
        vy=Y_dot_dot_ref-uy[0]
        vz=Z_dot_dot_ref-uz[0]
        # print(vx,vy,vz)
        # Compute phi, theta, U1
        a=vx/(vz+g)
        b=vy/(vz+g)
        c=np.cos(Psi_ref)
        d=np.sin(Psi_ref)
        tan_theta=a*c+b*d
        Theta_ref=np.arctan(tan_theta)

        if Psi_ref>=0:
            Psi_ref_singularity=Psi_ref-np.floor(abs(Psi_ref)/(2*np.pi))*2*np.pi
        else:
            Psi_ref_singularity=Psi_ref+np.floor(abs(Psi_ref)/(2*np.pi))*2*np.pi

        if ((np.abs(Psi_ref_singularity)<np.pi/4 or np.abs(Psi_ref_singularity)>7*np.pi/4) or \
            (np.abs(Psi_ref_singularity)>3*np.pi/4 and np.abs(Psi_ref_singularity)<5*np.pi/4)):
            tan_phi=np.cos(Theta_ref)*(np.tan(Theta_ref)*d-b)/c
        else:
            tan_phi=np.cos(Theta_ref)*(a-np.tan(Theta_ref)*c)/d
        Phi_ref=np.arctan(tan_phi)
        U1=(vz+g)*m/(np.cos(Phi_ref)*np.cos(Theta_ref))
        print(Phi_ref,Theta_ref,U1)
        return Phi_ref, Theta_ref, U1

    def LPV_cont_discrete(self,states,omega_total):
        '''This is an LPV model concerning the three rotational axis.'''

        # Get the necessary constants
        Ix=self.constants['Ix'] # kg*m^2
        Iy=self.constants['Iy'] # kg*m^2
        Iz=self.constants['Iz'] # kg*m^2
        Jtp=self.constants['Jtp'] #N*m*s^2=kg*m^2
        Ts=self.constants['Ts'] #s

        # Assign the states
        # States: [x,y,z,phi,theta,psi,u,v,w,p,q,r]
        phi=states[3]
        theta=states[4]
        psi=states[5]
        u=states[6]
        v=states[7]
        w=states[8]
        p=states[9]
        q=states[10]
        r=states[11]

        # Rotational matrix that relates u,v,w with x_dot,y_dot,z_dot
        R_x=np.array([[1, 0, 0],[0, np.cos(phi), -np.sin(phi)],[0, np.sin(phi), np.cos(phi)]])
        R_y=np.array([[np.cos(theta),0,np.sin(theta)],[0,1,0],[-np.sin(theta),0,np.cos(theta)]])
        R_z=np.array([[np.cos(psi),-np.sin(psi),0],[np.sin(psi),np.cos(psi),0],[0,0,1]])
        R_matrix=np.matmul(R_z,np.matmul(R_y,R_x))
        pos_vel_body=np.array([[u],[v],[w]])
        pos_vel_fixed=np.matmul(R_matrix,pos_vel_body)
        x_dot=pos_vel_fixed[0]
        y_dot=pos_vel_fixed[1]
        z_dot=pos_vel_fixed[2]
        x_dot=x_dot[0]
        y_dot=y_dot[0]
        z_dot=z_dot[0]

        # To get phi_dot, theta_dot, psi_dot, you need the T matrix
        # Transformation matrix that relates p,q,r with phi_dot,theta_dot,psi_dot
        T_matrix=np.array([[1,np.sin(phi)*np.tan(theta),np.cos(phi)*np.tan(theta)],\
            [0,np.cos(phi),-np.sin(phi)],\
            [0,np.sin(phi)/np.cos(theta),np.cos(phi)/np.cos(theta)]])
        rot_vel_body=np.array([[p],[q],[r]])
        rot_vel_fixed=np.matmul(T_matrix,rot_vel_body)
        phi_dot=rot_vel_fixed[0]
        theta_dot=rot_vel_fixed[1]
        psi_dot=rot_vel_fixed[2]
        phi_dot=phi_dot[0]
        theta_dot=theta_dot[0]
        psi_dot=psi_dot[0]

        # Create the continuous LPV A, B, C, D matrices
        A01=1
        A13=-omega_total*Jtp/Ix
        A15=theta_dot*(Iy-Iz)/Ix
        A23=1
        A31=omega_total*Jtp/Iy
        A35=phi_dot*(Iz-Ix)/Iy
        A45=1
        A51=(theta_dot/2)*(Ix-Iy)/Iz
        A53=(phi_dot/2)*(Ix-Iy)/Iz

        A=np.zeros((6,6))
        B=np.zeros((6,3))
        C=np.zeros((3,6))
        D=0

        A[0,1]=A01
        A[1,3]=A13
        A[1,5]=A15
        A[2,3]=A23
        A[3,1]=A31
        A[3,5]=A35
        A[4,5]=A45
        A[5,1]=A51
        A[5,3]=A53

        B[1,0]=1/Ix
        B[3,1]=1/Iy
        B[5,2]=1/Iz

        C[0,0]=1
        C[1,2]=1
        C[2,4]=1

        D=np.zeros((3,3))

        # Discretize the system (Forward Euler)
        Ad=np.identity(np.size(A,1))+Ts*A
        Bd=Ts*B
        Cd=C
        Dd=D

        return Ad,Bd,Cd,Dd,x_dot,y_dot,z_dot,phi,phi_dot,theta,theta_dot,psi,psi_dot

    def mpc_simplification(self, Ad, Bd, Cd, Dd, hz, y_max, y_min):
        '''This function creates the compact matrices for Model Predictive Control'''
        # db - double bar
        # dbt - double bar transpose
        # dc - double circumflex
        A_aug=np.concatenate((Ad,Bd),axis=1)
        temp1=np.zeros((np.size(Bd,1),np.size(Ad,1)))
        temp2=np.identity(np.size(Bd,1))
        temp=np.concatenate((temp1,temp2),axis=1)

        A_aug=np.concatenate((A_aug,temp),axis=0)
        B_aug=np.concatenate((Bd,np.identity(np.size(Bd,1))),axis=0)
        C_aug=np.concatenate((Cd,np.zeros((np.size(Cd,0),np.size(Bd,1)))),axis=1)
        D_aug=Dd


        Q=self.constants['Q']
        S=self.constants['S']
        R=self.constants['R']

        C_cm=self.constants['C_cm']
        C_cm_g=np.zeros((np.size(C_cm,0)*hz,np.size(C_cm,1)*hz))
        y_max_global=np.zeros((np.size(y_max,0)*hz,np.size(y_max,1)))
        y_min_global=np.zeros((np.size(y_min,0)*hz,np.size(y_min,1)))

        CQC=np.matmul(np.transpose(C_aug),Q)
        CQC=np.matmul(CQC,C_aug)

        CSC=np.matmul(np.transpose(C_aug),S)
        CSC=np.matmul(CSC,C_aug)

        QC=np.matmul(Q,C_aug)
        SC=np.matmul(S,C_aug)


        Qdb=np.zeros((np.size(CQC,0)*hz,np.size(CQC,1)*hz))
        Tdb=np.zeros((np.size(QC,0)*hz,np.size(QC,1)*hz))
        Rdb=np.zeros((np.size(R,0)*hz,np.size(R,1)*hz))
        Cdb=np.zeros((np.size(B_aug,0)*hz,np.size(B_aug,1)*hz))
        Adc=np.zeros((np.size(A_aug,0)*hz,np.size(A_aug,1)))

        for i in range(0,hz):
            if i == hz-1:
                Qdb[np.size(CSC,0)*i:np.size(CSC,0)*i+CSC.shape[0],np.size(CSC,1)*i:np.size(CSC,1)*i+CSC.shape[1]]=CSC
                Tdb[np.size(SC,0)*i:np.size(SC,0)*i+SC.shape[0],np.size(SC,1)*i:np.size(SC,1)*i+SC.shape[1]]=SC

                C_cm_g[np.size(C_cm,0)*i:np.size(C_cm,0)*i+C_cm.shape[0],np.size(C_cm,1)*i:np.size(C_cm,1)*i+C_cm.shape[1]]=C_cm
            else:
                Qdb[np.size(CQC,0)*i:np.size(CQC,0)*i+CQC.shape[0],np.size(CQC,1)*i:np.size(CQC,1)*i+CQC.shape[1]]=CQC
                Tdb[np.size(QC,0)*i:np.size(QC,0)*i+QC.shape[0],np.size(QC,1)*i:np.size(QC,1)*i+QC.shape[1]]=QC
                C_cm_g[np.size(C_cm,0)*i:np.size(C_cm,0)*i+C_cm.shape[0],np.size(C_cm,1)*i:np.size(C_cm,1)*i+C_cm.shape[1]]=C_cm

            Rdb[np.size(R,0)*i:np.size(R,0)*i+R.shape[0],np.size(R,1)*i:np.size(R,1)*i+R.shape[1]]=R

            for j in range(0,hz):
                if j<=i:
                    Cdb[np.size(B_aug,0)*i:np.size(B_aug,0)*i+B_aug.shape[0],np.size(B_aug,1)*j:np.size(B_aug,1)*j+B_aug.shape[1]]=\
                    np.matmul(np.linalg.matrix_power(A_aug,((i+1)-(j+1))),B_aug)

            Adc[np.size(A_aug,0)*i:np.size(A_aug,0)*i+A_aug.shape[0],0:0+A_aug.shape[1]]=np.linalg.matrix_power(A_aug,i+1)
            y_max_global[np.size(y_max,0)*i:np.size(y_max,0)*(i+1),0]=y_max[:,0]
            y_min_global[np.size(y_min,0)*i:np.size(y_min,0)*(i+1),0]=y_min[:,0]

        Hdb=np.matmul(np.transpose(Cdb),Qdb)
        Hdb=np.matmul(Hdb,Cdb)+Rdb

        temp=np.matmul(np.transpose(Adc),Qdb)
        temp=np.matmul(temp,Cdb)

        temp2=np.matmul(-Tdb,Cdb)
        Fdbt=np.concatenate((temp,temp2),axis=0)

        return Hdb,Fdbt,Cdb,Adc,C_cm_g,y_max_global,y_min_global

    def pid_controller(self, phi_ref, theta_ref, psi_ref, states):
        # --- Extract Attitude States ---
        phi = states[3]
        theta = states[4]
        psi = states[5]

        # print(phi_ref, theta_ref, psi_ref)

        Ts = self.constants['Ts']

        kp = self.constants['kp_att']
        ki = self.constants['ki_att']
        kd = self.constants['kd_att']

        prev_error_att = self.constants['prev_error_att']

        integral_att = self.constants['integral_att']

        U2_min = self.constants['U2_min']
        U2_max = self.constants['U2_max']
        U3_min = self.constants['U3_min']
        U3_max = self.constants['U3_max']
        U4_min = self.constants['U4_min']
        U4_max = self.constants['U4_max']

        # --- Attitude PID: Compute Errors ---
        errors_att = np.array([phi_ref - phi, theta_ref - theta, psi_ref - psi])

        # --- Update Integral Term ---
        integral_att += errors_att

        # --- Compute Filtered Derivative ---
        derive_att = (errors_att - prev_error_att) / Ts
        self.constants['prev_error_att'] = errors_att.copy()

        # --- PID Control: Compute Torques ---
        U2 = (kp[0] * errors_att[0] +
              ki[0] * integral_att[0] +
              kd[0] * derive_att[0])
        U3 = (kp[1] * errors_att[1] +
              ki[1] * integral_att[1] +
              kd[1] * derive_att[1])
        U4 = (kp[2] * errors_att[2] +
              ki[2] * integral_att[2] +
              kd[2] * derive_att[2])

        # --- Apply Constraints on Torques ---
        U2 = np.clip(U2, U2_min, U2_max)
        U3 = np.clip(U3, U3_min, U3_max)
        U4 = np.clip(U4, U4_min, U4_max)

        return U2, U3, U4

    def open_loop_new_states(self,states,omega_total,U1,U2,U3,U4):
        '''This function computes the new state vector for one sample time later'''

        # Get the necessary constants
        Ix=self.constants['Ix']
        Iy=self.constants['Iy']
        Iz=self.constants['Iz']
        m=self.constants['m']
        g=self.constants['g']
        Jtp=self.constants['Jtp']
        Ts=self.constants['Ts']
        noise_switch = self.constants['noise_switch']
        wind_inertial = self.constants['wind_inertial']

        # States: [x,y,z,phi,theta,psi,u,v,w,p,q,r]
        current_states=states
        new_states=current_states
        x = current_states[0]
        y = current_states[1]
        z = current_states[2]
        phi = current_states[3]
        theta = current_states[4]
        psi = current_states[5]
        u = current_states[6]
        v = current_states[7]
        w = current_states[8]
        p = current_states[9]
        q = current_states[10]
        r = current_states[11]
        sub_loop=self.constants['sub_loop']  #Chop Ts into 5 pieces
        states_ani=np.zeros((sub_loop,6))
        U_ani=np.zeros((sub_loop,4))

        # Drag force:
        drag_switch=self.constants['drag_switch']
        C_D_u=self.constants['C_D_u']
        C_D_v=self.constants['C_D_v']
        C_D_w=self.constants['C_D_w']
        A_u=self.constants['A_u']
        A_v=self.constants['A_v']
        A_w=self.constants['A_w']
        rho=self.constants['rho']

        # Runge-Kutta method
        x_or=x
        y_or=y
        z_or=z
        phi_or=phi
        theta_or=theta
        psi_or=psi
        u_or=u
        v_or=v
        w_or=w
        p_or=p
        q_or=q
        r_or=r

        Ts_pos=2

        for j in range(0,4):
            # Get the states in the inertial frame
            # Rotational matrix that relates u,v,w with x_dot,y_dot,z_dot
            R_x = np.array([[1, 0, 0], [0, np.cos(phi), -np.sin(phi)], [0, np.sin(phi), np.cos(phi)]])
            R_y = np.array([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]])
            R_z = np.array([[np.cos(psi), -np.sin(psi), 0], [np.sin(psi), np.cos(psi), 0], [0, 0, 1]])
            R_matrix = np.matmul(R_z, np.matmul(R_y, R_x))

            # Transform wind velocity from inertial to body frame
            if noise_switch == 2:
                wind_body = np.matmul(np.linalg.inv(R_matrix), wind_inertial)
                wind_u, wind_v, wind_w = wind_body[0], wind_body[1], wind_body[2]
            else:
                wind_u, wind_v, wind_w = 0.0, 0.0, 0.0  # No wind if noise_switch is off

            if drag_switch==1:
                Fd_u=0.5*C_D_u*rho*u**2*A_u
                Fd_v=0.5*C_D_v*rho*v**2*A_v
                Fd_w=0.5*C_D_w*rho*w**2*A_w
            elif drag_switch==0:
                Fd_u=0
                Fd_v=0
                Fd_w=0
            else:
                print("drag_switch variable must be either 0 or 1 in the init function")
                exit()

            # Compute wind-induced forces based on relative velocity
            Fw_u = 0.5 * C_D_u * rho * (u - wind_u) ** 2 * A_u * np.sign(u - wind_u)
            Fw_v = 0.5 * C_D_v * rho * (v - wind_v) ** 2 * A_v * np.sign(v - wind_v)
            Fw_w = 0.5 * C_D_w * rho * (w - wind_w) ** 2 * A_w * np.sign(w - wind_w)

            # Compute slopes k_x
            u_dot=(v*r-w*q)+g*np.sin(theta)-(Fd_u+Fw_u)/m
            v_dot=(w*p-u*r)-g*np.cos(theta)*np.sin(phi)-(Fd_v+Fw_v)/m
            w_dot=(u*q-v*p)-g*np.cos(theta)*np.cos(phi)+U1/m-(Fd_w+Fw_w)/m
            p_dot=q*r*(Iy-Iz)/Ix-Jtp/Ix*q*omega_total+U2/Ix
            q_dot=p*r*(Iz-Ix)/Iy+Jtp/Iy*p*omega_total+U3/Iy
            r_dot=p*q*(Ix-Iy)/Iz+U4/Iz

            pos_vel_body=np.array([[u],[v],[w]])
            pos_vel_fixed=np.matmul(R_matrix,pos_vel_body)
            x_dot=pos_vel_fixed[0]
            y_dot=pos_vel_fixed[1]
            z_dot=pos_vel_fixed[2]
            x_dot=x_dot[0]
            y_dot=y_dot[0]
            z_dot=z_dot[0]

            # To get phi_dot, theta_dot, psi_dot, you need the T matrix
            # Transformation matrix that relates p,q,r with phi_dot,theta_dot,psi_dot
            T_matrix=np.array([[1,np.sin(phi)*np.tan(theta),np.cos(phi)*np.tan(theta)],\
                [0,np.cos(phi),-np.sin(phi)],\
                [0,np.sin(phi)/np.cos(theta),np.cos(phi)/np.cos(theta)]])
            rot_vel_body=np.array([[p],[q],[r]])
            rot_vel_fixed=np.matmul(T_matrix,rot_vel_body)
            phi_dot=rot_vel_fixed[0]
            theta_dot=rot_vel_fixed[1]
            psi_dot=rot_vel_fixed[2]
            phi_dot=phi_dot[0]
            theta_dot=theta_dot[0]
            psi_dot=psi_dot[0]

            # Save the slopes:
            if j == 0:
                x_dot_k1=x_dot
                y_dot_k1=y_dot
                z_dot_k1=z_dot
                phi_dot_k1=phi_dot
                theta_dot_k1=theta_dot
                psi_dot_k1=psi_dot
                u_dot_k1=u_dot
                v_dot_k1=v_dot
                w_dot_k1=w_dot
                p_dot_k1=p_dot
                q_dot_k1=q_dot
                r_dot_k1=r_dot
            elif j == 1:
                x_dot_k2=x_dot
                y_dot_k2=y_dot
                z_dot_k2=z_dot
                phi_dot_k2=phi_dot
                theta_dot_k2=theta_dot
                psi_dot_k2=psi_dot
                u_dot_k2=u_dot
                v_dot_k2=v_dot
                w_dot_k2=w_dot
                p_dot_k2=p_dot
                q_dot_k2=q_dot
                r_dot_k2=r_dot
            elif j == 2:
                x_dot_k3=x_dot
                y_dot_k3=y_dot
                z_dot_k3=z_dot
                phi_dot_k3=phi_dot
                theta_dot_k3=theta_dot
                psi_dot_k3=psi_dot
                u_dot_k3=u_dot
                v_dot_k3=v_dot
                w_dot_k3=w_dot
                p_dot_k3=p_dot
                q_dot_k3=q_dot
                r_dot_k3=r_dot

                Ts_pos=1
            else:
                x_dot_k4=x_dot
                y_dot_k4=y_dot
                z_dot_k4=z_dot
                phi_dot_k4=phi_dot
                theta_dot_k4=theta_dot
                psi_dot_k4=psi_dot
                u_dot_k4=u_dot
                v_dot_k4=v_dot
                w_dot_k4=w_dot
                p_dot_k4=p_dot
                q_dot_k4=q_dot
                r_dot_k4=r_dot

            if j<3:
                # New states using k_x
                x=x_or+x_dot*Ts/Ts_pos
                y=y_or+y_dot*Ts/Ts_pos
                z=z_or+z_dot*Ts/Ts_pos
                phi=phi_or+phi_dot*Ts/Ts_pos
                theta=theta_or+theta_dot*Ts/Ts_pos
                psi=psi_or+psi_dot*Ts/Ts_pos
                u=u_or+u_dot*Ts/Ts_pos
                v=v_or+v_dot*Ts/Ts_pos
                w=w_or+w_dot*Ts/Ts_pos
                p=p_or+p_dot*Ts/Ts_pos
                q=q_or+q_dot*Ts/Ts_pos
                r=r_or+r_dot*Ts/Ts_pos
            else:
                # New states using average k_x
                x=x_or+1/6*(x_dot_k1+2*x_dot_k2+2*x_dot_k3+x_dot_k4)*Ts
                y=y_or+1/6*(y_dot_k1+2*y_dot_k2+2*y_dot_k3+y_dot_k4)*Ts
                z=z_or+1/6*(z_dot_k1+2*z_dot_k2+2*z_dot_k3+z_dot_k4)*Ts
                phi=phi_or+1/6*(phi_dot_k1+2*phi_dot_k2+2*phi_dot_k3+phi_dot_k4)*Ts
                theta=theta_or+1/6*(theta_dot_k1+2*theta_dot_k2+2*theta_dot_k3+theta_dot_k4)*Ts
                psi=psi_or+1/6*(psi_dot_k1+2*psi_dot_k2+2*psi_dot_k3+psi_dot_k4)*Ts
                u=u_or+1/6*(u_dot_k1+2*u_dot_k2+2*u_dot_k3+u_dot_k4)*Ts
                v=v_or+1/6*(v_dot_k1+2*v_dot_k2+2*v_dot_k3+v_dot_k4)*Ts
                w=w_or+1/6*(w_dot_k1+2*w_dot_k2+2*w_dot_k3+w_dot_k4)*Ts
                p=p_or+1/6*(p_dot_k1+2*p_dot_k2+2*p_dot_k3+p_dot_k4)*Ts
                q=q_or+1/6*(q_dot_k1+2*q_dot_k2+2*q_dot_k3+q_dot_k4)*Ts
                r=r_or+1/6*(r_dot_k1+2*r_dot_k2+2*r_dot_k3+r_dot_k4)*Ts

                print(phi*(np.pi/180), theta*(np.pi/180), psi*(np.pi/180))

        for k in range(0,sub_loop):
            states_ani[k,0]=x_or+(x-x_or)/Ts*(k/(sub_loop-1))*Ts
            states_ani[k,1]=y_or+(y-y_or)/Ts*(k/(sub_loop-1))*Ts
            states_ani[k,2]=z_or+(z-z_or)/Ts*(k/(sub_loop-1))*Ts
            states_ani[k,3]=phi_or+(phi-phi_or)/Ts*(k/(sub_loop-1))*Ts
            states_ani[k,4]=theta_or+(theta-theta_or)/Ts*(k/(sub_loop-1))*Ts
            states_ani[k,5]=psi_or+(psi-psi_or)/Ts*(k/(sub_loop-1))*Ts

        U_ani[:,0]=U1
        U_ani[:,1]=U2
        U_ani[:,2]=U3
        U_ani[:,3]=U4

        # End of Runge-Kutta method

        # Take the last states
        new_states[0] = x
        new_states[1] = y
        new_states[2] = z
        new_states[3] = phi
        new_states[4] = theta
        new_states[5] = psi
        new_states[6] = u
        new_states[7] = v
        new_states[8] = w
        new_states[9] = p
        new_states[10] = q
        new_states[11] = r

        return new_states, states_ani, U_ani
