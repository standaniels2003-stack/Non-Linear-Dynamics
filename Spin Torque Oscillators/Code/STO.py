import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.integrate import solve_ivp

# --- CONSTANTS & SYSTEM PARAMETERS ---
gamma = 1.9e11     # Gyromagnetic ratio (Hz/T) [cite: 380]
alpha = 1e-2       # Gilbert damping parameter [cite: 380]
beta = 10.0 / 3.0  # Spin torque material parameter [cite: 380]
H_a = 0.2          # External magnetic field (T) 
H_k = 0.05         # Uniaxial anisotropy field (T) 
H_dz = 1.6         # Demagnetization field (T) 

def get_UV(J_dc, J_ac=0.0):
    """Calculates parameters U and V based on system equations."""
    U = alpha * H_a + beta * (J_dc + J_ac)
    V = H_a - alpha * beta * (J_dc + J_ac)
    return U, V

def sto_derivatives(tau, state, J_dc, J_ac=0.0):
    """Computes the explicit derivatives dtheta/dtau and dphi/dtau."""
    theta, phi = state
    
    # Avoid numerical singularities at the sphere's poles [cite: 388]
    if sin_theta := np.sin(theta):
        pass
    else:
        sin_theta = 1e-12
        
    cos_theta = np.cos(theta)
    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)
    
    U, V = get_UV(J_dc, J_ac)
    anisotropy_term = H_dz + H_k * (cos_phi**2)
    
    # Equation 19: dtheta/dtau [cite: 387]
    dtheta = (U * cos_theta * cos_phi 
              + alpha * anisotropy_term * sin_theta * cos_theta 
              - V * sin_phi 
              - H_k * sin_phi * cos_phi * sin_theta)
    
    # Equation 20: dphi/dtau [cite: 388]
    dphi = (1.0 / sin_theta) * (-U * sin_phi 
                                - anisotropy_term * sin_theta * cos_theta 
                                - V * cos_phi * cos_theta 
                                - alpha * H_k * sin_phi * cos_phi * sin_theta)
    return [dtheta, dphi]

# --- RK4 INTEGRATOR IMPLEMENTATION ---
def rk4_step(f, tau, x, dt, *args):
    """Performs a single 4th-order Runge-Kutta step."""
    k1 = np.array(f(tau, x, *args))
    k2 = np.array(f(tau + 0.5*dt, x + 0.5*dt*k1, *args))
    k3 = np.array(f(tau + 0.5*dt, x + 0.5*dt*k2, *args))
    k4 = np.array(f(tau + dt, x + dt*k3, *args))
    return x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

def simulate_sto(J_dc, J_ac_func=None, tau_max=500.0, dt=0.01, init_state=[np.pi/3, 0.1]):
    """Simulates a single STO over a time grid using custom RK4."""
    N_steps = int(tau_max / dt)
    tau_grid = np.linspace(0, tau_max, N_steps)
    states = np.zeros((N_steps, 2))
    states[0] = init_state
    
    for i in range(1, N_steps):
        t = tau_grid[i-1]
        j_ac_val = J_ac_func(t) if J_ac_func else 0.0
        states[i] = rk4_step(sto_derivatives, t, states[i-1], dt, J_dc, j_ac_val)
        
    return tau_grid, states

# --- ANALYSIS FUNCTIONS ---
def calculate_frequency(tau, signal):
    """Extracts oscillation frequency via peak detection from time series data."""
    peaks, _ = find_peaks(signal)
    if len(peaks) < 2:
        return 0.0
    periods = np.diff(tau[peaks])
    return 2 * np.pi / np.mean(periods)

# --- EXECUTION OF NUMERICAL GOALS ---

print("Executing Goal 1: Custom Verification against solve_ivp...")
t_eval, x_rk4 = simulate_sto(J_dc=0.01, tau_max=100.0)
sol_ivp = solve_ivp(sto_derivatives, [0, 100], [np.pi/3, 0.1], args=(0.01, 0.0), t_eval=t_eval, rtol=1e-8)
print(f"-> Max Absolute Deviation: {np.max(np.abs(x_rk4[:, 0] - sol_ivp.y[0])):.3e}\n")


print("Executing Goal 2: Sweeping Driving Current J_dc...")
J_vals = np.linspace(0.001, 0.04, 30) # [cite: 404]
frequencies = []
for j in J_vals:
    t, st = simulate_sto(J_dc=j, tau_max=400.0)
    # Use the late-stage steady-state signal to bypass transients [cite: 251]
    freq = calculate_frequency(t[20000:], np.sin(st[20000:, 1]))
    frequencies.append(freq)

plt.figure(figsize=(7, 4))
plt.plot(J_vals, frequencies, 'o-', color='crimson', label="STO Frequency")
plt.xlabel(r"DC Current Bias $J_d$ (A)")
plt.ylabel(r"Oscillation Frequency $\omega$ (rad/$\tau$)")
plt.title("STO Frequency vs DC Current Drive")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("sto_frequency_sweep.png")
plt.close()
print("-> Sweeping complete. Saved plot to 'sto_frequency_sweep.png'.\n")


print("Executing Goal 3: Analyzing AC Phase Locking Behavior...")
J_dc_fixed = 0.01 # 
Omega_ac_vals = np.linspace(0.6, 0.9, 25) # 
A_ac_vals = np.linspace(0.0, 2e-3, 25) # 
locking_matrix = np.zeros((len(A_ac_vals), len(Omega_ac_vals)))

for idx_A, A in enumerate(A_ac_vals):
    for idx_O, Omega in enumerate(Omega_ac_vals):
        ac_func = lambda t: A * np.sin(Omega * t)
        t, st = simulate_sto(J_dc=J_dc_fixed, J_ac_func=ac_func, tau_max=600.0)
        
        # Calculate the natural frequency under AC drive [cite: 407]
        obs_freq = calculate_frequency(t[40000:], np.sin(st[40000:, 1]))
        
        # Check if the observed frequency locks to the driving frequency [cite: 407]
        if np.abs(obs_freq - Omega) < 0.01:
            locking_matrix[idx_A, idx_O] = 1 # 1: Locked
        elif np.abs(2*obs_freq - Omega) < 0.01 or np.abs(obs_freq - 2*Omega) < 0.01:
            locking_matrix[idx_A, idx_O] = 0.5 # 0.5: Higher harmonic lock

plt.figure(figsize=(8, 5))
plt.imshow(locking_matrix, extent=[0.6, 0.9, 0.0, 2e-3], origin='lower', aspect='auto', cmap='plasma')
plt.colorbar(label="Locking Regime (1=1:1 Lock, 0.5=Harmonic Lock)")
plt.xlabel(r"AC Drive Frequency $\Omega$")
plt.ylabel(r"AC Drive Amplitude $|J_a|$")
plt.title("Arnold Tongues: Phase Locking Dynamics")
plt.tight_layout()
plt.savefig("arnold_tongues.png")
plt.close()
print("-> AC analysis complete. Saved plot to 'arnold_tongues.png'.\n")


print("Executing Goals 4 & 5: Simulating Coupled STO Arrays...")
def simulate_coupled_stos(J_dc, C_a, C_b, D, tau_max=400.0, dt=0.02):
    """Simulates two series-coupled STO devices with feedback delays[cite: 409, 411]."""
    N_steps = int(tau_max / dt)
    delay_indices = int(D / dt)
    tau_grid = np.linspace(0, tau_max, N_steps)
    
    # State mapping: [theta1, phi1, theta2, phi2]
    states = np.zeros((N_steps, 4))
    states[0] = [np.pi/3, 0.1, np.pi/3 + 0.05, 0.15] # Asymmetric initial values
    
    for i in range(1, N_steps):
        t = tau_grid[i-1]
        
        # Evaluate delayed terms or default to initial conditions for t < D 
        hist_idx = max(0, i - 1 - delay_indices)
        st_past = states[hist_idx]
        
        coupling_sum = (np.sin(st_past[0]) * np.cos(st_past[1]) + 
                        np.sin(st_past[2]) * np.cos(st_past[3])) # 
        
        # Series coupling common current profile 
        J_shared = J_dc / (C_a - C_b * coupling_sum) # 
        
        # Derivative mapping for both oscillators
        derivs1 = sto_derivatives(t, states[i-1, 0:2], J_shared, 0.0)
        derivs2 = sto_derivatives(t, states[i-1, 2:4], J_shared, 0.0)
        total_deriv = np.array(derivs1 + derivs2)
        
        states[i] = states[i-1] + dt * total_deriv
        
    return tau_grid, states

# Parameters carefully chosen to ensure stability and coupling strength 
t_c, st_c = simulate_coupled_stos(J_dc=0.01, C_a=5.0, C_b=1.2, D=1.5)

plt.figure(figsize=(8, 4))
plt.plot(t_c[-5000:], st_c[-5000:, 1], label=r"$\phi_1$ (STO 1)", color='navy', alpha=0.8)
plt.plot(t_c[-5000:], st_c[-5000:, 3], '--', label=r"$\phi_2$ (STO 2)", color='orange', alpha=0.8)
plt.xlabel(r"Dimensionless Time $\tau$")
plt.ylabel(r"Azimuthal Phase $\phi$")
plt.title("Synchronization Trajectories of Two Coupled STOs")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("sto_coupled_sync.png")
plt.close()
print("-> Simulation complete. Output saved to 'sto_coupled_sync.png'. All goals processed successfully.")