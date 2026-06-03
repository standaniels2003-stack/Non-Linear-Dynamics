import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from scipy.integrate import odeint
import matplotlib.ticker as ticker

class Pendulum:
    def __init__(self, L=1.0, m=1.0, b=0.2):
        self.g = 9.81
        self.L = L
        self.m = m
        self.b = b
        self.base_path = os.path.dirname(os.path.abspath(__file__))
        
        self.results_dir = os.path.join(self.base_path, "results")
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)

    def equations(self, state, t):
        theta, omega = state
        return [omega, -(self.b/self.m)*omega - (self.g/self.L)*np.sin(theta)]

    def simulate(self, initial_state=[np.pi/2, 0], duration=20):
        steps = duration * 30
        self.t = np.linspace(0, duration, steps)
        sol = odeint(self.equations, initial_state, self.t)
        self.theta = sol[:, 0]
        self.omega = sol[:, 1]
        return self.theta

    def plot_phase_portrait_quiver(self, filename="phase_portrait_arrows.png"):
        if hasattr(self, 'theta'):
            x_min, x_max = self.theta.min() - 1, self.theta.max() + 1
            y_min, y_max = self.omega.min() - 1, self.omega.max() + 1
        else:
            x_min, x_max = -2*np.pi, 2*np.pi
            y_min, y_max = -8, 8

        th_range = np.linspace(x_min, x_max, 30)
        om_range = np.linspace(y_min, y_max, 30)
        TH, OM = np.meshgrid(th_range, om_range)

        U = OM
        V = -(self.b/self.m)*OM - (self.g/self.L)*np.sin(TH)
        
        N = np.hypot(U, V)
        N[N == 0] = 1.0
        U_norm, V_norm = U / N, V / N

        plt.style.use('dark_background')
        # Use subplots to get the 'ax' object correctly
        fig, ax = plt.subplots(figsize=(15, 10))
        
        # CRITICAL: Fix for the 'AttributeError'
        def format_pi(value, tick_number):
            n = value / np.pi
            if n == 0: return "0"
            if abs(n - 1) < 1e-6: return r"$\pi$"
            if abs(n + 1) < 1e-6: return r"$-\pi$"
            if n % 1 == 0: return fr"${int(n)}\pi$"
            return fr"${n:g}\pi$"

        # Use ax.xaxis instead of plt.xaxis
        ax.xaxis.set_major_locator(ticker.MultipleLocator(np.pi / 2))
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_pi))
        
        # Force equal aspect ratio so horizontal/vertical arrows are same size
        ax.set_aspect('equal', adjustable='box')

        # Reference Lines
        line_style = {'color': 'white', 'linestyle': '--', 'linewidth': 1.5, 'alpha': 1}
        ax.axhline(0, **line_style)
        n_start, n_end = int(np.ceil(x_min / np.pi)), int(np.floor(x_max / np.pi))
        for n in range(n_start, n_end + 1):
            ax.axvline(n * np.pi, **line_style)
        
        # Plot Quiver - using scale=1 and scale_units='xy' for true unit length
        Q = ax.quiver(TH, OM, U_norm, V_norm, N, cmap='plasma',
                    angles='xy', scale_units='xy', scale=2, 
                    pivot='mid', width=0.003, zorder=3)
        
        stable_pts = [n * np.pi for n in range(n_start, n_end + 1) if n % 2 == 0]
        unstable_pts = [n * np.pi for n in range(n_start, n_end + 1) if n % 2 != 0]

        # Stable (Green)
        ax.scatter(stable_pts, np.zeros_like(stable_pts), 
           color="#FFFFFF",      # Bright Lime
           s=100,                # Slightly larger
           marker='o',  
           zorder=6)

        # 2. The inner black "hole" to create the target look
        ax.scatter(stable_pts, np.zeros_like(stable_pts), 
            color='black', 
            s=80,                 # Smaller center
            marker='o',
            label='Stable Fixed Point',
            zorder=7)
        # Unstable (Red)
        ax.scatter(unstable_pts, np.zeros_like(unstable_pts), color="#FFFFFF", 
                   s=100, marker='o', label='Unstable Fixed Point', zorder=6)
        
        if hasattr(self, 'theta'):
            ax.plot(self.theta, self.omega, color='cyan', lw=2, label='Trajectory', zorder=4)
            ax.plot(self.theta[0], self.omega[0], 
            marker='o',
            color='lime',
            markersize=10, 
            label='Start', 
            zorder=5)
        ax.legend(loc='upper left', frameon=True, facecolor='black', edgecolor='white')

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_title(f"Phase Space Flow (b={self.b})")
        ax.set_xlabel(r'$\theta$ (rad)')
        ax.set_ylabel(r'$\omega$ (rad/s)')
        
        # Handle colorbar for the specific axes
        cbar = fig.colorbar(Q, ax=ax)
        cbar.set_label('Magnitude')
        
        save_path = os.path.join(self.results_dir, filename)
        plt.savefig(save_path, dpi=200, facecolor=fig.get_facecolor())
        plt.style.use('default') # Reset for the animation

    def animate_and_save(self, filename="pendulum_swing.mp4"):
        if not hasattr(self, 'theta'):
            self.simulate()

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.set_xlim(-self.L * 1.2, self.L * 1.2)
        ax.set_ylim(-self.L * 1.2, self.L * 1.2)
        ax.set_aspect('equal')
        ax.grid(True)

        line, = ax.plot([], [], 'o-', lw=2, color='navy', markersize=10)
        time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

        def init():
            line.set_data([], [])
            time_text.set_text('')
            return line, time_text

        def update(i):
            x = self.L * np.sin(self.theta[i])
            y = -self.L * np.cos(self.theta[i])
            line.set_data([0, x], [0, y])
            time_text.set_text(f'time = {self.t[i]:.1f}s')
            return line, time_text

        ani = FuncAnimation(fig, update, frames=len(self.theta), init_func=init, blit=True)
        save_path = os.path.join(self.results_dir, filename)
        
        print(f"Saving animation... (Silent process)")
        try:
            writer = FFMpegWriter(fps=30)
            ani.save(save_path, writer=writer)
        except Exception as e:
            save_path = save_path.replace(".mp4", ".gif")
            ani.save(save_path, writer='pillow')
        
        plt.close(fig)
        print(f"Success: {save_path}")

if __name__ == "__main__":
    p = Pendulum(L=1.0, m=1.0, b=0.5)
    p.simulate(initial_state=[np.radians(180), 5.9035445], duration=30)
    p.plot_phase_portrait_quiver(filename="pendulum_quiver.png")
    p.animate_and_save(filename="pendulum_simulation.mp4")