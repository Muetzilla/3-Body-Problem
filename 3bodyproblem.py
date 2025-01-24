import tkinter as tk
from tkinter import ttk
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

def three_body_equations(t, y, masses):
    r1, r2, r3 = y[:2], y[2:4], y[4:6]
    v1, v2, v3 = y[6:8], y[8:10], y[10:12]

    def compute_acceleration(ri, rj, mj):
        return mj * (rj - ri) / np.linalg.norm(rj - ri)**3

    a1 = compute_acceleration(r1, r2, masses[1]) + compute_acceleration(r1, r3, masses[2])
    a2 = compute_acceleration(r2, r1, masses[0]) + compute_acceleration(r2, r3, masses[2])
    a3 = compute_acceleration(r3, r1, masses[0]) + compute_acceleration(r3, r2, masses[1])

    return np.concatenate([v1, v2, v3, a1, a2, a3])

def run_simulation(masses, initial_conditions, t_span, t_eval):
    result = solve_ivp(
        three_body_equations, t_span, initial_conditions, t_eval=t_eval, args=(masses,)
    )
    return result.t, result.y

def create_gui():
    def update_initial_positions():
        ax.clear()
        ax.set_xlim(-width_var.get() / 2, width_var.get() / 2)
        ax.set_ylim(-height_var.get() / 2, height_var.get() / 2)
        ax.set_aspect('equal')
        initial_positions = [
            (x1_var.get(), y1_var.get()),
            (x2_var.get(), y2_var.get()),
            (x3_var.get(), y3_var.get()),
        ]
        sizes = [mass1_var.get() * 10, mass2_var.get() * 10, mass3_var.get() * 10]
        colors = [color1_var.get(), color2_var.get(), color3_var.get()]

        for pos, size, color in zip(initial_positions, sizes, colors):
            ax.plot(pos[0], pos[1], 'o', markersize=size, color=color)

        canvas.draw()

    def update_simulation():
        update_initial_positions()

    def start_simulation():
        nonlocal ani
        start_button.config(state=tk.DISABLED)

        masses = [mass1_var.get(), mass2_var.get(), mass3_var.get()]
        initial_conditions = [
            x1_var.get(), y1_var.get(),
            x2_var.get(), y2_var.get(),
            x3_var.get(), y3_var.get(),
            vx1_var.get(), vy1_var.get(),
            vx2_var.get(), vy2_var.get(),
            vx3_var.get(), vy3_var.get(),
        ]
        t_span = (0, 100)
        t_eval = np.linspace(0, 100, 10000)

        t, y = run_simulation(masses, initial_conditions, t_span, t_eval)

        fig.clear()
        ax = fig.add_subplot(111)
        ax.set_xlim(-width_var.get() / 2, width_var.get() / 2)
        ax.set_ylim(-height_var.get() / 2, height_var.get() / 2)
        ax.set_aspect('equal')

        sizes = [mass * 10 for mass in masses]
        colors = [color1_var.get(), color2_var.get(), color3_var.get()]
        nonlocal lines, trails
        lines = [ax.plot([], [], 'o', color=color)[0] for color in colors]
        trails = [ax.plot([], [], '-', lw=0.5, alpha=0.7, color=color)[0] for color in colors]

        def update(frame):
            idx = frame % len(t)
            for i, line in enumerate(lines):
                line.set_data([y[2*i, idx]], [y[2*i+1, idx]])
                line.set_markersize(sizes[i])
                trails[i].set_data(y[2*i, :idx+1], y[2*i+1, :idx+1])
            return lines + trails

        ani = FuncAnimation(fig, update, frames=range(100_000_000_000), interval=1000 // speed_var.get(), blit=True, repeat=True)
        canvas.draw()

    def pause_simulation():
        if ani:
            ani.event_source.stop()

    def resume_simulation():
        if ani:
            ani.event_source.start()

    def clear_simulation():
        nonlocal ani
        if ani:
            ani.event_source.stop()
            ani = None
        start_button.config(state=tk.NORMAL)
        update_initial_positions()

    root = tk.Tk()
    root.title("3-Body Problem Simulator")

    frame = ttk.Frame(root, padding="10")
    frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

    # Variables for sliders
    mass1_var = tk.DoubleVar(value=1.0)
    mass2_var = tk.DoubleVar(value=1.0)
    mass3_var = tk.DoubleVar(value=1.0)

    x1_var = tk.DoubleVar(value=-1.0)
    y1_var = tk.DoubleVar(value=0.0)
    x2_var = tk.DoubleVar(value=1.0)
    y2_var = tk.DoubleVar(value=0.0)
    x3_var = tk.DoubleVar(value=0.0)
    y3_var = tk.DoubleVar(value=1.0)

    vx1_var = tk.DoubleVar(value=0.0)
    vy1_var = tk.DoubleVar(value=1.0)
    vx2_var = tk.DoubleVar(value=0.0)
    vy2_var = tk.DoubleVar(value=-1.0)
    vx3_var = tk.DoubleVar(value=1.0)
    vy3_var = tk.DoubleVar(value=0.0)

    speed_var = tk.IntVar(value=100)

    width_var = tk.DoubleVar(value=10.0)
    height_var = tk.DoubleVar(value=10.0)

    color1_var = tk.StringVar(value="red")
    color2_var = tk.StringVar(value="green")
    color3_var = tk.StringVar(value="blue")

    def create_slider(label_text, variable, row, col, from_, to):
        ttk.Label(frame, text=label_text).grid(row=row, column=col)
        slider = ttk.Scale(frame, from_=from_, to=to, variable=variable, orient=tk.HORIZONTAL, command=lambda e: update_initial_positions())
        slider.grid(row=row, column=col+1)
        value_label = ttk.Label(frame, text=f"{variable.get():.2f}")
        value_label.grid(row=row, column=col+2)

        def update_label(*args):
            value_label.config(text=f"{variable.get():.2f}")

        variable.trace_add("write", update_label)

    def create_color_selector(label_text, variable, row, col):
        ttk.Label(frame, text=label_text).grid(row=row, column=col)
        color_selector = ttk.Combobox(frame, textvariable=variable, values=["red", "green", "blue", "yellow", "purple", "cyan"])
        color_selector.grid(row=row, column=col+1, columnspan=2)
        color_selector.bind("<<ComboboxSelected>>", lambda e: update_initial_positions())

    create_slider("Mass 1:", mass1_var, 0, 0, 0.1, 10.0)
    create_color_selector("Color 1:", color1_var, 0, 3)
    create_slider("Mass 2:", mass2_var, 1, 0, 0.1, 10.0)
    create_color_selector("Color 2:", color2_var, 1, 3)
    create_slider("Mass 3:", mass3_var, 2, 0, 0.1, 10.0)
    create_color_selector("Color 3:", color3_var, 2, 3)

    create_slider("Initial x1:", x1_var, 3, 0, -5.0, 5.0)
    create_slider("Initial y1:", y1_var, 3, 2, -5.0, 5.0)
    create_slider("Initial x2:", x2_var, 4, 0, -5.0, 5.0)
    create_slider("Initial y2:", y2_var, 4, 2, -5.0, 5.0)
    create_slider("Initial x3:", x3_var, 5, 0, -5.0, 5.0)
    create_slider("Initial y3:", y3_var, 5, 2, -5.0, 5.0)
    create_slider("Initial vx1:", vx1_var, 6, 0, -5.0, 5.0)
    create_slider("Initial vy1:", vy1_var, 6, 2, -5.0, 5.0)
    create_slider("Initial vx2:", vx2_var, 7, 0, -5.0, 5.0)
    create_slider("Initial vy2:", vy2_var, 7, 2, -5.0, 5.0)
    create_slider("Initial vx3:", vx3_var, 8, 0, -5.0, 5.0)
    create_slider("Initial vy3:", vy3_var, 8, 2, -5.0, 5.0)
    create_slider("Speed:", speed_var, 9, 0, 10, 200)
    create_slider("Width:", width_var, 10, 0, 5.0, 50.0)
    create_slider("Height:", height_var, 10, 2, 5.0, 50.0)

    start_button = ttk.Button(frame, text="Start Simulation", command=start_simulation)
    start_button.grid(row=12, column=0, columnspan=1)

    pause_button = ttk.Button(frame, text="Pause Simulation", command=pause_simulation)
    pause_button.grid(row=12, column=1, columnspan=1)

    resume_button = ttk.Button(frame, text="Resume Simulation", command=resume_simulation)
    resume_button.grid(row=12, column=2, columnspan=1)

    clear_button = ttk.Button(frame, text="Clear", command=clear_simulation)
    clear_button.grid(row=12, column=3, columnspan=1)

    fig = plt.Figure(figsize=(8, 8))
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.grid(row=13, column=0, columnspan=4, padx=10, pady=10)

    ax = fig.add_subplot(111)
    ax.set_xlim(-width_var.get() / 2, width_var.get() / 2)
    ax.set_ylim(-height_var.get() / 2, height_var.get() / 2)
    ax.set_aspect('equal')

    lines = []
    trails = []
    ani = None

    update_initial_positions()

    root.mainloop()

create_gui()
