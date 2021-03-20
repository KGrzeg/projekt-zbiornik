#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from math import sqrt, sin
from matplotlib.animation import FuncAnimation


def Q(t):
    if t < 2:
        return 1
    elif t < 8:
        return 3
    elif t < 11:
        return 0.5
    return 5


def zbiornik_model(x, t):
    A = 2  # pole powierzchni lustra wody
    g = 9.81  # warość przyśpieszenia ziemskiego
    s = 0.02  # pole powierzchni przekroju wypływu
    # Qin = 0.005  # przepływ wejściowy
    Qin = Q(t)
    h = x[0]

    dhdt = 1 / A * (Qin - s * sqrt(2 * g * h))

    return [dhdt]


# x0 = [0]
# t = np.linspace(0, 15, 1000)

# x = odeint(zbiornik_model, x0, t)
# h = x[:, 0]

# plt.plot(t, h)
# plt.show()

# animacja

fig = plt.figure()
ax = fig.add_subplot(111, autoscale_on=False, xlim=(-5, 5), ylim=(-5, 5))
ax.grid()

(line,) = ax.plot([], [], "o-", lw=2)
time_template = "time = %.1fs"
time_text = ax.text(0.05, 0.9, "", transform=ax.transAxes)

dt = 0.05
t = np.arange(0.0, 20, dt)


def static_objects():
    container_points = [[-3, 2], [-2.8, -2], [2.8, -2], [3, 2]]
    container = plt.Polygon(
        container_points, closed=None, edgecolor="k", fill=False, lw=4
    )

    pipe_in_points = [[-2.7, 2.3], [-5, 2.4], [-5, 2.2], [-2.7, 2.1]]
    pipe_in = plt.Polygon(pipe_in_points, closed=None, edgecolor="k", fill=False, lw=2)

    pipe_out_points = [[2, -2], [2.1, -4], [5, -4], [5, -3.8], [2.3, -3.8], [2.2, -2]]
    pipe_out = plt.Polygon(
        pipe_out_points, closed=None, edgecolor="k", fill=False, lw=2
    )

    plt.gca().add_patch(container)
    plt.gca().add_patch(pipe_in)
    plt.gca().add_patch(pipe_out)


def init():
    static_objects()
    line.set_data([], [])
    time_text.set_text("")
    return line, time_text


def animate(i):
    thisx = [0, sin(i * 0.1), 2]
    thisy = [0, -0.5, -1]

    line.set_data(thisx, thisy)
    time_text.set_text(time_template % (i * dt))
    print("Jestę %d" % i)
    return line, time_text


ani = FuncAnimation(
    fig, animate, np.arange(1, len(t)), interval=25, blit=True, init_func=init
)
# ani.save("preview.html", fps=15)
plt.show()
