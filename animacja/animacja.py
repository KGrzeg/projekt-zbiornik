#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from math import sqrt, sin
from matplotlib.animation import FuncAnimation


def Q(t):
    if t < 2:
        return 1
    elif t < 5:
        return 0
    elif t < 17:
        return 0.5
    return 5


def zbiornik_model(x, t):
    A = 2  # pole powierzchni lustra wody
    g = 9.81  # warość przyśpieszenia ziemskiego
    s = 0.02  # pole powierzchni przekroju wypływu
    Qin = Q(t)
    h = x[0]

    dhdt = 1 / A * (Qin - s * sqrt(2 * g * h))

    return [dhdt]


class SuperContainer:
    def __init__(self, h):
        container_points = [[-3, 2], [-3, -2], [3, -2], [3, 2]]
        pipe_in_points = [[-2.7, 2.3], [-5, 2.4], [-5, 2.2], [-2.7, 2.1]]
        pipe_out_points = [
            [2, -2],
            [2.1, -4],
            [5, -4],
            [5, -3.8],
            [2.3, -3.8],
            [2.2, -2],
        ]
        fill_points = self.__get_fill_points()

        self.outline = plt.Polygon(
            container_points, closed=None, edgecolor="k", fill=False, lw=4
        )
        self.fill = plt.Polygon(fill_points, facecolor="b", edgecolor=None, fill=True)
        self.pipe_in = plt.Polygon(
            pipe_in_points, closed=None, edgecolor="k", facecolor="b", lw=2
        )
        self.pipe_out = plt.Polygon(
            pipe_out_points, closed=None, edgecolor="k", facecolor="b", lw=2
        )
        self.__h = h

    def add_path(self, gca):
        gca.add_patch(self.fill)
        gca.add_patch(self.outline)
        gca.add_patch(self.pipe_in)
        gca.add_patch(self.pipe_out)

    def __wavy(self, x1, x2, y0, points, amp=1, offset=0, reverse=False):
        ax = np.linspace(x1, x2, points)
        ay = np.sin(np.linspace(0, (x2 - x1) * 5, points) + offset) * amp
        ay = ay + y0

        if reverse:
            ax = np.flip(ax)

        return np.vstack((ax, ay)).T

    def __get_fill_points(self, offset=0, h=0):
        h_max = 10
        y_min = -1.9
        y = y_min + h / h_max
        return [
            [-3, y],
            [-3, y_min],
            [3, y_min],
            *self.__wavy(-3, 3, y, 30, 0.1, offset * 0.1, True),
        ]

    def animate(self, i):
        h = self.__h[i]
        self.fill.set_xy(self.__get_fill_points(i, h))

        return (self.fill,)


def init():
    global superContainer, h
    superContainer = SuperContainer(h)
    superContainer.add_path(plt.gca())
    time_text.set_text("")
    return (time_text,)


def animate(i):
    global x
    time_text.set_text(time_template % (i * dt, x[i, 0]))
    return time_text, *superContainer.animate(i)


fig = plt.figure()

dt = 0.05
t = np.arange(0.0, 20, dt)

x0 = [0]
x = odeint(zbiornik_model, x0, t)
superContainer = None

h = x[:, 0]
Q = [Q(t) for t in t]
ax1 = fig.add_subplot(212, autoscale_on=True)
ax1.plot(t, h, label="h")
ax1.plot(t, Q, label="Q")
ax1.set_xlabel("t")

ax0 = fig.add_subplot(211, autoscale_on=False, xlim=(-5, 5), ylim=(-5, 5))
ax0.grid()

time_template = "time = %.1fs, h = %.1f"
time_text = ax0.text(0.05, 0.9, "", transform=ax0.transAxes)

ani = FuncAnimation(
    fig, animate, np.arange(1, len(t)), interval=25, blit=True, init_func=init
)
# ani.save("preview.html", fps=15)
plt.show()
