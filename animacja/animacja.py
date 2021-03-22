#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from math import sqrt, sin
from matplotlib.animation import FuncAnimation


def Q(t):
    if t < 4:
        return 0.2 * t
    elif t < 8:
        return 0
    elif t < 12:
        return 4 - 0.3 * t
    elif t < 16:
        return 4 - 0.1 * t
    return 0


def zbiornik_model(x, t):
    A = 2  # pole powierzchni lustra wody
    g = 9.81  # warość przyśpieszenia ziemskiego
    s = 0.02  # pole powierzchni przekroju wypływu
    Qin = Q(t)  # dopływ wody
    h = x[0]  # wysokość słupa wody

    dhdt = 1 / A * (Qin - s * sqrt(2 * g * h))

    return [dhdt]


class SuperContainer:
    def __init__(self, h, q_in, ax):
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
        pointer_points = [[3, 2], [5, 2]]
        stream_points = self.__stream(0)

        self.outline = plt.Polygon(
            container_points, closed=None, edgecolor="k", fill=False, lw=4
        )
        self.pointer = plt.Polygon(
            pointer_points, closed=None, edgecolor="r", fill=False, lw=1
        )
        self.fill = plt.Polygon(fill_points, facecolor="b", edgecolor=None, fill=True)
        self.pipe_in = plt.Polygon(
            pipe_in_points, closed=None, edgecolor="k", facecolor="b", lw=2
        )
        self.pipe_out = plt.Polygon(
            pipe_out_points, closed=None, edgecolor="k", facecolor="b", lw=2
        )

        self.stream = plt.Polygon(
            stream_points, closed=None, edgecolor="b", fill=False, lw=5
        )
        self.text = ax.text(4, 2, "h=0")
        self.__h = h
        self.__q = q_in

    def add_path(self, gca):
        gca.add_patch(self.fill)
        gca.add_patch(self.outline)
        gca.add_patch(self.pipe_in)
        gca.add_patch(self.pipe_out)
        gca.add_patch(self.pointer)
        gca.add_patch(self.stream)

    def __wavy(self, x1, x2, y0, points, amp=1, offset=0, reverse=False):
        ax = np.linspace(x1, x2, points)
        ay = np.sin(np.linspace(0, (x2 - x1) * 5, points) + offset) * amp
        ay = ay + y0

        if reverse:
            ax = np.flip(ax)

        return np.vstack((ax, ay)).T

    def __stream(self, q_in):
        q_max = 5
        q = q_in / q_max
        w_max = 1
        points = 20
        x = np.linspace(-2.7, -2.7 + q * w_max, points)
        y = -4 * np.sin(np.linspace(0, np.pi / 2, points)) + 2.3

        return np.vstack((x, y)).T

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
        q = self.__q[i]

        h_max = 10
        y_min = -1.9
        y = y_min + h / h_max

        self.fill.set_xy(self.__get_fill_points(i, h))
        self.pointer.set_xy([[3, y], [5, y]])
        self.stream.set_xy(self.__stream(q))
        self.text.set_position([3.5, y])
        self.text.set_text("h = %.1f" % h)

        return (self.fill, self.pointer, self.stream, self.text)


def init():
    global superContainer, h, Q, ax0
    superContainer = SuperContainer(h, Q, ax0)
    superContainer.add_path(plt.gca())
    return superContainer.animate(0)


def animate(i):
    return superContainer.animate(i)


fig = plt.figure()

dt = 0.05
t = np.arange(0.0, 20, dt)

x0 = [0]
x = odeint(zbiornik_model, x0, t)
superContainer = None

h = x[:, 0]
Q = [Q(t) for t in t]
ax1 = fig.add_subplot(212, autoscale_on=True)
ax1.plot(t, h, label="h [m]")
ax1.plot(t, Q, label="Q [m^3/s]")
ax1.set_xlim([t[0], t[-1]])
ax1.set_ylim((0, np.amax(h)))
ax1.set_xlabel("t")
ax1.legend(loc="upper left")

ax0 = fig.add_subplot(211, autoscale_on=False, xlim=(-5, 5), ylim=(-5, 5))
ax0.grid()

ani = FuncAnimation(
    fig, animate, np.arange(1, len(t)), interval=25, blit=True, init_func=init
)

plt.show()
