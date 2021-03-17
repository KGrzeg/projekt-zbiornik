#!/usr/bin/env python3

import numpy as np
from scipy.integrate import odeint
from math import sqrt
import matplotlib.pyplot as plt


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


x0 = [0]
t = np.linspace(0, 15, 1000)

x = odeint(zbiornik_model, x0, t)
h = x[:, 0]

plt.plot(t, h)
plt.show()
