# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import complex_ode

def f(t, y):
    g=1.+0j
    f0 = -1j*g/2*np.conj(y[1])*y[2]
    f1 = np.conj(+1j*g/2*np.conj(y[2])*y[0])
    f2 = -1j*g/2*y[0]*y[1]
    return [f0, f1, f2]


t0=0;
y0=np.array([100,1,1], dtype=np.complex128)

r = complex_ode(f)
r.set_initial_value(t0, y0)

t1 = 10
dt = 1

sol = np.array([], dtype = np.complex256)
t   = np.array([], dtype = np.complex256)

while (r.t < t1).all():
    t=np.append(t,r.t+dt)
    sol=np.append(sol, r.integrate(r.t+dt))

A1 = sol[:, 0]
A2 = sol[:, 1]
A3 = sol[:, 2]

plt.figure()
plt.plot(t, abs(A1), label='A1')
plt.plot(t, abs(A2), label='A2')
plt.plot(t, abs(A3), label='A3')
plt.xlabel('t')
plt.ylabel('A')
plt.title('A-t')
plt.legend(loc=0)