import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import complex_ode, ode
from scipy.linalg import expm

HBAR = 1.0
INTERACTION = 0.1
EPSILON = 1.

def hamiltonian(t=0):
    return np.array([[EPSILON, INTERACTION], [INTERACTION, -EPSILON]])

def rhs(t, psi):
    return - 1.0j / HBAR * hamiltonian().dot(psi)

# Create an `ode` instance to solve the system of differential
# equations defined by `hamiltonian`, and set the solver method to dopri5 (an alternative more precise RK-8 is dop853)
solver = complex_ode(rhs)
solver.set_integrator('dopri5')

# Give the value of omega to the solver. This is passed to
# `hamiltonian` when the solver calls it.

# Set the initial value z(0) = z0.
t0 = 0.0
population_0 = 0.1
population_1 = np.sqrt(1 - population_0 ** 2)
psi_0 = np.array([population_0, population_1]).astype(np.complex)
solver.set_initial_value(psi_0, t0)

# Create the array `t` of time values at which to compute
# the solution, and create an array to hold the solution.
# Put the initial value in the solution array.
MAX_TIME = 10
N_TIMES = 100
times = np.linspace(t0, MAX_TIME, N_TIMES)
psi_t = np.zeros((N_TIMES, 2)).astype(np.complex)
psi_t[0] = psi_0

# Repeatedly call the `integrate` method to advance the
# solution to time t[k], and save the solution in sol[k].
for i in range(1, times.shape[0]):
    t = times[i]
    if not solver.successful():
        break
    psi_t[i] = solver.integrate(t)

# Plot the solution...
plt.plot(times, psi_t[:, 0].real, label='psi_0_real')
plt.plot(times, psi_t[:, 0].imag, label='psi_0_imag')
plt.plot(times, psi_t[:, 1].real, label='psi_1_real')
plt.plot(times, psi_t[:, 1].imag, label='psi_1_imag')
plt.plot(times, psi_t.dot(psi_t.T.conj()).diagonal().real, label='normalization')
plt.xlabel('t')
plt.grid(True)
plt.legend()
plt.show()


psi_t = np.zeros((N_TIMES, 2)).astype(np.complex)

# For each time calculate the time evolution
for t in range(np.shape(times)[0]):
    time = times[t]
    psi_t[t] = expm(-1j * hamiltonian() * time).dot(psi_0)


# Plot the solution...
plt.plot(times, psi_t[:, 0].real, label='psi_0_real')
plt.plot(times, psi_t[:, 0].imag, label='psi_0_imag')
plt.plot(times, psi_t[:, 1].real, label='psi_1_real')
plt.plot(times, psi_t[:, 1].imag, label='psi_1_imag')
plt.plot(times, psi_t.dot(psi_t.T.conj()).diagonal().real, label='normalization')
plt.xlabel('t')
plt.grid(True)
plt.legend()
plt.show()

# Crank Nicolson propagator
psi_t = np.zeros((N_TIMES, 2)).astype(np.complex)
psi_t[0] = psi_0
ones = np.eye(hamiltonian().shape[0])
for t in range(1, np.shape(times)[0]):
    time = times[t]
    delta_t = times[t] - times[t-1]
    propagator = np.linalg.inv(ones + 1j * delta_t / 2  * hamiltonian())
    propagator = propagator.dot(ones - 1j * delta_t / 2  * hamiltonian())
    psi_t[t] = propagator.dot(psi_t[t-1])

# Plot the solution...
plt.plot(times, psi_t[:, 0].real, label='psi_0_real')
plt.plot(times, psi_t[:, 0].imag, label='psi_0_imag')
plt.plot(times, psi_t[:, 1].real, label='psi_1_real')
plt.plot(times, psi_t[:, 1].imag, label='psi_1_imag')
plt.plot(times, psi_t.dot(psi_t.T.conj()).diagonal().real, label='normalization')
plt.xlabel('t')
plt.grid(True)
plt.legend()
plt.show()

