import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import complex_ode, ode

hbar = 1.0

def hamiltonian(t=0):
    omega = 1.0
    chi = 0.1
    return np.array([[omega,  chi], [chi, - omega]])

def rhs(t, psi):
    """
    t	(float) Current time.
    z	(ndarray) Current variable values.
    """
    return - 1.0j /hbar * hamiltonian().dot(psi)

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
psi0 = np.array([population_0, population_1]).astype(np.complex)
solver.set_initial_value(psi0, t0)

# Create the array `t` of time values at which to compute
# the solution, and create an array to hold the solution.
# Put the initial value in the solution array.
t1 = 10
N = 100
times = np.linspace(t0, t1, N)
psi = np.zeros((N, 2)).astype(np.complex)
psi[0] = psi0

# Repeatedly call the `integrate` method to advance the
# solution to time t[k], and save the solution in sol[k].
for i in range(1, times.shape[0]):
    t = times[i]
    if not solver.successful():
        break
    psi[i] = solver.integrate(t)

# Plot the solution...
plt.plot(times, psi[:,0].real, label='psi_0_real')
plt.plot(times, psi[:,0].imag, label='psi_0_imag')
plt.plot(times, psi[:,1].real, label='psi_1_real')
plt.plot(times, psi[:,1].imag, label='psi_1_imag')
plt.plot(times, psi.dot(psi.T.conj()).diagonal().real, label='normalization')
plt.xlabel('t')
plt.grid(True)
plt.legend()
plt.show()
