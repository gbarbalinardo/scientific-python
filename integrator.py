import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import complex_ode, ode

def fun(t, y):
    """
    t	(float) Current time.
    z	(ndarray) Current variable values.
    """

    omega = 2 * np.pi
    chi = 10.
    f = [1j *  omega * y[0] - chi * y[1], chi * y[0] -  omega * y[1]]
    return f

# Create an `ode` instance to solve the system of differential
# equations defined by `fun`, and set the solver method to dopri5 (an alternative more precise RK-8 is dop853)
solver = complex_ode(fun)
solver.set_integrator('dopri5')

# Give the value of omega to the solver. This is passed to
# `fun` when the solver calls it.

# Set the initial value z(0) = z0.
t0 = 0.0
z0 = np.array([1, -0.25]).astype(np.complex)
solver.set_initial_value(z0, t0)

# Create the array `t` of time values at which to compute
# the solution, and create an array to hold the solution.
# Put the initial value in the solution array.
t1 = 2.5
N = 75
times = np.linspace(t0, t1, N)
sol = np.zeros((N, 2)).astype(np.complex)
sol[0] = z0

# Repeatedly call the `integrate` method to advance the
# solution to time t[k], and save the solution in sol[k].
for i in range(1, times.shape[0]):
    t = times[i]
    if not solver.successful():
        break
    sol[i] = solver.integrate(t)

# Plot the solution...
plt.plot(times, sol[:,0].real, label='x')
plt.plot(times, sol[:,1].real, label='y')
plt.xlabel('t')
plt.grid(True)
plt.legend()
plt.show()