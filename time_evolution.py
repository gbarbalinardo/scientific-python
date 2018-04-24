import numpy as np
# from scipy.linalg import expm
import matplotlib.pyplot as plt

def expm(matrix):
	''' Manually compute the exp of a matrix diagonalizing it'''
	eigvals, transformation = np.linalg.eig (matrix)
	return transformation.dot (np.exp(np.diag(eigvals))).dot (np.linalg.inv (transformation))

# constant go at the beginning
MAX_TIME = 20.
INTERACTION = 0.1
EPSILON = 1.
N_TIMES = 200

# Let's define the system here
psi_0 = np.array([0, 1])
times = np.linspace(0., MAX_TIME, N_TIMES)
psi_t = np.zeros((N_TIMES, 2)).astype(np.complex)
hamiltonian = np.array ([[EPSILON, INTERACTION], [INTERACTION, -EPSILON]])
energies, transformation = np.linalg.eig (hamiltonian)

# This is a test to check that we are diagonalizing the hamiltonian correctly
print(np.diag(energies) - np.linalg.inv(transformation).dot(hamiltonian).dot(transformation))

# For each time calculate the time evolution
for t in range(np.shape(times)[0]):
	time = times[t]
	psi_t[t] = expm(1j * hamiltonian * time).dot(psi_0)
	
# Here we plot the real part of the wavefunction
plt.plot(times, psi_t.dot(psi_0).real)
plt.show()
