import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt

max_time = 5.
interaction = 0.1
epsilon = 1.
n_times = 100

psi_0 = np.array([0, 1])
times = np.linspace(0., max_time, n_times)
psi_t = np.zeros((n_times, 2)).astype(np.complex)
hamiltonian = np.array ([[epsilon, interaction], [interaction, -epsilon]])
energies, transformation = np.linalg.eig (hamiltonian)
print(np.diag(energies) - np.linalg.inv(transformation).dot(hamiltonian).dot(transformation))

def expm(matrix):
	eigvals, transformation = np.linalg.eig (matrix)
	return transformation.dot (np.exp(eigvals)).dot (np.linalg.inv (transformation))


# print(np.diag(energies) - expm)

for t in range(np.shape(times)[0]):
	time = times[t]
	psi_t[t] = expm(1j * hamiltonian * time).dot(psi_0)
	
plt.plot(psi_t.dot(psi_0).real)
plt.show()
