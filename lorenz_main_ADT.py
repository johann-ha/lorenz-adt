# Main program for solving lorenz attactors (Rouson et al book)
# using ADT pattern

from lorenz_module import Lorenz
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt



def plot_lorenz_attractor(out, fname):
	fig = plt.figure()
	ax = fig.gca(projection='3d')
	x, y, z = out[:,0], out[:,1], out[:,2]
	ax.plot(x, y, z, label='Lorenz Strange Attractor')
	ax.legend()
	plt.savefig(fname, dpi=300)

if __name__ == "__main__":
	num_steps = 2000
	space_dimension = 3
	sigma, rho, beta, dt = 10., 28., 8./3., 1E-2

	initial_condition = np.array([1., 1., 1.])

	attractor = Lorenz(initial_condition, sigma, rho, beta)

	out = np.zeros(1, dtype='3float32')
	out[0] = attractor.output()

	for step in range(num_steps):
	    attractor = attractor.integrate(dt)
	    out = np.append(out, [attractor.output()], axis=0)

	fname = "lorenz_attractor.jpg"
	plot_lorenz_attractor(out, fname)
