import numpy as np

from KernelFunc import KernelFunc

def SupIntensity(para, sim_seq, mu, taumax):

	# print('sim_seq: ' + repr(sim_seq))

	array_inv = sim_seq[-1] - sim_seq

	tmp = KernelFunc(array_inv, para)

	tmp = np.sum(tmp[:-1])

	try:

		return tmp + mu

	except(TypeError):

		return mu