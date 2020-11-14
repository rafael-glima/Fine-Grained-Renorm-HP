
import numpy as np
import numpy.random as rand
from KernelFunc import KernelFunc
from SupIntensity import SupIntensity

def simHP(level, para, maxjumps, taumax, Delta):

	ratio = 1.

	eps = np.finfo(float).eps

	type = para['K1_Type']

	coeffs = para[type + '_coeffs']

	K1_Param = para

	if K1_Param['K1_Type'] == 'EXP':

		statcriter = K1_Param['EXP_statcriter']

	if K1_Param['K1_Type'] == 'PWL':

		statcriter = K1_Param['PWL_statcriter']

	if K1_Param['K1_Type'] == 'SQR':

		statcriter = K1_Param['SQR_statcriter']

	if K1_Param['K1_Type'] == 'SNS':

		statcriter = K1_Param['SNS_statcriter']

	if K1_Param['K1_Type'] == 'RAY':

		statcriter = K1_Param['RAY_statcriter']

	if K1_Param['K1_Type'] == 'QEXP':

		statcriter = K1_Param['QEXP_statcriter']

	if K1_Param['K1_Type'] == 'GSS':

		statcriter = K1_Param['GSS_statcriter']

	if statcriter/ratio >= 1.:

		print('Error: The sequence could not be modeled, because the estimated kernel is not stable.')

		print('kernel_type: ', K1_Param['K1_Type'], 'statcriter: ' + repr(statcriter))

		return np.zeros((maxjumps,))

	mu = Delta*(1-statcriter/ratio) + eps
	#
	# print('mu: ' + repr(mu))

	sim_seq = np.array([rand.exponential(1/mu)])

	n_of_jumps = 1

	time = sim_seq[0]

	while (n_of_jumps < maxjumps) and (time < taumax):

		l = 2*taumax

		u = rand.random()

		if type == 'EXP' or type == 'QEXP':

			mt = SupIntensity(para, sim_seq, mu, taumax) + coeffs[0]

		elif type == 'PWL':

			mt = SupIntensity(para, sim_seq, mu, taumax) + coeffs[0]/(np.power(coeffs[1],coeffs[2]))

		elif type == 'RAY':

			count = sim_seq[-1] - sim_seq

			#print(count, count[count < 1/np.sqrt(coeffs[1]*2)], 1/np.sqrt(coeffs[1]*2))

			count = len(list(count[count <= 1/np.sqrt(coeffs[1]*2)]))

			#print("count: ", count)

			mt = SupIntensity(para, sim_seq, mu, taumax) + count*coeffs[0]/np.sqrt(coeffs[1]*2*np.e)

		elif type == 'GSS':

			count = sim_seq[-1] - sim_seq

			#print(count, count[count <= coeffs[1]], coeffs[1])

			count = len(list(count[count < coeffs[1]]))

			#print("count: ", count)

			mt = SupIntensity(para, sim_seq, mu, taumax) + count*coeffs[0]

		#print('mt: ' + repr(mt))

		dt = rand.exponential(1/mt)

		intens_dt = SupIntensity(para, np.append(sim_seq, time + dt), mu, taumax)

		assert intens_dt/mt <= 1., "intens_dt/mt > 1 !!!"

		#print(intens_dt/mt)

		if (dt < l) and ((intens_dt/mt) < u):

			time += dt

			sim_seq = np.append(sim_seq, time)
			
			n_of_jumps += 1

		# else:
		#
		# 	time += l

	return sim_seq