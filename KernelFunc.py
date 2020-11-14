
import numpy as np

def KernelFunc(vector, para):

	K1_Param = para

	if K1_Param['K1_Type'] == 'EXP':

		coeffs = K1_Param['EXP_coeffs']

		alpha = coeffs[0]

		beta = coeffs[1]

		return alpha*np.exp(-beta*vector)

	if K1_Param['K1_Type'] == 'PWL':

		coeffs = K1_Param['PWL_coeffs']

		K = coeffs[0]

		c = coeffs[1]

		p = coeffs[2]

		return K*np.power(vector+c,-p)

	if K1_Param['K1_Type'] == 'SQR':

		coeffs = K1_Param['SQR_coeffs']

		B = coeffs[0]

		L = coeffs[1]

		# def SQR(vector, B, L):

		vector[vector > L] == 0

		vector[vector <= L] == B

		return vector

	if K1_Param['K1_Type'] == 'SNS':

		coeffs = K1_Param['SNS_coeffs']

		A = coeffs[0]

		omega = coeffs[1]

		# def SNS(vector, A, omega):

		vector[vector < np.pi/omega] = A*np.sin(omega*vector[vector < np.pi/omega])

		vector[vector >= np.pi/omega] = 0

		return vector

	if K1_Param['K1_Type'] == 'RAY':

		coeffs = K1_Param['RAY_coeffs']

		gamma = coeffs[0]

		eta = coeffs[1]

		return gamma*np.multiply(vector,np.exp(-eta*np.square(vector)))

	if K1_Param['K1_Type'] == 'QEXP':

		coeffs = K1_Param['QEXP_coeffs']

		a = coeffs[0]

		q = coeffs[1]

		if(q != 1.):

			vector = a*np.power((1 + (q - 1)*vector), 1 / (1 - q))

			vector[vector < 0] = 0

		elif(q == 1.):

			vector = a*np.exp(-vector)
		else:

			vector = np.zeros_like(vector)

		return vector

	if K1_Param['K1_Type'] == 'GSS':

		coeffs = K1_Param['GSS_coeffs']

		kappa = coeffs[0]

		tau = coeffs[1]

		sigma = coeffs[2]

		return kappa*np.exp(-np.square(vector-tau)/sigma)
