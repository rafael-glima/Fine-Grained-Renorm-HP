import scipy.io
from scipy.optimize import minimize
import numpy as np
from scipy.integrate import quad
from math import ceil
#import numpy.random as np.random
np.random.seed(2020)

def trainGD_RAY(seq,eps, M,method, train='full', frac=0.7, T=100):

	if train == 'full':

		assert frac == 1.0, "frac can not be different from 1 for full train"

	epsilon = np.finfo(float).eps

	if train == 'full':

		test_seq = seq

	elif train == 'partial':

		train_ind = ceil(frac*len(seq))

		test_seq = seq[seq > frac*T]

		test_seq -= test_seq[0]

		seq = seq[ seq < frac*T]

	else:

		raise ValueError("Select full or partial training!")

	gamma_0 = np.random.rand()
	eta_0 = np.random.rand() #2*gamma_0
	mu_0 = np.random.rand()

	# input_data = scipy.io.loadmat('4Kern_newSNS_10seqT100000_highMuandfreq0.15.mat')
	# seq = input_data['Seq2'][1][0][0]

	# seq = seq[:300]

	T = frac*T #seq[-1]#-seq[0]
	Delta = len(seq)/T

	#seq = np.array([1.,2.,3.,4.,5.])

	bnds = ((0,None),(0,None),(0,None))

	def logGD_RAY(RAY_coeffs):

		def funcRAY(x,gamma,eta):
			return gamma*x*np.exp(-eta*np.power(x,2))

		gamma = RAY_coeffs[1];

		eta = RAY_coeffs[2];

		mu = RAY_coeffs[0]

		if (gamma/(2*eta) < 1.) and (gamma/(2*eta) >= 0.):
			mu = mu#(1.-gamma/(2*eta))*Delta;
		else:
			mu = mu#0. 
			#return np.inf

		intens = np.zeros(len(seq));

		compens = mu*T

		for i in range(0,len(seq)):

			intens[i] += mu

			#############  REVER ESSA PARTE !!!!!!!!!!!!! ##################################################

			compens += (gamma/(2*eta))*(1-np.exp(-eta*np.power(T-seq[i],2)))#quad(funcRAY,0,T-seq[i], args=(gamma,eta))[0]

			#print('compens: '+repr(compens))

			for j in range(0,i):

				intens[i] += gamma*(seq[i]-seq[j])*np.exp(-eta*np.power(seq[i] - seq[j],2))			

			#print('intens_i: '+repr(intens))

		intens[intens < 0.] = 0.	

		print ('Loglikelihood Train GD: ' + repr(np.sum(np.nan_to_num(np.log(intens+epsilon))) - compens) + '\n')

		return - np.sum(np.nan_to_num(np.log(intens+epsilon))) + max(compens,0.)

	par = minimize(logGD_RAY, [mu_0, gamma_0, eta_0], method=method, tol=1e-2, options={'maxiter':10})

	print('Final Parameters: '+ repr(par.x)+'\n')

	RAY_statcriter = abs(par.x[1]/(2*par.x[2]))

	print('RAY_statcriter: ' + repr(RAY_statcriter))

	if par.x[0] < 0:

		fin_llh = np.inf

	else:

		fin_llh = log_RAY(par.x, test_seq, T)

	fin_llh = (-1)*fin_llh

#	if np.isinf(fin_llh) or np.isnan(fin_llh) or (fin_llh > 0.):

	par_renorm_gamma = [Delta*(1.-1./(1.+eps)),par.x[1]/(RAY_statcriter*(1+eps)),par.x[2]]

	par_renorm_eta = [Delta*(1.-1./(1.+eps)),par.x[1],par.x[2]*(RAY_statcriter*(1+eps))]

	for i in range(1,M):

		if i == 1:

			par_renorm_sqrt = [Delta*(1.-1./(1.+eps)),par.x[1]/np.power(RAY_statcriter*(1+eps), i/M),par.x[2]*np.power(RAY_statcriter*(1+eps), (M-i)/M)]

		else:

			par_renorm_sqrt = par_renorm_sqrt + [Delta * (1. - 1. / (1. + eps)), par.x[1] / np.power(RAY_statcriter * (1 + eps), i / M), par.x[2] * np.power(RAY_statcriter * (1 + eps), (M - i) / M)]

	assert len(par_renorm_sqrt) == (M-1)*3, "Length of par_renorm_sqrt_RAY is wrong!!"

	par_renorm_sqrt = np.reshape(par_renorm_sqrt, (-1,3))

	llh_renorm_gamma = log_RAY(par_renorm_gamma,seq,T)

	llh_renorm_eta = log_RAY(par_renorm_eta, seq, T)

	for i in range(1, M):

		if i == 1:

			llh_renorm_sqrt = [log_RAY(par_renorm_sqrt[i-1], seq, T)]

		else:

			llh_renorm_sqrt = llh_renorm_sqrt + [log_RAY(par_renorm_sqrt[i-1], seq, T)]

	llh_renorm_gamma *= -1

	llh_renorm_eta *= -1

	llh_renorm_sqrt  = [-1*item for item in llh_renorm_sqrt]

	print('par_renorm_gamma: '+repr(par_renorm_gamma))

	print('par_renorm_eta: '+repr(par_renorm_eta))

	print('par_renorm_sqrt: '+repr(par_renorm_sqrt))

	print('llh_renorm_gamma: '+repr(llh_renorm_gamma))

	print('llh_renorm_eta: '+repr(llh_renorm_eta))

	print('llh_renorm_sqrt: '+repr(llh_renorm_sqrt))


	par_max_sqrt = par_renorm_sqrt[np.argmax(llh_renorm_sqrt)]

	print("Par_max_sqrt: ", par_max_sqrt)

	llh_arr = [fin_llh] + [llh_renorm_gamma] + [np.max(llh_renorm_sqrt)] + [llh_renorm_eta]

	par_arr = [par.x] + [par_renorm_gamma] + [par_max_sqrt] + [par_renorm_eta]

	par_max = par_arr[np.argmax(llh_arr)]

	print("par_max: ", par_max)

	if train == 'partial':

		par_max_test_llh = -log_RAY(par_max, test_seq, (1-frac)*T/frac)

	else:

		par_max_test_llh = None

	print('par_max_test_llh: ', par_max_test_llh)

	K1_Param = {'par_max_test_llh': par_max_test_llh, 'RAY_coeffs': par.x, 'K1_Type': 'RAY', 'RAY_statcriter': par.x[1]/par.x[2],\
			'final_llh': fin_llh, 'par_renorm_gamma': par_renorm_gamma,'llh_renorm_gamma': llh_renorm_gamma, 'par_renorm_eta': par_renorm_eta,\
			'llh_renorm_eta': llh_renorm_eta, 'par_renorm_sqrt': par_renorm_sqrt, 'llh_renorm_sqrt': llh_renorm_sqrt}

	# else:

	# 	llh_renorm_gamma = fin_llh

	# 	llh_renorm_eta = fin_llh

	# 	llh_renorm_sqrt = fin_llh

	# 	K1_Param = {'RAY_coeffs': par.x, 'K1_Type': 'RAY', 'RAY_statcriter': par.x[1]/par.x[2], 'final_llh': fin_llh,\
	# 	'llh_renorm_gamma': llh_renorm_gamma, 'llh_renorm_eta': llh_renorm_eta, 'llh_renorm_sqrt':llh_renorm_sqrt}

	return K1_Param

def log_RAY(RAY_coeffs, seq, T):

	epsilon = np.finfo(float).eps
	T = T #seq[-1]#-seq[0]
	Delta = len(seq)/T

	def funcRAY(x,gamma,eta):
		return gamma*x*np.exp(-eta*np.power(x,2))

	gamma = RAY_coeffs[1];

	eta = RAY_coeffs[2];

	mu = RAY_coeffs[0]

	if (gamma/(2*eta) < 1.) and (gamma/(2*eta) >= 0.):
		mu = mu#(1.-gamma/(2*eta))*Delta;
	else:
		mu = mu#0.
		#return np.inf

	intens = np.zeros(len(seq));

	compens = mu*T

	for i in range(0,len(seq)):

		intens[i] += mu

		#############  REVER ESSA PARTE !!!!!!!!!!!!! ##################################################

		compens += (gamma/(2*eta))*(1-np.exp(-eta*np.power(T-seq[i],2)))#quad(funcRAY,0,T-seq[i], args=(gamma,eta))[0]

		#print('compens: '+repr(compens))

		for j in range(0,i):

			intens[i] += gamma*(seq[i]-seq[j])*np.exp(-eta*np.power(seq[i] - seq[j],2))

		#print('intens_i: '+repr(intens))

	intens[intens < 0.] = 0.

	print ('Loglikelihood Train GD: ' + repr(np.sum(np.nan_to_num(np.log(intens+epsilon))) - compens) + '\n')

	return - np.sum(np.nan_to_num(np.log(intens+epsilon))) + max(compens,0.)
