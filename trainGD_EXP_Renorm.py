import scipy.io
from scipy.optimize import minimize
import numpy as np
from scipy.integrate import quad
from math import ceil
#import numpy.random as np.random
np.random.seed(2020)

def trainGD_EXP(seq,eps, M, method, train='full', frac=0.7, T=100):

	epsilon = np.finfo(float).eps

	if train == 'full':

		test_seq = seq

	elif train == 'partial':

		train_ind = ceil(frac*len(seq))

		test_seq = seq[seq < frac*T]

		test_seq -= test_seq[0]

		seq = seq[seq > frac*T]

	else:

		raise ValueError("Select full or partial training!")

	alpha_0 = np.random.rand()
	beta_0 = np.random.rand() #2*alpha_0
	mu_0 = np.random.rand()

	# input_data = scipy.io.loadmat('4Kern_newSNS_10seqT100000_highMuandfreq0.15.mat')
	# seq = input_data['Seq2'][1][0][0]

	# seq = seq[:300]

	T = frac*T #seq[-1]#-seq[0]
	Delta = len(seq)/T

	#seq = np.array([1.,2.,3.,4.,5.])

	bnds = ((0,None),(0,None),(0,None))

	def logGD_EXP(EXP_coeffs):

		def funcexp(x,alpha,beta):
			return alpha*np.exp(-beta*x)

		alpha = EXP_coeffs[1];

		beta = EXP_coeffs[2];

		mu = EXP_coeffs[0]

		if (alpha/beta < 1.) and (alpha/beta >= 0.):
			mu = mu#(1.-alpha/beta)*Delta;
		else:
			mu = mu#0. 
			#return np.inf

		intens = np.zeros(len(seq));

		compens = mu*T;

		for i in range(0,len(seq)):

			intens[i] += mu;

			compens += (alpha/beta)*(1-np.exp(-beta*(T-seq[i])))#quad(funcexp,0,T-seq[i], args=(alpha,beta))[0]

			for j in range(0,i):

				intens[i] += alpha*np.exp(-beta*(seq[i] - seq[j]))
		
		intens[intens < 0.] = 0.			

		print ('Loglikelihood Train GD: ' + repr(np.sum(np.nan_to_num(np.log(intens+epsilon))) - compens) + '\n')

		return - np.sum(np.nan_to_num(np.log(intens+epsilon))) + max(compens,0.+epsilon)

	par = minimize(logGD_EXP, [mu_0, alpha_0, beta_0], method=method, tol=1e-2, options={'maxiter':10})

	print('Final Parameters: '+ repr(par.x)+'\n')

	EXP_statcriter = abs(par.x[1]/par.x[2])

	print('EXP_statcriter:' + repr(EXP_statcriter))

	if par.x[0] < 0:

		fin_llh = np.inf

	else:

		fin_llh = log_EXP(par.x, seq, frac*T)

	fin_llh = (-1)*fin_llh

#	if np.isinf(fin_llh) or np.isnan(fin_llh) or (fin_llh > 0.):

	par_renorm_alpha = [Delta*(1.-1./(1.+eps)),par.x[1]/(EXP_statcriter*(1+eps)),par.x[2]]

	par_renorm_beta = [Delta*(1.-1./(1.+eps)),par.x[1],par.x[2]*EXP_statcriter*(1+eps)]

	for i in range(1,M):

		if i ==1:

			par_renorm_sqrt = [Delta*(1.-1./(1.+eps)),par.x[1]/np.power(EXP_statcriter*(1+eps),i/M),par.x[2]*np.power(EXP_statcriter*(1+eps), (M-i)/M)]

		else:

			par_renorm_sqrt = par_renorm_sqrt + [Delta*(1.-1./(1.+eps)),par.x[1]/np.power(EXP_statcriter*(1+eps),i/M),par.x[2]*np.power(EXP_statcriter*(1+eps), (M-i)/M)]

	par_renorm_sqrt = np.reshape(par_renorm_sqrt, (-1,3))

	print('par_renorm_alpha: '+repr(par_renorm_alpha))

	print('par_renorm_beta: ' + repr(par_renorm_beta))

	print('par_renorm_sqrt: ' + repr(par_renorm_sqrt))

	llh_renorm_alpha = log_EXP(par_renorm_alpha, seq, T)

	llh_renorm_beta = log_EXP(par_renorm_beta, seq, T)

	for i in range(1,M):

		#print("EXP")

		if i == 1:

			llh_renorm_sqrt = [log_EXP(par_renorm_sqrt[i-1], seq, T)]

			#print("llh_renorm_sqrt: ", llh_renorm_sqrt)

		else:

			llh_renorm_sqrt = llh_renorm_sqrt + [log_EXP(par_renorm_sqrt[i-1], seq, T)]

			#print("llh_renorm_sqrt: ", llh_renorm_sqrt)

	llh_renorm_alpha *= -1

	llh_renorm_beta *= -1

	llh_renorm_sqrt = [-1*item for item in llh_renorm_sqrt]

	print('llh_renorm_alpha: '+repr(llh_renorm_alpha))

	print('llh_renorm_beta: '+repr(llh_renorm_beta))

	print('check llh_renorm_sqrt: '+ repr(llh_renorm_sqrt))


	par_max_sqrt = par_renorm_sqrt[np.argmax(llh_renorm_sqrt)]

	print("Par_max_sqrt: ", par_max_sqrt)

	llh_arr = [fin_llh] + [llh_renorm_alpha] + [np.max(llh_renorm_sqrt)] + [llh_renorm_beta]

	par_arr = [par.x] + [par_renorm_alpha] + [par_max_sqrt] + [par_renorm_beta]

	par_max = par_arr[np.argmax(llh_arr)]

	print("par_max: ", par_max)

	if train == 'partial':

		par_max_test_llh = -log_EXP(par_max, test_seq, (1-frac)*T/frac)

	else:

		par_max_test_llh = None

	print('par_max_test_llh: ', par_max_test_llh)


#	else:

		# llh_renorm_alpha = fin_llh

		# llh_renorm_beta = fin_llh

		# llh_renorm_sqrt = fin_llh

	K1_Param = {'par_max_test_llh': par_max_test_llh, 'EXP_coeffs': par.x, 'K1_Type': 'EXP', 'EXP_statcriter': par.x[1]/par.x[2],\
				'final_llh': fin_llh, 'llh_renorm_alpha': llh_renorm_alpha, 'llh_renorm_beta': llh_renorm_beta, 'llh_renorm_sqrt': llh_renorm_sqrt}

	return K1_Param

def log_EXP(EXP_coeffs, seq, T):

	epsilon = np.finfo(float).eps
	T = T #seq[-1]#-seq[0]
	Delta = len(seq)/T

	def funcexp(x, alpha, beta):
		return alpha * np.exp(-beta * x)

	alpha = EXP_coeffs[1];

	beta = EXP_coeffs[2];

	mu = EXP_coeffs[0]

	if (alpha / beta < 1.) and (alpha / beta >= 0.):
		mu = mu  # (1.-alpha/beta)*Delta;
	else:
		mu = mu  # 0.
	# return np.inf

	intens = np.zeros(len(seq));

	compens = mu * T;

	for i in range(0, len(seq)):

		intens[i] += mu;

		compens += (alpha / beta) * (
					1 - np.exp(-beta * (T - seq[i])))  # quad(funcexp,0,T-seq[i], args=(alpha,beta))[0]

		for j in range(0, i):
			intens[i] += alpha * np.exp(-beta * (seq[i] - seq[j]))

	intens[intens < 0.] = 0.

	print('Loglikelihood Train GD: ' + repr(np.sum(np.nan_to_num(np.log(intens + epsilon))) - compens) + '\n')

	return - np.sum(np.nan_to_num(np.log(intens + epsilon))) + max(compens, 0. + epsilon)