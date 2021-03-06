import scipy.io
from scipy.optimize import minimize
import numpy as np
from scipy.integrate import quad
from scipy.special import lambertw
from math import ceil
#import numpy.random as np.random
np.random.seed(2020)

def trainGD_PWL(seq,eps, M,method, train='full', frac=0.7, T=100):

	if train == 'full':

		test_seq = seq

	elif train == 'partial':

		train_ind = ceil(frac*len(seq))

		test_seq = seq[seq > frac*T]

		test_seq -= test_seq[0]

		seq = seq[seq < frac*T]

	else:

		raise ValueError("Select full or partial training!")

	T = frac*T #seq[-1]#-seq[0]
	Delta = len(seq)/T

	mu_0 = Delta
	# K_0 = 0.59571517
	# c_0 = 1.94173945
	# p_0 = 2.13085285

	K_0 = np.random.rand()
	c_0 = np.random.rand() #2*K_0
	p_0 = 1. + np.random.rand()

	# input_data = scipy.io.loadmat('4Kern_newSNS_10seqT100000_highMuandfreq0.15.mat')
	# seq = input_data['Seq2'][1][0][0]

	# seq = seq[:300]

	#seq = np.array([1.,2.,3.,4.,5.])

	bnds = ((0,None),(0,None),(0,None))

	def logGD_PWL(PWL_coeffs):

		epsilon = np.finfo(float).eps

		def funcpwl(x,K,c,p):
			return K*np.power(x+c,-p)

		K = PWL_coeffs[0]

		c = PWL_coeffs[1]

		p = PWL_coeffs[2]

		mu = PWL_coeffs[3]
		
		phi = K*np.power(c,1.-p)/(1.-p)

		if (phi < 1.) and (K >= 0.) and (c > 0.) and (p > 1.):
		 	mu = mu#(1.-phi)*Delta;
		else:
			mu = mu#0. 
			#print("ill-conditioned parameters!")
			#return np.inf

		intens = np.zeros(len(seq));

		compens = mu*T;

		for i in range(0,len(seq)):

			intens[i] += mu;

			#print("T-seq[i]+c+epsilon: "+ repr(T-seq[i]+c+epsilon))

			#print("c:" + repr(c))

			#print("T-seq[i]: "+ repr(T-seq[i]))

			#print(K*np.power(T-seq[i]+c+epsilon,1-p)/(1-p))

			#print(K*np.power(c+epsilon,1-p)/(1-p))

			#print('(1-p): ' + repr((1-p)))

			compens += K*np.power(T-seq[i]+c+epsilon,1-p)/(1-p+epsilon) - K*np.power(c+epsilon,1-p)/(1-p+epsilon)#quad(funcpwl,0,T-seq[i], args=(K,c,p))[0] #(alpha/beta)*(1-np.exp(-beta*(T-seq[i])))

			#print('compens: ' + repr(compens))

			for j in range(0,i):

				intens[i] += K*np.power((seq[i] - seq[j])+c+epsilon,-p)

				#print('intens: ' + repr(intens))			

		intens[intens < 0.] = 0.

		print ('Loglikelihood Train GD: ' + repr(np.sum(np.nan_to_num(np.log(intens+epsilon))) - compens) + '\n')

		return - np.sum(np.nan_to_num(np.log(intens+epsilon))) + max(compens,0.+epsilon)

	par = minimize(logGD_PWL, [K_0, c_0, p_0, mu_0], method=method, tol=1e-2, options={'maxiter':10})

	print('Final Parameters: '+ repr(par.x)+'\n')

	PWL_statcriter = abs(((par.x[2]-1)*par.x[0]*(par.x[1]**(1-par.x[2]))))

	print('PWL_statcriter: ' + repr(PWL_statcriter))

	if par.x[3] < 0:

		fin_llh = np.inf

	else:

		fin_llh = log_PWL(par.x, seq, T)

	fin_llh = (-1)*fin_llh

	# if np.isinf(fin_llh) or np.isnan(fin_llh) or (fin_llh > 0.):

	par_renorm_K = [par.x[0]/(PWL_statcriter*(1+eps)),par.x[1],par.x[2],Delta*(1.-1./(1.+eps))]

	par_renorm_c = [par.x[0],np.power((1+eps)*PWL_statcriter,1/par.x[2])*par.x[1],par.x[2],Delta*(1.-1./(1.+eps))]

	Delta_p = PWL_statcriter*(1+eps)*(par.x[2]-1)*np.power(par.x[1],par.x[2]-1.)

	par_renorm_p = [par.x[0],par.x[1],1+ lambertw(Delta_p*np.log(par.x[1])).real/np.log(par.x[1]),Delta*(1.-1./(1.+eps))]

	for i in range(1,M):

		if i == 1:

			par_renorm_Kc = [par.x[0] / np.power(PWL_statcriter * (1 + eps), i/M),
							 par.x[1] * np.power(np.power(PWL_statcriter * (1 + eps), (M-i)/M), 1 / (2 * (par.x[2] - 1))), par.x[2],
							 Delta * (1. - 1. / (1. + eps))]

			Delta_Kp = np.power(PWL_statcriter * (1 + eps), (M-i)/M) * (par.x[2] - 1) * np.power(par.x[1], par.x[2] - 1.)

			par_renorm_Kp = [par.x[0] / np.power(PWL_statcriter * (1 + eps), i/M), par.x[1],
							 1 + lambertw(Delta_Kp * np.log(par.x[1])).real / np.log(par.x[1]),
							 Delta * (1. - 1. / (1. + eps))]
		else:

			par_renorm_Kc = par_renorm_Kc + [par.x[0] / np.power(PWL_statcriter * (1 + eps), i/M),
							 par.x[1] * np.power(np.power(PWL_statcriter * (1 + eps), (M-i)/M), 1 / (2 * (par.x[2] - 1))), par.x[2],
							 Delta * (1. - 1. / (1. + eps))]

			Delta_Kp = np.power(PWL_statcriter * (1 + eps), (M-i)/M) * (par.x[2] - 1) * np.power(par.x[1], par.x[2] - 1.)

			par_renorm_Kp = par_renorm_Kp + [par.x[0] / np.power(PWL_statcriter * (1 + eps), i/M), par.x[1],
							 1 + lambertw(Delta_Kp * np.log(par.x[1])).real / np.log(par.x[1]),
							 Delta * (1. - 1. / (1. + eps))]

	assert len(par_renorm_Kp) == (M-1)*4, "length of par_renorm_Kp is wrong!"

	assert len(par_renorm_Kc) == (M-1)*4, "length of par_renorm_Kc is wrong!"

	# par_renorm_Kc = [par.x[0]/np.sqrt(PWL_statcriter*(1+eps)),np.power((1+eps)*PWL_statcriter,\
	# 	1/(2*par.x[2]))*par.x[1],par.x[2],par.x[3]]

	# par_renorm_Kp = [par.x[0]/np.sqrt(PWL_statcriter*(1+eps)),par.x[1],par.x[2]+ \
	# np.log(np.sqrt((1+eps)*PWL_statcriter))/np.log(par.x[1]),par.x[3]]

	# par_renorm_sqrt = [par.x[0]/(PWL_statcriter*np.power(1+eps,1/3)),np.power(np.power(1+eps,1/3)*PWL_statcriter,\
	# 	1/(par.x[2]+ np.log(np.power(1+eps,1/3)*PWL_statcriter)/np.log(np.power(np.power(1+eps,1/3)*PWL_statcriter,1/par.x[2])*par.x[1])))*par.x[1],\
	# par.x[2]+ np.log(np.power(1+eps,1/3)*PWL_statcriter)/np.log(np.power(np.power(1+eps,1/3)*PWL_statcriter,1/par.x[2])*par.x[1]),par.x[3]]

	# par_renorm_sqrt = [par.x[0]/(PWL_statcriter*np.power(1+eps,1/3)),par.x[1]*np.power(PWL_statcriter,1/par.x[2]+np.log(PWL_statcriter*\
	# 	np.power(1+eps,1/3))/np.log(par.x[1]))*np.power(1+eps,1/3*(par.x[2]+np.log(PWL_statcriter*np.power(1+eps,1/3))/np.log(par.x[1]))),\
	# par.x[2]+np.log(PWL_statcriter*np.power(1+eps,1/3))/np.log(par.x[1]),par.x[3]]

	par_renorm_Kc = np.reshape(par_renorm_Kc, (-1,4))

	par_renorm_Kp = np.reshape(par_renorm_Kp, (-1,4))

	llh_renorm_K = log_PWL(par_renorm_K, seq, T)

	llh_renorm_c = log_PWL(par_renorm_c, seq, T)

	llh_renorm_p = log_PWL(par_renorm_p, seq, T)

	for i in range(1,M):

		if i == 1:

			llh_renorm_Kc = [log_PWL(par_renorm_Kc[i-1], seq, T)]

			llh_renorm_Kp = [log_PWL(par_renorm_Kp[i-1], seq, T)]

		else:

			llh_renorm_Kc = llh_renorm_Kc + [log_PWL(par_renorm_Kc[i-1], seq, T)]

			llh_renorm_Kp = llh_renorm_Kp + [log_PWL(par_renorm_Kp[i-1], seq, T)]


	llh_renorm_K *= -1

	llh_renorm_c *= -1

	llh_renorm_p *= -1

	llh_renorm_Kc  = [-1*item for item in llh_renorm_Kc]

	llh_renorm_Kp = [-1*item for item in llh_renorm_Kp]

	print('par_renorm_K: '+repr(par_renorm_K))

	print('par_renorm_c: '+repr(par_renorm_c))

	print('par_renorm_p: '+repr(par_renorm_p))

	print('par_renorm_Kc: '+repr(par_renorm_Kc))

	print('par_renorm_Kp: '+repr(par_renorm_Kp))

	print('llh_renorm_K: '+repr(llh_renorm_K))

	print('llh_renorm_c: '+ repr(llh_renorm_c))

	print('llh_renorm_p: '+repr(llh_renorm_p))

	print('llh_renorm_Kc:'+repr(llh_renorm_Kc))

	print('llh_renorm_Kp: '+repr(llh_renorm_Kp))

	# else:

	# 	llh_renorm_K = fin_llh

	# 	llh_renorm_c = fin_llh

	# 	llh_renorm_p = fin_llh

	# 	llh_renorm_Kc = fin_llh

	# 	llh_renorm_Kp = fin_llh

	par_max_Kc = par_renorm_Kc[np.argmax(llh_renorm_Kc)]

	par_max_Kp = par_renorm_Kp[np.argmax(llh_renorm_Kp)]

	print("Par_max_Kc: ", par_max_Kc, "Par_max_Kp: ", par_max_Kp)

	llh_arr = [fin_llh] + [llh_renorm_K] + [np.max(llh_renorm_Kc)] + [llh_renorm_c] + [np.max(llh_renorm_Kp)] + [llh_renorm_p]

	par_arr = [par.x] + [par_renorm_K] + [par_max_Kc] + [par_renorm_c] + [par_max_Kp] + [par_renorm_p]

	par_max = par_arr[np.argmax(llh_arr)]

	print("par_max: ", par_max)

	if train == 'partial':

		par_max_test_llh = -log_PWL(par_max, test_seq, (1-frac)*T/frac)

	else:

		par_max_test_llh = None

	print('par_max_test_llh: ', par_max_test_llh)

	K1_Param = {'par_max_test_llh': par_max_test_llh, 'PWL_coeffs': par.x, 'K1_Type': 'PWL', 'PWL_statcriter': ((par.x[2]-1)*par.x[0]*(par.x[1]**(1-par.x[2]))), 'final_llh': fin_llh,\
	 'llh_renorm_K': llh_renorm_K, 'llh_renorm_c': llh_renorm_c, 'llh_renorm_p': llh_renorm_p, 'llh_renorm_Kc': llh_renorm_Kc,\
	  'llh_renorm_Kp': llh_renorm_Kp}#<1)*(par.x[2]>1)}

	return K1_Param

def log_PWL(PWL_coeffs, seq, T):

	epsilon = np.finfo(float).eps
	T = T #seq[-1]#-seq[0]
	Delta = len(seq)/T

	def funcpwl(x, K, c, p):
		return K * np.power(x + c, -p)

	K = PWL_coeffs[0]

	c = PWL_coeffs[1]

	p = PWL_coeffs[2]

	mu = PWL_coeffs[3]

	phi = K * np.power(c, 1. - p) / (1. - p)

	if (phi < 1.) and (K >= 0.) and (c > 0.) and (p > 1.):
		mu = mu  # (1.-phi)*Delta;
	else:
		mu = mu  # 0.
	# print("ill-conditioned parameters!")
	# return np.inf

	intens = np.zeros(len(seq));

	compens = mu * T;

	for i in range(0, len(seq)):

		intens[i] += mu;

		# print("T-seq[i]+c+epsilon: "+ repr(T-seq[i]+c+epsilon))

		# print("c:" + repr(c))

		# print("T-seq[i]: "+ repr(T-seq[i]))

		# print(K*np.power(T-seq[i]+c+epsilon,1-p)/(1-p))

		# print(K*np.power(c+epsilon,1-p)/(1-p))

		# print('(1-p): ' + repr((1-p)))

		compens += K * np.power(T - seq[i] + c + epsilon, 1 - p) / (1 - p + epsilon) - K * np.power(c + epsilon,
																									1 - p) / (
							   1 - p + epsilon)  # quad(funcpwl,0,T-seq[i], args=(K,c,p))[0] #(alpha/beta)*(1-np.exp(-beta*(T-seq[i])))

		# print('compens: ' + repr(compens))

		for j in range(0, i):
			intens[i] += K * np.power((seq[i] - seq[j]) + c + epsilon, -p)

	# print('intens: ' + repr(intens))

	intens[intens < 0.] = 0.

	print('Loglikelihood Train GD: ' + repr(np.sum(np.nan_to_num(np.log(intens + epsilon))) - compens) + '\n')

	return - np.sum(np.nan_to_num(np.log(intens + epsilon))) + max(compens, 0. + epsilon)