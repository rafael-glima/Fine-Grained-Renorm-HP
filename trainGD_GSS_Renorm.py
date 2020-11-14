import scipy.io
from scipy.optimize import minimize
import numpy as np
from scipy.integrate import quad
import math
from math import ceil
#import numpy.random as np.random
np.random.seed(2020)

def trainGD_GSS(seq,eps, M,method, train='full', frac=0.7, T=100):

	epsilon = np.finfo(float).eps

	if train == 'full':

		test_seq = seq

	elif train == 'partial':

		train_ind = ceil(frac*len(seq))

		test_seq = seq[seq > frac*T]

		test_seq -= test_seq[0]

		seq = seq[seq < frac*T]

	else:

		raise ValueError("Select full or partial training!")

	kappa_0 = np.random.rand()
	tau_0 = np.random.rand() #2*gamma_0
	sigma_0 = np.random.rand()
	mu_0 = np.random.rand()

	# input_data = scipy.io.loadmat('4Kern_newSNS_10seqT100000_highMuandfreq0.15.mat')
	# seq = input_data['Seq2'][1][0][0]

	# seq = seq[:300]

	T = frac*T #seq[-1] #-seq[0]
	Delta = len(seq)/T

	#seq = np.array([1.,2.,3.,4.,5.])

	bnds = ((0,None),(0,None),(0,None))

	def logGD_GSS(GSS_coeffs):

		def funcGSS(x,kappa,tau,sigma):
			return kappa*np.exp(-np.power(x-tau,2)/sigma)

		kappa = GSS_coeffs[1]

		tau = np.nan_to_num(GSS_coeffs[2])

		sigma = GSS_coeffs[3]

		mu = GSS_coeffs[0]

		print(" mu: ", mu, "kappa: ", kappa, " tau: ", tau, " sigma: ", sigma)

		# if(gamma/(2*eta) < 1.) and (gamma/(2*eta) >= 0.):
		mu = mu #(1.-gamma/(2*eta))*Delta;
		# else:
		# 	mu = mu#0.
			#return np.inf

		intens = np.zeros(len(seq));

		compens = mu*T;

		for i in range(0,len(seq)):

			intens[i] += mu

			#############  REVER ESSA PARTE !!!!!!!!!!!!! ##################################################
			if(T - seq[i] < tau):

				compens += 0.5*np.sqrt(np.pi*sigma)*kappa*(scipy.special.erf(tau/np.sqrt(sigma)) - scipy.special.erf((tau - T + seq[i])/np.sqrt(sigma))) #(1-np.exp(-eta*np.power(T-seq[i],2)))#quad(funcRAY,0,T-seq[i], args=(gamma,eta))[0]

			elif(T - seq[i] >= tau):

				compens += 0.5*np.sqrt(np.pi*sigma)*kappa*(scipy.special.erf(tau/np.sqrt(sigma)) + scipy.special.erf((T-seq[i] - tau)/np.sqrt(sigma)))

			#print('compens: '+repr(compens))

			for j in range(0,i):

				intens[i] += kappa*np.exp(-1*np.power(seq[i] - seq[j]-tau,2)/sigma)

			#print('intens_i: '+repr(intens))

		intens[intens < 0.] = 0. + epsilon

		print ('Loglikelihood Train GD: ' + repr(np.sum(np.nan_to_num(np.log(intens+epsilon))) - compens) + '\n')

		return - np.sum(np.nan_to_num(np.log(intens+epsilon))) + max(compens,0.+ epsilon)

	par = minimize(logGD_GSS, [mu_0, kappa_0, tau_0, sigma_0], method=method, tol=1e-2, options={'maxiter':10})

	print('Final Parameters: '+ repr(par.x)+'\n')

	par.x[2] = np.nan_to_num(par.x[2])

	GSS_statcriter = par.x[1]*(np.sqrt(np.pi*par.x[3])/2)*(scipy.special.erf(par.x[2]/np.sqrt(par.x[3])) + 1) #abs(par.x[1]/(2*par.x[2]))

	print('GSS_statcriter: ' + repr(GSS_statcriter))

	if par.x[0] < 0:

		fin_llh = np.inf

	else:

		fin_llh = logGD_GSS(par.x)

	fin_llh = (-1)*fin_llh

#	if np.isinf(fin_llh) or np.isnan(fin_llh) or (fin_llh > 0.):

	par_renorm_kappa = [Delta*(1.-1./(1.+eps)),par.x[1]/(GSS_statcriter*(1+eps)),par.x[2], par.x[3]]

	print("renorm_parameters: ", 2/(par.x[1]*np.sqrt(np.pi*par.x[3])*(1+eps)+epsilon) - 1, scipy.special.erfinv(min(1. - epsilon,2/(par.x[1]*np.sqrt(np.pi*par.x[3])*(1+eps)+epsilon) - 1)),scipy.special.erfinv(2*GSS_statcriter/(par.x[1]*np.sqrt(math.pi)*np.sqrt(par.x[3])) - 1))

	par_renorm_tau = [Delta*(1.-1./(1.+eps)),par.x[1],par.x[2]*scipy.special.erfinv(min(1.0-epsilon,2/(par.x[1]*np.sqrt(np.pi*par.x[3])*(1+eps)+epsilon) - 1))/(scipy.special.erfinv(min(1-epsilon,2*GSS_statcriter/(par.x[1]*np.sqrt(math.pi)*np.sqrt(par.x[3])) - 1))+epsilon), par.x[3]]

	for i in range(1,M):

		if(i == 1):

			par_renorm_sqrt = [Delta*(1.-1./(1.+eps)),par.x[1]/np.power(GSS_statcriter*(1+eps), i/M),par.x[2]*scipy.special.erfinv(min(1.0-epsilon,2/(par.x[1]*np.sqrt(np.pi*par.x[3])*np.power(1+eps,(M-i)/M)+epsilon) - 1))/(scipy.special.erfinv(min(1.0-epsilon,2*np.power(GSS_statcriter,(M-i)/M)/(eps+par.x[1]*np.sqrt(math.pi)*np.sqrt(par.x[3])) - 1))+epsilon), par.x[3]]

		else:

			par_renorm_sqrt = par_renorm_sqrt + [Delta*(1.-1./(1.+eps)),par.x[1]/np.power(GSS_statcriter*(1+eps), i/M),par.x[2]*(scipy.special.erfinv(min(1.0-epsilon,2/(par.x[1]*np.sqrt(np.pi*par.x[3])*np.power(1+eps,(M-i)/M)+epsilon) - 1))/(scipy.special.erfinv(min(1.0-epsilon,2*np.power(GSS_statcriter,(M-i)/M)/(eps+par.x[1]*np.sqrt(math.pi)*np.sqrt(par.x[3])) - 1)))+epsilon), par.x[3]]

	llh_renorm_kappa = logGD_GSS(par_renorm_kappa)

	llh_renorm_tau = logGD_GSS(par_renorm_tau)

	par_renorm_sqrt = np.reshape(par_renorm_sqrt, (-1,4))

	for i in range(1,M):

		if(i == 1):

			print(logGD_GSS(par_renorm_sqrt[i-1]))

			llh_renorm_sqrt = [logGD_GSS(par_renorm_sqrt[i-1])]

		else:

			print(logGD_GSS(par_renorm_sqrt[i-1]))

			llh_renorm_sqrt = llh_renorm_sqrt + [logGD_GSS(par_renorm_sqrt[i-1])]

	llh_renorm_kappa *= -1

	llh_renorm_tau *= -1

	llh_renorm_sqrt = [-1*item for item in llh_renorm_sqrt]

	print('par_renorm_kappa: '+repr(par_renorm_kappa))

	print('par_renorm_tau: '+repr(par_renorm_tau))

	print('par_renorm_sqrt: '+repr(par_renorm_sqrt))

	print('llh_renorm_kappa: '+repr(llh_renorm_kappa))

	print('llh_renorm_tau: '+repr(llh_renorm_tau))

	print('llh_renorm_sqrt: '+repr(llh_renorm_sqrt))

	par_max_sqrt = par_renorm_sqrt[np.argmax(llh_renorm_sqrt)]

	print("Par_max_sqrt: ", par_max_sqrt)

	llh_arr = [fin_llh] + [llh_renorm_kappa] + [np.max(llh_renorm_sqrt)] + [llh_renorm_tau]

	par_arr = [par.x] + [par_renorm_kappa] + [par_max_sqrt] + [par_renorm_tau]

	par_max = par_arr[np.argmax(llh_arr)]

	print("par_max: ", par_max)

	if train == 'partial':

		par_max_test_llh = -log_GSS(par_max, test_seq, (1-frac)*T/frac)

	else:

		par_max_test_llh = None

	print('par_max_test_llh: ', par_max_test_llh)

	K1_Param = {'par_max_test_llh': par_max_test_llh, 'GSS_coeffs': par.x, 'K1_Type': 'GSS', 'GSS_statcriter': par.x[1]*(scipy.special.erf(par.x[2]/np.sqrt(par.x[3])) + 0.5), 'final_llh': fin_llh, 'par_renorm_kappa': par_renorm_kappa,\
	'llh_renorm_kappa': llh_renorm_kappa, 'par_renorm_tau': par_renorm_tau, 'llh_renorm_tau': llh_renorm_tau, 'par_renorm_sqrt': par_renorm_sqrt, 'llh_renorm_sqrt': llh_renorm_sqrt}

	# else:

	# 	llh_renorm_gamma = fin_llh

	# 	llh_renorm_eta = fin_llh

	# 	llh_renorm_sqrt = fin_llh

	# 	K1_Param = {'RAY_coeffs': par.x, 'K1_Type': 'RAY', 'RAY_statcriter': par.x[1]/par.x[2], 'final_llh': fin_llh,\
	# 	'llh_renorm_gamma': llh_renorm_gamma, 'llh_renorm_eta': llh_renorm_eta, 'llh_renorm_sqrt':llh_renorm_sqrt}

	return K1_Param

def log_GSS(GSS_coeffs, seq, T):

	epsilon = np.finfo(float).eps
	T = T #seq[-1]#-seq[0]
	Delta = len(seq)/T

	def funcGSS(x,kappa,tau,sigma):
		return kappa*np.exp(-np.power(x-tau,2)/sigma)

	kappa = GSS_coeffs[1]

	tau = np.nan_to_num(GSS_coeffs[2])

	sigma = GSS_coeffs[3]

	mu = GSS_coeffs[0]

	print(" mu: ", mu, "kappa: ", kappa, " tau: ", tau, " sigma: ", sigma)

	# if(gamma/(2*eta) < 1.) and (gamma/(2*eta) >= 0.):
	mu = mu #(1.-gamma/(2*eta))*Delta;
	# else:
	# 	mu = mu#0.
		#return np.inf

	intens = np.zeros(len(seq));

	compens = mu*T;

	for i in range(0,len(seq)):

		intens[i] += mu

		#############  REVER ESSA PARTE !!!!!!!!!!!!! ##################################################
		if(T - seq[i] < tau):

			compens += 0.5*np.sqrt(np.pi*sigma)*kappa*(scipy.special.erf(tau/np.sqrt(sigma)) - scipy.special.erf((tau - T + seq[i])/np.sqrt(sigma))) #(1-np.exp(-eta*np.power(T-seq[i],2)))#quad(funcRAY,0,T-seq[i], args=(gamma,eta))[0]

		elif(T - seq[i] >= tau):

			compens += 0.5*np.sqrt(np.pi*sigma)*kappa*(scipy.special.erf(tau/np.sqrt(sigma)) + scipy.special.erf((T-seq[i] - tau)/np.sqrt(sigma)))

		#print('compens: '+repr(compens))

		for j in range(0,i):

			intens[i] += kappa*np.exp(-1*np.power(seq[i] - seq[j]-tau,2)/sigma)

		#print('intens_i: '+repr(intens))

	intens[intens < 0.] = 0. + epsilon

	print ('Loglikelihood Train GD: ' + repr(np.sum(np.nan_to_num(np.log(intens+epsilon))) - compens) + '\n')

	return - np.sum(np.nan_to_num(np.log(intens+epsilon))) + max(compens,0.+ epsilon)