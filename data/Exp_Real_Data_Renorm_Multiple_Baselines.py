
import argparse
import numpy as np
import scipy.io
import math
import matplotlib.pyplot as plt
import random as rd
import pickle
rd.seed(2018)
np.random.seed(2018)

import pandas as pd
import datetime as DT
import time
from math import ceil
from llh_EM import llh_EM

from joblib import Parallel, delayed
import multiprocessing
import parmap
from multiprocessing import Pool, freeze_support
import itertools

from trainGD_EXP_Renorm import trainGD_EXP
from trainGD_PWL_Renorm import trainGD_PWL
from trainGD_QEXP_Renorm_New import trainGD_QEXP
from trainGD_RAY_Renorm import trainGD_RAY
from trainGD_GSS_Renorm import trainGD_GSS

from tick.hawkes import HawkesEM, HawkesBasisKernels, HawkesConditionalLaw
# from tick.simulation import SimuHawkes, HawkesKernelTimeFunc, HawkesKernelExp
# from tick.base import TimeFunction
# from tick.plot import plot_hawkes_kernels
# from tick.inference import HawkesExpKern, HawkesSumExpKern
# from tick.inference import HawkesBasisKernels

from SimHawkesProcesses import simHP

#input_data = scipy.io.loadmat('4Kern_Renorm_10seq_T1000.mat')

DATA_TYPE = 'retweet'

parser = argparse.ArgumentParser(description='Get parameters for Experiment: eps, n_of_seq, max_jumps, M and T.')
parser.add_argument('--eps', metavar='N', type=float, help='eps: the border distance parameter')
parser.add_argument('--n_of_seq', type=int, help='number of sequences')
parser.add_argument('--max_jumps', type=int,help='max number of jumps in each sequence')
parser.add_argument('--M', type = int,help='Stability Resolution Criterion')
parser.add_argument('--T', type=float, help='Time horizon for sequence simulation.')
parser.add_argument('--mu', type=float, help='Baseline Intensity.')
parser.add_argument('--method', type=str, help='Baseline Intensity.')

args = parser.parse_args()

args = args.__dict__

eps = args['eps']
n_of_seq = args['n_of_seq']
max_jumps = args['max_jumps']
M = args['M']
T = args['T']
mu = args['mu']
method = args['method']
train_method = 'partial'
train_frac = 0.7
min_seq_length = 20

assert M >= 3, "M has to be greater or equal to 3 !!"

print("eps: ", eps, "n_of_seq: ", n_of_seq, "max_jumps: ", max_jumps, "M: ", M, "T: ", T, "mu: ", mu)

print("Simulating sequences")

if DATA_TYPE == 'simulated':

	kernel_list = ['EXP','PWL','QEXP','RAY','GSS']

	kernel_coeffs = {'EXP': [1.,1.1], 'PWL': [0.9,1.,2.], 'GSS': [0.5,0.5,1.], 'RAY': [1.2,1.], 'QEXP': [0.8,1.1]}

	def compute_stationary_criterions(coeffs, type):

			if type == 'EXP':

				return coeffs[0]/coeffs[1]

			if type == 'PWL':

				assert coeffs[2] > 1, "p from PWL kernel is lower than 1 !!!"

				return coeffs[0]*np.power(coeffs[1],1-coeffs[2])/(coeffs[2]-1)

			if type == 'RAY':

				return coeffs[0]/(2*coeffs[1])

			if type == 'GSS':

				return (1 + scipy.special.erf(coeffs[1]/coeffs[2]))*coeffs[0]*np.sqrt(np.pi*coeffs[2])/2

			if type == 'QEXP':

				#assert (q >= 1) and (q < 2), "q parameter from QEXP is not in [1,2) !!!"

				return coeffs[0]/(2 - coeffs[1])

	for item in kernel_list:

		kernel_coeffs[item + '_statcriter'] = compute_stationary_criterions(kernel_coeffs[item], item)

else:

	kernel_list = ['REAL']

if DATA_TYPE == 'simulated':

	simulated_sequences = {}

	for i in range(n_of_seq):

		for kernel in kernel_list:

			print(kernel, ": ", kernel_coeffs[kernel + '_statcriter'])

			simulated_sequences[kernel + '_' + str(i)] = simHP(1, {'K1_Type': kernel, kernel + '_statcriter': kernel_coeffs[kernel + '_statcriter'],
													  kernel + '_coeffs': kernel_coeffs[kernel]}, max_jumps, T, mu/(1-kernel_coeffs[kernel + '_statcriter']))

	with open("simulated_sequences_eps={}_n_of_seq={}_max_jumps={}_M={}_T={}_mu={}.pkl".format(eps,n_of_seq,max_jumps,M,T,mu),"wb") as f:

		pickle.dump(simulated_sequences, f)

	f.close()

	with open("kernel_parameters_eps={}_n_of_seq={}_max_jumps={}_M={}_T={}_mu={}.pkl".format(eps,n_of_seq,max_jumps,M,T,mu), "wb") as f:
		pickle.dump(kernel_coeffs, f)

	f.close()

else:

	if (DATA_TYPE in ['conttime', 'hawkes', 'hawkesinhib', 'meme', 'missing', 'retweet']):

		train_pkl = '../../RLPP/data/data_' + DATA_TYPE + '/train.pkl'

		test_pkl = '../../RLPP/data/data_' + DATA_TYPE + '/test.pkl'

	elif (DATA_TYPE in ['bookorder', 'mimic', 'so']):

		train_pkl = '../../RLPP/data/data_' + DATA_TYPE + '/fold1/train.pkl'

		test_pkl = '../../RLPP/data/data_' + DATA_TYPE + '/fold1/test.pkl'

	else:
		raise ValueError(
			"Incorrect data type! Please select one among: \n['conttime','hawkes','hawkesinhib','meme','missing','retweet','bookorder','mimic','so']")

	for pkl_file in [train_pkl, test_pkl]:

		with open(pkl_file, 'rb') as f:
			data_temp = pickle.load(f, encoding='latin1')

		f.close()

		with open(pkl_file.replace('.pkl', '.txt'), 'w') as f:

			for s in data_temp[pkl_file.split('/')[-1].replace('.pkl', '')]:
				seq = []
				for item in s:
					seq = seq + [item['time_since_start']]
				#seq = seq / seq[-1]
				seq = [str(item) for item in seq]
				f.write('[' + ','.join(seq) + ']\n')

		f.close()

	train_sequences = np.genfromtxt(train_pkl.replace('.pkl', '.txt'), dtype=str)  # , delimiter='\n')
	test_sequences = np.genfromtxt(test_pkl.replace('.pkl', '.txt'), dtype=str)  # , delimiter='\n')


	def convert_sequences(original_sequences):

		max_len = 0

		for seq in original_sequences:

			if (len([float(item) for item in seq.strip('[]').split(',')]) > max_len):
				max_len = len([float(item) for item in seq.strip('[]').split(',')])

		# print(max_len)

		converted_array = np.array([[float(item) for item in original_sequences[0].strip('[]').split(',')] + [0] * (
					max_len - len([float(item) for item in original_sequences[0].strip('[]').split(',')]))],
								   dtype=np.float32)

		# print(converted_array.shape)

		for seq in original_sequences[1:]:
			converted_array = np.append(converted_array, np.array([[float(item) for item in
																	seq.strip('[]').split(',')] + [0] * (max_len - len(
				[float(item) for item in seq.strip('[]').split(',')]))], dtype=np.float32), axis=0)

		# converted_array = np.reshape(converted_array,(-1,max_len))

		return converted_array


	simulated_sequences = convert_sequences(train_sequences)
	n_of_seq = simulated_sequences.shape[0]
	print("n_of_seq: ", n_of_seq)

	# test_sequences = convert_sequences(test_sequences)

print("Starting Optimization ...")

llh_GD_EXP = np.array([])
llh_GD_PWL = np.array([])
llh_GD_QEXP = np.array([])
llh_GD_RAY = np.array([])
llh_GD_GSS = np.array([])
llh_EM__ = np.array([])
llh_Zhou_ = np.array([])
llh_CL__ = np.array([])

llh_GD_EXP_Renorm_alpha = np.array([])
llh_GD_PWL_Renorm_K = np.array([])
llh_GD_QEXP_Renorm_a = np.array([])
llh_GD_RAY_Renorm_gamma = np.array([])
llh_GD_GSS_Renorm_kappa = np.array([])

llh_GD_EXP_Renorm_beta = np.array([])
llh_GD_PWL_Renorm_c = np.array([])
llh_GD_PWL_Renorm_p = np.array([])
llh_GD_QEXP_Renorm_q = np.array([])
llh_GD_RAY_Renorm_eta = np.array([])
llh_GD_GSS_Renorm_tau = np.array([])

llh_GD_EXP_Renorm_alphabeta = np.array([])
llh_GD_PWL_Renorm_Kc = np.array([])
llh_GD_QEXP_Renorm_aq = np.array([])
llh_GD_RAY_Renorm_gammaeta = np.array([])
llh_GD_GSS_Renorm_kappatau = np.array([])

llh_GD_PWL_Renorm_Kp = np.array([])

for kernel in kernel_list:

	print(kernel)

	for j in range(n_of_seq):

		try:

			print(j)

			if DATA_TYPE == 'simulated':

				seq = simulated_sequences[kernel + "_" + str(i)]

				seq_size = 1

			else:

				seq = simulated_sequences[j]

				seq = seq[seq > 0]

				if (len(seq) > min_seq_length):

					T = seq[-1]

					print("len_seq: ", len(seq))

					#seq = seq - seq[0]

					train_ind = ceil(train_frac * len(seq))

					test_seq = seq[seq > train_frac*T]

					test_seq -= test_seq[0]

					seq_size = len(test_seq)


			if ((len(seq) > min_seq_length) and (DATA_TYPE != 'simulated')) or (DATA_TYPE == 'simulated'):

				EXP_Param = trainGD_EXP(seq,eps,M, method, train_method, train_frac, T)

				PWL_Param = trainGD_PWL(seq,eps,M,method, train_method, train_frac, T)

				QEXP_Param = trainGD_QEXP(seq,eps,M,method, train_method, train_frac, T)

				RAY_Param = trainGD_RAY(seq,eps,M,method, train_method, train_frac, T)

				GSS_Param = trainGD_GSS(seq,eps,M,method, train_method, train_frac, T)

				llh_GD_EXP = np.append(llh_GD_EXP,EXP_Param['par_max_test_llh']/seq_size)
				llh_GD_PWL = np.append(llh_GD_PWL,PWL_Param['par_max_test_llh']/seq_size)
				llh_GD_QEXP = np.append(llh_GD_QEXP,QEXP_Param['par_max_test_llh']/seq_size)
				llh_GD_RAY = np.append(llh_GD_RAY,RAY_Param['par_max_test_llh']/seq_size)
				llh_GD_GSS = np.append(llh_GD_GSS, GSS_Param['par_max_test_llh']/seq_size)

				###################### Mohler E-M Method ###########################

				if train_method == 'full':

					assert frac == 1.0, "For full train, frac must be set to 1."

				# if train_method == 'full':
				#
				# 	test_seq = seq
				#
				# elif train_method == 'partial':
				#
				# 	train_ind = ceil(train_frac * len(seq))
				#
				# 	test_seq = seq[seq > train_frac*T]
				#
				# 	test_seq -= test_seq[0]
				#
				# 	seq = seq[seq < train_frac*T]
				#
				# else:
				#
				# 	raise ValueError("Select full or partial training!")
				#
				taumax = np.max(np.diff(np.array(seq)))

				print('taumax: ', taumax)

				n_bins = 20

				em = HawkesEM(taumax,n_bins,8)#(kernel_support=taumax, kernel_size=n_bins, max_n_threads=8)#, verbose=False, tol=1e-3)

				print(seq.dtype)

				try:
					em.fit([[np.array(seq, dtype=np.float64)]])
				except:
					em.fit([seq])

				kernel_support = np.linspace(0., taumax, n_bins + 1)

				kernel_values = em.get_kernel_values(0, 0, kernel_support)

				mu = em.baseline

				# print('timestamps: '+repr([seq][0]))

				if train_method == 'partial':

					llh_EM_ = llh_EM(test_seq, mu, kernel_support, kernel_values, (1-train_frac)*T)

				elif train_method == 'full': #em.score([test_seq]) #llh_EM(test_seq, mu, kernel_support, kernel_values)

					llh_EM_ = llh_EM(test_seq, mu, kernel_support, kernel_values, train_frac*T)

				# print('llh_EM: '+repr(em.score([seq])))

				print('llh_EM: ' + repr(llh_EM_))

				llh_EM__ = np.append(llh_EM__, llh_EM_/seq_size)

				###################### Zhou Method ###########################

				end_time = 1e9
				C = 1e-3
				max_iter = 100

				zhou = HawkesBasisKernels(kernel_support=10, n_basis=3) #kernel_support=taumax, kernel_dt=tauman_bins, kernel_tmax=taumax, alpha=0.1, max_n_threads)
			# kernel_support=taumax)#,
		#     n_basis=None,
		#     kernel_size=10,
		#     tol=1e-05,
		#     C=0.1,
		#     max_iter=100,
		#     verbose=False,
		#     print_every=10,
		#     record_every=10,
		#     n_threads=1,
		#     ode_max_iter=100,
		#     ode_tol=1e-05,
		# )
				try:
					zhou.fit([[np.array(seq, dtype=np.float64)]])
				except:
					zhou.fit([seq])

				kernel_support = np.linspace(0., 10, 10)

				kernel_values = zhou.get_kernel_values(0, 0, kernel_support)

				mu = zhou.baseline

				# print('timestamps: '+repr([seq][0]))

				llh_Zhou = llh_EM(test_seq, mu, kernel_support, kernel_values, (1-train_frac)*T) #zhou.score([test_seq]) #

				# print('llh_EM: '+repr(em.score([seq])))

				print('llh_Zhou: ' + repr(llh_Zhou))

				llh_Zhou_ = np.append(llh_Zhou_, llh_Zhou/seq_size)

				############ Conditional Law ###################

				# cl = HawkesConditionalLaw(claw_method="log", delta_lag=0.1, min_lag=0.002,
				#                  max_lag=100, quad_method="log", n_quad=n_bins,
				#                  min_support=0.002, max_support=taumax, n_threads=-1)
				# print(seq.shape)
				#
				# cl.fit([seq])
				#
				# kernel_support = np.linspace(0., taumax, n_bins + 1)
				#
				# kernel_values = cl.kernels(0, 0, kernel_support)
				#
				# print(kernel_values)
				#
				# mu = cl.baseline
				#
				# # print('timestamps: '+repr([seq][0]))
				#
				# if train_method == 'partial':
				#
				# 	llh_CL_ = llh_EM(test_seq, mu, kernel_support, kernel_values, (1-train_frac)*T)
				#
				# elif train_method == 'full': #em.score([test_seq]) #llh_EM(test_seq, mu, kernel_support, kernel_values)
				#
				# 	llh_CL_ = llh_EM(test_seq, mu, kernel_support, kernel_values, train_frac*T)
				#
				# # print('llh_EM: '+repr(em.score([seq])))
				#
				# print('llh_CL: ' + repr(llh_CL_))
				#
				# llh_CL__ = np.append(llh_CL__, llh_CL_)


				llh_GD_EXP_Renorm_alpha = np.append(llh_GD_EXP_Renorm_alpha,EXP_Param['llh_renorm_alpha']/seq_size)
				llh_GD_PWL_Renorm_K = np.append(llh_GD_PWL_Renorm_K,PWL_Param['llh_renorm_K']/seq_size)
				llh_GD_QEXP_Renorm_a = np.append(llh_GD_QEXP_Renorm_a,QEXP_Param['llh_renorm_alpha']/seq_size)
				llh_GD_RAY_Renorm_gamma = np.append(llh_GD_RAY_Renorm_gamma,RAY_Param['llh_renorm_gamma']/seq_size)
				llh_GD_GSS_Renorm_kappa = np.append(llh_GD_GSS_Renorm_kappa, GSS_Param['llh_renorm_kappa']/seq_size)

				llh_GD_EXP_Renorm_beta = np.append(llh_GD_EXP_Renorm_beta,EXP_Param['llh_renorm_beta']/seq_size)
				llh_GD_PWL_Renorm_c = np.append(llh_GD_PWL_Renorm_c,PWL_Param['llh_renorm_c']/seq_size)
				llh_GD_PWL_Renorm_p = np.append(llh_GD_PWL_Renorm_p,PWL_Param['llh_renorm_p']/seq_size)
				llh_GD_QEXP_Renorm_q = np.append(llh_GD_QEXP_Renorm_q,QEXP_Param['llh_renorm_q']/seq_size)
				llh_GD_RAY_Renorm_eta = np.append(llh_GD_RAY_Renorm_eta,RAY_Param['llh_renorm_eta']/seq_size)
				llh_GD_GSS_Renorm_tau = np.append(llh_GD_GSS_Renorm_tau, GSS_Param['llh_renorm_tau']/seq_size)

				llh_GD_EXP_Renorm_alphabeta = np.append(llh_GD_EXP_Renorm_alphabeta,[item/seq_size for item in EXP_Param['llh_renorm_sqrt']])
				llh_GD_PWL_Renorm_Kc = np.append(llh_GD_PWL_Renorm_Kc,[item/seq_size for item in PWL_Param['llh_renorm_Kc']])
				llh_GD_QEXP_Renorm_aq = np.append(llh_GD_QEXP_Renorm_aq,[item/seq_size for item in QEXP_Param['llh_renorm_sqrt']])
				llh_GD_RAY_Renorm_gammaeta = np.append(llh_GD_RAY_Renorm_gammaeta,[item/seq_size for item in RAY_Param['llh_renorm_sqrt']])
				llh_GD_GSS_Renorm_kappatau = np.append(llh_GD_GSS_Renorm_kappatau, [item/seq_size for item in GSS_Param['llh_renorm_sqrt']])

				llh_GD_PWL_Renorm_Kp = np.append(llh_GD_PWL_Renorm_Kp,[item/seq_size for item in PWL_Param['llh_renorm_Kp']])
		except:

			print("Problem with sequence {}".format(j))

		
print('llh_GD_EXP: ' + repr(llh_GD_EXP) + '\n')
# print('llh_GD_EXP_Renorm_alpha: ' + repr(llh_GD_EXP_Renorm_alpha) + '\n')
# print('llh_GD_EXP_Renorm_beta: ' + repr(llh_GD_EXP_Renorm_beta) + '\n')
# print('llh_GD_EXP_Renorm_alphabeta: ' + repr(llh_GD_EXP_Renorm_alphabeta) + '\n')
max_EXP_Renorm = np.maximum.reduce(np.concatenate((llh_GD_EXP_Renorm_alpha,llh_GD_EXP_Renorm_beta,llh_GD_EXP_Renorm_alphabeta)))
#max_EXP_Renorm = np.maximum.reduce([llh_GD_EXP,max_EXP_Renorm])

print('llh_GD_PWL: ' + repr(llh_GD_PWL) + '\n')
# print('llh_GD_PWL_Renorm_K: ' + repr(llh_GD_PWL_Renorm_K) + '\n')
# print('llh_GD_PWL_Renorm_c: ' + repr(llh_GD_PWL_Renorm_c) + '\n')
# print('lllh_GD_PWL_Renorm_p: ' + repr(llh_GD_PWL_Renorm_p) + '\n')
# print('llh_GD_PWL_Renorm_Kc: ' + repr(llh_GD_PWL_Renorm_Kc) + '\n')
# print('llh_GD_PWL_Renorm_Kp: ' + repr(llh_GD_PWL_Renorm_Kp) + '\n')
max_PWL_Renorm = np.maximum.reduce(np.concatenate((llh_GD_PWL_Renorm_K,llh_GD_PWL_Renorm_c,llh_GD_PWL_Renorm_p,llh_GD_PWL_Renorm_Kc,llh_GD_PWL_Renorm_Kp)))
#max_PWL_Renorm = np.maximum.reduce(np.concatenate((max_PWL_Renorm,llh_GD_PWL_Renorm_Kc,llh_GD_PWL_Renorm_Kp)))
#max_EXP_Renorm = np.maximum.reduce([llh_GD_PWL,max_PWL_Renorm])

print('llh_GD_QEXP: ' + repr(llh_GD_QEXP) + '\n')
# print('llh_GD_QEXP_Renorm_a: ' + repr(llh_GD_QEXP_Renorm_a) + '\n')
# print('lllh_GD_QEXP_Renorm_q: ' + repr(llh_GD_QEXP_Renorm_q) + '\n')
# print('llh_GD_QEXP_Renorm_aq: ' + repr(llh_GD_QEXP_Renorm_aq) + '\n')
max_QEXP_Renorm = np.maximum.reduce(np.concatenate((llh_GD_QEXP_Renorm_a,llh_GD_QEXP_Renorm_q,llh_GD_QEXP_Renorm_aq)))
#max_QEXP_Renorm = np.maximum.reduce([llh_GD_QEXP,max_QEXP_Renorm])

print('llh_GD_RAY: ' + repr(llh_GD_RAY) + '\n')
# print('llh_GD_RAY_Renorm_gamma: ' + repr(llh_GD_RAY_Renorm_gamma) + '\n')
# print('llh_GD_RAY_Renorm_eta: ' + repr(llh_GD_RAY_Renorm_eta) + '\n')
# print('llh_GD_RAY_Renorm_gammaeta: ' + repr(llh_GD_RAY_Renorm_gammaeta) + '\n')
max_RAY_Renorm = np.maximum.reduce(np.concatenate((llh_GD_RAY_Renorm_gamma,llh_GD_RAY_Renorm_eta,llh_GD_RAY_Renorm_gammaeta)))
#max_RAY_Renorm = np.maximum.reduce([llh_GD_RAY,max_RAY_Renorm])

print('llh_GD_GSS: ' + repr(llh_GD_GSS) + '\n')
# print('llh_GD_GSS_Renorm_kappa: ' + repr(llh_GD_GSS_Renorm_kappa) + '\n')
# print('llh_GD_GSS_Renorm_tau: ' + repr(llh_GD_GSS_Renorm_tau) + '\n')
# print('llh_GD_GSS_Renorm_kappatau: ' + repr(llh_GD_GSS_Renorm_kappatau) + '\n')
max_GSS_Renorm = np.maximum.reduce(np.concatenate((llh_GD_GSS_Renorm_kappa,llh_GD_GSS_Renorm_tau,llh_GD_GSS_Renorm_kappatau)))
#max_RAY_Renorm = np.maximum.reduce([llh_GD_RAY,max_RAY_Renorm])

print('llh_EM: ' + repr(llh_EM__))
print('llh_Zhou: ' + repr(llh_Zhou_))
print('llh_CL: ' + repr(llh_CL__))

f = open('Exp_Synthetic_Renorm_no_negative_mu_eps={}_n_of_seq={}_max_jumps={}_M={}_T={}_mu={}_method={}.txt'.format(eps,n_of_seq,max_jumps,M,T,mu,method),'w')
f.write('llh_GD_EXP: ' + repr(llh_GD_EXP) + '\n')
f.write('llh_GD_EXP_Renorm_alpha: ' + repr(llh_GD_EXP_Renorm_alpha) + '\n')
f.write('llh_GD_EXP_Renorm_beta: ' + repr(llh_GD_EXP_Renorm_beta) + '\n')
f.write('llh_GD_EXP_Renorm_alphabeta: ' + repr(llh_GD_EXP_Renorm_alphabeta) + '\n')
f.write('max_EXP_Renorm: ' + repr(max_EXP_Renorm) + '\n')

f.write('llh_GD_PWL: ' + repr(llh_GD_PWL) + '\n')
f.write('llh_GD_PWL_Renorm_K: ' + repr(llh_GD_PWL_Renorm_K) + '\n')
f.write('llh_GD_PWL_Renorm_c: ' + repr(llh_GD_PWL_Renorm_c) + '\n')
f.write('lllh_GD_PWL_Renorm_p: ' + repr(llh_GD_PWL_Renorm_p) + '\n')
f.write('llh_GD_PWL_Renorm_Kc: ' + repr(llh_GD_PWL_Renorm_Kc) + '\n')
f.write('llh_GD_PWL_Renorm_Kp: ' + repr(llh_GD_PWL_Renorm_Kp) + '\n')
f.write('max_PWL_Renorm: ' + repr(max_PWL_Renorm) + '\n')

f.write('llh_GD_QEXP: ' + repr(llh_GD_QEXP) + '\n')
f.write('llh_GD_QEXP_Renorm_a: ' + repr(llh_GD_QEXP_Renorm_a) + '\n')
f.write('lllh_GD_QEXP_Renorm_q: ' + repr(llh_GD_QEXP_Renorm_q) + '\n')
f.write('llh_GD_QEXP_Renorm_aq: ' + repr(llh_GD_QEXP_Renorm_aq) + '\n')
f.write('max_QEXP_Renorm: ' + repr(max_QEXP_Renorm) + '\n')

f.write('llh_GD_RAY: ' + repr(llh_GD_RAY) + '\n')
f.write('llh_GD_RAY_Renorm_gamma: ' + repr(llh_GD_RAY_Renorm_gamma) + '\n')
f.write('llh_GD_RAY_Renorm_eta: ' + repr(llh_GD_RAY_Renorm_eta) + '\n')
f.write('llh_GD_RAY_Renorm_gammaeta: ' + repr(llh_GD_RAY_Renorm_gammaeta) + '\n')
f.write('max_RAY_Renorm: ' + repr(max_RAY_Renorm) + '\n')

f.write('llh_GD_GSS: ' + repr(llh_GD_GSS) + '\n')
f.write('llh_GD_GSS_Renorm_kappa: ' + repr(llh_GD_GSS_Renorm_kappa) + '\n')
f.write('llh_GD_GSS_Renorm_tau: ' + repr(llh_GD_GSS_Renorm_tau) + '\n')
f.write('llh_GD_GSS_Renorm_kappatau: ' + repr(llh_GD_GSS_Renorm_kappatau) + '\n')
f.write('max_GSS_Renorm: ' + repr(max_GSS_Renorm) + '\n')

f.close()

np.savez('llh_arrays_simulated_sequences_{}_data_fixed_T_eps={}_n_of_seq={}_max_jumps={}_M={}_T={}_mu={}_method={}.npz'.format(DATA_TYPE,eps,n_of_seq,max_jumps,M,T,mu,method),\
		 llh_GD_EXP=llh_GD_EXP,llh_GD_EXP_Renorm_alpha=llh_GD_EXP_Renorm_alpha,llh_GD_EXP_Renorm_beta=llh_GD_EXP_Renorm_beta,\
		 llh_GD_EXP_Renorm_alphabeta=llh_GD_EXP_Renorm_alphabeta,llh_GD_PWL=llh_GD_PWL,llh_GD_PWL_Renorm_K=llh_GD_PWL_Renorm_K,\
		 llh_GD_PWL_Renorm_c=llh_GD_PWL_Renorm_c,llh_GD_PWL_Renorm_p=llh_GD_PWL_Renorm_p,llh_GD_PWL_Renorm_Kc=llh_GD_PWL_Renorm_Kc,\
		 llh_GD_PWL_Renorm_Kp=llh_GD_PWL_Renorm_Kp,llh_GD_QEXP=llh_GD_QEXP,llh_GD_QEXP_Renorm_a=llh_GD_QEXP_Renorm_a,\
		 llh_GD_QEXP_Renorm_q=llh_GD_QEXP_Renorm_q,llh_GD_QEXP_Renorm_aq=llh_GD_QEXP_Renorm_aq,llh_GD_RAY=llh_GD_RAY,\
		 llh_GD_RAY_Renorm_gamma=llh_GD_RAY_Renorm_gamma,llh_GD_RAY_Renorm_eta=llh_GD_RAY_Renorm_eta,llh_GD_RAY_Renorm_gammaeta=llh_GD_RAY_Renorm_gammaeta,\
		 max_EXP_Renorm=max_EXP_Renorm,max_PWL_Renorm=max_PWL_Renorm,max_QEXP_Renorm=max_QEXP_Renorm,max_RAY_Renorm=max_RAY_Renorm,\
		 llh_GD_GSS=llh_GD_GSS, llh_GD_GSS_Renorm_kappa=llh_GD_GSS_Renorm_kappa, llh_GD_GSS_Renorm_tau=llh_GD_GSS_Renorm_tau,\
		 llh_GD_GSS_Renorm_kappatau=llh_GD_GSS_Renorm_kappatau, max_GSS_Renorm=max_GSS_Renorm, llh_EM__=llh_EM__, llh_Zhou_=llh_Zhou, llh_CL__=llh_CL__)

