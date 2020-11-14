
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

from SimHawkesProcesses import simHP

#input_data = scipy.io.loadmat('4Kern_Renorm_10seq_T1000.mat')



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
train_method = 'full'
train_frac = 1.0

assert M >= 3, "M has to be greater or equal to 3 !!"

print("eps: ", eps, "n_of_seq: ", n_of_seq, "max_jumps: ", max_jumps, "M: ", M, "T: ", T, "mu: ", mu)

print("Simulating sequences")

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

print("Starting Optimization ...")

llh_GD_EXP = np.array([])
llh_GD_PWL = np.array([])
llh_GD_QEXP = np.array([])
llh_GD_RAY = np.array([])
llh_GD_GSS = np.array([])

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

		print(j)

		seq = simulated_sequences[kernel + "_" + str(i)]

		EXP_Param = trainGD_EXP(seq,eps,M, method, train_method, train_frac, T)

		PWL_Param = trainGD_PWL(seq,eps,M,method, train_method, train_frac, T)

		QEXP_Param = trainGD_QEXP(seq,eps,M,method, train_method, train_frac, T)

		RAY_Param = trainGD_RAY(seq,eps,M,method, train_method, train_frac, T)

		GSS_Param = trainGD_GSS(seq,eps,M,method, train_method, train_frac, T)

		llh_GD_EXP = np.append(llh_GD_EXP,EXP_Param['final_llh'])
		llh_GD_PWL = np.append(llh_GD_PWL,PWL_Param['final_llh'])
		llh_GD_QEXP = np.append(llh_GD_QEXP,QEXP_Param['final_llh'])
		llh_GD_RAY = np.append(llh_GD_RAY,RAY_Param['final_llh'])
		llh_GD_GSS = np.append(llh_GD_GSS, GSS_Param['final_llh'])

		llh_GD_EXP_Renorm_alpha = np.append(llh_GD_EXP_Renorm_alpha,EXP_Param['llh_renorm_alpha'])
		llh_GD_PWL_Renorm_K = np.append(llh_GD_PWL_Renorm_K,PWL_Param['llh_renorm_K'])
		llh_GD_QEXP_Renorm_a = np.append(llh_GD_QEXP_Renorm_a,QEXP_Param['llh_renorm_alpha'])
		llh_GD_RAY_Renorm_gamma = np.append(llh_GD_RAY_Renorm_gamma,RAY_Param['llh_renorm_gamma'])
		llh_GD_GSS_Renorm_kappa = np.append(llh_GD_GSS_Renorm_kappa, GSS_Param['llh_renorm_kappa'])

		llh_GD_EXP_Renorm_beta = np.append(llh_GD_EXP_Renorm_beta,EXP_Param['llh_renorm_beta'])
		llh_GD_PWL_Renorm_c = np.append(llh_GD_PWL_Renorm_c,PWL_Param['llh_renorm_c'])
		llh_GD_PWL_Renorm_p = np.append(llh_GD_PWL_Renorm_p,PWL_Param['llh_renorm_p'])
		llh_GD_QEXP_Renorm_q = np.append(llh_GD_QEXP_Renorm_q,QEXP_Param['llh_renorm_q'])
		llh_GD_RAY_Renorm_eta = np.append(llh_GD_RAY_Renorm_eta,RAY_Param['llh_renorm_eta'])
		llh_GD_GSS_Renorm_tau = np.append(llh_GD_GSS_Renorm_tau, GSS_Param['llh_renorm_tau'])

		llh_GD_EXP_Renorm_alphabeta = np.append(llh_GD_EXP_Renorm_alphabeta,EXP_Param['llh_renorm_sqrt'])
		llh_GD_PWL_Renorm_Kc = np.append(llh_GD_PWL_Renorm_Kc,PWL_Param['llh_renorm_Kc'])
		llh_GD_QEXP_Renorm_aq = np.append(llh_GD_QEXP_Renorm_aq,QEXP_Param['llh_renorm_sqrt'])
		llh_GD_RAY_Renorm_gammaeta = np.append(llh_GD_RAY_Renorm_gammaeta,RAY_Param['llh_renorm_sqrt'])
		llh_GD_GSS_Renorm_kappatau = np.append(llh_GD_GSS_Renorm_kappatau, GSS_Param['llh_renorm_sqrt'])

		llh_GD_PWL_Renorm_Kp = np.append(llh_GD_PWL_Renorm_Kp,PWL_Param['llh_renorm_Kp'])


		
print('llh_GD_EXP: ' + repr(llh_GD_EXP) + '\n')
print('llh_GD_EXP_Renorm_alpha: ' + repr(llh_GD_EXP_Renorm_alpha) + '\n')
print('llh_GD_EXP_Renorm_beta: ' + repr(llh_GD_EXP_Renorm_beta) + '\n')
print('llh_GD_EXP_Renorm_alphabeta: ' + repr(llh_GD_EXP_Renorm_alphabeta) + '\n')
max_EXP_Renorm = np.maximum.reduce(np.concatenate((llh_GD_EXP_Renorm_alpha,llh_GD_EXP_Renorm_beta,llh_GD_EXP_Renorm_alphabeta)))
#max_EXP_Renorm = np.maximum.reduce([llh_GD_EXP,max_EXP_Renorm])

print('llh_GD_PWL: ' + repr(llh_GD_PWL) + '\n')
print('llh_GD_PWL_Renorm_K: ' + repr(llh_GD_PWL_Renorm_K) + '\n')
print('llh_GD_PWL_Renorm_c: ' + repr(llh_GD_PWL_Renorm_c) + '\n')
print('lllh_GD_PWL_Renorm_p: ' + repr(llh_GD_PWL_Renorm_p) + '\n')
print('llh_GD_PWL_Renorm_Kc: ' + repr(llh_GD_PWL_Renorm_Kc) + '\n')
print('llh_GD_PWL_Renorm_Kp: ' + repr(llh_GD_PWL_Renorm_Kp) + '\n')
max_PWL_Renorm = np.maximum.reduce(np.concatenate((llh_GD_PWL_Renorm_K,llh_GD_PWL_Renorm_c,llh_GD_PWL_Renorm_p,llh_GD_PWL_Renorm_Kc,llh_GD_PWL_Renorm_Kp)))
#max_PWL_Renorm = np.maximum.reduce(np.concatenate((max_PWL_Renorm,llh_GD_PWL_Renorm_Kc,llh_GD_PWL_Renorm_Kp)))
#max_EXP_Renorm = np.maximum.reduce([llh_GD_PWL,max_PWL_Renorm])

print('llh_GD_QEXP: ' + repr(llh_GD_QEXP) + '\n')
print('llh_GD_QEXP_Renorm_a: ' + repr(llh_GD_QEXP_Renorm_a) + '\n')
print('lllh_GD_QEXP_Renorm_q: ' + repr(llh_GD_QEXP_Renorm_q) + '\n')
print('llh_GD_QEXP_Renorm_aq: ' + repr(llh_GD_QEXP_Renorm_aq) + '\n')
max_QEXP_Renorm = np.maximum.reduce(np.concatenate((llh_GD_QEXP_Renorm_a,llh_GD_QEXP_Renorm_q,llh_GD_QEXP_Renorm_aq)))
#max_QEXP_Renorm = np.maximum.reduce([llh_GD_QEXP,max_QEXP_Renorm])

print('llh_GD_RAY: ' + repr(llh_GD_RAY) + '\n')
print('llh_GD_RAY_Renorm_gamma: ' + repr(llh_GD_RAY_Renorm_gamma) + '\n')
print('llh_GD_RAY_Renorm_eta: ' + repr(llh_GD_RAY_Renorm_eta) + '\n')
print('llh_GD_RAY_Renorm_gammaeta: ' + repr(llh_GD_RAY_Renorm_gammaeta) + '\n')
max_RAY_Renorm = np.maximum.reduce(np.concatenate((llh_GD_RAY_Renorm_gamma,llh_GD_RAY_Renorm_eta,llh_GD_RAY_Renorm_gammaeta)))
#max_RAY_Renorm = np.maximum.reduce([llh_GD_RAY,max_RAY_Renorm])

print('llh_GD_GSS: ' + repr(llh_GD_GSS) + '\n')
print('llh_GD_GSS_Renorm_kappa: ' + repr(llh_GD_GSS_Renorm_kappa) + '\n')
print('llh_GD_GSS_Renorm_tau: ' + repr(llh_GD_GSS_Renorm_tau) + '\n')
print('llh_GD_GSS_Renorm_kappatau: ' + repr(llh_GD_GSS_Renorm_kappatau) + '\n')
max_GSS_Renorm = np.maximum.reduce(np.concatenate((llh_GD_GSS_Renorm_kappa,llh_GD_GSS_Renorm_tau,llh_GD_GSS_Renorm_kappatau)))
#max_RAY_Renorm = np.maximum.reduce([llh_GD_RAY,max_RAY_Renorm])

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

np.savez('llh_arrays_simulated_sequences_no_negative_mu_eps={}_n_of_seq={}_max_jumps={}_M={}_T={}_mu={}_method={}.npz'.format(eps,n_of_seq,max_jumps,M,T,mu,method),\
		 llh_GD_EXP=llh_GD_EXP,llh_GD_EXP_Renorm_alpha=llh_GD_EXP_Renorm_alpha,llh_GD_EXP_Renorm_beta=llh_GD_EXP_Renorm_beta,\
		 llh_GD_EXP_Renorm_alphabeta=llh_GD_EXP_Renorm_alphabeta,llh_GD_PWL=llh_GD_PWL,llh_GD_PWL_Renorm_K=llh_GD_PWL_Renorm_K,\
		 llh_GD_PWL_Renorm_c=llh_GD_PWL_Renorm_c,llh_GD_PWL_Renorm_p=llh_GD_PWL_Renorm_p,llh_GD_PWL_Renorm_Kc=llh_GD_PWL_Renorm_Kc,\
		 llh_GD_PWL_Renorm_Kp=llh_GD_PWL_Renorm_Kp,llh_GD_QEXP=llh_GD_QEXP,llh_GD_QEXP_Renorm_a=llh_GD_QEXP_Renorm_a,\
		 llh_GD_QEXP_Renorm_q=llh_GD_QEXP_Renorm_q,llh_GD_QEXP_Renorm_aq=llh_GD_QEXP_Renorm_aq,llh_GD_RAY=llh_GD_RAY,\
		 llh_GD_RAY_Renorm_gamma=llh_GD_RAY_Renorm_gamma,llh_GD_RAY_Renorm_eta=llh_GD_RAY_Renorm_eta,llh_GD_RAY_Renorm_gammaeta=llh_GD_RAY_Renorm_gammaeta,\
		 max_EXP_Renorm=max_EXP_Renorm,max_PWL_Renorm=max_PWL_Renorm,max_QEXP_Renorm=max_QEXP_Renorm,max_RAY_Renorm=max_RAY_Renorm,\
		 llh_GD_GSS=llh_GD_GSS, llh_GD_GSS_Renorm_kappa=llh_GD_GSS_Renorm_kappa, llh_GD_GSS_Renorm_tau=llh_GD_GSS_Renorm_tau,\
		 llh_GD_GSS_Renorm_kappatau=llh_GD_GSS_Renorm_kappatau, max_GSS_Renorm=max_GSS_Renorm)

