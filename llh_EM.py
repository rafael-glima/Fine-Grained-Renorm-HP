import numpy as np

def llh_EM(seq, mu, kernel_support, kernel_values, T):

    intens=np.array([mu])

    taumax = kernel_support[-1]

    for i in range(1,len(seq)):

        t = seq[i] - seq[:i]

        t = t[t <= taumax]

        if len(t) == 0:

            intens_i = mu

            intens = np.append(intens, intens_i)

            continue

        ind_bin = np.digitize(t,kernel_support,right=True)

        intens_i = mu

        for j in range(len(ind_bin)):

            intens_i += kernel_values[ind_bin[j]]

        intens = np.append(intens,intens_i)

    llh = np.sum(np.log(intens))

    #### Compensator ##############

    delta = kernel_support[1] - kernel_support[0]

    T = T #seq[-1]

    compens = mu*T

    for i in range(len(seq)):

        tmp = T-seq[i]

        if tmp >= taumax:

            compens += delta*np.sum(kernel_values)

        else:

            ind_tmp = np.digitize(tmp,kernel_support,right=True)

            if ind_tmp > 1:

                compens += np.sum(delta*kernel_values[:ind_tmp]) + delta*kernel_values[ind_tmp-1]*(tmp-kernel_support[ind_tmp-1])

            else:

                compens += delta*(tmp-kernel_support[ind_tmp-1])*kernel_values[ind_tmp-1]


    return llh - compens
