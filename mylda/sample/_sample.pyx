# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

from cython.operator cimport preincrement as inc, predecrement as dec
from libc.stdlib cimport malloc, free, srand, rand, RAND_MAX
import numpy as np
cimport numpy as np

cpdef _train(int[:,:] n_mk, int[:,:] n_kt, int [:] n_kt_sum,
            int[:] W, int[:] Z, int[:] N_m, int[:] I_m,
            double alpha, double beta):
    cdef:
        int m, i, num_z_change = 0
        int M = n_mk.shape[0]
        int K = n_kt.shape[0]
        int T = n_kt.shape[1]
        double *theta_estimate = <double *>malloc(K*sizeof(double))
        double *phi_t_estimate = <double *>malloc(K*sizeof(double))
        double *pval = <double *>malloc(K*sizeof(double))

    for m in range(M):
        for i in range(N_m[m]):
            num_z_change += _sample_topic(n_mk, n_kt, n_kt_sum, W, Z, I_m, N_m, m, i, alpha, beta, theta_estimate, phi_t_estimate, pval)

    free(theta_estimate)
    free(phi_t_estimate)
    free(pval)
    return num_z_change


cdef _sample_topic(int[:,:] n_mk, int[:,:] n_kt, int[:] n_kt_sum,
                   int[:] W, int[:] Z, int[:] I_m, int[:] N_m,
                   int m, int i,
                   double alpha, double beta,
                   double *theta_estimate, double *phi_t_estimate, double *pval):
    cdef:
        int k, t
        double cum
        int w = W[I_m[m] + i]
        int z = Z[I_m[m] + i]
        int z_orig = z
        int K = n_kt.shape[0]
        int T = n_kt.shape[1]

    dec(n_mk[m, z])
    dec(n_kt[z, w])
    dec(n_kt_sum[z])

    cum = N_m[m] + K * alpha
    for k in range(K):
        theta_estimate[k] = (n_mk[m,k] + alpha) / cum

    for k in range(K):
        cum = n_kt_sum[k]
        phi_t_estimate[k] = (n_kt[k, w] + beta)/(cum + T * beta)
        pval[k] = theta_estimate[k] * phi_t_estimate[k]

    cum = 0
    for k in range(K):
        cum += pval[k]
    for k in range(K):
        pval[k] /= cum


    cum = <double>rand() / <double>RAND_MAX

    z = K - 1
    for k in range(K):
        cum -= pval[k]
        if cum < 0:
            z = k
            break

    Z[I_m[m]+i] = z
    inc(n_kt[z, w])
    inc(n_mk[m, z])
    inc(n_kt_sum[z])
    if z != z_orig:
        return 1
    else:
        return 0


cdef _predict_word(double[:,:] phi, int[:] doc, int[:] n_z, int n, int N,
                   double alpha, double *theta_estimate, double *pval):
    cdef:
        int k, t, z
        int w_i = doc[n]
        int K = phi.shape[0]
        int T = phi.shape[1]
        double cum = N + K * alpha

    cum = N + K * alpha
    for k in range(K):
        theta_estimate[k] = (n_z[k] + alpha) / cum
        pval[k] = theta_estimate[k] * phi[k, w_i]

    cum = 0
    for k in range(K):
        cum += pval[k]
    for k in range(K):
        pval[k] /= cum

    cum = <double>rand() / <double>RAND_MAX

    z = K - 1
    for k in range(K):
        cum -= pval[k]
        if cum < 0:
            z = k
            break
    return z


cpdef _predict(double[:,:] phi, int[:] doc, int[:] z, int[:] n_z, double alpha):
    cdef:
        int n, z_i_old, z_i_new
        int N = len(doc)
        int K = phi.shape[0]
        double *theta_estimate = <double *>malloc(K*sizeof(double))
        double *pval = <double *>malloc(K*sizeof(double))

    for n in range(N):
        z_i_old = z[n]
        dec(n_z[z_i_old])
        z_i_new = _predict_word(phi, doc, n_z, n, N, alpha,
                                theta_estimate, pval)
        inc(n_z[z_i_new])
        z[n] = z_i_new

    free(theta_estimate)
    free(pval)
    return z

