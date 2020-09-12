import numpy as np
import scipy.linalg as slin
import scipy.optimize as sopt
from scipy.special import expit as sigmoid
from Representation import GRAIL
import scipy.stats as st
from time import time


def notears_linear(X, lambda1, loss_type, rho_val, max_iter=100, h_tol=1e-8, rho_max=1e+16, w_threshold=0.3, mode = "base"):
    """Solve min_W L(W; X) + lambda1 ‖W‖_1 s.t. h(W) = 0 using augmented Lagrangian.

    Args:
        X (np.ndarray): [n, d] sample matrix
        lambda1 (float): l1 penalty parameter
        loss_type (str): l2, logistic, poisson
        max_iter (int): max num of dual ascent steps
        h_tol (float): exit if |h(w_est)| <= htol
        rho_max (float): exit if rho >= rho_max
        w_threshold (float): drop edge if |weight| < threshold

    Returns:
        W_est (np.ndarray): [d, d] estimated DAG
    """
    def _loss(W):
        """Evaluate value and gradient of loss."""
        M = X @ W
        if loss_type == 'l2':
            R = X - M
            loss = 0.5 / X.shape[0] * (R ** 2).sum()
            G_loss = - 1.0 / X.shape[0] * X.T @ R
        elif loss_type == 'logistic':
            loss = 1.0 / X.shape[0] * (np.logaddexp(0, M) - X * M).sum()
            G_loss = 1.0 / X.shape[0] * X.T @ (sigmoid(M) - X)
        elif loss_type == 'poisson':
            S = np.exp(M)
            loss = 1.0 / X.shape[0] * (S - X * M).sum()
            G_loss = 1.0 / X.shape[0] * X.T @ (S - X)
        else:
            raise ValueError('unknown loss type')
        return loss, G_loss

    def _h(W):
        """Evaluate value and gradient of acyclicity constraint."""
        #     E = slin.expm(W * W)  # (Zheng et al. 2018)
        #     h = np.trace(E) - d
        M = np.eye(d) + W * W / d  # (Yu et al. 2019)
        E = np.linalg.matrix_power(M, d - 1)
        h = (E.T * M).sum() - d
        G_h = E.T * W * 2
        return h, G_h

    def _adj(w):
        """Convert doubled variables ([2 d^2] array) back to original variables ([d, d] matrix)."""
        return (w[:d * d] - w[d * d:]).reshape([d, d])

    def _func(w):
        """Evaluate value and gradient of augmented Lagrangian for doubled variables ([2 d^2] array)."""
        W = _adj(w)
        loss, G_loss = _loss(W)
        h, G_h = _h(W)
        obj = loss + 0.5 * rho * h * h + alpha * h + lambda1 * w.sum()
        G_smooth = G_loss + (rho * h + alpha) * G_h
        g_obj = np.concatenate((G_smooth + lambda1, - G_smooth + lambda1), axis=None)
        return obj, g_obj

    n, d = X.shape

    if mode == "grail":
        grail = GRAIL(d = 10)
        representation = grail.get_representation(np.transpose(X))
        dist = np.zeros((d,d))
        for i in range(d):
            for j in range(d):
                dist[i,j] = np.linalg.norm(representation[i,:] - representation[j,:])
        mx = np.amax(dist)
        for i in range(d):
            for j in range(d):
                dist[i,j] = 1 - dist[i,j]/mx

        sim_threshold = np.percentile(dist, 80)
        for i in range(d):
            for j in range(d):
                if dist[i,j] <= sim_threshold:
                    dist[i,j] = 0

    w_est, rho, alpha, h = np.zeros(2 * d * d), rho_val, 0.0, np.inf  # double w_est into (w_pos, w_neg)
    if mode == "true":
        dummy = np.genfromtxt("W_true.csv", delimiter=",")
        w_est[:d*d] = dummy.flatten()
    if mode == "grail":
        w_est[:d*d] = dist.flatten()

    bnds = [(0, 0) if i == j else (0, None) for _ in range(2) for i in range(d) for j in range(d)]
    if loss_type == 'l2':
        X = X - np.mean(X, axis=0, keepdims=True)
    t = time()
    final_iter = max_iter
    final_time = 0
    for iterat in range(max_iter):
        w_new, h_new = None, None
        while rho < rho_max:
            sol = sopt.minimize(_func, w_est, method='L-BFGS-B', jac=True, bounds=bnds)
            w_new = sol.x
            h_new, _ = _h(_adj(w_new))
            if h_new > 0.25 * h:
                rho *= 10
            else:
                break
        w_est, h = w_new, h_new
        alpha += rho * h
        if h <= h_tol or rho >= rho_max:
            final_iter = iterat
            break
    W_est = _adj(w_est)
    W_est[np.abs(W_est) < w_threshold] = 0
    final_time = time() - t
    return W_est, final_iter, final_time


if __name__ == '__main__':
    import notears_utils as utils

    for mode in ["base", "true", "grail"]:
        rho = 1
        while rho < 1000000:
            acc_arr_fdr = np.zeros(25)
            acc_arr_tpr = np.zeros(25)
            acc_arr_fpr = np.zeros(25)
            acc_arr_shd = np.zeros(25)
            acc_arr_nnz = np.zeros(25)
            time_arr = np.zeros(25)
            iter_arr = np.zeros(25)
            for i in range(0,25):
                print(i)
                utils.set_random_seed(i)

                n, d, s0, graph_type, sem_type = 100, 20, 20, 'ER', 'gauss'
                B_true = utils.simulate_dag(d, s0, graph_type)
                W_true = utils.simulate_parameter(B_true)
                np.savetxt('W_true.csv', W_true, delimiter=',')

                X = utils.simulate_linear_sem(W_true, n, sem_type)
                np.savetxt('X.csv', X, delimiter=',')

                W_est, final_iter, final_time = notears_linear(X, rho_val=rho, lambda1=0.1, loss_type='l2', mode = mode)
                assert utils.is_dag(W_est)
                np.savetxt('W_est.csv', W_est, delimiter=',')
                acc = utils.count_accuracy(B_true, W_est != 0)
                acc_arr_tpr[i] = acc["tpr"]
                acc_arr_fdr[i] = acc["fdr"]
                acc_arr_fpr[i] = acc["fpr"]
                acc_arr_shd[i] = acc["shd"]
                acc_arr_nnz[i] = acc["nnz"]
                time_arr[i] = final_time
                iter_arr[i] = final_iter
            print(mode, " Results for rho = ", rho)
            print("iterations: ", np.mean(iter_arr))
            print("time: ", np.mean(time_arr))
            print("fdr = ", np.mean(acc_arr_fdr), "tpr = ", np.mean(acc_arr_tpr), "fpr = ", np.mean(acc_arr_fpr), "shd = ", np.mean(acc_arr_shd), "nnz = "
                  , np.mean(acc_arr_nnz))
            rho *= 10

