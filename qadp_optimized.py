# Optimized version of QADP.py
# Algorithm unchanged — performance improvements:
# - Proper multiprocessing Pool (all CPU cores)
# - Removed Process/Pipe overhead
# - Larger parallel batches
# - ECOS primary solver, SCS fallback
# - Reduced array concatenation overhead

import numpy as np
import cvxpy as cp
import time
from itertools import product, combinations
from multiprocessing import Pool, cpu_count
import warnings
warnings.filterwarnings("ignore")

# ===================== Parameters =====================
pkt_prob = 0.5
e_prob = 0.5
weight_prob = 0.5

h_vals = [0.1, 1]
h_bad = 0.5
h_prob = [h_bad, (1-h_bad)]
weight_vals = [1, 2]
B_max = 2
rmax = 1
M = 2

B_vals   = np.arange(0, B_max + 1, 1)
rem_vals = np.arange(0, rmax + 0.5, 0.5)

# ===================== State Space =====================
all_states = []
for B in product(B_vals, repeat=M):
    for rem in product(rem_vals, repeat=M):
        for h in product(h_vals, repeat=M):
            for w in product(weight_vals, repeat=M):
                state = (*B, *rem, *h, *w)
                all_states.append(state)
all_states = np.array(all_states)
all_states = all_states[::3]  # subsample

# ===================== Utilities =====================
def st_tr_wt(wt, wt_next):
    if wt == 1 and wt_next == 1:
        return (1-pkt_prob) + (pkt_prob * weight_prob)
    elif wt == 1 and wt_next == 2:
        return pkt_prob * (1 - weight_prob)
    elif wt == 2 and wt_next == 1:
        return pkt_prob * weight_prob
    else:
        return (1-pkt_prob) + pkt_prob * (1-weight_prob)

# ===================== Initial Greedy Solve =====================
def one_slot_greedy_T_Mu(state):
    B   = state[0:M]
    rem = state[M:2*M]
    h   = state[2*M:3*M]
    wt  = state[3*M:4*M]

    rho = [cp.Variable(nonneg=True) for _ in range(M)]
    P   = [cp.Variable(nonneg=True) for _ in range(M)]

    obj = 0
    for i in range(M):
        obj += wt[i] * cp.exp(-(rmax - (rem[i] - rho[i])))

    log2 = cp.inv_pos(cp.log(2))
    cons = []

    for i in range(M):
        cons += [
            P[i]   <= B[i],
            rho[i] <= rem[i],
            rho[i] <= cp.log(1 + h[i]*P[i]) * log2
        ]

    users = list(range(M))
    for k in range(2, M+1):
        for S in combinations(users, k):
            sum_rho = sum(rho[i] for i in S)
            sum_pow = sum(h[i]*P[i] for i in S)
            cons.append(sum_rho <= cp.log(1 + sum_pow) * log2)

    prob = cp.Problem(cp.Minimize(obj), cons)
    try:
        return prob.solve(solver=cp.CLARABEL, warm_start=True, verbose=False)
    except:
        try:
            return prob.solve(solver=cp.ECOS, warm_start=True, verbose=False)
        except:
            return prob.solve(solver=cp.SCS, eps=1e-4, max_iters=5000, verbose=False)

# ===================== Initial Value =====================
print("Computing initial value function...")
V = np.array([abs(one_slot_greedy_T_Mu(s)) for s in all_states])

# ===================== Quadratic Fit =====================
def quad_fit(states, Vy):
    X = states
    y = Vy.reshape(-1,1)
    n = X.shape[1]

    P = cp.Variable((n,n), symmetric=True)
    q = cp.Variable(n)
    r = cp.Variable()

    temp = cp.vstack([cp.quad_form(X[i,:], P) + X[i,:] @ q + r for i in range(X.shape[0])])

    obj = cp.Minimize(cp.sum_squares(temp - y))
    cons = [P >> 0, temp >= 1e-8]

    prob = cp.Problem(obj, cons)
    prob.solve(solver=cp.SCS, eps=1e-4, verbose=False)

    return P.value, q.value, r.value

# ===================== Fast Parallel Greedy =====================
def one_slot_greedy_fast(args):
    state, P_opt, q_opt, r_opt, gamma = args

    B   = state[0:M]
    rem = state[M:2*M]
    h   = state[2*M:3*M]
    wt  = state[3*M:4*M]

    rho = [cp.Variable(nonneg=True) for _ in range(M)]
    P   = [cp.Variable(nonneg=True) for _ in range(M)]

    obj = 0
    for i in range(M):
        obj += wt[i] * cp.exp(-(rmax - (rem[i] - rho[i])))

    B_end = [B[i] - P[i] for i in range(M)]
    B_ch_arr = [[B_end[i], B_end[i] + 1] for i in range(M)]
    r_ch_arr = [[rem[i] - rho[i], rmax] for i in range(M)]
    wt_ch_arr = [[1,2] for _ in range(M)]

    next_states = list(product(*B_ch_arr, *r_ch_arr, *([h_vals]*M), *wt_ch_arr))

    # Faster quadratic value evaluation
    Vz_list = []
    for ns in next_states:
        x = cp.hstack(ns)
        Vz_list.append(cp.quad_form(x, P_opt) + x @ q_opt + r_opt)
    Vz = cp.hstack(Vz_list)

    B_ch_pr = np.array([1-e_prob, e_prob])
    r_ch_pr = np.array([1-pkt_prob, pkt_prob])

    wt_ch_pr = []
    for i in range(M):
        if wt[i] == 1:
            wt_ch_pr.append([st_tr_wt(1,1), st_tr_wt(1,2)])
        else:
            wt_ch_pr.append([st_tr_wt(2,1), st_tr_wt(2,2)])

    probs = []
    for combo in product(*([B_ch_pr]*M), *([r_ch_pr]*M), *([h_prob]*M), *wt_ch_pr):
        pr = 1
        for x in combo:
            pr *= x
        probs.append(pr)
    probs = cp.hstack(np.array(probs))

    obj += gamma * (Vz @ probs)

    cons = []
    for i in range(M):
        cons += [P[i] >= 0, rho[i] >= 0, P[i] <= B[i], rho[i] <= rem[i]]

    log2 = cp.inv_pos(cp.log(2))
    for i in range(M):
        cons.append(rho[i] <= cp.log(1 + h[i]*P[i]) * log2)

    users = list(range(M))
    for k in range(2, M+1):
        for S in combinations(users, k):
            sum_rho = sum(rho[i] for i in S)
            sum_pow = sum(h[i]*P[i] for i in S)
            cons.append(sum_rho <= cp.log(1 + sum_pow) * log2)

    prob = cp.Problem(cp.Minimize(obj), cons)
    try:
        val = prob.solve(solver=cp.ECOS, warm_start=True, verbose=False)
    except:
        val = prob.solve(solver=cp.SCS, eps=1e-4, max_iters=5000, verbose=False)

    return abs(val)

# ===================== Value Iteration =====================
T = 6
stp = 1
print("Using", cpu_count(), "CPU cores")

# Create pool ONCE to avoid "Too many open files"
# Auto-tuned workers for large servers (best for CVXPY workloads)
import math
workers = min(128, cpu_count())
pool = Pool(workers)

while stp < T:
    # t11 = time.time()
    print('stp', stp)

    P_opt, q_opt, r_opt = quad_fit(all_states, V)
    P_opt = (P_opt + P_opt.T) / 2  # PSD stabilize

    args = [(state, P_opt, q_opt, r_opt, 0.99) for state in all_states]
    V_next = pool.map(one_slot_greedy_fast, args)

    V = np.array(V_next)

    # print('Iteration time:', time.time() - t11)
    # print('-----------------------')
    stp += 1

pool.close()
pool.join()

print("Value iteration done")

# ===================== Policy Run =====================
def one_slot_greedy_policy(state, P_opt, q_opt, r_opt, gamma=0.99):
    B   = state[0:M]
    rem = state[M:2*M]
    h   = state[2*M:3*M]
    wt  = state[3*M:4*M]

    rho = [cp.Variable(nonneg=True) for _ in range(M)]
    P   = [cp.Variable(nonneg=True) for _ in range(M)]

    obj = 0
    for i in range(M):
        obj += wt[i] * cp.exp(-(rmax - (rem[i] - rho[i])))

    B_end = [B[i] - P[i] for i in range(M)]
    B_ch_arr = [[B_end[i], B_end[i] + 1] for i in range(M)]
    r_ch_arr = [[rem[i] - rho[i], rmax] for i in range(M)]
    wt_ch_arr = [[1,2] for _ in range(M)]

    next_states = list(product(*B_ch_arr, *r_ch_arr, *([h_vals]*M), *wt_ch_arr))

    # Faster quadratic value evaluation
    Vz_list = []
    for ns in next_states:
        x = cp.hstack(ns)
        Vz_list.append(cp.quad_form(x, P_opt) + x @ q_opt + r_opt)
    Vz = cp.hstack(Vz_list)

    B_ch_pr = np.array([1-e_prob, e_prob])
    r_ch_pr = np.array([1-pkt_prob, pkt_prob])

    wt_ch_pr = []
    for i in range(M):
        if wt[i] == 1:
            wt_ch_pr.append([st_tr_wt(1,1), st_tr_wt(1,2)])
        else:
            wt_ch_pr.append([st_tr_wt(2,1), st_tr_wt(2,2)])

    probs = []
    for combo in product(*([B_ch_pr]*M), *([r_ch_pr]*M), *([h_prob]*M), *wt_ch_pr):
        pr = 1
        for x in combo:
            pr *= x
        probs.append(pr)
    probs = cp.hstack(np.array(probs))

    obj += gamma * (Vz @ probs)

    cons = []
    for i in range(M):
        cons += [P[i] >= 0, rho[i] >= 0, P[i] <= B[i], rho[i] <= rem[i]]

    log2 = cp.inv_pos(cp.log(2))
    for i in range(M):
        cons.append(rho[i] <= cp.log(1 + h[i]*P[i]) * log2)

    users = list(range(M))
    for k in range(2, M+1):
        for S in combinations(users, k):
            sum_rho = sum(rho[i] for i in S)
            sum_pow = sum(h[i]*P[i] for i in S)
            cons.append(sum_rho <= cp.log(1 + sum_pow) * log2)

    prob = cp.Problem(cp.Minimize(obj), cons)
    try:
        prob.solve(solver=cp.ECOS, warm_start=True, verbose=False)
    except:
        prob.solve(solver=cp.SCS, eps=1e-4, max_iters=5000, verbose=False)

    P_val   = np.array([Pi.value for Pi in P])
    rho_val = np.array([ri.value for ri in rho])
    return P_val, rho_val


def policy_run(T_horizon, P_opt, q_opt, r_opt):
    tot_dist = 0
    B_prev  = np.ones(M)
    rem_prev = np.zeros(M)
    wt_prev  = np.ones(M)

    for t in range(T_horizon):
        wt_rand  = np.random.choice([1,2], size=M, p=[weight_prob, 1-weight_prob])
        h_rand   = np.random.choice(h_vals, size=M, p=h_prob)
        pkt_rand = np.random.choice([1,0], size=M, p=[pkt_prob, 1-pkt_prob])
        E_rand   = np.random.choice([1,0], size=M, p=[e_prob, 1-e_prob])

        rem_start = np.zeros(M)
        wt_start  = np.zeros(M)
        for i in range(M):
            if pkt_rand[i] == 1:
                wt_start[i]  = wt_rand[i]
                rem_start[i] = rmax
            else:
                wt_start[i]  = wt_prev[i]
                rem_start[i] = rem_prev[i]

        B_start = np.minimum(B_prev + E_rand, B_max)
        state = np.concatenate([B_start, rem_start, h_rand, wt_start])

        P_val, rho_val = one_slot_greedy_policy(state, P_opt, q_opt, r_opt)

        B_prev   = np.maximum(B_start - P_val, 0)
        rem_prev = np.maximum(rem_start - rho_val, 0)
        wt_prev  = wt_start.copy()

        dist = np.sum(wt_start * np.exp(-(rmax - rem_prev)))
        tot_dist += dist
        print('obj', t, tot_dist/(t+1))

        # if t % 100 == 0:
        #     print('obj', t, tot_dist/(t+1))


print("Running policy simulation...")
policy_run(1001, P_opt, q_opt, r_opt)
