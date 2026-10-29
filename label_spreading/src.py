import numpy as np
import scipy
import time
from sklearn.neighbors import NearestNeighbors
import scipy.sparse as sparse
from scipy.sparse.csgraph import connected_components
import json
import pyamgx
import faiss

# pyamgx.initialize() sets up the process-wide AMGX/CUDA library context. It is run
# once per process (not per solve) to avoid repeated init/finalize cycles during long
# hyperparameter searches; per-solve objects are still created and destroyed per call.
_AMGX_INITIALIZED = False


def _ensure_amgx_initialized():
    global _AMGX_INITIALIZED
    if not _AMGX_INITIALIZED:
        pyamgx.initialize()
        _AMGX_INITIALIZED = True


def _apply_object_conditioning(dataset):
    """Rescales p_hat's non-background class mass by the objectness estimate in
    score_est, producing p_hat_mod / prediction_mod: p(class | object present) *
    p(object present), with p(no object) kept as the last column. Shared by every
    spreading variant that independently estimates p_hat and score_est."""
    score = np.vstack(dataset["score_est"].to_numpy())
    p_no, p_obj = score[:, 0], score[:, 1]

    p_hat = np.vstack(dataset["p_hat"].to_numpy())
    p_bg_old = p_hat[:, -1]   # probability mass in the last class, "background"
    eps = 1e-12
    mass_obj_old = np.clip(1.0 - p_bg_old, eps, None)
    p_cond = p_hat[:, :-1] / mass_obj_old[:, None]
    p_hat_mod = np.hstack([p_cond * p_obj[:, None], p_no[:, None]])

    dataset["p_hat_mod"] = list(p_hat_mod)
    dataset["prediction_mod"] = p_hat_mod.argmax(axis=1)



def prob_label_spreading_object(dataset_name, only_objects, dataset, data_space, label_column, prob_label_column, score_column, n_data, k, alpha, n_samples, indices = None, config = "configs/FGMRES.json"):

    if n_data == "all":
        n_data = len(dataset)
    else:
        dataset = dataset.sample(n = n_data) # random data subset
        dataset.reset_index(drop=True, inplace=True)

    classes = sorted(dataset[label_column].dropna().unique())

    start=time.time()

    if indices is not None: # use pre-defined indices
        assert len(indices) == n_samples, "Length of indices must match n_samples"
        print("use pre-defined indices")
        

    else:
        indices = np.random.choice(dataset.index, size=n_samples, replace=True) # this draws n_samples many indices randomly
        print("use random indices")

    if alpha != 0: # if alpha is zero we do not need to set up the graph structure nor solve the linear system
        # set graph and linear system
        S, I, time_NN, time_graph = set_graph(k, dataset, data_space, n_data)

        # connected-component info
        labels, sizes = graph_report_from_S(S)
        dataset["cc_id"] = labels.astype(np.int32)
        dataset["cc_size"] = sizes[labels].astype(np.int32)

        F, trials, F_object, trials_object, ind, time_spread, time_setup, time_solver = sampling_and_spreading_efficient(alpha, S, dataset, only_objects, prob_label_column, score_column, classes, config, indices)

        
        dataset["prediction"] = [x.argmax() for x in F]
        dataset['trials'] = trials
        dataset["p_hat"] = [ (F[i] + ((1e-4)/len(classes))) /(trials[i] + 1e-4) for i in range(len(trials))]
        dataset["score_est"] = [ (F_object[i] + ((1e-4)/2)) /(trials_object[i] + 1e-4) for i in range(len(trials_object))]
        dataset["trials_object"] = trials_object

    else:
        # MLE estimate (histogram) for the objectness score and the probabilistic label
        # histogram estimate where information was observed and uniform class distribution otherwise
        object_feedback = np.zeros( (n_data, 2)) # counts the number of sampled object presences for every datapoint -> histogram that corresponds to the prob. distr.
        feedback = np.zeros( (n_data, len(classes))) # counts the number of sampled classes for every datapoint -> histogram that corresponds to the prob. distr.
        for i in indices: # for every sampled data point update the estimated prob. distr.
             c = np.random.choice(classes, p = dataset[prob_label_column][i]) # sample a class according to the prob. label
             o = np.random.choice([0,1], p = [1 - dataset[score_column][i], dataset[score_column][i]]) # sample whether an object is present according to the score
             feedback[i, c] += 1
             object_feedback[i, o] += 1
        # insert uniform distr. for data points that have not been sampled
        trials = [x.sum() for x in feedback]
        feedback = [x if x.sum() > 0 else np.ones(len(classes)) for x in feedback]
        feedback = [x / x.sum() for x in feedback] # normalize 

        object_feedback = [x if x.sum() > 0 else np.ones(2) for x in object_feedback]
        object_feedback = [x / x.sum() for x in object_feedback]   # normalize 

        dataset["prediction"] = [x.argmax() for x in feedback] # this will assign class zero if the distr. is uniform
        dataset["p_hat"] = [x for x in feedback]
        dataset["score_est"] = [x for x in object_feedback]

        time_NN = time_graph = time_setup = time_solver = time_spread = 0.0

    _apply_object_conditioning(dataset)

    elapsed_time = time.time() - start

    dataset.attrs["run_meta"] = dict(
        dataset_name=dataset_name, only_objects=only_objects, data_space=data_space,
        label_column=label_column, prob_label_column=prob_label_column,
        n_data=n_data, k=k, alpha=alpha, n_samples=n_samples, config=config,
        elapsed_time=elapsed_time, time_NN=time_NN, time_graph=time_graph,
        time_setup=time_setup, time_solver=time_solver, time_spread=time_spread,
    )
    return dataset

def set_graph(k_NN, train, data_space, n_data = None, gpu_id=0, bandwidth_scale=1.0, metric="l2"):
        # kNN graph (exact search via faiss) with Gaussian-kernel edge weights.
        # bandwidth_scale multiplies mu (the mean squared kNN distance used as the
        # Gaussian-kernel bandwidth below); 1.0 is the unscaled bandwidth. Values
        # below ~0.05-0.1 make the similarity matrix near-degenerate (unstable
        # solves), so stay >= 0.1.
        #
        # metric="l2" (default): exact L2 on the raw embeddings.
        # metric="cosine": L2-normalizes each embedding first; for unit vectors
        # ||a-b||^2 = 2 - 2*cos(a,b), so the same IndexFlatL2 search yields exact
        # cosine nearest-neighbor rankings with the same kernel/bandwidth machinery.
        if n_data == None:
                n_data = len(train)
        # compute k nearest neighbors and setup a graph where each node has a link to its k nearest neighbors
        time_nn_anfang=time.time()

        x = np.vstack(train[data_space].to_numpy()).astype(np.float32, copy=False)
        x = np.ascontiguousarray(x)

        if metric == "cosine":
            norms = np.linalg.norm(x, axis=1, keepdims=True)
            norms[norms == 0] = 1.0   # guard degenerate all-zero embeddings, if any
            x = np.ascontiguousarray(x / norms)
        elif metric != "l2":
            raise ValueError(f"set_graph: unknown metric {metric!r} -- expected 'l2' or 'cosine'")

        # query k_NN neighbors including self, then drop self -> k_NN-1 edges
        k_query = k_NN

        res = faiss.StandardGpuResources()
        index_cpu = faiss.IndexFlatL2(x.shape[1])   # exact L2 (on unit vectors when metric="cosine")
        index = faiss.index_cpu_to_gpu(res, gpu_id, index_cpu)

        index.add(x)  # add all points as database
        D, I = index.search(x, k_query)  # D: squared L2 distances, I: indices

        time_nn_ende=time.time()
        time_setup_graph_anfang=time.time()

        # drop the self neighbor (usually at position 0, but handle any position)
        row = np.arange(n_data)[:, None]
        is_self = (I == row)
        if is_self.any():
            order = np.argsort(is_self, axis=1)  # False first, True last
            I = np.take_along_axis(I, order, axis=1)
            D = np.take_along_axis(D, order, axis=1)

        # keep the first k_NN-1 neighbors
        I = I[:, :k_NN-1]
        D = D[:, :k_NN-1]  # already squared distances

        # preprocessing for setting up a sparse adjacency matrix
        indptr = np.arange(0, n_data + 1, dtype=np.int64) * (k_NN - 1)
        raw_dists = D.ravel()  # already squared
        indices = I.ravel()

        # transform distances into similarities
        mu = float(raw_dists.mean()) * bandwidth_scale
        similarity = np.exp(raw_dists * (-1.0 / (2.0 * mu)))

        # form the sparse matrix
        C = scipy.sparse.csr_matrix((similarity, indices, indptr), shape=(n_data, n_data), dtype=np.float64)
       
        # symmetrize it
        W = (C + C.T) / 2.0

        # degree vector (row sums)
        Dsum = np.asarray(W.sum(axis=1)).ravel()
        # avoid division by zero (rare but possible)
        Dsum[Dsum == 0] = 1.0
       
        # symmetrically normalized affinity S = D^-1/2 W D^-1/2
        D_inv_sqrt = scipy.sparse.spdiags(Dsum ** (-0.5), 0, n_data, n_data)
        S = D_inv_sqrt @ W @ D_inv_sqrt

        I_eye = scipy.sparse.eye(n_data, format="csr")
        time_setup_graph_ende=time.time()
        time_NN=time_nn_ende-time_nn_anfang
        time_graph=time_setup_graph_ende-time_setup_graph_anfang
        return S,I_eye, time_NN,time_graph


# construct the right hand side of the linear system for class-wise label spreading     
def make_rhs(n_samples, n_classes, train, data_space, n_data = None):
        if n_data == None:
                n_data = len(train)
        ind=[]
        data_train = ( np.asarray(train[[data_space]]), np.asarray(train[["label"]]) )
        y = np.asarray( train[['label']] ).ravel().astype(int)
        divide=int(n_samples/n_classes) 
        for i in range(n_classes):
                    random_i=np.random.choice( np.where(data_train[1] == i)[0],divide,replace=False )  
                    ind.append( random_i )
        Y=np.zeros((n_data, n_classes))
        for i in ind:
            y_i=y[i]
            Y[i,y_i]=1
        
        return Y 


def solve_probabilistic(Matrix, cfg, ind, length, dataset, only_objects, prob_label_column, score_column, sampling_matrix, classes, F, trials, F_object, trials_object):
    """ Solves the linear system involved in label spreading with algebraic multigrid methods using the AMGX library."""
    # suppress the repetitive output of the AMGX library
    pyamgx.register_print_callback(lambda *args, **kwargs : None)
    start_solver=time.time()
    _ensure_amgx_initialized()

    # Initialize config and resources (which solver to use, etc. as specified in config file):
    cfg = pyamgx.Config().create_from_file(cfg)
    rsc = pyamgx.Resources().create_simple(cfg)

    # Create matrices and vectors:
    A = pyamgx.Matrix().create(rsc)
    b = pyamgx.Vector().create(rsc)
    x = pyamgx.Vector().create(rsc)

    # Create solver:
    solver = pyamgx.Solver().create(rsc, cfg)

    # Upload system and solve. try/finally ensures A/b/x/solver/rsc/cfg are freed
    # even if solver.setup or solver.solve raises (AMGXError, at small
    # alpha/k/bandwidth on large sparse graphs), so callers that catch the error
    # and skip a combo do not leak GPU memory.
    try:
        M = sparse.csr_matrix(Matrix)
        A.upload_CSR(M)
        t_a=time.time()
        solver.setup(A)
        t_c=time.time()
        t_setup=t_c-t_a
        t_g=time.time()
        t_spread_start=time.time()
        for i in ind:
            e_i = np.zeros(length)
            e_i[i] = 1  #e_i is the i-th unit vector, solution is the information spread by the i-th node throughout the graph

            rhs = np.array(e_i)
            sol = np.zeros(np.size(e_i), dtype=np.float64)
            b.upload(rhs)
            x.upload(sol)
            # Setup and solve system:
            solver.solve(b, x, zero_initial_guess=True)
            # Download solution
            x.download(sol)
            solution=sol

            if np.isnan(solution).all():
                solution = np.zeros(length) # it might happen that the solver occasionally does not find the solution (for alpha and k very small)

            # the solution is scaled such that the maximum entry is 1, i.e. the node that spreads the information gets a value of 1
            m = np.max(solution)
            if m > 0: # always the case if the solution is found, otherwise solution is set to zero anyways so no update necessary
                solution = solution/m

            # Here, we have the solution of the linear system, i.e. the information spread by the i-th node throughout the graph
            # First, let us sample whether there is an object present or not according to the score of the i-th point
            u_object = np.random.uniform(size=1)
            if u_object <= dataset[score_column].iloc[i]: # object present
                trials_object += solution
                F_object[:,1] += solution # spread the information that there is an object present
                object_present = True
            else: # no object present
                trials_object += solution
                F_object[:,0] += solution # spread the information that there is no object present
                object_present = False

            ### spread only if "object_present" is true
            if only_objects:
                if object_present:
                    trials += solution
                    # sample a class according to the probabilistic label of the i-th point to spread the information of this class
                    u = np.random.uniform(size=1) # for sampling the class according to the prob. label randomly
                    p_sum = 0
                    for j in range(len(classes)):
                        p_sum = p_sum + dataset[prob_label_column].iloc[i][j]
                        if j == 0 and u <=dataset[prob_label_column].iloc[i][j]:
                            break
                        if j > 0 and u <= p_sum:
                            break

                    # j is now the sampled class index and we can spread the information for this class and store it in F
                    if m > 0:
                        F[:,j] += solution

                    sampling_matrix[i][j] += 1 # store how often class j was sampled for point i for comparison
            ### spread all data
            else:
                trials += solution
                # sample a class according to the probabilistic label of the i-th point to spread the information of this class
                u = np.random.uniform(size=1) # for sampling the class according to the prob. label randomly
                p_sum = 0
                for j in range(len(classes)):
                    p_sum = p_sum + dataset[prob_label_column].iloc[i][j]
                    if j == 0 and u <=dataset[prob_label_column].iloc[i][j]:
                        break
                    if j > 0 and u <= p_sum:
                        break

                # j is now the sampled class index and we can spread the information for this class and store it in F
                if m > 0:
                    F[:,j] += solution

                sampling_matrix[i][j] += 1 # store how often class j was sampled for point i for comparison

        # cleanup
        t_spread_end=time.time()
        t_spread=t_spread_end-t_spread_start
        t_b=time.time()
    finally:
        A.destroy()
        x.destroy()
        b.destroy()
        solver.destroy()
        rsc.destroy()
        cfg.destroy()

    # no pyamgx.finalize(): the library context persists for the process lifetime
    # (see _ensure_amgx_initialized)
    end_solver=time.time()
    t_solver=end_solver-start_solver
    return sampling_matrix, F, trials, F_object, trials_object, t_setup, t_spread, t_solver

def sampling_and_spreading_efficient(alpha, S, dataset, only_objects, prob_label_column, score_column, classes, config, ind):
       
    length = len(dataset)

    # store the cumulative feedback obtained for every data point in the vector trials 
    trials = np.zeros(length)
    trials_object = np.zeros(length)

    # the matrix F stores the cumulative spreaded information for every class and every data point
    # sampling_matrix stores how often a class was sampled for every data point
    F = np.zeros(length*len(classes)).reshape(length,len(classes))
    F_object = np.zeros(length*2).reshape(length,2)
    sampling_matrix = np.zeros(length*len(classes)).reshape(length, len(classes))

    I = scipy.sparse.eye(length)
    M = I - alpha * S

    sampling_matrix, F, trials, F_object, trials_object, time_setup, time_spread, time_solver = solve_probabilistic(M, config, ind, length, dataset, only_objects,
                                                                                                prob_label_column, score_column, sampling_matrix, classes, F, trials, F_object, trials_object)
                        
    return F, trials, F_object, trials_object, ind, time_spread, time_setup, time_solver

# connected-component report of the graph
def graph_report_from_S(S, topk=10, report=None):
    # make sure its sparse CSR
    if not sparse.issparse(S):
        raise TypeError("S must be a scipy sparse matrix")
    S = S.tocsr()
    S.eliminate_zeros()

    n = S.shape[0]
    nnz = S.nnz

    # connected components (undirected)
    n_comp, labels = connected_components(S, directed=False, return_labels=True)

    sizes = np.bincount(labels, minlength=n_comp)
    sizes_sorted = np.sort(sizes)[::-1]

    # degree on the unweighted pattern (counts neighbors, not weights)
    A = S.copy()
    A.data[:] = 1
    deg = np.asarray(A.sum(axis=1)).ravel()

    if report:
        print("=== Graph report (from S nonzeros) ===")
        print("nodes:", n)
        print("edges (undirected approx):", nnz // 2)   # S is symmetric, so nnz ~ 2|E|
        print("components:", n_comp)
        print("largest component:", sizes_sorted[0], f"({sizes_sorted[0]/n:.2%})")
        print("isolated nodes:", int((sizes == 1).sum()))
        print("top component sizes:", sizes_sorted[:topk])
        print("degree min/mean/median/max:", int(deg.min()), float(deg.mean()), float(np.median(deg)), int(deg.max()))

    return labels, sizes


def solve_conventional(Matrix, cfg, ind, length, dataset, prob_label_column, score_column, classes):
    """Solves the linear system for *conventional* label spreading: a single label
    matrix Y is built from the (soft) labels of every sampled annotation at once,
    and the system is solved once per class/objectness column (len(classes) + 2
    solves total), instead of once per sampled annotation like solve_probabilistic
    does. This is the standard label-spreading formulation (Zhou et al., 2004).

    Each entry of `ind` (drawn with replacement, so an index can repeat) is one
    independent virtual annotation event: a single one-hot class is sampled from
    that node's soft_label distribution and counted into Y, so Y[i, c] is the
    number of times class c was drawn for node i (0, 1, 2, ... -- not the soft
    distribution itself). This reproduces solve_probabilistic's per-annotation
    Monte-Carlo sampling exactly (same generative model, same sampling noise),
    just batched into one solve per class instead of one solve per annotation."""
    pyamgx.register_print_callback(lambda *args, **kwargs: None)

    start_solver = time.time()
    _ensure_amgx_initialized()

    cfg = pyamgx.Config().create_from_file(cfg)
    rsc = pyamgx.Resources().create_simple(cfg)

    A = pyamgx.Matrix().create(rsc)
    b = pyamgx.Vector().create(rsc)
    x = pyamgx.Vector().create(rsc)
    solver = pyamgx.Solver().create(rsc, cfg)

    # solver.setup/solver.solve can raise AMGXError at small alpha/k/bandwidth on
    # large sparse graphs; try/finally frees A/b/x/solver/rsc/cfg in that case so
    # callers that catch the error and skip the combo do not leak GPU memory.
    try:
        M = sparse.csr_matrix(Matrix)
        A.upload_CSR(M)
        t_a = time.time()
        solver.setup(A)
        t_setup = time.time() - t_a

        # build the label matrix: one one-hot class sample per entry of `ind`,
        # drawn from that node's soft_label distribution via inverse-CDF sampling,
        # then counted (np.add.at accumulates repeats -- a node visited twice by
        # the with-replacement budget can end up with a row summing to 2, split
        # across one or two classes depending on what was drawn each time).
        p = np.vstack(dataset[prob_label_column].to_numpy())
        s = dataset[score_column].to_numpy()

        p_ind = p[ind]
        cdf = np.cumsum(p_ind, axis=1)
        u = np.random.uniform(size=(len(ind), 1))
        sampled_class = np.clip((u > cdf).sum(axis=1), 0, len(classes) - 1)

        Y = np.zeros((length, len(classes)))
        np.add.at(Y, (ind, sampled_class), 1.0)

        Y_object = np.zeros((length, 2))
        Y_object[ind, 0] = 1.0 - s[ind]
        Y_object[ind, 1] = s[ind]

        def _solve_columns(Y_mat):
            out = np.zeros_like(Y_mat, dtype=np.float64)
            for c in range(Y_mat.shape[1]):
                rhs = np.ascontiguousarray(Y_mat[:, c])
                sol = np.zeros(length, dtype=np.float64)
                b.upload(rhs)
                x.upload(sol)
                solver.solve(b, x, zero_initial_guess=True)
                x.download(sol)
                if np.isnan(sol).all():
                    sol = np.zeros(length)  # solver occasionally fails to converge (small alpha/k)
                out[:, c] = sol
            return out

        t_spread_start = time.time()
        F = _solve_columns(Y)
        F_object = _solve_columns(Y_object)
        t_spread = time.time() - t_spread_start
    finally:
        A.destroy()
        x.destroy()
        b.destroy()
        solver.destroy()
        rsc.destroy()
        cfg.destroy()
        # no pyamgx.finalize(): see _ensure_amgx_initialized

    t_solver = time.time() - start_solver
    return F, F_object, t_setup, t_spread, t_solver


def spreading_conventional(alpha, S, dataset, prob_label_column, score_column, classes, config, ind):
    length = len(dataset)
    I_eye = scipy.sparse.eye(length)
    M = I_eye - alpha * S

    F, F_object, time_setup, time_spread, time_solver = solve_conventional(M, config, ind, length, dataset, prob_label_column, score_column, classes)

    return F, F_object, time_spread, time_setup, time_solver


def label_spreading_object_conventional(dataset_name, only_objects, dataset, data_space, label_column, prob_label_column, score_column, n_data, k, alpha, n_samples, indices=None, config="configs/FGMRES.json", bandwidth_scale=1.0):
    """Conventional label spreading: every sampled annotation's one-hot class draw
    (sampled from its soft_label distribution, see solve_conventional) is spread
    simultaneously by solving one linear system per class (len(classes) + 2 solves
    total), instead of prob_label_spreading_object's approach of solving the
    system once per sampled annotation (one solve per entry in `indices`). This
    batches the same per-annotation Monte-Carlo sampling prob_label_spreading_object
    does into a single linear solve per class, which is much cheaper when n_samples
    is large, without changing the underlying generative model. Drop-in replacement
    with the same signature and return value (process_data, i.e. `dataset`) as
    prob_label_spreading_object -- call evaluation.evaluate_result(process_data)
    separately to get the metrics row."""

    if n_data == "all":
        n_data = len(dataset)
    else:
        dataset = dataset.sample(n=n_data) # random data subset
        dataset.reset_index(drop=True, inplace=True)

    classes = sorted(dataset[label_column].dropna().unique())

    start = time.time()

    if indices is not None: # use pre-defined indices
        assert len(indices) == n_samples, "Length of indices must match n_samples"
        print("use pre-defined indices")
    else:
        indices = np.random.choice(dataset.index, size=n_samples, replace=True)
        print("use random indices")

    if alpha != 0: # if alpha is zero there is no propagation, only the sampled rows carry information
        S, I_eye, time_NN, time_graph = set_graph(k, dataset, data_space, n_data, bandwidth_scale=bandwidth_scale)

        labels, sizes = graph_report_from_S(S)
        dataset["cc_id"] = labels.astype(np.int32)
        dataset["cc_size"] = sizes[labels].astype(np.int32)

        F, F_object, time_spread, time_setup, time_solver = spreading_conventional(alpha, S, dataset, prob_label_column, score_column, classes, config, indices)

        # row-normalize the propagated mass into probability distributions; nodes
        # that received no propagated mass at all (e.g. isolated in the graph)
        # fall back to a uniform distribution
        row_sum = F.sum(axis=1)
        uniform = np.full(len(classes), 1.0 / len(classes))
        p_hat = np.divide(F, row_sum[:, None], out=np.tile(uniform, (n_data, 1)), where=row_sum[:, None] > 0)

        obj_row_sum = F_object.sum(axis=1)
        score_est = np.divide(F_object, obj_row_sum[:, None], out=np.full((n_data, 2), 0.5), where=obj_row_sum[:, None] > 0)

        dataset["p_hat"] = list(p_hat)
        dataset["prediction"] = p_hat.argmax(axis=1)
        dataset["score_est"] = list(score_est)
        # raw total propagated mass per node (same "trials" name/meaning as
        # prob_label_spreading_object's Monte-Carlo trial count): p_hat is
        # normalized, so the raw propagated mass F is recoverable as
        # p_hat * trials[:, None] (0 for isolated nodes, matching F itself).
        dataset["trials"] = row_sum

    else:
        p_hat = np.tile(1.0 / len(classes), (n_data, len(classes)))
        score_est = np.full((n_data, 2), 0.5)

        p = np.vstack(dataset[prob_label_column].to_numpy())
        s = dataset[score_column].to_numpy()
        p_hat[indices] = p[indices]
        score_est[indices, 0] = 1.0 - s[indices]
        score_est[indices, 1] = s[indices]

        dataset["p_hat"] = list(p_hat)
        dataset["prediction"] = p_hat.argmax(axis=1)
        dataset["score_est"] = list(score_est)
        # no propagation happened (alpha == 0): only the sampled indices carry
        # any raw mass (exactly their own soft label, which sums to 1).
        trials = np.zeros(n_data)
        trials[indices] = 1.0
        dataset["trials"] = trials

        time_NN = time_graph = time_setup = time_solver = time_spread = 0.0

    _apply_object_conditioning(dataset)

    elapsed_time = time.time() - start

    dataset.attrs["run_meta"] = dict(
        dataset_name=dataset_name, only_objects=only_objects, data_space=data_space,
        label_column=label_column, prob_label_column=prob_label_column,
        n_data=n_data, k=k, alpha=alpha, n_samples=n_samples, config=config,
        bandwidth_scale=bandwidth_scale,
        elapsed_time=elapsed_time, time_NN=time_NN, time_graph=time_graph,
        time_setup=time_setup, time_solver=time_solver, time_spread=time_spread,
    )
    return dataset


def _faiss_labeled_neighbors(dataset, data_space, indices, m, n_data=None, gpu_id=0):
    """Builds a FAISS index over the embeddings of the *labeled* subset (`indices`)
    only, then queries every point in `dataset` against it. Returns squared-L2
    distances `D` and *local* neighbor positions `J` (0..len(indices)-1, i.e.
    indices into arrays already restricted to `indices`), shape (n_data, m) each,
    sorted by increasing distance. Shared by knn_regression_object and gkr_object
    (and reused as a single cached query across a whole gamma/k grid at fixed
    budget in a sweep, since neither depends on gamma/k)."""
    if n_data is None:
        n_data = len(dataset)

    x_all = np.vstack(dataset[data_space].to_numpy()).astype(np.float32, copy=False)
    x_all = np.ascontiguousarray(x_all)
    x_labeled = np.ascontiguousarray(x_all[indices])

    m = min(m, len(indices))

    res = faiss.StandardGpuResources()
    index_cpu = faiss.IndexFlatL2(x_labeled.shape[1])
    index = faiss.index_cpu_to_gpu(res, gpu_id, index_cpu)
    index.add(x_labeled)

    D, J = index.search(x_all, m)
    return D, J


def knn_regression_object(dataset_name, only_objects, dataset, data_space, label_column, prob_label_column, score_column, n_data, k, n_samples, indices=None, config=None, neighbor_cache=None):
    """kNN regression baseline: for every point, find its k nearest *labeled*
    neighbors (a FAISS index built only over the `indices` subset, queried by
    every point) and estimate p_hat / score_est as the plain unweighted average
    of those neighbors' (soft) label / objectness. Drop-in replacement with the
    same signature and return value (process_data) as
    label_spreading_object_conventional, so it plugs into the same
    evaluation.evaluate_result/sweep code unchanged.

    `neighbor_cache`, if given, is a precomputed (D, J) pair from
    _faiss_labeled_neighbors (m >= k) — lets a sweep over budgets share one
    FAISS query between the kNN and GKR baselines instead of re-querying per
    hyperparameter value."""

    if n_data == "all":
        n_data = len(dataset)
    else:
        dataset = dataset.sample(n=n_data)
        dataset.reset_index(drop=True, inplace=True)

    start = time.time()

    if indices is not None:
        assert len(indices) == n_samples, "Length of indices must match n_samples"
    else:
        indices = np.random.choice(dataset.index, size=n_samples, replace=True)

    if neighbor_cache is not None:
        D, J = neighbor_cache
        J = J[:, :k]
        time_NN = 0.0
    else:
        t0 = time.time()
        D, J = _faiss_labeled_neighbors(dataset, data_space, indices, k, n_data)
        time_NN = time.time() - t0

    t0 = time.time()
    p_labeled   = np.vstack(dataset[prob_label_column].to_numpy())[indices]                    # (n_samples, n_classes)
    s_labeled   = dataset[score_column].to_numpy()[indices]                                     # (n_samples,)
    obj_labeled = np.stack([1.0 - s_labeled, s_labeled], axis=1)                                 # (n_samples, 2)

    p_hat     = p_labeled[J].mean(axis=1)      # (n_data, n_classes)
    score_est = obj_labeled[J].mean(axis=1)    # (n_data, 2)
    time_spread = time.time() - t0

    dataset["p_hat"] = list(p_hat)
    dataset["prediction"] = p_hat.argmax(axis=1)
    dataset["score_est"] = list(score_est)

    _apply_object_conditioning(dataset)

    elapsed_time = time.time() - start

    dataset.attrs["run_meta"] = dict(
        dataset_name=dataset_name, only_objects=only_objects, data_space=data_space,
        label_column=label_column, prob_label_column=prob_label_column,
        n_data=n_data, k=k, alpha=np.nan, n_samples=n_samples, config=config,
        elapsed_time=elapsed_time, time_NN=time_NN, time_graph=0.0,
        time_setup=0.0, time_solver=0.0, time_spread=time_spread,
    )
    return dataset


def gkr_object(dataset_name, only_objects, dataset, data_space, label_column, prob_label_column, score_column, n_data, gamma, n_samples, indices=None, config=None, m=500, neighbor_cache=None):
    """Gaussian Kernel Regression (Nadaraya-Watson) baseline: for every point,
    estimate p_hat / score_est as a Gaussian-kernel-weighted average of the
    (soft) label / objectness of its labeled neighbors, weight(i, j) =
    exp(-gamma * ||x_i - x_j||^2). Truncated to the `m` nearest labeled
    neighbors (found via the same FAISS index as knn_regression_object) rather
    than every labeled point — the Gaussian weight of a far-away point is
    already ~0, so for reasonable gamma this is a close approximation of the
    untruncated estimator at a fraction of the cost, and lets GKR share a
    single FAISS query with the kNN baseline (see `neighbor_cache`). Drop-in
    replacement with the same signature/return value (process_data) as
    label_spreading_object_conventional."""

    if n_data == "all":
        n_data = len(dataset)
    else:
        dataset = dataset.sample(n=n_data)
        dataset.reset_index(drop=True, inplace=True)

    start = time.time()

    if indices is not None:
        assert len(indices) == n_samples, "Length of indices must match n_samples"
    else:
        indices = np.random.choice(dataset.index, size=n_samples, replace=True)

    if neighbor_cache is not None:
        D, J = neighbor_cache
        D, J = D[:, :m], J[:, :m]
        time_NN = 0.0
    else:
        t0 = time.time()
        D, J = _faiss_labeled_neighbors(dataset, data_space, indices, m, n_data)
        time_NN = time.time() - t0

    t0 = time.time()
    p_labeled   = np.vstack(dataset[prob_label_column].to_numpy())[indices]                    # (n_samples, n_classes)
    s_labeled   = dataset[score_column].to_numpy()[indices]                                     # (n_samples,)
    obj_labeled = np.stack([1.0 - s_labeled, s_labeled], axis=1)                                 # (n_samples, 2)

    weights = np.exp(-gamma * D)                    # (n_data, m)
    w_sum   = weights.sum(axis=1, keepdims=True)
    w_sum[w_sum == 0] = 1.0   # only possible if gamma*D underflows to 0 everywhere; falls back to a zero estimate

    p_hat     = (weights[:, :, None] * p_labeled[J]).sum(axis=1)   / w_sum
    score_est = (weights[:, :, None] * obj_labeled[J]).sum(axis=1) / w_sum
    time_spread = time.time() - t0

    dataset["p_hat"] = list(p_hat)
    dataset["prediction"] = p_hat.argmax(axis=1)
    dataset["score_est"] = list(score_est)

    _apply_object_conditioning(dataset)

    elapsed_time = time.time() - start

    dataset.attrs["run_meta"] = dict(
        dataset_name=dataset_name, only_objects=only_objects, data_space=data_space,
        label_column=label_column, prob_label_column=prob_label_column,
        n_data=n_data, k=m, alpha=gamma, n_samples=n_samples, config=config,
        elapsed_time=elapsed_time, time_NN=time_NN, time_graph=0.0,
        time_setup=0.0, time_solver=0.0, time_spread=time_spread,
    )
    return dataset