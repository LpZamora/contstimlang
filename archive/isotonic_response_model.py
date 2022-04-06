import time

import numpy as np
import scipy.special, scipy.interpolate
import quadprog
import osqp
from scipy import sparse


def griddata_linear_interp_nearest_exterp(points, values, xi, rescale=False):
    """linearly interpolate and use nearest neighbors for extrapolation."""
    y_hat_linear = scipy.interpolate.griddata(
        points, values, xi, rescale=rescale, fill_value=np.nan, method="linear"
    )
    y_hat_nearest = scipy.interpolate.griddata(
        points, values, xi, rescale=rescale, method="nearest"
    )

    # linearly interpolate, use nearest neighbors for extrapolation.
    y_hat = np.where(np.isnan(y_hat_linear), y_hat_nearest, y_hat_linear)

    y_hat = y_hat_linear
    return y_hat


def plot_grid(x, y, z, resolution=1024, vmin=None, vmax=None):
    min_s = min(np.min(x), np.min(y))
    max_s = min(np.max(x), np.max(y))

    xi, yi = np.meshgrid(
        np.linspace(min_s, max_s, num=resolution),
        np.linspace(min_s, max_s, num=resolution),
    )
    interp_z = griddata_linear_interp_nearest_exterp(
        points=(x, y), values=z, xi=(xi, yi)
    )

    plt.imshow(
        interp_z,
        cmap="bwr",
        aspect="equal",
        vmin=vmin,
        vmax=vmax,
        origin="lower",
        extent=(min_s, max_s, min_s, max_s),
    )
    plt.colorbar()
    plt.plot((min_s, max_s), (min_s, max_s), "k--")


def overfitted_isotonic_mapping(s1, s2, y):
    """get the best bivariate isotonic mapping from s1 and s2 to predict y."""

    mapping = ModelProbsToHumanRating(y_symmetry_point=3.5, y_range=5)
    mapping.fit(s1, s2, y)
    y_hat = mapping.predict(s1, s2)
    return y_hat


def crossvalidated_isotonic_mapping():
    raise NotImplementedError


class ModelProbsToHumanRating:
    def __init__(self, y_symmetry_point=0, y_range=None):
        self.y_symmetry_point = y_symmetry_point
        self.y_range = y_range

    def recode_vars(self, s1, s2, y=None):
        # recode the variables such that s2[i]>=s1[i] for all i,
        # and y is shifted by the symmetry point and reflected when s1 and s2 are swapped.
        do_swap = s2 < s1
        new_s1 = np.where(do_swap, s2, s1)
        new_s2 = np.where(do_swap, s1, s2)
        if y is not None:
            new_y = np.where(
                do_swap, self.y_symmetry_point - y, y - self.y_symmetry_point
            )
        else:
            new_y = None

        return new_s1, new_s2, new_y, do_swap

    def unrecode_vars(self, y, do_swap):
        """undo recode_vars's effect on y"""

        new_y = np.where(do_swap, self.y_symmetry_point - y, y + self.y_symmetry_point)
        return new_y

    def fit(self, s1, s2, y):
        """Given vectors of model sentence scores s1 and s2 and human ratings r, learn f(s1,s2) that minimizes sum_i w[i]*(f(s1[i],s2[i])-y[i])**2

        args:
        s1 (np.ndarray)
        s2 (np.ndarray)
        y (np.ndarray)

        """

        assert s1.ndim == 1 and s2.ndim == 1 and y.ndim == 1
        n_trials = len(y)
        s1, s2, y, _ = self.recode_vars(s1, s2, y=y)

        # We cast the problem as quadratic programming problem.

        #     Minimize     1/2 x^T P x + q^T x
        #     Subject to   l <= x <= u

        # https://osqp.org/docs/solver/index.html

        # TODO: introduce weights
        P = sparse.eye(n_trials, format="csc")
        q = -y

        # %% build constraint matrix A and lower and upper bound vectors l and u
        A = []
        l = []
        u = []

        # isotonic response mapping

        # find all trials pairs such that
        # s2[i_trial]>=s2[j_trial] and s1[i_trial]<=s1[j_trial]
        condition_mat = np.logical_and(
            np.atleast_2d(s2).T >= np.atleast_2d(s2),
            np.atleast_2d(s1).T <= np.atleast_2d(s1),
        )
        np.fill_diagonal(condition_mat, False)  # eliminate i_trial==j_trial
        i_trial, j_trial = np.nonzero(condition_mat)
        for i, j in zip(i_trial, j_trial):
            assert s2[i] >= s2[j] and s1[i] <= s1[j]

        n_constraints = len(i_trial)
        constraint = sparse.csc_matrix(
            (+np.ones(n_constraints), (np.arange(n_constraints), i_trial)),
            shape=(n_constraints, n_trials),
        ) + sparse.csc_matrix(
            (-np.ones(n_constraints), (np.arange(n_constraints), j_trial)),
            shape=(n_constraints, n_trials),
        )
        A.append(constraint)
        l.append(np.zeros(n_constraints))
        u.append(np.ones(n_constraints) * np.inf)

        # response range limiting
        if self.y_range is not None:
            constraint = sparse.eye(n_trials)
            A.append(constraint)
            l.append(np.zeros(n_trials))
            u.append(np.ones(n_trials) * self.y_range / 2)

        A = sparse.vstack(A, format="csc")
        l = np.concatenate(l)
        u = np.concatenate(u)

        assert A.shape[0] == len(l) and A.shape[1] == n_trials

        # solve the quadratic programming problem
        prob = osqp.OSQP()
        prob.setup(P=P, q=q, A=A, l=l, u=u, polish=False)
        res = prob.solve()
        assert res.info.status == "solved"
        self.training_y_hat_ = res.x

        self.training_s1_ = s1
        self.training_s2_ = s2
        self.n_training_points_ = len(self.training_y_hat_)

    def fit_slow(self, s1, s2, y, qp_framework="osqp"):
        """Given vectors of model sentence scores s1 and s2 and human ratings r, learn f(s1,s2) that minimizes sum_i w[i]*(f(s1[i],s2[i])-y[i])**2

        args:
        s1 (np.ndarray)
        s2 (np.ndarray)
        y (np.ndarray)

        """

        assert s1.ndim == 1 and s2.ndim == 1 and y.ndim == 1
        n_trials = len(y)
        s1, s2, y, _ = self.recode_vars(s1, s2, y=y)

        # We cast the problem as quadratic programming problem.

        #     Minimize     1/2 x^T G x - a^T x
        #     Subject to   C.T x >= b

        # TODO: condense trials and set y
        w = np.ones(n_trials)
        G = np.diag(w)
        a = w * y

        # %% build constraint matrix C and vector b
        C_cols = []
        b = []

        # TODO: since osqp is much faster than quadprog, we can eliminate quadprog and build the matrices as sparse matrices.

        # isotonic response mapping
        for i_trial in range(n_trials):
            for j_trial in range(n_trials):
                if i_trial == j_trial:
                    continue
                if s2[i_trial] >= s2[j_trial] and s1[i_trial] <= s1[j_trial]:
                    # then y[i_trial] >= y[j_trial]
                    C_col = np.zeros(n_trials)
                    C_col[i_trial] = 1.0
                    C_col[j_trial] = -1.0
                    C_cols.append(C_col)
                    b.append(0.0)

        # response range limiting
        if self.y_range is not None:
            ub = self.y_range / 2
            for i_trial in range(n_trials):
                C_col = np.zeros(n_trials)
                C_col[i_trial] = -1.0
                C_cols.append(C_col)
                b.append(-ub)

        lb = 0
        for i_trial in range(n_trials):
            C_col = np.zeros(n_trials)
            C_col[i_trial] = 1.0
            C_cols.append(C_col)
            b.append(lb)

        C = np.stack(C_cols, axis=1)
        b = np.asarray(b)
        assert C.shape[0] == n_trials and C.shape[1] == len(b)

        # solve the quadratic programming problem
        if qp_framework == "quadprog":
            self.training_y_hat_ = quadprog.solve_qp(G, a, C, b)[0]
        elif qp_framework == "osqp":
            prob = osqp.OSQP()
            prob.setup(
                P=sparse.csc_matrix(G),
                q=-a,
                A=sparse.csc_matrix(C.T),
                l=b,
                u=np.ones(b.shape) * np.inf,
                polish=True,
            )
            res = prob.solve()
            assert res.info.status == "solved"
            self.training_y_hat_ = res.x
        else:
            raise ValueError("invalid qp_framework " + qp_framework)

        self.training_s1_ = s1
        self.training_s2_ = s2
        self.n_training_points_ = len(self.training_y_hat_)

    def predict(self, s1, s2):

        s1, s2, _, do_swap = self.recode_vars(s1, s2)

        y_hat = np.zeros(s1.shape)
        for i_point in range(self.n_training_points_):
            # check which evaluation points are dominating i_point
            mask = np.logical_and(
                s2 >= self.training_s2_[i_point], s1 <= self.training_s1_[i_point]
            )
            # these evalatuion points should have scores at least as high as the dominated data point.
            if np.any(mask):
                y_hat[mask] = np.maximum(y_hat[mask], self.training_y_hat_[i_point])

        # # for each point, find dominating training points
        # y_hat=griddata_linear_interp_nearest_exterp(points=(self.training_s1_,self.training_s2_),
        #                                             values=self.training_y_hat_, xi=(s1,s2))
        y_hat = self.unrecode_vars(y_hat, do_swap)
        return y_hat

    def plot_response_function(
        self, min_s, max_s, resolution=1024, vmin=None, vmax=None
    ):
        s1, s2 = np.meshgrid(
            np.linspace(min_s, max_s, num=resolution),
            np.linspace(min_s, max_s, num=resolution),
        )
        z = self.predict(s1, s2)
        plt.imshow(
            z,
            cmap="bwr",
            aspect="equal",
            vmin=vmin,
            vmax=vmax,
            origin="lower",
            extent=(min_s, max_s, min_s, max_s),
        )
        plt.xlabel("s1")
        plt.ylabel("s2")
        plt.colorbar()
        plt.plot((min_s, max_s), (min_s, max_s), "k--")


def recode_vars(s1, s2, y=None, y_symmetry_point=3.5):
    # recode the variables such that s2[i]>=s1[i] for all i,
    # and y is shifted by the symmetry point and reflected when s1 and s2 are swapped.
    do_swap = s2 < s1
    new_s1 = np.where(do_swap, s2, s1)
    new_s2 = np.where(do_swap, s1, s2)
    if y is not None:
        new_y = np.where(do_swap, y_symmetry_point - y, y - y_symmetry_point)
    else:
        new_y = None

    return new_s1, new_s2, new_y, do_swap


def nonparametric_model_human_consistency(lp_1, lp_2, rating):
    n_trials = len(lp_1)

    lp_1 = np.asarray(lp_1)
    lp_2 = np.asarray(lp_2)
    rating = np.asarray(rating)

    lp_1, lp_2, rating, do_swap = recode_vars(lp_1, lp_2, y=rating)

    n_non_tied_trial_pairs = 0
    n_inconsistent_trial_pairs = 0

    # slow, readable version
    for i_trial in range(n_trials):
        for j_trial in range(n_trials):
            if (
                rating[i_trial] > rating[j_trial]
            ):  # human rating in i_trial favors more sentence2 than in j_trial
                n_non_tied_trial_pairs += 1
                if (lp_2[i_trial] <= lp_2[j_trial]) and (
                    lp_1[i_trial] >= lp_1[j_trial]
                ):
                    n_inconsistent_trial_pairs += 1

    return 1 - n_inconsistent_trial_pairs / n_non_tied_trial_pairs


def nonparametric_model_human_consistency_NC(nc, rating):
    n_trials = len(nc)

    nc = np.asarray(nc)
    rating = np.asarray(rating)
    n_non_tied_trial_pairs = 0
    n_inconsistent_trial_pairs = 0

    # slow, readable version
    for i_trial in range(n_trials):
        for j_trial in range(n_trials):
            if (
                rating[i_trial] > rating[j_trial]
            ):  # human rating in i_trial favors more sentence2 than in j_trial
                n_non_tied_trial_pairs += 1
                if nc[i_trial] <= nc[j_trial]:
                    n_inconsistent_trial_pairs += 1
    return 1 - n_inconsistent_trial_pairs / n_non_tied_trial_pairs


def measure_nonparametric_model_human_consistency(df):
    models = get_models(df) + ["mean_rating_NC"]

    df2 = []
    for subject in tqdm(df["Participant Private ID"].unique()):
        mask = df["Participant Private ID"] == subject
        reduced_df = df[mask]
        cur_result = {"Participant Private ID": subject}
        for model in models:
            if model != "mean_rating_NC":
                cur_result[model] = nonparametric_model_human_consistency(
                    reduced_df["sentence1_" + model + "_prob"],
                    reduced_df["sentence2_" + model + "_prob"],
                    reduced_df["rating"],
                )
            else:
                cur_result[model] = nonparametric_model_human_consistency_NC(
                    reduced_df["mean_rating_NC"], reduced_df["rating"]
                )
        df2.append(cur_result)
    return pd.DataFrame(df2)


# print('non-measure_nonparametric_model_human_consistency:')
# df2=measure_nonparametric_model_human_consistency(df)
# print(df2.mean())


if __name__ == "__main__":
    n = 500
    s1 = np.random.randn(n)
    s2 = np.random.randn(n)
    noise = np.random.randn(n) * 1.0
    y = scipy.special.expit(s2 - s1 + noise) * 5 + 1

    response_mdl = ModelProbsToHumanRating(y_symmetry_point=3.5, y_range=5.0)

    t0 = time.time()
    response_mdl.fit_slow(s1, s2, y, qp_framework="quadprog")
    y_quadprog = response_mdl.training_y_hat_
    print("quadprog: MSE=", np.sum((y_quadprog - y) ** 2), "t=", time.time() - t0)

    t0 = time.time()
    response_mdl.fit(s1, s2, y)
    y_osqp = response_mdl.training_y_hat_
    print("osqp: y_osqp=", np.sum((y_osqp - y) ** 2), "t=", time.time() - t0)

    import matplotlib

    matplotlib.use("TkAgg")

    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(121)
    plot_grid(s1, s2, y)
    plt.xlabel("s1")
    plt.ylabel("s2")

    ax = fig.add_subplot(122)

    response_mdl.plot_response_function(-4, 4, vmin=1, vmax=6)
    plt.scatter(s1, s2, s=20, c=y, cmap="bwr", vmin=1, vmax=6, edgecolors="k")
    plt.show()
