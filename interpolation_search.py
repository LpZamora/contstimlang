import numpy as np
import sklearn.linear_model
import matplotlib.pyplot as plt
import pandas as pd


def expand_to_matrix(x):
    x = np.asarray(x)
    if x.ndim == 0:
        x = np.expand_dims(x, -1)
    if x.ndim == 1:
        x = np.expand_dims(x, -1)
    return x


class SetInterpolationSearch:
    def __init__(
        self,
        loss_fun,
        g,
        initial_observed_xs=None,
        initial_observed_ys=None,
        initial_xs_guesses=None,
        h_method="LinearRegression",
    ):
        """
        Minimize loss_fun(f(x)) for a variable x
        which belongs to a discrete closed set,
        where y=f(x) is expensive to compute but
        can be approximated by a cheap, precomputed g(x).

        Given already observed (x, f(x)) pairs, we form
        y_aprx~=h(g(x)), a linear (or non-linear) prediction of y=f(x)
        as a function of g(x). This prediction is used to find our
        best guess for an x (for which we haven't yet observed f(x)),
        that will minimize loss_fun(f(x)).

        If we are unsatisfied with this guess after evaluating f(x),
        we can update the object with the new (x,f(x)) data point and
        retrieve a more informed guess for a new x to evaluate.

        This function supports a (scalar) loss function defined over more
        than one f(x): loss_fun(f_1(x),f_2(x),..,f_k(x))
        In that case, each f_k(x) is approximated by a corresponding g_k(x).

        Parameters
        ----------
        loss_fun : Python function, a function from (N,K) y values to (N,) loss values
        g: array_like, precomputed g(x) for each x, as an (N,K) numpy array.
        initial_observed_xs: array_like, (M,) numpy array of x indecis for which we already observed f_k(x)
        initial_observed_ys: array_like (M,K) numpy array of observed f_k(x) values
        initial_xs_guesses: list, x indecis to evaluate before approximating f (first index is evaluated first)
        h_method: str, name of regression class used. default - 'LinearRegression'
        """

        self.loss_fun = loss_fun
        self.g = expand_to_matrix(g)
        self._N = self.g.shape[0]
        self._K = self.g.shape[1]

        self.ys = np.empty(shape=(self._N, self._K))
        self.ys[:] = np.nan

        if initial_observed_xs is not None:
            assert (
                initial_observed_ys is not None
            ), "observed ys must be provided if observed xs are provided"
            assert len(initial_observed_ys) == len(
                initial_observed_xs
            ), "len(.) of initial observed xs and ys must match"
            self.update_query_result(xs=initial_observed_xs, ys=initial_observed_ys)

        if initial_xs_guesses is not None:
            self.initial_xs_guesses = list(initial_xs_guesses)
        else:
            self.initial_xs_guesses = []

        self.h_method = h_method
        self._h_class = {
            "LinearRegression": sklearn.linear_model.LinearRegression,
        }[h_method]

    def update_query_result(self, xs, ys, k=None):
        """
        Record observed results.

        Parameters
        ----------
        xs : integer array_like (M,), indices of observed data points
        ys : array_like (M,K), observed f(x) values, or,
             if k is not None, ys is expected to be an (M,)
             vector which is used to update ys[:,k].
        """

        if xs is None or ys is None:
            return

        xs = np.atleast_1d(np.asarray(xs, dtype=int))

        if k is not None:  # update particular variable
            ys = np.atleast_1d(np.asarray(ys))
            assert ys.ndim == 1
            not_nan_mask = np.logical_not(np.isnan(ys))
            self.ys[xs, k] = np.where(not_nan_mask, ys, self.ys[xs, k])
        else:
            ys = expand_to_matrix(ys)
            assert ys.shape[1] == self._K
            not_nan_mask = np.logical_not(np.isnan(ys))
            self.ys[xs] = np.where(not_nan_mask, ys, self.ys[xs])

    def _calc_y_aprx(self, xs_to_predict, return_ground_truth_when_available=True):
        xs_to_predict = np.asarray(xs_to_predict, dtype=int)
        y_aprx = np.empty(shape=(len(xs_to_predict), self._K))
        y_aprx[:] = np.nan
        for k in range(self._K):
            if return_ground_truth_when_available:
                # check which observations are already observed
                is_observed_xs = np.logical_not(np.isnan(self.ys[xs_to_predict, k]))
                y_aprx[is_observed_xs, k] = self.ys[xs_to_predict[is_observed_xs], k]
                if is_observed_xs.all():
                    continue
            else:
                is_observed_xs = np.zeros_like(xs_to_predict, dtype=bool)

            # some values are missing, or return_ground_truth_when_available is False. Run regression.
            # for fitting the regression, select only values for which we have both the predictor and criterion.
            not_nan_mask = np.logical_and(
                np.logical_not(np.isnan(self.ys[:, k])),
                np.logical_not(np.isnan(self.g[:, k])),
            )
            assert (
                not_nan_mask.sum() >= 2
            ), "insufficient number of observed predictor-criterion pairs for fitting a regression."
            h = self._h_class()
            h.fit(
                X=expand_to_matrix(self.g[not_nan_mask, k]), y=self.ys[not_nan_mask, k]
            )
            should_predict_xs = np.logical_and(
                np.logical_not(is_observed_xs),
                np.logical_not(np.isnan(self.g[xs_to_predict, k])),
            )
            y_aprx[should_predict_xs, k] = h.predict(
                X=expand_to_matrix(self.g[xs_to_predict[should_predict_xs], k])
            )
        return y_aprx

    def fully_observed_obs(self):
        # this line can be optimized for speed by incremental updating
        return np.flatnonzero(np.all(np.logical_not(np.isnan(self.ys)), axis=1))

    def get_observed_loss_minimum(self):
        """
        yield the best observed x that minimizes loss_fun, and the corresponding loss
        """
        fully_observed_obs = self.fully_observed_obs()
        if len(fully_observed_obs) == 0:
            return None, None, None

        observed_loss = self.loss_fun(self.ys[fully_observed_obs])

        minimum_loss = np.nanmin(observed_loss)
        minima_indices_in_observed_xs = np.flatnonzero(observed_loss == minimum_loss)

        if len(minima_indices_in_observed_xs) > 1:
            print("found {} exact minima.".format(len(minima_indices_in_observed_xs)))
            minimum_index_in_observed_xs = np.random.choice(
                minima_indices_in_observed_xs
            )
        else:
            minimum_index_in_observed_xs = minima_indices_in_observed_xs.item()

        minimum_index = fully_observed_obs[minimum_index_in_observed_xs]
        minimum_ys = self.ys[minimum_index, :]
        return minimum_index, minimum_loss, minimum_ys

    def get_unobserved_loss_minimum(self):
        """
        yield the best yet unobserved x that minimizes loss_fun

        returns the index of the loss minimizer, the *predicted* loss at the point,
        and the variable indices (e.g. [0,1]) of the missing variables at that point
        """

        # use initial guesses first, if available
        if len(self.initial_xs_guesses) > 0:
            minimum_index = self.initial_xs_guesses.pop(0)
            minimum_loss = None
        else:
            # this line can be optimized for speed by incremental updating
            unobserved_obs = np.flatnonzero(np.any(np.isnan(self.ys), axis=1))

            if len(unobserved_obs) == 0:
                return None, None, []
            y_aprx = self._calc_y_aprx(unobserved_obs)
            predicted_loss = self.loss_fun(y_aprx)

            minimum_loss = np.nanmin(predicted_loss)
            minima_indices_in_unobserved_xs = np.flatnonzero(
                predicted_loss == minimum_loss
            )

            if len(minima_indices_in_unobserved_xs) > 1:
                print("found {} minima.".format(len(minima_indices_in_unobserved_xs)))
                minimum_index_in_unobserved_xs = np.random.choice(
                    minima_indices_in_unobserved_xs
                )
            else:
                minimum_index_in_unobserved_xs = minima_indices_in_unobserved_xs
            minimum_index = unobserved_obs[minimum_index_in_unobserved_xs].item()
        which_variables_are_missing = np.flatnonzero(np.isnan(self.ys[minimum_index]))
        return minimum_index, minimum_loss, which_variables_are_missing

    def get_loss_for_x(self, x):
        """
        yield the exact loss for scalar index x.

        args:
        x (int) observation index
        """
        ys = self.ys[slice(x, x + 1)]
        assert not np.isnan(ys).any(), "exact ys must be already observed"
        x_exact_loss = self.loss_fun(ys)
        if isinstance(x_exact_loss, np.ndarray):
            x_exact_loss = x_exact_loss.item()
        return x_exact_loss, ys

    def debugging_figure(self):
        fig = plt.figure()

        all_xs = np.arange(self._N)
        y_aprx = self._calc_y_aprx(all_xs)
        predicted_loss = self.loss_fun(y_aprx)

        for k in range(self._K):

            fig.add_subplot(2, self._K, k * 2 + 1)

            IX = np.argsort(self.g[:, k])

            # plot predictions
            plt.plot(self.g[IX, k], y_aprx[IX, k], "k--")

            # plot observed xs and f(xs)
            observed_obs_mask = np.logical_not(np.isnan(self.ys[:, k]))
            plt.scatter(self.g[observed_obs_mask, k], self.ys[observed_obs_mask, k])
            plt.ylabel("f_" + str(k + 1) + "(x)")
            plt.xlabel("g_" + str(k + 1) + "(x)")

            fully_obs_mask = np.all(np.logical_not(np.isnan(self.ys)), axis=1)
            plt.subplot(2, self._K, k * 2 + 2)
            plt.plot(
                self.g[fully_obs_mask, k], self.loss_fun(y_aprx[fully_obs_mask]), "r--"
            )
            plt.scatter(
                self.g[fully_obs_mask, k], self.loss_fun(self.ys[fully_obs_mask])
            )
            plt.xlabel("g_" + str(k + 1) + "(x)")
            plt.ylabel("loss_fun(f(x))")

        plt.tight_layout()
        plt.show()


class SetInterpolationSearchPandas(SetInterpolationSearch):
    # use a pandas dataframe instead
    def __init__(
        self, loss_fun, df, initial_xs_guesses=None, h_method="LinearRegression"
    ):

        g = df.filter(regex="^approximate_", axis=1).to_numpy()
        initial_observed_ys = df.filter(regex="^exact_", axis=1).to_numpy()
        mask = np.logical_not(np.isnan(initial_observed_ys)).any(axis=1)
        initial_observed_ys = initial_observed_ys[mask]
        initial_observes_xs = np.flatnonzero(mask)

        super().__init__(
            loss_fun,
            g,
            initial_observed_xs=initial_observes_xs,
            initial_observed_ys=initial_observed_ys,
            initial_xs_guesses=initial_xs_guesses,
            h_method=h_method,
        )


if __name__ == "__main__":

    # toy example
    x = np.arange(100)

    g_1 = np.asarray(x, dtype=float)
    g_1[11] = np.nan  # simulate missing variable

    g_2 = 100 - np.asarray(x, dtype=float)
    g_2[11] = np.nan  # simulate missing variable
    g = np.stack([g_1, g_2], axis=-1)
    f_1 = 0.9 * g_1 + np.random.normal(size=g_1.shape) * 1e2

    f_2 = 2 * g_2 + np.random.normal(size=g_2.shape) * 1e2
    f = np.stack([f_1, f_2], axis=-1)

    loss_fun = lambda f: abs(f[:, 0] ** 2 - f[:, 1] ** 2)

    initial_observed_xs = np.asarray([0, 11, 99])
    initial_observed_ys = f[initial_observed_xs]

    opt = SetInterpolationSearch(
        loss_fun=loss_fun,
        g=g,
        initial_observed_xs=initial_observed_xs,
        initial_observed_ys=initial_observed_ys,
        initial_xs_guesses=[],
    )

    print("real minimum is at ", np.nanargmin(loss_fun(f[x])).item())

    for i in range(1000):
        minimum_index, best_loss = opt.get_observed_loss_minimum()
        if minimum_index == np.nanargmin(loss_fun(f[x])).item():
            print("root found")
            opt.update_query_result(xs=next_x, ys=next_y)
            break

        next_x, predicted_loss, missing_variables = opt.get_unobserved_loss_minimum()
        if next_x is None:
            print("predictions depleted")
            break
        next_y = f[next_x, :].reshape(1, -1)
        next_loss = loss_fun(next_y)
        print(
            "next_x:",
            next_x,
            "next_y:",
            next_y,
            "next_loss:",
            next_loss,
            "missing_variables:",
            missing_variables,
        )

        # to test the code, let's update just one variable
        variable_to_update = np.random.choice(missing_variables)
        print("updating :", variable_to_update)
        opt.update_query_result(
            xs=next_x, ys=next_y[:, variable_to_update], k=variable_to_update
        )

    opt.debugging_figure()
