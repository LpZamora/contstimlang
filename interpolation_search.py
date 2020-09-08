import numpy as np
import sklearn.linear_model
import matplotlib.pyplot as plt

def expand_to_matrix(x):
    x=np.asarray(x)
    if x.ndim==0:
        x=np.expand_dims(x,-1)
    if x.ndim==1:
        x=np.expand_dims(x,-1)
    return x

class SetInterpolationSearch:
    def __init__(self,loss_fun,g,initial_observed_xs=None,initial_observed_ys=None,initial_xs_guesses=None,h_method='LinearRegression'):
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
        g: array_like, precomputed g(x) for each x, as a (N,K) numpy array.
        initial_observed_xs: array_like, (M,) numpy array of x indecis for which we already observed f_k(x)
        initial_observed_ys: array_like (M,K) numpy array of observed f_k(x) values
        initial_xs_guesses: list, x indecis to evaluate before approximating f (first index is evaluated first)
        h_method: str, name of regression class used. default - 'LinearRegression'
        """

        self.loss_fun=loss_fun
        self.g=expand_to_matrix(g)
        self._N=self.g.shape[0]
        self._K=self.g.shape[1]
        self._xs_was_observed=np.zeros((self._N,),dtype=bool)

        if initial_observed_xs is not None:
            assert initial_observed_ys is not None, "observed ys must be provided if observed xs are provided"
            assert len(initial_observed_ys)==len(initial_observed_xs), "len(.) of initial observed xs and ys must match"
            self.xs=np.atleast_1d(np.asarray(initial_observed_xs,dtype=int))
            self.ys=expand_to_matrix(initial_observed_ys)
            assert self.ys.shape[1]==self._K, "g.shape[1] must match shape(initial_observed_ys)[1]"
            self._xs_was_observed[self.xs]=True
        else:
            self.xs=np.empty(dtype=int,shape=(0,))
            self.ys=np.empty(shape=(0,self._K))

        if initial_xs_guesses is not None:
            self.initial_xs_guesses=list(initial_xs_guesses)
        else:
            self.initial_xs_guesses=[]

        self.h_method=h_method
        self._h_class={'LinearRegression':sklearn.linear_model.LinearRegression,
            }[h_method]

    def update_query_result(self,xs,ys):
        """
        Record observed results.

        Parameters
        ----------
        xs : integer array_like (M,), indices of observed data points
        ys : array_like (M,K), observed f(x) values
        """

        if xs is None or ys is None:
            return

        xs=np.atleast_1d(np.asarray(xs,dtype=int))
        ys=expand_to_matrix(ys)

        if len(xs)==0 or len(ys)==0:
            return

        self.xs=np.concatenate([self.xs,xs],0)
        self.ys=np.concatenate([self.ys,ys],0)
        self._xs_was_observed[xs]=True
    def _calc_y_aprx(self,xs_to_predict):
        y_aprx=np.empty(shape=(len(xs_to_predict),self._K))
        for k in range(self._K):
            h=self._h_class()
            h.fit(X=expand_to_matrix(self.g[self.xs,k]),y=self.ys[:,k])
            y_aprx[:,k]=h.predict(X=expand_to_matrix(self.g[xs_to_predict,k]))
        return y_aprx

    def yield_next_x(self):
        """
        yield the best yet unobserved x that minimizes loss_fun
        """

        # use initial guesses first, if available
        if len(self.initial_xs_guesses)>0:
            return self.initial_xs_guesses.pop(0)

        unobserved_x_indecis=np.nonzero(np.logical_not(self._xs_was_observed))[0]

        if len(unobserved_x_indecis)==0:
            return None
        y_aprx=self._calc_y_aprx(unobserved_x_indecis)
        predicted_loss=self.loss_fun(y_aprx)
        return unobserved_x_indecis[np.argmin(predicted_loss)]

    def debugging_figure(self):
        fig=plt.figure()

        all_xs=np.arange(self._N)
        y_aprx=self._calc_y_aprx(all_xs)
        predicted_loss=self.loss_fun(y_aprx)

        for k in range(self._K):

            fig.add_subplot(2,self._K,k*2+1)

            IX=np.argsort(self.g[:,k])

            # plot predictions
            plt.plot(self.g[IX,k],y_aprx[IX,k],'k--')

            # plot observed xs and f(xs)
            plt.scatter(self.g[self.xs,k],self.ys[:,k],c=np.arange(len(self.xs)),cmap='viridis')
            plt.ylabel('f_'+str(k+1)+'(x)')
            plt.xlabel('g_'+str(k+1)+'(x)')

            plt.subplot(2,self._K,k*2+2)
            plt.plot(self.g[self.xs,k],self.loss_fun(self.ys),'r--')
            plt.scatter(self.g[self.xs,k],self.loss_fun(self.ys),c=np.arange(len(self.xs)),cmap='viridis')
            plt.xlabel('g_'+str(k+1)+'(x)')
            plt.ylabel('loss_fun(f(x))')

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":

    # toy example
    x=np.arange(100)
    g_1=x
    g_2=100-x
    g=np.stack([g_1,g_2],axis=-1)
    f_1=0.9*g_1+np.random.normal(size=g_1.shape)
    f_2=2*g_2+np.random.normal(size=g_2.shape)-1
    f=np.stack([f_1,f_2],axis=-1)
    loss_fun=lambda f: abs(f[:,0]-f[:,1])

    initial_observed_xs=np.asarray([20])
    initial_observed_ys=f[initial_observed_xs]

    opt=SetInterpolationSearch(loss_fun=loss_fun,
        g=g,initial_observed_xs=initial_observed_xs,
        initial_observed_ys=initial_observed_ys,
        initial_xs_guesses=[21])

    print('real minimum is at ',np.argmin(loss_fun(f[x])).item())

    for i in range(100):
        next_x=opt.yield_next_x()
        if next_x is None:
            print('predictions depleted')
            break
        next_y=f[next_x,:].reshape(1,-1)
        next_loss=loss_fun(next_y)
        opt.update_query_result(xs=next_x,ys=next_y)
        print('next_x:',next_x,'next_y:',next_y,'next_loss:',next_loss)
        if next_x==np.argmin(loss_fun(f[x])).item():
            print('root found')
            opt.update_query_result(xs=next_x,ys=next_y)
            break
    opt.debugging_figure()
