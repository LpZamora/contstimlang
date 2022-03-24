import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine
import matplotlib.pyplot as plt
import seaborn as sns

def _rank_with_avg_ties(x):
     my_vec = pd.Series(x)
     return np.asarray(my_vec.rank(method='average'))

def _rank_with_random_tie_breaking(x):
     my_vec = pd.Series(x)
     return np.asarray(my_vec.sample(frac=1).rank(method='first').reindex_like(my_vec))

def calc_signed_rank_with_avg_ties(x):
    non_zero_mask = np.not_equal(x,0)
    r = np.zeros_like(x, dtype=np.float64)
    r[non_zero_mask] = np.sign(x[non_zero_mask])*_rank_with_avg_ties(np.abs(x[non_zero_mask]))
    return r

def norm_of_ranked_vector(n):
    return np.sqrt(n*(n+1)*(2*n+1)/6)

def calc_signed_rank_cosine_similarity_analytical_RAE(a,b):
     """ Calculate signed rank cosine similarity with expectancy over random tie breaking by means of an analytical solution """

     assert len(a) == len(b)

     a = np.asarray(a)
     b = np.asarray(b)

     if not (np.any(a) and np.any(b)): # cosine similarity is undefined for all zero vectors
          return np.nan

     signed_ranks_a = calc_signed_rank_with_avg_ties(a)
     signed_ranks_b = calc_signed_rank_with_avg_ties(b)

     a_norm = norm_of_ranked_vector(np.sum(np.not_equal(a,0)))
     b_norm = norm_of_ranked_vector(np.sum(np.not_equal(b,0)))
     return np.dot(signed_ranks_a, signed_ranks_b) / (a_norm * b_norm)

def calc_semi_signed_rank_cosine_similarity_analytical_RAE(a,b):
     """ a variant where b is not ranked """

     assert len(a) == len(b)

     a = np.asarray(a)
     b = np.asarray(b)

     if not (np.any(a) and np.any(b)): # cosine similarity is undefined for all zero vectors
          return np.nan

     signed_ranks_a = calc_signed_rank_with_avg_ties(a)

     a_norm = norm_of_ranked_vector(np.sum(np.not_equal(a,0)))
     b_norm = np.linalg.norm(b)
     return np.dot(signed_ranks_a, b) / (a_norm * b_norm)

def calc_expected_normalized_RAE_signed_rank_response_pattern(x):
     r = calc_signed_rank_with_avg_ties(x)
     n_non_zero = np.sum(np.not_equal(x,0))
     norm_r = norm_of_ranked_vector(n_non_zero)
     return r/norm_r

# from here, sampling related calculations (used only for validating the analytical solution)

def _calc_RAE_signed_rank_sampling(x):
    """ slow and noisy, use only for testing purposes"""
    x = np.asarray(x)
    non_zero_mask = x!=0
    r = np.zeros_like(x,dtype=np.float64)
    r[non_zero_mask] = np.sign(x[non_zero_mask])*_rank_with_random_tie_breaking(np.abs(x[non_zero_mask]))
    return r

def _calc_signed_rank_cosine_similarity_sampling_RAE(a,b, n_samples=100):
     """ Calculate signed rank cosine similarity with expectancy over random tie breaking by means of sampling
     this is slow and inexact, currently not used in the analysis.
     See _calc_signed_rank_cosine_similarity_analytical_RAE instead
     """
     assert len(a) == len(b)

     a = np.asarray(a)
     b = np.asarray(b)

     result = []
     for i_sample in range(n_samples):
          a_signed_ranks = _calc_RAE_signed_rank_sampling(a)
          b_signed_ranks = _calc_RAE_signed_rank_sampling(b)
          result.append(1-cosine(a_signed_ranks,b_signed_ranks))
     return np.mean(result)

def upper_bound_on_reliability_closed_form(x):
     """
     args:
     x (np.asarray) n_observations x n_realizations matrix of observations

     returns the an upper bound on the
     """

     r = np.ones_like(x, dtype=np.float64)
     for i_realization in range(x.shape[1]):
          r[:,i_realization]=calc_expected_normalized_RAE_signed_rank_response_pattern(
               x[:,i_realization])
     average_pattern = r.mean(axis=1) # n_obs long vector

     return (average_pattern/np.linalg.norm(average_pattern)) @ r

def _upper_bound_on_reliability_optimization(x):
     """
     args:
     x (np.asarray) n_observations x n_realizations matrix of observations
     """

     n_obs, n_realizations = x.shape
     r = np.ones_like(x, dtype=np.float64)
     for i_realization in range(n_realizations):
          r[:,i_realization]=calc_expected_normalized_RAE_signed_rank_response_pattern(
               x[:,i_realization])

     import torch

     r = torch.as_tensor(r, dtype=torch.float32)
     best_prediction = torch.randn(n_obs,dtype=torch.float32)

     best_prediction.requires_grad = True
     optimizer = torch.optim.LBFGS([best_prediction])
     def closure():
        optimizer.zero_grad()
        bound = ((best_prediction/torch.norm(best_prediction)) @ r).mean()
        loss = -bound
        loss.backward()
        return loss
     for i in range(100):
        optimizer.step(closure)

     bound = ((best_prediction/torch.norm(best_prediction)) @ r).mean().item()
     return bound

def _test_upper_bound():
     for n_obs in [3,5,10,100]:
          for n_realizations in [1,2,5,10,100]:
               print (f'testing {n_obs} by {n_realizations} data')
               for i_rep in range(10):
                    x = None
                    while x is None:
                         x = np.random.randint(low=-3,high=4,size=(n_obs,n_realizations))
                         if (x == 0).all(axis=0).any():
                              x = None # don't use examples with all zero vectors
                    closed_form = upper_bound_on_reliability_closed_form(x).mean()
                    optimized = _upper_bound_on_reliability_optimization(x)
                    assert np.isclose(closed_form,optimized) or closed_form>optimized

def _test_calc_signed_rank_cosine_similarity_analytical_RAE(n_simulations=3,n_samples=100):
     """ sanity test for the analytical RAE estimate """
     n_range = [3]

     c_analytical = []
     c_sampling = []

     for n in n_range:
          for a_magnitude in [2,3,5,10]:
               for b_magnitude in [2,3,5,10]:
                    for i_sim in range(n_simulations):
                         a = np.random.randint(a_magnitude+1,size=n)-a_magnitude/2
                         b = np.random.randint(b_magnitude+1,size=n)-b_magnitude/2

                         if not (np.any(a) and np.any(b)):
                              continue # skip the all zero case, cosine similarity is undefined
                         c_analytical.append(calc_signed_rank_cosine_similarity_analytical_RAE(a,b))
                         c_sampling.append(_calc_signed_rank_cosine_similarity_sampling_RAE(a,b, n_samples=n_samples))

     sns.scatterplot(x=c_analytical,y=c_sampling)
     plt.xlabel('analytical','sampling')
     plt.plot([-1,1],[-1,1],'k')
     plt.show()

if __name__ == '__main__':

  _test_upper_bound()