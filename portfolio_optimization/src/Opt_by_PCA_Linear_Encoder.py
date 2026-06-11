


# Trading Strategy based on PCA
# 
# Welcome to your course project. This exercise gives you a hands-on experience to use PCA to:
# 
# - construct eigen-portfolios
# - implement a measure of market systemic risk
# - develop simple trading strategy
# 

import pandas as pd
import numpy as np
import sklearn.decomposition
import tensorflow as tf
from tensorflow.contrib.layers import fully_connected
import sys

import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

print("Package Versions:")
print("  scikit-learn: %s" % sklearn.__version__)
print("  tensorflow: %s" % tf.__version__)

sys.path.append("..")
import grading

try:
    import sklearn.model_selection
    import sklearn.linear_model
except:
    print("Looks like an older version of sklearn package")

try:
    import pandas as pd
    print("  pandas: %s"% pd.__version__)
except:
    print("Missing pandas package")



# #### Dataset:  daily prices of  stocks from S&P 500 index  ####
# For this exercise we will be working with S&P 500 Index stock prices dataset. 
# The following cell computes returns based for a subset of S&P 500 index stocks. It starts with stocks price data:


import os


asset_prices = pd.read_csv('/home/jovyan/work/readonly/spx_holdings_and_spx_closeprice.csv',
                     date_parser=lambda dt: pd.to_datetime(dt, format='%Y-%m-%d'),
                     index_col = 0).dropna()

import os
os.path.exists("/home/jovyan/work/readonly/spx_holdings_and_spx_closeprice.csv")

import shutil

shutil.copy(
    "/home/jovyan/work/readonly/spx_holdings_and_spx_closeprice.csv",
    "spx_holdings_and_spx_closeprice.csv"
)




n_stocks_show = 12
print('Asset prices shape', asset_prices.shape)
asset_prices.iloc[:, :n_stocks_show].head()




asset_returns = np.log(asset_prices) - np.log(asset_prices.shift(1))
asset_returns = asset_prices.pct_change(periods=1)
asset_returns = asset_returns.iloc[1:, :]
asset_returns.iloc[:, :n_stocks_show].head()


def center_returns(r_df):
    """
    Normalize, i.e. center and divide by standard deviation raw asset returns data

    Arguments:
    r_df -- a pandas.DataFrame of asset returns

    Return:
    normed_df -- normalized returns
    """
    mean_r = r_df.mean(axis=0)
    sd_r = r_df.std(axis=0)
    normed_df = (r_df - mean_r) / sd_r
    return normed_df



normed_r = center_returns(asset_returns)
normed_r.iloc[:, :n_stocks_show].head()


def exponent_weighting(n_periods, half_life = 252):
    """
    Calculate exponentially smoothed normalized (in probability density function sense) weights

    Arguments:
    n_periods -- number of periods, an integer, N in the formula above
    half_life -- half-life, which determines the speed of decay, h in the formula
    
    Return:
    exp_probs -- exponentially smoothed weights, np.array
    """
    
    exp_probs = np.zeros(n_periods) 

    sum=0
    for j in range(n_periods):
        exp_probs[j]= np.exp(-np.log(2)*j/half_life)
        sum += exp_probs[j]
    exp_probs =exp_probs/sum
    return exp_probs




exp_probs = exponent_weighting(252*1)
plt.plot(exp_probs, linewidth=3)




def absorption_ratio(explained_variance, n_components):
    """
    Calculate absorption ratio via PCA. absorption_ratio() is NOT to be used with Auto-Encoder. 
    
    Arguments:
    explained_variance -- 1D np.array of explained variance by each pricincipal component, in descending order
    
    n_components -- an integer, a number of principal components to compute absorption ratio
    
    Return:
    ar -- absorption ratio
    """
    ar = np.sum(explained_variance[:n_components]) / np.sum(explained_variance)
    return ar




def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)
    
class LinearAutoEncoder:

    def __init__(self, n_inputs, n_codings, learning_rate=0.01):
        self.learning_rate = learning_rate
        n_outputs = n_inputs
        self.destroy()
        reset_graph()
    
        # the inputs are n_inputs x n_inputs covariance matrices
        self.X = tf.placeholder(tf.float32, shape=[None, n_inputs, n_inputs])
        
        with tf.name_scope("lin_ae"):

            # -------------------
            # Encoder
            # -------------------
            
            self.codings_layer = tf.layers.dense(self.X, n_codings)  
            self.outputs = tf.layers.dense(self.codings_layer, n_outputs)
            
 
        with tf.name_scope("loss"):
            self.reconstruction_loss = None
            self.training_op = None
            ### START CODE HERE ### (≈ 4-5 lines of code)
            self.reconstruction_loss = tf.losses.mean_squared_error(self.X, self.outputs)
            self.training_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.reconstruction_loss)
            #optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            #self.training_op = optimizer.minimize(self.reconstruction_loss)


            ### END CODE HERE ###
            self.init = tf.global_variables_initializer()
            
    def destroy(self):
        if hasattr(self, 'sess') and self.sess is not None:
            self.sess.close()
            self.sess = None

    def absorption_ratio(self, test_input):
        """
        Calculate absorption ratio based on already trained model
        """
        if self.outputs is None:
            return test_input, 0.
        
        with self.sess.as_default():  # do not close session
            codings = self.codings_layer.eval(feed_dict={self.X: test_input})

            # calculate variance explained ratio
            result_ = self.outputs.eval(feed_dict={self.X: test_input})
            var_explained = np.sum(np.diag(result_.squeeze())) / np.sum(np.diag(test_input.squeeze()))

        return codings[0,:,:], var_explained
    
    def next_batch(self, X_train, batch_size):

        y_batch = None

        selected_idx = np.random.choice(tuple(range(X_train.shape[0])), size=batch_size)
        X_batch = X_train[selected_idx, :, :]
        return X_batch, y_batch

    def train(self, X_train, X_test, n_epochs=5, batch_size=2, verbose=False):

        if self.outputs is None:
            return X_test, 0.
        
        n_examples = len(X_train)  # number of training examples
        self.sess = tf.Session()
        
        # as_default context manager does not close the session when you exit the context,
        # and you must close the session explicitly.
        with self.sess.as_default():
            self.init.run()
            for epoch in range(n_epochs):
                n_batches = n_examples // min(n_examples, batch_size)
                for _ in range(n_batches):
                    X_batch, y_batch = self.next_batch(X_train, batch_size)
                    self.sess.run(self.training_op, feed_dict={self.X: X_batch})
                
                if verbose:
                    # last covariance matrix from the training sample
                    if X_train.shape[0] == 1:
                        mse_train = self.reconstruction_loss.eval(feed_dict={self.X: X_train})
                    else:
                        mse_train = self.reconstruction_loss.eval(feed_dict={self.X: np.array([X_train[-1, :, :]])})
                    mse_test = self.reconstruction_loss.eval(feed_dict={self.X: X_test})
                    print('Epoch %d. MSE Train %.4f, MSE Test %.4f' % (epoch, mse_train, mse_test))

            # calculate variance explained ratio
            test_input = np.array([X_train[-1, :, :]])
            result_ = self.outputs.eval(feed_dict={self.X: test_input})
            var_explained = np.sum(np.diag(result_.squeeze())) / np.sum(np.diag(test_input.squeeze()))
            # print('Linear Auto-Encoder: variance explained: %.2f' % var_explained)

            codings = self.codings_layer.eval(feed_dict={self.X: X_test})
            # print('Done training linear auto-encoder')

        return codings[0,:,:], var_explained


ix_offset = 1000
stock_tickers = asset_returns.columns.values[:-1]
assert 'SPX' not in stock_tickers, "By accident included SPX index"

step_size = 60
num_samples = 5
lookback_window = 252 * 2   # in (days)
num_assets = len(stock_tickers)
cov_matricies = np.zeros((num_samples, num_assets, num_assets)) # hold training data

# collect training and test data
ik = 0
for ix in range(ix_offset, min(ix_offset + num_samples * step_size, len(normed_r)), step_size):
    ret_frame = normed_r.iloc[ix_offset - lookback_window:ix_offset, :-1]
    print("time index and covariance matrix shape", ix, ret_frame.shape)
    cov_matricies[ik, :, :] = ret_frame.cov()
    ik += 1

# the last covariance matrix determines the absorption ratio
lin_ae = LinearAutoEncoder(n_inputs=num_assets, n_codings=200)
np.array([cov_matricies[-1, :, :]]).shape
lin_codings, test_absorp_ratio = lin_ae.train(cov_matricies[ : int((2/3)*num_samples), :, :],
                                                np.array([cov_matricies[-1, :, :]]),
                                                n_epochs=10, 
                                                batch_size=5)
lin_codings, in_sample_absorp_ratio = lin_ae.absorption_ratio(np.array([cov_matricies[0, :, :]]))



### GRADED PART (DO NOT EDIT) ###
part_2=[test_absorp_ratio, in_sample_absorp_ratio]
try:
    part2 = " ".join(map(repr, part_2))
except TypeError:
    part2 = repr(part_2)
submissions[all_parts[1]]=part2
grading.submit(COURSERA_EMAIL, COURSERA_TOKEN, assignment_key,all_parts[:2],all_parts,submissions)
[test_absorp_ratio, in_sample_absorp_ratio]


# In[15]:


stock_tickers = asset_returns.columns.values[:-1]
assert 'SPX' not in stock_tickers, "By accident included SPX index"

half_life = 252             # in (days)
lookback_window = 252 * 2   # in (days)
num_assets = len(stock_tickers)
step_size = 1          # days : 5 - weekly, 21 - monthly, 63 - quarterly

# require of that much variance to be explained. How many components are needed?
var_threshold = 0.8     

# fix 20% of principal components for absorption ratio calculation. How much variance do they explain?
absorb_comp = int((1 / 5) * num_assets)  

print('Half-life = %d' % half_life)
print('Lookback window = %d' % lookback_window)
print('Step size = %d' % step_size)
print('Variance Threshold = %d' % var_threshold)
print('Number of stocks = %d' % num_assets)
print('Number of principal components = %d' % absorb_comp)



# indexes date on which to compute PCA
days_offset = 4 * 252
num_days = 6 * 252 + days_offset
pca_ts_index = normed_r.index[list(range(lookback_window + days_offset, min(num_days, len(normed_r)), step_size))]

# allocate arrays for storing absorption ratio
pca_components = np.array([np.nan]*len(pca_ts_index))
absorp_ratio = np.array([np.nan]*len(pca_ts_index))
lae_ar = np.array([np.nan]*len(pca_ts_index))  # absorption ratio computed by Auto-Encoder 

# keep track of covariance matricies as we would need them for training Auto-Encoder
buf_size = 5
cov_matricies = np.zeros((buf_size, num_assets, num_assets))

exp_probs = exponent_weighting(lookback_window, half_life)
assert 'SPX' not in normed_r.iloc[:lookback_window, :-1].columns.values, "By accident included SPX index"



import time
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd

ik = 0
use_ewm = False

time_start = time.time()

for ix in range(lookback_window + days_offset, min(num_days, len(normed_r)), step_size):

    # rolling window of returns (UNCHANGED)
    #ret_frame = normed_r.iloc[ix - lookback_window:ix, :-1]

    #if use_ewm:
        #ret_frame = (ret_frame.T * exp_probs).T

    #circ_idx = ik % buf_size

    # compute PCA only on same schedule as before
    if ik == 0 or ik % 21 == 0:
        
        ret_frame = normed_r.iloc[ix - lookback_window:ix, :-1]
        # exact covariance as original logic
        cov_mat = ret_frame.cov()

        # PCA identical to original
        pca = PCA().fit(cov_mat.values)

        cum_var = np.cumsum(pca.explained_variance_ratio_)
        k = int(np.argmax(cum_var >= var_threshold) + 1)

        pca_components[ik] = k

        absorp_ratio[ik] = absorption_ratio(
            pca.explained_variance_,
            absorb_comp
        )

    else:
        # forward fill (exact original behavior)
        absorp_ratio[ik] = absorp_ratio[ik - 1]
        pca_components[ik] = pca_components[ik - 1]
    
    
    if ik % 105 == 0:
        print("Absorption Ratio")
    ik += 1

print('Absorption Ratio done! Time elapsed: {} seconds'.format(time.time() - time_start))

ts_pca_components = pd.Series(pca_components, index=pca_ts_index)
ts_absorb_ratio = pd.Series(absorp_ratio, index=pca_ts_index)


# In[ ]:


# run the main loop computing PCA and absorption at each step using moving window of returns  
# run this loop using both exponentially weighted returns and equally weighted returns
import time
from sklearn.decomposition import PCA
ik = 0
use_ewm = False
lin_ae = None
time_start = time.time()
for ix in range(lookback_window + days_offset, min(num_days, len(normed_r)), step_size):
    ret_frame = normed_r.iloc[ix - lookback_window:ix, :-1]  # fixed window
    # ret_frame = normed_r.iloc[:ix, :-1]  # ever-growing window
    if use_ewm:
        ret_frame = (ret_frame.T * exp_probs).T
    
    cov_mat = ret_frame.cov()
    circ_idx = ik % buf_size
    cov_matricies[circ_idx, :, :] = cov_mat.values

    if ik == 0 or ik % 21 == 0:
        ### START CODE HERE ### (≈ 4-5 lines of code)
        ### fit PCA, compute absorption ratio by calling absorption_ratio()
        ### store result into pca_components for grading
        
        # fit PCA on covariance matrix
        # fit PCA


        pca = PCA().fit(cov_mat.values)

        cum_var = np.cumsum(pca.explained_variance_ratio_)

        k = int(np.argmax(cum_var >= var_threshold) + 1)

        pca_components[ik] = k

        absorp_ratio[ik] = absorption_ratio(
            pca.explained_variance_,
            absorb_comp
        )

        ### END CODE HERE ###
    else:
        absorp_ratio[ik] = absorp_ratio[ik-1] 
        pca_components[ik] = pca_components[ik-1]
    
    if ik == 0 or ik % 252 == 0:        
        if lin_ae is not None:
            lin_ae.destroy()

        print('Trainging AE', normed_r.index[ix])
        lin_ae = LinearAutoEncoder(cov_mat.shape[0], absorb_comp)
        lin_codings, lae_ar[ik] = lin_ae.train(cov_matricies[:circ_idx + 1, :, :], np.array([cov_mat.values]),batch_size=2)
    else:
        lin_codings, lae_ar[ik] = lin_ae.absorption_ratio(np.array([cov_mat.values]))

    ik += 1
    
print ('Absorption Ratio done! Time elapsed: {} seconds'.format(time.time() - time_start))    
ts_pca_components = pd.Series(pca_components, index=pca_ts_index)
ts_absorb_ratio = pd.Series(absorp_ratio, index=pca_ts_index)
ts_lae_absorb_ratio = pd.Series(lae_ar, index=pca_ts_index)


# In[ ]:


import time
import numpy as np
import pandas as pd

#from sklearn.decomposition import PCA  # (kept only for compatibility elsewhere)

ik = 0
use_ewm = False
lin_ae = None

time_start = time.time()

# ----------------------------
# PRECOMPUTE NUMPY ARRAY ONCE
# ----------------------------
data = normed_r.values[:, :-1].astype(np.float64)


for ix in range(lookback_window + days_offset, min(num_days, len(normed_r)), step_size):
    ret_frame = normed_r.iloc[ix - lookback_window:ix, :-1]  # fixed window
    # ret_frame = normed_r.iloc[:ix, :-1]  # ever-growing window
    if use_ewm:
        ret_frame = (ret_frame.T * exp_probs).T
    
    cov_mat = ret_frame.cov()
    circ_idx = ik % buf_size
    cov_matricies[circ_idx, :, :] = cov_mat.values
    # ----------------------------
    # PCA / ABSORPTION RATIO STEP
    # ----------------------------
    if ik == 0 or ik % 21 == 0:

        # FAST EIGENVALUE DECOMPOSITION (replaces PCA)
        eigvals = np.linalg.eigvalsh(cov_mat)
        eigvals = np.sort(eigvals)[::-1]

        # Number of components needed to explain var_threshold variance
        cum_var = np.cumsum(eigvals / np.sum(eigvals))
        k = np.searchsorted(cum_var, var_threshold) + 1
        pca_components[ik] = k
        # Absorption ratio using a fixed 20% of principal components
        absorp_ratio[ik] = absorption_ratio(eigvals,absorb_comp) 
    else:
        absorp_ratio[ik] = absorp_ratio[ik - 1]
        pca_components[ik] = pca_components[ik - 1]

    # ----------------------------
    # AUTOENCODER UPDATE (UNCHANGED)
    # ----------------------------
    if ik == 0 or ik % 252 == 0:

        if lin_ae is not None:
            lin_ae.destroy()

        print('Training AE', normed_r.index[ix])

        lin_ae = LinearAutoEncoder(cov_mat.shape[0], absorb_comp)
        lin_codings, lae_ar[ik] = lin_ae.train(cov_matricies[:circ_idx + 1, :, :], np.array([cov_mat.values]),batch_size=2)
    
    else:
        lin_codings, lae_ar[ik] = lin_ae.absorption_ratio(np.array([cov_mat.values]))

    ik += 1
        
    """    
        lin_ae = LinearAutoEncoder(cov_mat.shape[0], absorb_comp)

        lin_codings, lae_ar[ik] = lin_ae.train(
            cov_matricies[:circ_idx + 1],
            np.array([cov_mat]),
            batch_size=2
        )
    else:
        lin_codings, lae_ar[ik] = lin_ae.absorption_ratio(np.array([cov_mat]))

    ik += 1
    """    
print('Absorption Ratio done! Time elapsed:', time.time() - time_start)

ts_pca_components = pd.Series(pca_components, index=pca_ts_index)
ts_absorb_ratio = pd.Series(absorp_ratio, index=pca_ts_index)
ts_lae_absorb_ratio = pd.Series(lae_ar, index=pca_ts_index)


"""
for ix in range(lookback_window + days_offset,
               min(num_days, len(normed_r)),
               step_size):

    # ----------------------------
    # FAST WINDOW EXTRACTION
    # ----------------------------
    window = data[ix - lookback_window:ix]

    if use_ewm:
        window = window * exp_probs  # assumes aligned already

    # ----------------------------
    # FAST COVARIANCE (NUMPY)
    # ----------------------------
    cov_mat = np.cov(window, rowvar=False)

    circ_idx = ik % buf_size
    cov_matricies[circ_idx] = cov_mat
"""
    



import time
import numpy as np
import pandas as pd

from sklearn.decomposition import PCA  # (kept only for compatibility elsewhere)

ik = 0
use_ewm = False
lin_ae = None

time_start = time.time()


last_cov_mat = None
last_eigvals = None

for ix in range(lookback_window + days_offset, min(num_days, len(normed_r)), step_size):

    circ_idx = ik % buf_size

    # ----------------------------
    # ONLY COMPUTE WHEN NEEDED
    # ----------------------------
    run_pca = (ik == 0 or ik % 21 == 0)
    run_ae = (ik == 0 or ik % 252 == 0)

    if run_pca or run_ae:

        ret_frame = normed_r.iloc[ix - lookback_window:ix, :-1]

        if use_ewm:
            ret_frame = (ret_frame.T * exp_probs).T

        cov_mat = ret_frame.cov()

        # store covariance ONLY when computed
        cov_matricies[circ_idx, :, :] = cov_mat.values

        last_cov_mat = cov_mat

    else:
        # reuse last computed covariance (no recompute)
        cov_mat = last_cov_mat

    # ----------------------------
    # PCA / ABSORPTION RATIO STEP
    # ----------------------------
    if run_pca:

        eigvals = np.linalg.eigvalsh(cov_mat.values)
        eigvals = np.sort(eigvals)[::-1]

        last_eigvals = eigvals

        cum_var = np.cumsum(eigvals / np.sum(eigvals))
        k = np.searchsorted(cum_var, var_threshold) + 1

        pca_components[ik] = k

        absorp_ratio[ik] = absorption_ratio(
            eigvals,
            absorb_comp
        )

    else:
        pca_components[ik] = pca_components[ik - 1]
        absorp_ratio[ik] = absorp_ratio[ik - 1]

    # ----------------------------
    # AUTOENCODER UPDATE
    # ----------------------------
    if run_ae:

        if lin_ae is not None:
            lin_ae.destroy()

        print('Training AE', normed_r.index[ix])

        lin_ae = LinearAutoEncoder(cov_mat.shape[0], absorb_comp)

        lin_codings, lae_ar[ik] = lin_ae.train(
            cov_matricies[:circ_idx + 1, :, :],
            np.array([cov_mat.values]),
            batch_size=2
        )

    else:
        lin_codings, lae_ar[ik] = lin_ae.absorption_ratio(
            np.array([cov_mat.values])
        )

    ik += 1

print('Absorption Ratio done! Time elapsed:', time.time() - time_start)

ts_pca_components = pd.Series(pca_components, index=pca_ts_index)
ts_absorb_ratio = pd.Series(absorp_ratio, index=pca_ts_index)
ts_lae_absorb_ratio = pd.Series(lae_ar, index=pca_ts_index)


# In[ ]:


import time
import numpy as np
import pandas as pd

ik = 0
use_ewm = False
lin_ae = None

time_start = time.time()

data = normed_r.values[:, :-1]   # FAST ACCESS (no pandas in loop)

for ix in range(lookback_window + days_offset, min(num_days, len(normed_r)), step_size):

    circ_idx = ik % buf_size

    run_pca = (ik == 0 or ik % 21 == 0)
    run_ae  = (ik == 0 or ik % 252 == 0)

    # ----------------------------
    # FAST WINDOW (IDENTICAL DATA)
    # ----------------------------
    window = data[ix - lookback_window:ix]

    if use_ewm:
        window = window * exp_probs  # only valid if exp_probs already numpy-aligned

    # ----------------------------
    # EXACT SAME COVARIANCE AS PANDAS
    # ----------------------------
    cov_mat = np.cov(window, rowvar=False, ddof=1)

    cov_matricies[circ_idx] = cov_mat

    # ----------------------------
    # PCA / ABSORPTION RATIO
    # ----------------------------
    if run_pca:

        eigvals = np.sort(np.linalg.eigvalsh(cov_mat))[::-1]

        cum_var = np.cumsum(eigvals / np.sum(eigvals))
        k = np.searchsorted(cum_var, var_threshold) + 1

        pca_components[ik] = k
        absorp_ratio[ik] = absorption_ratio(eigvals, absorb_comp)

    else:
        pca_components[ik] = pca_components[ik - 1]
        absorp_ratio[ik] = absorp_ratio[ik - 1]

    # ----------------------------
    # AUTOENCODER (IDENTICAL LOGIC)
    # ----------------------------
    if run_ae:

        if lin_ae is not None:
            lin_ae.destroy()

        lin_ae = LinearAutoEncoder(cov_mat.shape[0], absorb_comp)

        lin_3d = cov_mat[None, :, :]  # same as np.array([cov_mat])

        lin_codings, lae_ar[ik] = lin_ae.train(
            cov_matricies[:circ_idx + 1],
            lin_3d,
            batch_size=2
        )

    else:
        lin_codings, lae_ar[ik] = lin_ae.absorption_ratio(
            cov_mat[None, :, :]
        )

    ik += 1

print('Absorption Ratio done! Time elapsed:', time.time() - time_start)

ts_pca_components = pd.Series(pca_components, index=pca_ts_index)
ts_absorb_ratio = pd.Series(absorp_ratio, index=pca_ts_index)
ts_lae_absorb_ratio = pd.Series(lae_ar, index=pca_ts_index)


# In[ ]:


print("pc_comp",pca_components)
print("absorp_rat",absorp_ratio)
print(ts_lae_absorb_ratio )


# In[ ]:


ts_absorb_ratio.plot(figsize=(12,6), title='Absorption Ratio via PCA', linewidth=3)
plt.savefig("Absorption_Ratio_SPX.png", dpi=900)


# In[ ]:


ts_lae_absorb_ratio.plot(figsize=(12,6), title='Absorption Ratio via Auto-Encoder', linewidth=3)


# Having computed daily (this means the step size is 1) Absorption Ratio times series, we further follow M. Kritzman to make use of AR to define yet another measure: AR Delta. In particular:
# $$ AR\delta = \frac{AR_{15d} - AR_{1y}}{ AR\sigma_{1y}}$$
# We use  $AR\delta$ to build simple portfolio trading strategy

# In[ ]:


# following Kritzman and computing AR_delta = (15d_AR -1yr_AR) / sigma_AR
ts_ar = ts_absorb_ratio
ar_mean_1yr = ts_ar.rolling(252).mean()
ar_mean_15d = ts_ar.rolling(15).mean()
ar_sd_1yr = ts_ar.rolling(252).std()
ar_delta = (ar_mean_15d - ar_mean_1yr) / ar_sd_1yr    # standardized shift in absorption ratio

df_plot = pd.DataFrame({'AR_delta': ar_delta.values, 'AR_1yr': ar_mean_1yr.values, 'AR_15d': ar_mean_15d.values}, 
                       index=ts_ar.index)
df_plot = df_plot.dropna()
if df_plot.shape[0] > 0:
    df_plot.plot(figsize=(12, 6), title='Absorption Ratio Delta', linewidth=3)



def get_weight(ar_delta):
    '''
    Calculate EQ / FI portfolio weights based on Absorption Ratio delta
    Arguments:
    ar_delta -- Absorption Ratio delta
    
    Return: 
        wgts -- a vector of portfolio weights
    '''
    
    ### START CODE HERE ### (≈ 6 lines of code)
    ### ....
        
    # Extreme high regime → risk-off (100% FI)
    if ar_delta > 1:
        wgts = [0.0, 1.0]
        
    # Extreme low regime → risk-on (100% EQ)
    elif ar_delta < -1:
        wgts = [1.0, 0.0]
        
    # Neutral regime → balanced portfolio
    else:
        wgts = [0.5, 0.5]
    

    return wgts
    ### END CODE HERE ###






etf_r= pd.read_csv('/home/jovyan/work/readonly/pca_hw5_etf_returns.csv',
                     date_parser=lambda dt: pd.to_datetime(dt, format='%Y-%m-%d'),
                     index_col = 0)
etf_prices = pd.read_csv('/home/jovyan/work/readonly/millenials_portfolio_etfs.csv',
                         date_parser=lambda dt: pd.to_datetime(dt, format='%Y-%m-%d'),
                         index_col = 0)
etf_returns = etf_prices.pct_change(periods=1)
etf_returns = etf_returns.iloc[1450:, :]
etf_r.head()


# #### Part 4 (Calculate performance of backtested strategy)
# 
# **Instructions:**
# 
# Implement function backtest_strategy which given a DataFrame of strategy weights and a DataFrame asset returns annualized return, volatility and Sharpe ratio of a strategy.

# In[ ]:


def backtest_strategy(strat_wgts, asset_returns, periods_per_year = 252):
    '''
    Calculate portfolio returns and return portfolio strategy performance
    Arguments:
    
    strat_wgts -- pandas.DataFrame of weights of the assets
    asset_returns -- pandas.DataFrame of asset returns
    periods_per_year -- number of return observations per year
    
    Return: 
        (ann_ret, ann_vol, sharpe) -- a tuple of (annualized return, annualized volatility, sharpe ratio)
    '''

    ### START CODE HERE ### (≈ 10 lines of code)

    # return 0., 0., 1. # annualized return,  annualized volatility,  sharp ratio
    ### END CODE HERE ###

