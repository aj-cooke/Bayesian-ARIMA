import pystan
import numpy as np
import pandas as pd

with open('ARMAX_normal.stan') as f:
    model = f.read()

with open('ARMAX_product_normal.stan') as f:
    model_pn = f.read()


class BayesARIMA:
    def __init__(self):
        self.model = model
        self.model_pn = model_pn
        self.sm = pystan.StanModel(model_code=self.model, extra_compile_args=["-w"])
        
    def fit(self, endog: pd.Series, exog, dates: pd.Series, p: int, 
            q: int, prior = 'normal', num_normal = 5, scale = 5, iters = 2000, 
            chains = 4, warmup = 500, seed = 42, stan_override = None):
        '''
        Begin sampling of stan model given data. Saves means of parameters
        for prediction.

        Parameters
        ----------
        endog : pd.Series
            Series being predicted.
        exog : pd.DataFrame
            Exogenous time series.
        dates : pd.Series
            Date column for the series.
        p : int
            AR component lags.
        q : int
            MA component lags.
        prior : str, optional
            Either normal or product normal. Product normal approximates spike 
            and slab. The default is 'normal'.
        num_normal : int, optional
            Number of normal distributions to multiply together for product normal. The default is 5.
        scale : int, optional
            Sigma for product normal priors. The default is 5.
        iters : int, optional
            Iterations for each chain during NUTS sampling. The default is 2000.
        chains : int, optional
            Number of chains for NUTS sampling. The default is 4.
        warmup : int, optional
            Number of iterations to burn before official samples begin. The default is 500.
        seed : int, optional
            Random seed. The default is 42.
        stan_override : dict, optional
            Stan data dictionary to provide more detailed arguments to Stan model. The default is None.

        '''
        self.endog = endog
        self.p = p
        self.q = q
        if stan_override != None:
            self.data = stan_override
        elif prior == 'normal':
            self.data = {'N': len(endog), 'y': endog, 'p': p, 'q':q, 'K': exog.shape[1],
                         'exog': exog, 'scale':scale, 'max_pq': max(p,q)}
        elif prior == 'product normal':
            self.data = {'N': len(endog), 'y': endog, 'p': p, 'q':q, 'K': exog.shape[1],
                         'exog': exog, 'nn_phi': num_normal, 'max_pq': max(p,q),
                         'nn_exog':num_normal, 'nn_theta':num_normal,
                         'scale_phi':scale, 'scale_exog':scale, 'scale_theta': scale}
        self.fit = self.sm.sampling(data=self.data, iter=iters, chains=chains, 
                                    warmup=warmup, thin=1, seed=seed, 
                                    control=dict(max_treedepth=12,adapt_delta = 0.85)
                                    )
        self.fit_df = self.fit.to_dataframe()
        self.theta = [np.nanmean(self.fit[f'theta[{x}]']) for x in range(1, q+1)]
        self.phi = [np.nanmean(self.fit[f'phi[{x}]']) for x in range(1, p+1)]
        self.phi_x = np.zeros(shape=(p, exog.shape[1]))
        for x in range(1,p):
            for y in range(1, exog.shape[1]):
                self.phi_x[x,y] = np.nanmean(self.fit[f"phi_x[{x},{y}]"])
        self.train_preds = [0 for x in range(max(p,q))]
        errors = [0 for x in range(max(p,q))]
        for i in range(max(p,q), exog.shape[0]):
            self.train_preds.append(np.sum([np.matmul(self.phi, endog.iloc[i-p:i]),
                                      np.matmul(self.theta, errors[i-q:i]),
                                      np.sum(np.sum(np.multiply(self.phi_x, exog.iloc[i-p:i, :])))]))
            errors.append(endog.iloc[i] - self.train_preds[i])
        self.pred_df = pd.DataFrame({'Date': dates, 'pred': self.train_preds, 'error': errors})
            
    def predict(self, endog, exog, dates):
        new_pred_df = pd.concat([dates, endog, exog], axis=1, ignore_index = False)
        start = self.pred_df.shape[0]
        self.pred_df = pd.merge(self.pred_df, new_pred_df, on = 'Date', how = 'outer')
        for i in range(start, self.pred_df.shape[0]):
            self.pred_df['pred'].iloc[i] = np.sum([np.matmul(self.phi, self.pred_df[endog.name].iloc[i-self.p:i]),
                                          np.matmul(self.theta, self.pred_df['error'].iloc[i-self.q:i]),
                                          np.sum(np.sum(np.multiply(self.phi_x, np.array(self.pred_df[exog.columns].iloc[i-self.p:i, :]))))])
            self.pred_df['error'].iloc[i] = self.pred_df[endog.name].iloc[i] - self.pred_df['pred'].iloc[i]
        self.pred_df = self.pred_df[['Date', 'pred', 'error']]
        return self.pred_df['pred'].iloc[start:]
            
    def forecast(self, endog: pd.Series, exog: pd.DataFrame, dates: pd.Series):
        '''
        Given trained model, forecast next index's value. 

        Parameters
        ----------
        endog : pd.Series
            Series being predicted.
        exog : pd.DataFrame
            Exogenous time series.
        dates : pd.Series
            Date column for the series.

        Returns
        -------
        float
            Prediction for next index.

        '''
        new_pred_df = pd.concat([dates, endog, exog], axis=1, ignore_index = False)
        new_pred_df = pd.merge(self.pred_df, new_pred_df, on = 'Date', how = 'outer')
        return np.sum([np.matmul(self.phi, new_pred_df[endog.name].iloc[new_pred_df.shape[0]-self.p:]),
                                          np.matmul(self.theta, new_pred_df['error'].iloc[new_pred_df.shape[0]-self.q:]),
                                          np.sum(np.sum(np.multiply(self.phi_x, np.array(new_pred_df[exog.columns].iloc[new_pred_df.shape[0]-self.p:, :]))))])
    
    def summary(self):
        '''
        Returns
        -------
        pd.DataFrame
            Summary dataframe from PyStan
        '''
        print(self.fit_df)
        return self.fit_df
    
    def plot(self):
        '''
        Plots histograms of all parameters from Stan sampling.
        '''
        self.fit_df.hist(bins=10, figsize=(25, 20))
            
    
