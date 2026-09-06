import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.api import VAR
from scipy.stats import chi2
import numpy as np
import warnings
from statsmodels.stats.diagnostic import acorr_breusch_godfrey

warnings.filterwarnings("ignore")


def optimal_lags(df, maxlags=15, alpha=0.05, output_path=None):
    print('\n Likelihood ratio test')
    #Fixed sample size
    T= len(df)-maxlags #effective sample
    n_vars= df.shape[1]
    dof_chi2= n_vars ** 2
    optimal_k=1
    results=[]

    #Covariance determinant
    def get_det(k):
        model_fixed= VAR(df.iloc[maxlags-k:])
        res=model_fixed.fit(k)
        return np.linalg.det(res.sigma_u_mle)

    prev_det=get_det(1)

    for k in range (2, maxlags+1):
        try:
            curr_det = get_det(k)
            lr_stat= T* (np.log(prev_det)-np.log(curr_det))
            p_val= chi2.sf(lr_stat, dof_chi2)
            print(f"\n Lag {k}: LR Stat = {lr_stat:.2f}, p_val = {p_val:.4f}")
            results.append({
                "Lag": k,
                "LR Statistic": lr_stat,
                "p-value": p_val,
                "Significant": p_val<alpha
            })

            if p_val < alpha:
                optimal_k=k
            prev_det=curr_det

        except ValueError:
            break
    results_df=pd.DataFrame(results)
    results_df.to_csv(output_path, index=False)
    print (f"\n Optimal lag length:{optimal_k}")
    print("Lag selection results saved.")
    return optimal_k, VAR(df)

def augmented_var(model, k, d_max):
    print("Estimation Toda-Yamamoto LA VAR")
    #Augmented lags
    total_lags= k+d_max
    var_results=model.fit(total_lags)
    return var_results

def model_diagnostics(var_results, k=None):
    print("\n--- Model Diagnostics ---")

    # ============================================================
    # 1. ADJUSTED R-SQUARED
    # ============================================================

    print("Equation Adjusted R-squared values:")

    # Residuals from the estimated VAR
    resid = np.array(var_results.resid)

    # Reconstruct Y:
    # Y = fitted values + residuals
    Y = np.array(var_results.fittedvalues) + resid

    # Number of lags actually estimated in the augmented VAR
    p_lags = var_results.k_ar

    # Original endogenous variables
    df_endog = pd.DataFrame(
        var_results.model.endog,
        columns=var_results.names
    )

    # Reconstruct the VAR's lagged explanatory variables
    X_df = pd.DataFrame()

    for lag in range(1, p_lags + 1):
        for col in df_endog.columns:
            X_df[f'{col}_L{lag}'] = df_endog[col].shift(lag)

    # Add constant and remove observations lost because of lags
    X_orig = sm.add_constant(X_df.dropna()).values

    # Number of observations and parameters
    T_obs, num_params = X_orig.shape

    # Calculate adjusted R-squared for each equation
    for i, eq in enumerate(var_results.names):

        ssr = np.sum(resid[:, i] ** 2)

        sst = np.sum(
            (Y[:, i] - np.mean(Y[:, i])) ** 2
        )

        adj_r2 = 1 - (
            (ssr / (T_obs - num_params))
            /
            (sst / (T_obs - 1))
        )

        print(f"  {eq}: {round(adj_r2, 3)}")


    # ============================================================
    # 2. MULTIVARIATE LM TEST FOR SERIAL CORRELATION
    # ============================================================

    print("\nMultivariate Lagrange Multiplier (LM) Test for Serial Correlation:")

    print(
        "Null Hypothesis: No serial correlation at specific lag h "
        "(p > 0.05 indicates good fit)"
    )

    # T = number of observations
    # K = number of equations
    T, K_vars = resid.shape

    # Residual covariance matrix from the estimated VAR
    sigma_u = np.dot(resid.T, resid) / T

    # Inverse of residual covariance matrix
    inv_sigma_u = np.linalg.inv(sigma_u)

    # Lags used in the original paper
    lags_to_test = [1, 4, 12]

    # Four equations -> K^2 = 16 degrees of freedom
    dof_chi2 = K_vars ** 2


    # Test each lag separately
    for h in lags_to_test:

        # --------------------------------------------------------
        # Keep only observations for which lag-h residuals exist
        # --------------------------------------------------------

        resid_current = resid[h:]
        resid_lagged = resid[:-h]
        X_current = X_orig[h:]

        # Combine:
        # Original VAR regressors
        # +
        # Lagged residuals
        X_aux = np.hstack([
            X_current,
            resid_lagged
        ])


        # --------------------------------------------------------
        # Run auxiliary regression for each equation
        # --------------------------------------------------------

        resid_aux = np.zeros_like(resid_current)

        for i in range(K_vars):

            model_aux = sm.OLS(
                resid_current[:, i],
                X_aux
            ).fit()

            resid_aux[:, i] = model_aux.resid


        # --------------------------------------------------------
        # Calculate covariance matrix of auxiliary residuals
        # --------------------------------------------------------

        T_h = len(resid_current)

        sigma_v = np.dot(
            resid_aux.T,
            resid_aux
        ) / T_h


        # --------------------------------------------------------
        # Calculate multivariate LM statistic
        # --------------------------------------------------------

        trace_val = np.trace(
            np.dot(
                sigma_v,
                inv_sigma_u
            )
        )

        lm_stat = T_h * (
            K_vars - trace_val
        )


        # --------------------------------------------------------
        # Calculate p-value
        # --------------------------------------------------------

        pval = chi2.sf(
            lm_stat,
            dof_chi2
        )


        # --------------------------------------------------------
        # Interpretation
        # --------------------------------------------------------

        if pval < 0.05:
            sig_flag = "FAIL (Serial Correlation Detected)"
        else:
            sig_flag = "PASS"


        print(
            f"  Lag {h:2d}: "
            f"LM Stat = {lm_stat:.2f} | "
            f"p-value = {pval:.4f} -> {sig_flag}"
        )