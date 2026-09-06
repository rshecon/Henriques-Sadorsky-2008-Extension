import os
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VAR

def compute_girf(var_results, max_horizon):
    """Calculates Pesaran-Shin (1998) GIRF point estimates for a 1 S.D. shock."""
    ma_coefs = var_results.ma_rep(max_horizon)  
    
    sigma_u = var_results.sigma_u
    sigma_u = sigma_u.values if hasattr(sigma_u, 'values') else sigma_u
    sd = np.sqrt(np.diag(sigma_u))
    K = sigma_u.shape[0]

    girf = np.zeros((max_horizon + 1, K, K))
    for n in range(max_horizon + 1):
        for j in range(K):
            # Pesaran-Shin Formula: (A_n * Sigma * e_j) / sqrt(sigma_jj)
            girf[n, :, j] = (ma_coefs[n] @ sigma_u[:, j]) / sd[j]
            
    return girf

def plot_impulse_responses(var_results, steps=10, n_sims=1000, output_path=None):
    np.random.seed(42)
    print(f"\n--- Phase 5: Generating Recursive Bootstrap GIRFs ({steps} weeks) ---")
    K = var_results.neqs
    names = var_results.names
    p = var_results.k_ar
    
    # 10 discrete points map to horizons n=0 through n=9
    target_horizons = steps - 1 
    girf_point = compute_girf(var_results, max_horizon=target_horizons)

    # Recursive VAR Bootstrap
    print(f"Executing recursive VAR bootstrap with {n_sims} simulations...")
    
    Y_orig = var_results.model.endog
    Y_orig = Y_orig.values if hasattr(Y_orig, 'values') else Y_orig
    resid = var_results.resid
    resid = resid.values if hasattr(resid, 'values') else resid
    
    coefs = var_results.coefs
    intercept = var_results.intercept
    T = len(Y_orig)
    
    girf_sims = np.zeros((n_sims, target_horizons + 1, K, K))

    for sim in range(n_sims):
        # 1. Resample residuals
        idx = np.random.randint(0, len(resid), len(resid))
        u_star = resid[idx]
        
        # 2. Recursively generate simulated series Y*
        Y_star = np.zeros((T, K))
        Y_star[:p] = Y_orig[:p]  # Seed with actual history
        
        for t in range(p, T):
            y_hat = intercept.copy()
            for i in range(p):
                y_hat += coefs[i] @ Y_star[t - 1 - i]
            Y_star[t] = y_hat + u_star[t - p]
            
        # 3. Refit VAR on simulated path
        sim_model = VAR(Y_star).fit(p)
        girf_sims[sim] = compute_girf(sim_model, max_horizon=target_horizons)

    # Extract standard errors
    se = np.std(girf_sims, axis=0, ddof=1)
    upper_band = girf_point + 2 * se
    lower_band = girf_point - 2 * se

    # Plotting
    fig, axes = plt.subplots(K, K, figsize=(16, 12))
    fig.suptitle('Response to Generalized One S.D. Innovations $\pm$ 2 S.E.', fontsize=16)

    # Align x-axis to exactly 1 through 10
    x_ticks = np.arange(1, steps + 1)

    for i in range(K):
        for j in range(K):
            ax = axes[i, j]
            ax.plot(x_ticks, girf_point[:, i, j], color='blue', linewidth=1.5, label='GIRF')
            ax.plot(x_ticks, upper_band[:, i, j], color='red', linestyle='--', linewidth=1, label='+ 2 S.E.')
            ax.plot(x_ticks, lower_band[:, i, j], color='red', linestyle='--', linewidth=1, label='- 2 S.E.')
            ax.axhline(0, color='black', linewidth=1)
            
            ax.set_title(f"Response of {names[i]} to {names[j]}", fontsize=10)
            ax.set_xticks(x_ticks)
            
            if i == 0 and j == K - 1:
                ax.legend(loc='upper right', fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=300)
    print(f"Saved Impulse Response plot to {output_path}")