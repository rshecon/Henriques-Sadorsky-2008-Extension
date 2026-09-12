import os
from statsmodels.tsa.api import VAR
from src.data import clean_data
from src.summary_stats import sum_stats
from src.multifactor_model import multifactor_model
from src.transform import transform_data
from src.tests import unit_root_test
from src.var_model import optimal_lags
from src.var_model import augmented_var, model_diagnostics
from src.granger import ty_causality
from src.plots import plot_impulse_responses
from src.robustness import lotech

def main():
    #Project Paths
    raw_path=os.path.join('data','raw','ECO_2004-2026.csv')
    cleaned_path=os.path.join('data', 'cleaned', 'wed_data.csv')
    log_path=os.path.join('data', 'cleaned', 'log_data.csv')
    summary_stats_path=os.path.join('results','summary_stats.csv')
    mf_path=os.path.join('results','mf.csv')
    unit_test_path=os.path.join('results','unit_root_tests.csv')
    lag_selection_path=os.path.join('results','lag_selection.csv')
    stat_path=os.path.join('results', 'stat_table.csv')
    pval_path=os.path.join('results', 'pval_table.csv')
    figure_path=os.path.join('results', 'figure')
    rob_log_path=os.path.join('results', 'rob_log.csv')
    rob_unit_test_path=os.path.join('results','rob_unit_root_tests.csv')
    rob_lags_path=os.path.join('results','rob_lag.csv')
    rob_fig_path=os.path.join('results', 'rob_figure')
    rob_stat_path=os.path.join('results', 'rob_stat_table.csv')
    rob_pval_path=os.path.join('results', 'rob_pval_table.csv')


    #Clean and combine data
    cleaned_df=clean_data(raw_path, cleaned_path)
    print('Cleaned Wednesday data saved.')

    #Summary statistics
    sum_stats(cleaned_df, summary_stats_path)
    multifactor_model(cleaned_df, mf_path)

    #Log transformation
    log_df= transform_data(cleaned_df, log_path)
    print('Log-transformed data saved.')


    #Unit root test
    test_results=unit_root_test(log_path, unit_test_path)
    print('Unit-root test result saved.')

    #Toda-Yamamoto integration oder
    d_max=1

    #VAR Estimation
    var_df=log_df[['LECO', 'LPSE', 'LOIL', 'LRATE']]
    k, var_model= optimal_lags (var_df, output_path=lag_selection_path)
    var_results= augmented_var(var_model, k=k, d_max=d_max)
    model_diagnostics(var_results, k=k)

    #Granger Causality
    stat_table, pval_table=ty_causality(var_df, k=k, d_max=d_max, stat_path=stat_path, pval_path=pval_path)

    #Impulse Response
    plot_impulse_responses(var_results, output_path=figure_path)

    #Robustness check
    df_robust= lotech(log_df.dropna(), output_path=rob_log_path)
    robust_var= df_robust[['LECO', 'LOTECH', 'LOIL', 'LRATE']]
    robust_test_results=unit_root_test(rob_log_path, rob_unit_test_path, variables=['LECO', 'LOTECH', 'LOIL', 'LRATE'])
    #reestimating
    robust_model=VAR(robust_var)
    k_rob, robust_lag_result= optimal_lags(robust_var, maxlags=12, output_path=rob_lags_path)
    robust_results=augmented_var(robust_model, k=k_rob, d_max=d_max)
    rob_stat_table, rob_pval_table= ty_causality(robust_var, k=k_rob, d_max=d_max, stat_path=rob_stat_path, pval_path=rob_pval_path)
    model_diagnostics(robust_results,k=k_rob)
    plot_impulse_responses(robust_results, steps=10, output_path=rob_fig_path)






if __name__ == "__main__":
    main()
