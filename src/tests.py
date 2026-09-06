import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss
from arch.unitroot import PhillipsPerron

def unit_root_test(data_path, output_path, variables=None):
    print("\n Unit Root Test")
    df=pd.read_csv(data_path, index_col='Date', parse_dates=True)
    if variables is None:
        variables= ['LECO', 'LPSE', 'LOIL', 'LRATE']
    results=[]

    for var in variables:
        series=df[var].dropna()
        
        #test stationarity at level
        adf_lvl=adfuller(series,regression='c', autolag='BIC')
        pp_lvl=PhillipsPerron(series, trend='c')
        kpss_lvl=kpss(series, regression='c', nlags='auto')

        #result-
        # LECO: ADF- non-stat, PP- non-stat, KPSS- non-stat
        # LPSE: ADF- non-stat, PP- non-stat, KPSS- non-stat
        # LOIL: ADF- stat, PP- stat, KPSS- stat
        # LRATE: ADF- stat, PP- stat, KPSS- non-stat

        #first difference
        diff_series=series.diff().dropna()
        adf_diff=adfuller(diff_series,regression='c', autolag='BIC')
        pp_diff=PhillipsPerron(diff_series, trend='c')
        kpss_diff=kpss(diff_series, regression='c', nlags='auto')

        #result first difference-
        # LECO: ADF- stat, PP- stat, KPSS- stat
        # LPSE: ADF- stat, PP- stat, KPSS- stat
        # LOIL: ADF- stat, PP- stat, KPSS- stat - at both level and first diff
        # LRATE: ADF- stat, PP- stat, KPSS- stat 


        results.append({
            'Variable': var,
            'ADF Level': round(adf_lvl[0],3),
            'ADF p-val (level)': round(adf_lvl[1],3),
            'PP Level': round(pp_lvl.stat, 3),
            'PP p-val (level)': round(pp_lvl.pvalue, 3),
            'KPSS Level': round(kpss_lvl[0],3),
            'KPSS p-val (level)': round(kpss_lvl[1],3),
            'ADF Diff': round(adf_diff[0],3),
            'ADF p-val (diff)': round(adf_diff[1],3),
            'PP Diff': round(pp_diff.stat, 3),
            'PP p-val (diff)': round(pp_diff.pvalue, 3),
            'KPSS Diff': round(kpss_diff[0],3),
            'KPSS p-val (diff)': round(kpss_diff[1],3)
        })

    results_df=pd.DataFrame(results).set_index('Variable')
    print("\n Unit Root test:")
    print(results_df.to_string())
    results_df.to_csv(output_path)

    return results_df
