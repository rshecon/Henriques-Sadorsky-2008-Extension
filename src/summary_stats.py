import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.stats import jarque_bera, kurtosis

def sum_stats(df, summary_stat_path):
    variables= ['ECO', 'PSE', 'SP500', 'OIL']
    returns= np.log(df[variables] / df[variables].shift(1)).dropna()
    print("\n Summary Statistics for Weekly Returns")
    stats= pd.DataFrame(index=variables)
    stats['Mean']=returns.mean() * 100
    stats['Median']=returns.median() * 100
    stats['Max']=returns.max() * 100
    stats['Min']=returns.min() * 100
    stats['Std Dev']=returns.std() * 100
    stats['Skewness']=returns.skew() 
    stats['Kurtosis']=returns.apply(lambda x: kurtosis(x, fisher=False))
    jb_results = {col: jarque_bera(returns[col]) for col in variables}
    stats['Jarque-Bera'] = [jb_results[col][0] for col in variables]
    stats['Probability'] = [ f"{jb_results[col][1]:.3g}" for col in variables]
    stats['Nobs']= returns.count()

    # Annualized Sharpe ratio
    annual_return = returns.mean() * 52
    annual_std = returns.std() * np.sqrt(52)
    annual_rf = df['RATE'].mean() / 100
    stats['Sharpe Ratio'] = (annual_return - annual_rf) / annual_std
    print (stats.round(2))
    stats.to_csv(summary_stat_path)
