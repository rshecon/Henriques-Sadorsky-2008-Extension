import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.stattools import durbin_watson


def multifactor_model(df, output_path):

    variables = ['ECO', 'PSE', 'SP500', 'OIL']

    # Weekly continuously compounded returns
    returns = np.log(df[variables] / df[variables].shift(1)).dropna()

    # Weekly change in the 3-month Treasury bill rate
    rate_change = df['RATE'].diff()

    # Align rate changes with return observations
    data = returns.copy()
    data['DRATE'] = rate_change
    data = data.dropna()

    models = {'ECO Model 1': (data['ECO'], data[['SP500']]),
        'ECO Model 2': (data['ECO'], data[['SP500', 'OIL', 'DRATE']]),
        'PSE Model 1': (data['PSE'], data[['SP500']]),
        'PSE Model 2': (data['PSE'], data[['SP500', 'OIL', 'DRATE']])}

    results = {}

    for name, (y, X) in models.items():
        X = sm.add_constant(X)
        model = sm.OLS(y, X).fit(
            cov_type='HAC',
            cov_kwds={'maxlags': 1}
        )
        results[name] = model
        print(f"\n{name}")
        print(model.summary().tables[1])
        print(
            f"Adjusted R-squared: {model.rsquared_adj:.3f} | "
            f"F-test p-value: {model.f_pvalue:.4f} | "
            f"Durbin-Watson: {sm.stats.stattools.durbin_watson(model.resid):.3f}"
        )
        columns = [
        'ECO Model 1',
        'ECO Model 2',
        'PSE Model 1',
        'PSE Model 2'
    ]

    rows = [
        'Constant',
        'Market',
        'Oil',
        'Rate',
        'Adjusted R-squared',
        'Durbin-Watson',
        'F-test p-value'
    ]

    table = pd.DataFrame(
        index=rows,
        columns=columns,
        dtype=float
    )

    for name, model in results.items():

        table.loc['Constant', name] = model.params.get(
            'const', np.nan
        )

        table.loc['Market', name] = model.params.get(
            'SP500', np.nan
        )

        table.loc['Oil', name] = model.params.get(
            'OIL', np.nan
        )

        table.loc['Rate', name] = model.params.get(
            'DRATE', np.nan
        )

        table.loc['Adjusted R-squared', name] = (
            model.rsquared_adj
        )

        table.loc['Durbin-Watson', name] = (
            durbin_watson(model.resid)
        )

        table.loc['F-test p-value', name] = (
            model.f_pvalue
        )

    print("\nMultifactor Risk Comparison")
    print(table.round(4))

    # Save CSV
    if output_path is not None:
        table.to_csv(output_path)
        print(f"\nTable 2 saved to: {output_path}")