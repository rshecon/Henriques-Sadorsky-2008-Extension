import pandas as pd
import statsmodels.api as sm


def ty_causality(df, k, d_max=1, stat_path=None, pval_path=None, variables=None):

    print("\nToda-Yamamoto LA-VAR")
    total_lags = k + d_max
    lagged_df = df.copy()

    for col in df.columns:
        for lag in range(1, total_lags + 1):
            lagged_df[f'{col}_L{lag}'] = df[col].shift(lag)

    # Remove observations lost because of lagging
    lagged_df = lagged_df.dropna()
    lagged_df = sm.add_constant(lagged_df)
    if variables is None:
        variables = list(df.columns)
    result = []

    for dep_var in variables:

        indep_cols = (
            ['const'] +
            [f'{c}_L{lag}' for c in variables for lag in range(1, total_lags + 1)]
        )

        y = lagged_df[dep_var]
        x = lagged_df[indep_cols]
        model = sm.OLS(y, x).fit()
        for causing_var in variables:

            # Don't test a variable against itself
            if dep_var == causing_var:
                continue

    
            # Restrictions on first k lags only
            restrictions = [
                f"{causing_var}_L{lag} = 0"
                for lag in range(1, k + 1)
            ]

            hypothesis = ",".join(restrictions)

            
            # Toda-Yamamoto modified Wald test
            wald_res = model.wald_test(
                hypothesis,
                scalar=True,
                use_f=False
            )

            chi_square = float(wald_res.statistic)
            p_value = float(wald_res.pvalue)

    
            #Significance stars
    
            if p_value < 0.01:
                stars = "***"
            elif p_value < 0.05:
                stars = "**"
            elif p_value < 0.10:
                stars = "*"
            else:
                stars = ""

    
            # Store raw numerical results
            result.append({
                'Dependent': dep_var,
                'Independent': causing_var,
                'Chi-Square': chi_square,
                'p-value': p_value,
                'Stars': stars
            })

    # Convert results to DataFrame
    results_df = pd.DataFrame(result)

    # Create Table 6 chi-square matrix
    stat_pivot = results_df.pivot(
        index='Dependent',
        columns='Independent',
        values='Chi-Square'
    )

    # Create p-value matrix
    pval_pivot = results_df.pivot(
        index='Dependent',
        columns='Independent',
        values='p-value'
    )

    # Create presentation matrix
    results_df['Chi-Square Table'] = (
        results_df['Chi-Square'].map(lambda x: f"{x:.3f}")
        + results_df['Stars']
    )

    table_pivot = results_df.pivot(
        index='Dependent',
        columns='Independent',
        values='Chi-Square Table'
    )

    #Print results
    print("\nToda-Yamamoto LA-VAR Wald Tests")
    print(table_pivot.fillna('-'))

    print("\nWald Test P-values:")
    print(pval_pivot.round(4).fillna('-'))

    #Save numerical results
    if stat_path is not None:
        stat_pivot.to_csv(stat_path)

    if pval_path is not None:
        pval_pivot.to_csv(pval_path)

    #Return results
    return stat_pivot, pval_pivot