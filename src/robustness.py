import statsmodels.api as sm

def lotech(df, output_path):
    print("Robustness check")

    #regressing lpse and lsp500
    y=df['LPSE']
    x=sm.add_constant(df['LSP500'])
    model= sm.OLS(y,x).fit()

    #residuals
    df = df.copy()
    df['LOTECH']=model.resid
    print("created orthogonalized tech stock variable")
    df.to_csv(output_path)
    return df