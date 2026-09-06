import numpy as np

def transform_data(df, output_path):
    log_df=df.copy()
    for col in ['ECO', 'PSE', 'OIL', 'SP500']:
        log_df[f'L{col}']= np.log(log_df[col])

        #7 values of rate touches 0 and negative
        log_df['LRATE']=np.log(log_df['RATE'].clip(lower=0.0001))
        #Setting index to date
        log_df.index.name='Date'
        
        log_df.to_csv(output_path)
    return log_df