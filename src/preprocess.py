import pandas as pd

def engineer_longitudinal_features(df):
    df = df.sort_values(['Subject ID', 'Visit'])
    
    # CDR change between visits (progression rate)
    df['CDR_change'] = df.groupby('Subject ID')['CDR'].diff()
    
    # MMSE decline rate
    df['MMSE_change'] = df.groupby('Subject ID')['MMSE'].diff()
    
    # Brain volume loss rate
    df['nWBV_change'] = df.groupby('Subject ID')['nWBV'].diff()
    
    # Number of visits (patient history length)
    df['visit_count'] = df.groupby('Subject ID')['Visit'].transform('count')
    
    # Baseline CDR (first visit)
    df['baseline_CDR'] = df.groupby('Subject ID')['CDR'].transform('first')
    
    return df




