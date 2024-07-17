import pandas as pd 
import numpy as np
import datetime 

fund_type_mapping = {'VC': 1,
                    'MediumBO': 2,
                    'MegaBO': 3}

def create_date_features(df):
    
    """
    Computes a group of features based around the datetime object
    There should be some signal around dates given the dotcom bubble, GFC etc 

    Parameters:
        dataframe (pd.DataFrame): A pd.DataFrame 

    Returns:
        df (pd.DataFrame): A dataframe containing the new date features 

    """    
    
    df = df.copy()
    
    df['Year'] = df['Date_Quarter'].dt.year
    df['Quarter'] = df['Date_Quarter'].dt.quarter
    df['Month'] = df['Date_Quarter'].dt.month
    df['Day'] = df['Date_Quarter'].dt.day
    df['DayOfWeek'] = df['Date_Quarter'].dt.dayofweek
    df['DayOfYear'] = df['Date_Quarter'].dt.dayofyear
    df['WeekOfYear'] = df['Date_Quarter'].dt.isocalendar().week
    
    return df               

def compute_cost_distributions(group):
    
    """
    Computes the distribution and costs by quarter for each fund. 

    Parameters:
        group (pd.groupby object): A pd.groupby object which has been grouped by the Fund_ID

    Returns:
        df_1 (pd.DataFrame): A dataframe containing the distribution and costs of each of the funds

    """    
    
    # Computing the difference between the cumulative values and filling the NaN with the original value
    distributions = group['Dist_USD_cumulative'].diff().fillna(group['Dist_USD_cumulative'])
    
    # Computing the difference between the cumulative values and filling the NaN with the original value
    costs = group['Cost_USD_cumulative'].diff().fillna(group['Cost_USD_cumulative'])

    df_1 = pd.DataFrame({'Distributions': distributions, 
                        'Costs': costs})
    
    return df_1

def compute_uplift(group):
    
    """
    Computes the Uplift target function 

    Parameters:
        group (pd.groupby object): A pd.groupby object which has been grouped by the Fund_ID

    Returns:
        uplift: The uplift feature computed for each fund and for each point in the time series
   
    """
    
    # Define the future distribution and costs required for Uplift function
    future_distributions = group['Distributions'][::-1].cumsum()[::-1].shift(-1).fillna(0)
    future_costs = group['Costs'][::-1].cumsum()[::-1].shift(-1).fillna(0)
    
    # Compute the uplift function 
    uplift = (future_distributions + group['FMV_USD_cumulative'].iloc[-1]) / (future_costs + group['FMV_USD_cumulative'])
    
    return uplift 

def strategy_mapping(dataframe, mapping=fund_type_mapping):
    
    """
    Creates a new feature encoding for the fund strategy 

    Parameters:
        dataframe (pd.DataFrame): The pd.DataFrame which requires the 'Strategy' column 
        mapping (dict): dictionary to map the values in the strategy column to integer encodings 
                        (default is the given fund_type_mapping)

    Returns:
        dataframe: A dataframe containing a new column 'Fund_strategy_mapping'

    """    
    
    dataframe['Strategy_mapping'] = dataframe['Strategy'].map(mapping)
    
    return dataframe

def cost_commitment_ratio(dataframe):
    
    """
    Creates two new features:
        The cost / the fund commitment at time t 
        The cumulative cost / the fund commitment at time t 

    Parameters:
        dataframe (pd.DataFrame): The main pd.DataFrame 

    Returns:
        dataframe: A dataframe containing the new columns Cost_commitment_ratio and Cost_commitment_ratio_cumulative
   
    """  
    
    dataframe['Cost_commitment_ratio'] = dataframe['Costs'] / dataframe['Commitment_USD_cumulative']
    
    dataframe['Cost_commitment_ratio_cumulative'] = dataframe['Cost_USD_cumulative'] / dataframe['Commitment_USD_cumulative']
    
    return dataframe 

def fund_size_standardizer(group):
    
    """
    Computes a new feature which has the standardized 'Commitment_USD_cumulative' for each Strategy

    Parameters:
        group (pd.groupby object): A pd.groupby object which has been grouped by the Strategy_mapping

    Returns:
        dataframe (pd.DataFrame): A dataframe containing the new feature 
        
    Notes:
        This should be run with the 'apply' function so that the standardization is done to each group

    """      
    
    fund_commitment = group['Commitment_USD_cumulative']
    
    expanding_mean = fund_commitment.expanding().mean()
    expanding_std = fund_commitment.expanding().std()
    
    group['Fund_commitment_z_score_Expanding'] = (fund_commitment - expanding_mean)/expanding_std
    
    # Fill with 0 the NaNs created at the beginning 
    group['Fund_commitment_z_score_Expanding'] = group['Fund_commitment_z_score_Expanding'].fillna(0)
    
    return group

def fund_vintage_normalizer(group):
    
    """
    Computes a new feature which is the normalized fund vintage grouped by the strategy 

    Parameters:
        group (pd.groupby object): A pd.groupby object which has been grouped by the Strategy_mapping

    Returns:
        dataframe (pd.DataFrame): A dataframe containing the new feature 
        
    Notes:
        This should be run with the 'apply' function so that the standardization is done to each group as explained in notebook

    """   
    
    fund_vintage = group['Fund_Vintage']
    
    expanding_min = fund_vintage.expanding().min()
    expanding_max = fund_vintage.expanding().max()
    
    group['Fund_Vintage_MinMax_Expanding'] = (fund_vintage - expanding_min)/(expanding_max - expanding_min)
    
    # Fill with 0 the NaNs created at the beginning 
    group['Fund_Vintage_MinMax_Expanding'] = group['Fund_Vintage_MinMax_Expanding'].fillna(0)
    
    return group 

def compute_momentum_factors(group, horizon = [2, 4, 8, 10, 12]):
    
    """
    Computes some momentum factors over the distributions, costs, FMV, TCPI, DPI, RVPI  for the fund 

    Parameters:
        group (pd.groupby object): A pd.groupby object which has been grouped by the Fund_ID

    Returns:
        dataframe (pd.DataFrame): A dataframe containing the new features 
        
    Notes: 
        There may will be some instances where we get np.inf or -np.inf values when running the pct_change 
        We want to bfill these and have done so on the notebook

    """       
    
    # Compute the percentage change over the time horizon. NaN values filled with 1 
    for x in horizon:
        group[f'Distribution_growth_{x}'] = group['Distributions'].pct_change(x).fillna(1)
        group[f'Cost_growth_{x}'] = group['Costs'].pct_change(x).fillna(1)
        group[f'FMV_cumulative_growth_{x}'] = group['FMV_USD_cumulative'].pct_change(x).fillna(1)
        group[f'TVPI_{x}'] = group['TVPI'].pct_change(x).fillna(1)
        group[f'DPI_{x}'] = group['DPI'].pct_change(x).fillna(1)
        group[f'RVPI_{x}'] = group['RVPI'].pct_change(x).fillna(1)
    
    return group

    
