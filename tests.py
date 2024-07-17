import pandas as pd 
import numpy as np
import xgboost 
from xgboost import XGBRegressor


def check_dist_cost_calcs(dataframe):
    
    """
    Checks if the sum of distributions and costs and the cumulative value is the same 
    
    parameters:
        dataframe (pd.DataFrame) : Our working dataframe containing the distributions and costs 
        
    returns:
        None: 
    
    """
    
    samples = dataframe['Fund_ID'].sample(n=20)
    
    print('Costs\n')
    
    for sample in samples:
        final_costs = dataframe[dataframe['Fund_ID']==sample]['Cost_USD_cumulative'].iloc[-1]
        cumulative_costs = dataframe[dataframe['Fund_ID']==sample]['Costs'].sum()
        delta_costs = final_costs - cumulative_costs

        print(f"The delta between the cumulative cost of fund {sample} and the sum of costs {delta_costs}")

    print('\nDistributions\n')    

    for sample in samples:
        
        final_dist = dataframe[dataframe['Fund_ID']==sample]['Dist_USD_cumulative'].iloc[-1]
        cumulative_dist = dataframe[dataframe['Fund_ID']==sample]['Distributions'].sum()
        delta_dist = final_dist - cumulative_dist

        print(f"The delta between the cumulative distributions of fund {sample} and the sum of distributions is {delta_dist}")
    
    if delta_costs != 0 and delta_dists != 0:
        raise ValueError("There is an error in the costs or distributions. The delta should be zero. Please check the formula.")
    
    return None 


def check_uplift_calc_1(dataframe):
    
    """
    Check if the uplift calculation is working properly for the first five funds in the dataframe 
    
    parameters:
    dataframe (pd.DataFrame): Dataframe with the calculations computed to date 
    
    returns:
        None 
    
    """
    
    # Checking the first five Fund_ID's
    ID_to_check = dataframe['Fund_ID'].values[0:5]

    # Variables to track the cumulative uplift using the helper function and from the computation below 
    uplift_from_function = 0
    uplift_calc = 0 

    # Since this is the end of the DataFrame there should be no distributions/costs at time t+1
    for sample in ID_to_check:

        # Create a subset of the dataframe for the given Fund_ID
        filtered_df = dataframe[dataframe['Fund_ID']==sample]

        # Create the variables needed for the calculation 
        future_dists = filtered_df['Distributions'][::-1].cumsum()[::-1].shift(-1).fillna(0)
        future_costs = filtered_df['Costs'][::-1].cumsum()[::-1].shift(-1).fillna(0)
        final_FMV = filtered_df['FMV_USD_cumulative'].iloc[-1]
        residual_FMV = filtered_df['FMV_USD_cumulative']

        # Calculated uplift 
        x_1 = (future_dists + final_FMV)/(future_costs + residual_FMV)
        # Helper function uplift 
        x_2 = filtered_df['Uplift']

        # Add to the cumulative tracker variables 
        uplift_calc += x_1.sum()
        uplift_from_function += x_2.sum()
        difference_in_uplift = uplift_calc - uplift_from_function

        # Check the difference 
        print(f'The computed uplift is: {uplift_calc} vs. the check {uplift_from_function}, resulting in a delta of {difference_in_uplift}')
    
        if difference_in_uplift !=0:
            raise ValueError(f"There was an error. The delta was not zero for Fund_ID: {sample}. Please check the uplift calc. ")
    
    return None 


def check_uplift_calc_2(dataframe, indexer=[0, 2, 5], fund_id = 105324):
    
    """
    We want to check the uplift calculation at different points in time for a given fund(s)
    
    parameters:
        dataframe (pd.DataFrame): The dataframe with the data to date
        indexer (list): Define the quarters we are going to check
                       Default value of quarter 0, 2 and 5
        fund_id (int): integer of a selected fund_ID to check
        
    returns:
        None 
    
    """
    
    for time in indexer:

        # Create a variable for the slice
        sliced_dataframe = dataframe[dataframe['Fund_ID']==fund_id]

        # Current and final FMV
        current_fmv = sliced_dataframe.iloc[time]['FMV_USD_cumulative']
        final_fmv = sliced_dataframe['FMV_USD_cumulative'].iloc[-1]

        # Sum of future distributions and costs
        sum_of_future_dists = sliced_dataframe.iloc[time:]['Distributions'].shift(-1).fillna(0).sum()
        sum_of_future_costs = sliced_dataframe.iloc[time:]['Costs'].shift(-1).fillna(0).sum()

        # Uplift calculation
        uplift = (sum_of_future_dists+final_fmv)/(sum_of_future_costs+current_fmv)

        # Uplift from df 
        df_uplift = sliced_dataframe.iloc[time]['Uplift']

        delta_uplift = uplift - df_uplift

        print(f'At quarter {time} the difference vs. helper function value = {delta_uplift}')
        if delta_uplift != 0:
            raise ValueError("There is a delta in the uplift. Try checking the distributions and costs using the check_dist_cost_calcs function in the test file")
    
    return None


def check_fund_proportions(train_data):
    
    """
    Returns the proportion of funds by strategy type 
    
    parameters: 
        pd.DataFrame
    
    returns:
        list: Proportions of the total data of each fund strategy type 
    
    """
    
    unique_vals = train_data['Strategy_mapping'].unique()
    proportions = {x: len(train_data[train_data['Strategy_mapping'] == x])/len(train_data) for x in unique_vals}
    
    return proportions 

def compute_accuracy(actual, preds):

    """
    Computes the accuracy (proportion of predicted values > 1 / actual values > 1) for a given set of actuals and predictions
    
    parameters:
        actual (pd.Series): actual Uplift values 
        preds (pd.Series): predicted Uplift values 
    
    returns: 
        None: prints the accuracy
    
    """
    
    score = 0 
    
    actual = actual > 1
    preds = preds > 1

    for idx in actual.index:
        if actual.loc[idx] == preds.loc[idx]:
            score+=1
        else:
            pass 

    accuracy = score / len(actual) 

    print(f'Prediction accuracy: {accuracy:.1%}\n')
    
    return None 

def accuracy_pipeline(train, target, saved_model='XGB_model.json', strats = [1,2,3]):
    
    """
    Split the data by the fund strategy type and make model predictions
    To be used in preparation for the compute_accuracy function 
    
    parameters:
        train (pd.DataFrame): The training data 
        target (pd.Series): The target data 
        saved_model (json format): XGBRegressor model saved 
        strats (list): default value of the fund mappings  
        
    returns:
        model_predictions (list of tuples): Contains the actual test data and the model predictions

    """
    
    train_data = []
    test_data = []
    
    for fund in strats:
        
        x = train[train['Strategy_mapping']==fund]
        y = target.loc[x.index]
        
        # Append to train_data
        train_data.append(x)
    
        # Append to test_data
        test_data.append(y)
    
    all_data = zip(train_data, test_data)
    
    model_predictions = []
    
    pred_model = XGBRegressor()
    pred_model.load_model(saved_model)
    
    for batch in all_data:
        
        # Make the model prediction 
        prediction = pred_model.predict(batch[0])
        
        # Convert it to a series with the same index as the input 
        y_preds = pd.Series(prediction, index = batch[1].index)
        
        # Append to the model predictions
        model_predictions.append((batch[1], y_preds))

    return model_predictions

