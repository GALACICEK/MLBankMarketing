# data_collection_and_cleaning.py

import pandas as pd
import numpy as np
from collections import namedtuple

class DataCollection:
        
    def __init__(self):
        
        return
        
    def data_readcsv(self,path):
        """
        Read a CSV file  into a DataFrame.

        Returns:
        pd.DataFrame: The DataFrame containing the read data.
        """

        df = pd.read_csv(path,delimiter=";")
        df.rename(columns={'y':'deposit'}, inplace=True)

        return df
    
    def unique_values_dtypes(self,df,data_type='object'):
        """
        Print unique values for each object-type column in the DataFrame.

        Parameters:
        - df (pd.DataFrame): The input DataFrame.
        - data_type (str): Type of columns to include ('int64', 'object', 'float64')

        Returns:
        result_dict: A dictionary containing the columns and their unique values.
        """
        if not isinstance(df,pd.DataFrame):
            raise ValueError("Input 'df' must be a Dataframe")
        
        result_dict = {}

        for col in df.select_dtypes(include=data_type).columns:
            unique_values = df[col].unique()
            result_dict[col] = unique_values

        return result_dict
    
    def check_duplication(self,df):
        """
        Check for duplicated rows in the DataFrame and drop them if any.

        Parameters:
        - df (pd.DataFrame): The input DataFrame.

        Returns:
        NamedTuple: A named tuple containing the message and DataFrame.

        Use Example:
            result = check_duplication(my_df)
            print(result.message)
            df = result.dataframe 
        """

        if not isinstance(df,pd.DataFrame):
            raise ValueError("Input 'df' must be a Dataframe")
    
        if df.duplicated().sum() > 0:
            dropped_df = df.drop_duplicates()
            message = f'Deleted Duplicates: {df.duplicated().sum()}'
        else :
            dropped_df = df.copy()
            message = 'No duplicates found.'

        Result = namedtuple('Result', ['message', 'dataframe'])
        return Result(message, dropped_df)
    


class DataCleaning:

    def __init__(self, df):
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input 'df' must be a DataFrame.")
        self.df = df.copy()

    def find_key_columns(self, key:str):

        """    Find columns in the DataFrame where the specified 'key' value is present,
        and calculate the frequency of each unique value in those columns.

        Parameters:
        - df (pd.DataFrame): The input DataFrame to analyze.
        - key (str): The value to search for in the DataFrame columns.

        Returns:
        result_dict: A dictionary containing the columns with 'key' values and their
            corresponding frequency DataFrames.
        """
        result_dict = {}

        for col in self.df.columns:

            key_rows = (self.df[col] == key)
            #  missing data is indicated as key.
            if key_rows.sum() > 0:    

                # finding frequency in the dataset
                df_freq = self.df.groupby(by=[col]).size()/len(self.df)
                result_dict[col] = df_freq.reset_index(name='Frequency')
                
        return result_dict
    
    def fill_missing_with_mode(self, cols, key:str):
        """
        Fill missing values in the specified column with the mode of that column.

        Parameters:
        - cols (str): The name of the columns in which missing values will be filled.
        - key (str): The value to search for in the DataFrame columns.
        
        Returns:
        pd.Series: The modified column with missing values filled.
        """
        self.df.replace(to_replace=key,value=np.NaN,inplace=True)

        for col in cols:
            self.df[col].fillna(self.df[col].mode()[0], inplace=True)

        
        return self.df
    
    def drop_missing_values(self, key:str):
        """
        Drop missing values in the specified column.

        Parameters:
        - key (str): The value to search for in the DataFrame columns.

        Returns:
        pd.Series: The modified datafreme with drop missing values.
        """
        self.df.replace(to_replace=key,value=np.NaN,inplace=True)

        self.df.dropna(inplace=True)


        return self.df
