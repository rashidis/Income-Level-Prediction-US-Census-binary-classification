import pandas as pd
from sklearn.preprocessing import LabelEncoder


def drop_num_outliers(df_num, num_features,  multiplier):
    """Drop outliers from a DataFrame based on the interquartile range (IQR) method.

    :param df_num: (pd.DataFrame) Input DataFrame containing numerical data.
    :param num_features: list of strings as column values
    :param multiplier: (float) Multiplier for determining the outlier threshold.
    :return: DataFrame with outliers removed.
    """
    for column in num_features:
        Q1 = df_num[column].quantile(0.25)
        Q3 = df_num[column].quantile(0.75)
        IQR = Q3 - Q1
        
        outliers = (df_num[column] < Q1 - multiplier * IQR) | (df_num[column] > Q3 + multiplier * IQR)
    
        print(f"number of Outliers of column {column} is {df_num[outliers].shape[0]}")
        df_num_no_outlier = df_num[~outliers]
    return df_num_no_outlier

def drop_cat_outliers(df, categorical_features, percentage=200):
    """For categorical data, we simply check the frequency of each category. 
    
    Categories with significantly lower frequencies than others may be considered outliers.
    This is set through percentage value

    :param df_num: (pd.DataFrame) Input DataFrame containing numerical data.
    :param categorical_features: list of strings as column values
    :param percentage: (int) percentage for determining the outlier threshold.
    :return: DataFrame with outliers removed.
    """
    for column in categorical_features:
        category_counts = df[column].value_counts()
    
        threshold = category_counts.max()//percentage
        outliers = (category_counts < threshold)
        outliers = outliers[outliers==True].index.values
        print(f"number of Outliers of column {column} is {outliers[outliers==True].sum()}")
        df = df[df[column].isin(outliers) == False]
    return df

def preprocessing(df, numerical_features, cols2drop, unknown_values):
    """ Preprocessing steps on dataset
    
    DROP unknown values
    DROP Duplicates
    Separate numerical and categorical features
    Drop numerical and categorical outliers
    
    :param df:(pd.DataFrame) Input DataFrame containing numerical data.
    :param numerical_features: list of strings as numerical features
    :param cols2drop: list of strings as columns to drop
    :param unknown_values: list of strings as unknown values
    :return: preprocessed dataset
    """
    df = df.map(lambda x: str(x).lower().strip())  # to match all entries
    
    # Drop unknown values
    df_clean = df.drop(columns=cols2drop) 
    df_clean = df_clean[~df_clean.isin(unknown_values).any(axis=1)]
    print(f"shape of dataframe after dropping unknown values {df_clean.shape}")
    
    # Drop Duplicates
    df_clean = df_clean.drop_duplicates()
    df_clean.reset_index(drop=True, inplace=True)
    print(f"shape of dataframe after dropping the duplicated rows is : {df_clean.shape}")
    
    # Separate numerical and categorical features
    categorical_features = list(set(df_clean.columns.tolist()) - set(numerical_features) - set(['income']))
    df_num = df_clean[numerical_features]
    df_num = df_num.apply(pd.to_numeric, errors='coerce')
    df_cat = df_clean[categorical_features]
    label = df_clean['income']
    df_clean = pd.concat([df_num, df_cat, label], axis=1 )
    print(f'df_clean shape is {df_clean.shape}')
    
    # Drop numerical and categorical outliers
    df_clean = drop_num_outliers(df_clean, numerical_features, multiplier=3)
    print(f"shape of dataframe after dropping numerical outliers {df_clean.shape}")
    df_clean = drop_cat_outliers(df_clean, categorical_features, percentage=200)
    print(f"shape of dataframe after dropping categorical outliers {df_clean.shape}")
    
    # reset index
    df_clean.reset_index(drop=True, inplace=True)
    
    return df_clean

def encoding(df_clean, categorical_features):
    """ Encodes the dataset given its categorical features as a list of strings

    :param df_clean: dataframe
    :param categorical_features: a list of categorical features as strings
    :return: the dataframe with categorical columns
    """
    label_encoder = LabelEncoder()
    for col_name in categorical_features:
        df_clean[col_name] = label_encoder.fit_transform(df_clean[col_name])
        
        # Print the encoding information for each categorical feature
        print(f"Encoding information for {col_name}:")
        print(f"Original classes: {label_encoder.classes_}")
        print(f"Encoded values: {list(range(len(label_encoder.classes_)))}")
        print()
        
    # separately encode label 
    df_clean['income'] = df_clean['income'].apply(lambda x: 0 if x == "-50000" else 1)
    df_clean['income'].value_counts()
    return df_clean