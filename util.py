import pandas as pd
from sklearn.model_selection import train_test_split

def convert_dataframe_to_vectors(df, target_variable='', target_variables=[]):
    X, y = vectorize_dataframe(df, target_variable, target_variables)
    return X, y

def preprocess_dataframe(df, columns_to_drop=[]):
    df = fill_missing_values(df)
    df = drop_columns(df, columns_to_drop)
    df = one_hot_encode_dataframe(df)
    return df
    

def count_rows_with_nan_values(df):
    missing_values_count = df.isnull().any(axis=1).value_counts()
    try:
        return missing_values_count.get_value(True)
    except:
        no_of_missing_values = 0
        return no_of_missing_values 
    
def count_number_of_features(df, target_variables=[]):
    return len(df.columns) - len(target_variables)
    
def display_rows_with_nan_values(df):
    return df[df.isnull().any(axis=1)]

def fill_missing_values(df, replacement_value='UNKNOWN'):
    for col in df:
        dtype = df[col].dtype 
        if dtype == int or dtype == float:
            df[col] = df[col].fillna(0)
        else:
            df[col] = df[col].fillna(replacement_value)
        
    assert count_rows_with_nan_values(df) == 0
    return df

def drop_columns(df, columns):
    existing_columns = df.columns
    for col in columns:
        if col in existing_columns:
            df = df.drop(col, axis=1)
        else:
            print('Skipping "{}". Column not found in dataframe'.format(col))
    return df

def one_hot_encode_dataframe(df, columns_to_skip_encoding=[], numerical_columns_to_encode=[]):
    columns_to_one_hot_encode = []
    columns = [col for col in df.columns if col not in columns_to_skip_encoding]
    for col in columns:
        dtype = df[col].dtype 
        
        if dtype == object: # find columns with string data
            columns_to_one_hot_encode.append(col)
            
    columns_to_one_hot_encode = columns_to_one_hot_encode + numerical_columns_to_encode
    
    df = pd.get_dummies(df, columns=columns_to_one_hot_encode)
    return df

def vectorize_dataframe(df, target_variable='', target_variables=[]):
    X_variables = [col for col in df.columns if col not in target_variables]
    
    X = df[X_variables]
    y = df[target_variable]
    
    assert X.shape == (len(df), len(X_variables))
    assert y.shape == (len(df), )
    
    return X, y

def plot_feature_importances(model, feature_names):
    feature_importances = model.feature_importances_
    
    sorted_feature_importances = sorted(zip(feature_importances, feature_names), reverse=True)
    
    plt.plot([x[0] for x in sorted_feature_importances], 'o')
    plt.xticks(range(len(df.columns)), [x[1] for x in sorted_feature_importances], rotation=90);
