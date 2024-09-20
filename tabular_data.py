import pandas as pd
from ast import literal_eval

def clean_tabular_data(df):

    def remove_rows_with_missing_ratings(df):
        ratings_columns = df.columns[df.columns.str.contains('rating')]
        df = df.dropna(subset=ratings_columns)
        
        return df
    
    df = remove_rows_with_missing_ratings(df)

    def combine_description_strings(df):
        df = df.dropna(subset=['Description'])
        
        def clean_description(description):
            
            try:
                description = literal_eval(description)
                description = [item for item in description if item.strip()]
                
                if 'About this space' in description:
                    description.remove('About this space')
                    
                description = ' '.join(description)
                description = description.replace('\n', ' ')
                
                return description
            
            except (ValueError, SyntaxError):
                
                return description
        
        df['Description'] = df['Description'].apply(clean_description)
        
        return df
    
    df = combine_description_strings(df)

    def set_default_feature_values(df):
        columns = ['guests', 'beds', 'bathrooms', 'bedrooms']
        df[columns] = df[columns].fillna(value=1)
        
        return df
    
    df = set_default_feature_values(df)
    
    return df

def load_airbnb(df, label):
    string_columns = list(df.dtypes[df.dtypes == 'object'].index)
    number_df = df.drop(string_columns, axis=1)
    number_df = number_df.drop(['Unnamed: 19'], axis=1)
    labels = number_df[label]
    features = number_df.drop([label], axis=1)

    return (features, labels)

if __name__ == '__main__':
    df = pd.read_csv('listing.csv')
    df = clean_tabular_data(df)
    df.to_csv('cleaned_tabular_data.csv')
