import pandas as pd
import json

def load_data(filepath, label):
    """Loads datasets used in experiments.
    
    Args: 
        filepath: Path to file(s) for dataset. Yelp uses `yelp_academic_dataset_tip.json`, 
            Foursquare uses `dataset_TSMC2014_NYC.txt` and `dataset_TSMC2014_TKY.txt`
            (which should be passed as a list) and Gowalla uses `loc-gowalla_totalCheckins.txt`.
        label: Dataset name. Either `yelp`, `foursquare` or `gowalla`.
    
    Returns:
        The corresponding dataset on a pandas DataFrame containing two columns `element` and `user`.
    """
    
    if label == 'yelp':
        with open(filepath, "r") as f:
            data = [json.loads(row) for row in f]

        # convert the list of dictionaries to a DataFrame
        yelp = pd.DataFrame(data)

        yelp = yelp[['business_id', 'user_id']]
        
        yelp.columns = ['element','user']
        
        return yelp.copy()
    
    elif label == 'foursquare':
        # Foursquare gets argument `filepath` as list.
        
        df1 = pd.read_csv(filepath[0], sep='\t', header=None,encoding = "ISO-8859-1")
        df2 = pd.read_csv(filepath[1], sep='\t', header=None,encoding = "ISO-8859-1")

        df1.columns = ['userid','venid','vencatid','venname','lat','long','tz','time']
        df1 = df1.drop(['vencatid','venname','lat','long','tz','time'], axis=1)
        df1.columns = ['user','element']

        df2.columns = ['userid','venid','vencatid','venname','lat','long','tz','time']
        df2 = df2.drop(['vencatid','venname','lat','long','tz','time'], axis=1)
        df2.columns = ['user','element']
        
        foursquare = pd.concat([df1,df2])
        
        return foursquare.copy()
    
    elif label == 'gowalla':
        
        gowalla = pd.read_csv(filepath, sep='\t', header=None)
        gowalla.columns = ['userid','timestamp','latitude','longitude','spotid']
        gowalla = gowalla.drop(['timestamp','latitude','longitude'], axis=1)
        gowalla.columns = ['user', 'element']
        
        return gowalla.copy()
    
    
def pre_process_data(df):
    """Processes the dataset to get a score vector that allows a user to contribute a 
        value of 1 for each element, irrespective of how many times the user appears 
        with the element in the dataset.
    
    Args:
        df: Dataset as a Pandas DataFrame containing two columns: `element` and `user`.
        
    Returns:
        Number of distinct users
        Number of distinct elements
        Processed score vector
    """
    
    # Creates column for score that will not count an element more than once
    df['val'] = 1
    
    # Remove duplicates, so that one user just gives a single 1 (score) per element
    df.drop_duplicates(inplace=True) 

    # Count number of users
    nr_users = df['user'].nunique()

    # Aggregate
    df = df.groupby(['element'], as_index=False)['val'].sum()

    # Isolate score vector
    score = df['val'].values
    score = score.astype('int32')
    score = score[score.argsort()[::-1]]
    
    return nr_users, len(score), score