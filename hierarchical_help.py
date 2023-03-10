from sklearn import tree
import pandas as pd 
from tqdm import tqdm

def all_pairs(countries: List[str]) -> List[Tuple[str, str]]:
    """Return all pairs of countries.
    
    This function takes a list of countries as input and returns a list of all
    possible pairs of countries. The order of the countries in each pair does
    not matter (i.e. (A, B) is considered the same as (B, A)).

    Args:
        countries (List[str]): A list of country names.

    Returns:
        List[Tuple[str, str]]: A list of tuples, where each tuple contains the
        names of two countries.
    """
    return [(countries[p1], countries[p2]) for p1 in range(len(countries)) for p2 in range(p1+1,len(countries))]

def create_country_df_dict(countries: List[str], df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Create a dictionary of dataframes for each country.
    
    This function takes a list of countries and a dataframe as inputs and returns
    a dictionary mapping the name of each country to a dataframe containing only
    the rows corresponding to that country.

    Args:
        countries (List[str]): A list of country names. (The names are encoded as numbers.)
        df (pd.DataFrame): The original dataframe.

    Returns:
        Dict[str, pd.DataFrame]: A dictionary mapping the names of the countries
        to their corresponding dataframes.
    """
    country_df_dict = dict()
    for country in tqdm(countries):
        df_country = pd.DataFrame() 
        df_country = df[df['WP5: Country']==country].drop('WP5: Country', axis=1)
        country_df_dict[str(country)+'-'] = df_country
    return country_df_dict


def create_country_tree_dict(countries, df_original, country_df_dict, depth):
    """
    create_country_tree_dict(countries, df_original, country_df_dict, depth)
    
    This function creates a dictionary containing the decision trees for a given list of countries.
    
    Parameters:
    - countries: A list of countries for which to create decision trees.
    - df_original: The original dataframe containing all data.
    - country_df_dict: A dictionary mapping country names to dataframes containing only data for that country.
    - depth: The maximum depth of the decision tree.
    
    Returns:
    - A dictionary mapping country names to decision tree models.
    """
    country_tree_dict = dict()
    for country in tqdm(countries):
        country = str(country)
        clf = calc_tree(country_df_dict[country+'-'], df_original, [country], depth)
        country_tree_dict[str(country)+'-'] = clf
    return country_tree_dict


def init_clusters(countries):
    """
    This function initializes a dictionary of clusters, where each cluster contains only one country.
    
    Parameters:
    - countries: A list of countries to initialize clusters for.
    
    Returns:
    - A dictionary mapping cluster names to lists of countries in the cluster.
    """
    clusters = dict()       
    for country in countries: 
        clusters[str(country)+'-'] = [country]
    return clusters


def init_distances(clusters: Dict[str, List[str]], country_df_dict: Dict[str, pd.DataFrame], country_tree_dict: Dict[str, Any], df_original: pd.DataFrame) -> Tuple[Dict[str, float], Dict[str, List[List[int]]]]:
    """
    Calculates the distances between the given clusters.
    
    Parameters:
    - clusters: A dictionary mapping cluster names to lists of countries in the cluster.
    - country_df_dict: A dictionary mapping country names to dataframes containing only data for that country.
    - country_tree_dict: A dictionary mapping country names to decision tree models.
    - df_original: The original dataframe containing all data.
    
    Returns:
    - A tuple containing two dictionaries:
        1. A dictionary mapping cluster pairs to distances.
        2. A dictionary mapping cluster pairs to the corresponding lists of countries in the pair.
    """

    # init distances  
    dist_dict = dict()      # country set where there are two clusters and the distance for this 
    name_to_list = dict()   # the name of the above maping to the two clusers (list of two lists)

    for pair in tqdm(all_pairs(list(clusters.keys()))): 
        country1 = pair[0]
        country1_name = country1[:-1]
        country2 = pair[1]
        country2_name = country2[:-1]
        df1 = country_df_dict[country1]
        
        acc = calc_dist(df1,country_df_dict[country2], [country1_name], [country2_name], country_tree_dict[country1], country_tree_dict[country2], df_original)
        c1 = [int(c) for c in clusters[country1]]
        c2 = [int(c) for c in clusters[country2]]
        clist = sorted([c1, c2])
        cname = ''.join(str(elem)[1:-1]+'-' for elem in clist)
        name_to_list[cname] = [c1, c2]
        dist_dict[cname] = acc

    return dist_dict, name_to_list

def calc_X(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates X from the given DataFrame.
    X is calculated by excluding the column 'WP1325: Move Permanently to Another Country' from the DataFrame,
    dividing each column by its sum, and filling in any NaN values with 0.
    
    Args:
        df: The DataFrame to calculate X from.
    
    Returns:
        A new DataFrame containing the calculated X values.
    """
    X = df.loc[:, ~df.columns.isin(['WP1325: Move Permanently to Another Country'])]
    X = X.div(X.sum(axis=0), axis=1)
    return X.fillna(0)

def calc_Y(df_original: pd.DataFrame, countrylist: Union[str, List[int]]) -> pd.Series:
    """
    Calculates Y from the given DataFrame and country list.
    Y is calculated by selecting the 'WP1325: Move Permanently to Another Country' column from the DataFrame
    for the given country or countries and converting the values to integers.
    
    Args:
        df_original: The original DataFrame to calculate Y from.
        countrylist: A string or list of integers representing the country or countries to select from the DataFrame.
    
    Returns:
        A pandas Series containing the calculated Y values.
    """
    if type(countrylist)==str:
        list = [int(countrylist)]
        Y = df_original[df_original['WP5: Country'].isin(list)]['WP1325: Move Permanently to Another Country'].astype(int)
    else:
        list = [int(c) for c in countrylist]
        Y = df_original[df_original['WP5: Country'].isin(list)]['WP1325: Move Permanently to Another Country'].astype(int)
    return Y

def calc_tree(df: pd.DataFrame, df_original: pd.DataFrame, countrylist: Union[str, List[int]], depth: int) -> tree.DecisionTreeClassifier:
    """
    Calculates a decision tree from the given DataFrame, country list, and tree depth.
    The decision tree is trained using the X and Y values calculated from the given DataFrame and country list,
    and the maximum depth of the tree is specified by the given depth value.
    
    Args:
        df: The DataFrame to calculate X from and use for training the decision tree.
        df_original: The original DataFrame to calculate Y from.
        countrylist: A string or list of integers representing the country or countries to select from the DataFrame.
        depth: The maximum depth of the decision tree.
    
    Returns:
        A DecisionTreeClassifier object trained on the calculated X and Y values with the specified maximum depth.
    """
    X = calc_X(df)
    Y = calc_Y(df_original, countrylist)
    clf = tree.DecisionTreeClassifier(max_depth=depth)
    clf = clf.fit(X, Y)
    return clf

def calc_dist(df1: pd.DataFrame, df2: pd.DataFrame, clist1: Union[str, List[int]], clist2: Union[str, List[int]], tree1: tree.DecisionTreeClassifier, tree2: tree.DecisionTreeClassifier, df_original: pd.DataFrame) -> float:
    """
    Calculates the average accuracy of two decision trees on each other's datasets.
    The average accuracy is calculated by using each tree to predict the target values from the other tree's dataset,
    and taking the average of the two resulting accuracy values.
    
    Args:
        df1: The DataFrame to calculate X1 and use for testing tree2.
        df2: The DataFrame to calculate X2 and use for testing tree1.
        clist1: A string or list of integers representing the country or countries to select from df_original for Y1.
        clist2: A string or list of integers representing the country or countries to select from df_original for Y2.
        tree1: The DecisionTreeClassifier to use for predicting Y2 from X2.
        tree2: The DecisionTreeClassifier to use for predicting Y1 from X1.
        df_original: The original DataFrame to calculate Y1 and Y2 from.
    
    Returns:
        A float value representing the average accuracy of the two decision trees on each other's datasets.
    """
    X1 = calc_X(df1)
    X2 = calc_X(df2)
    Y1 = calc_Y(df_original, clist1)
    Y2 = calc_Y(df_original, clist2)

    acc1 = sum(tree1.predict(X2) == Y2) / len(Y2)
    acc2 = sum(tree2.predict(X1) == Y1) / len(Y1)

    return (acc1 + acc2)/2

def get_key(dictionary: dict, val) -> any:
    """
    Returns the key of the given value in the given dictionary.
    If the value is not found in the dictionary, None is returned.
    """
    for key, value in dictionary.items():
        if val == value:
            return key
    return None

def clustering(num: int, clusters: dict, name_to_list: dict, dist_dict: dict, country_df_dict: dict, country_tree_dict: dict, df_original: pd.DataFrame, depth: int) -> dict:
    """
    Performs hierarchical clustering on the given data using a decision tree classifier.
    
    Arguments:
    num -- The number of clusters to generate.
    clusters -- A dictionary mapping cluster names to the countries in each cluster.
    name_to_list -- A dictionary mapping cluster names to the list of countries in each cluster.
    dist_dict -- A dictionary mapping cluster pairs to the distances between them.
    country_df_dict -- A dictionary mapping countries to their data frames.
    country_tree_dict -- A dictionary mapping countries to their decision trees.
    df_original -- A data frame containing the original input data.
    depth -- The maximum depth of the decision trees.
    
    Returns:
    A dictionary mapping the names of the generated clusters to the countries in each cluster.
    """

    for i in tqdm(range(len(clusters)-num)):
        print(f"============= iteration {i} =============")

        # pick biggest acc
        countries_to_merge = name_to_list[max(dist_dict, key=dist_dict.get)]
        countries_to_merge_flatten = sum(name_to_list[max(dist_dict, key=dist_dict.get)], [])
        print("pair to merge")
        print(countries_to_merge)
        print(dist_dict[max(dist_dict, key=dist_dict.get)])

        # calculate common tree
        df_common = pd.DataFrame()
        for country in countries_to_merge:
            country = get_key(clusters, country)
            df_common = df_common + country_df_dict[country]

        # add to the dictionaries
        countries_to_merge_name = ''.join(str(elem)+'-' for elem in sorted(countries_to_merge_flatten))

        common_clf = calc_tree(df_common, df_original, countries_to_merge_flatten, depth)

        clusters[countries_to_merge_name] = countries_to_merge_flatten
        country_df_dict[countries_to_merge_name] = df_common
        country_tree_dict[countries_to_merge_name] = common_clf 

        # remove the existing unmerged countries
        for country in countries_to_merge:
            country = get_key(clusters, country)
            del clusters[country]
            del country_df_dict[country]
            del country_tree_dict[country]

        # calc distances and remove distances from the countries in countries_to_merge
        for country in clusters.keys(): 
            if country != countries_to_merge_name: 
                dist = calc_dist(country_df_dict[country], df_common, clusters[country], countries_to_merge_flatten, country_tree_dict[country], common_clf, df_original)
                clist = sorted(clusters[country]+ countries_to_merge_flatten)
                cname = ''.join(str(elem)+'-' for elem in clist)
                name_to_list[cname] = [clusters[country], countries_to_merge_flatten ]
                dist_dict[cname] = dist

        keys = list(name_to_list.keys())
        for dist_cs in keys:
            # remove dist for the countries in countries_to_merge
            help = sum(name_to_list[dist_cs], [])
            c0 = countries_to_merge[0]
            c1 = countries_to_merge[1]
            if bool(set(c0) & set(help))==True:
                if bool(set(c1) & set(help))==False: 
                    del name_to_list[dist_cs]
                    del dist_dict[dist_cs]  
            elif bool(set(c1) & set(help))==True: 
                if bool(set(c0) & set(help))==False:
                    del name_to_list[dist_cs]
                    del dist_dict[dist_cs] 

        del name_to_list[countries_to_merge_name]
        del dist_dict[countries_to_merge_name]

    return clusters

def clustering_round(num: int, round:int, clusters: dict, name_to_list: dict, dist_dict: dict, country_df_dict: dict, country_tree_dict: dict, df_original: pd.DataFrame, depth: int) -> dict:
    """
    Performs hierarchical clustering on the given data using a decision tree classifier. In each step, 
    a constraint is used on the size of the clustered groups.
    
    Arguments:
    num -- The number of clusters to generate.
    round -- Constraint on the number of countries together.
    clusters -- A dictionary mapping cluster names to the countries in each cluster.
    name_to_list -- A dictionary mapping cluster names to the list of countries in each cluster.
    dist_dict -- A dictionary mapping cluster pairs to the distances between them.
    country_df_dict -- A dictionary mapping countries to their data frames.
    country_tree_dict -- A dictionary mapping countries to their decision trees.
    df_original -- A data frame containing the original input data.
    depth -- The maximum depth of the decision trees.
    
    Returns:
    A dictionary mapping the names of the generated clusters to the countries in each cluster.
    """
    for i in tqdm(range(len(clusters)-num)):
        print(f"============= iteration {i} =============")

        # pick biggest acc
        disti = {k: v for k, v in dist_dict.items() if len(name_to_list[k][0])<=round and len(name_to_list[k][1])<=round}
        if len(disti)==0:
            maxi = max(dist_dict, key=dist_dict.get)
        else: 
            maxi = max(disti, key=dist_dict.get)
        countries_to_merge = name_to_list[maxi]
        countries_to_merge_flatten = sum(name_to_list[maxi], [])
        print("pair to merge")
        print(countries_to_merge)
        print(dist_dict[maxi])

        # calculate common tree
        df_common = pd.DataFrame()
        for country in countries_to_merge:
            country = get_key(clusters, country)
            df_common = df_common + country_df_dict[country]

        # add to the dictionaries
        countries_to_merge_name = ''.join(str(elem)+'-' for elem in sorted(countries_to_merge_flatten))

        common_clf = calc_tree(df_common, df_original, countries_to_merge_flatten, depth)

        clusters[countries_to_merge_name] = countries_to_merge_flatten
        country_df_dict[countries_to_merge_name] = df_common
        country_tree_dict[countries_to_merge_name] = common_clf 

        # remove the existing unmerged countries
        for country in countries_to_merge:
            country = get_key(clusters, country)
            del clusters[country]
            del country_df_dict[country]
            del country_tree_dict[country]

        # calc distances and remove distances from the countries in countries_to_merge
        for country in clusters.keys(): 
            if country != countries_to_merge_name: 
                dist = calc_dist(country_df_dict[country], df_common, clusters[country], countries_to_merge_flatten, country_tree_dict[country], common_clf, df_original)
                clist = sorted(clusters[country]+ countries_to_merge_flatten)
                cname = ''.join(str(elem)+'-' for elem in clist)
                name_to_list[cname] = [clusters[country], countries_to_merge_flatten ]
                dist_dict[cname] = dist

        keys = list(name_to_list.keys())
        for dist_cs in keys:
            # remove dist for the countries in countries_to_merge
            help = sum(name_to_list[dist_cs], [])
            c0 = countries_to_merge[0]
            c1 = countries_to_merge[1]
            if bool(set(c0) & set(help))==True:
                if bool(set(c1) & set(help))==False: 
                    del name_to_list[dist_cs]
                    del dist_dict[dist_cs]  
            elif bool(set(c1) & set(help))==True: 
                if bool(set(c0) & set(help))==False:
                    del name_to_list[dist_cs]
                    del dist_dict[dist_cs]
        
        del name_to_list[countries_to_merge_name]
        del dist_dict[countries_to_merge_name]

    return clusters
    