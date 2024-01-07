import pandas as pd
import re
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import MinMaxScaler


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)

# Reads the recipes dataset and reviews dataset
df_recipes=pd.read_csv('recipes/recipes.csv')
##df_reviews=pd.read_csv('recipes/reviews.csv')

#This funtion transforms PT0H0M to minutes as in integer
def parse_time_string(time_str):
    #Collects the pattern of time
    pattern = r'PT(?:(\d+)H)?(?:(\d+)M)?'
    #Matches with the format that is given
    match = re.match(pattern, time_str)


    #If there is a match, it is going to collect Hour and Minutes and returns in minutes format
    if match:
        hours = int(match.group(1)) if match.group(1) else 0
        minutes = int(match.group(2)) if match.group(2) else 0
        total_minutes = hours * 60 + minutes
        return total_minutes
    else:
        return None

#This funtion fills NA values with desired input
def fill_missing(dataframe, col, value):
    return df_recipes[col].fillna(value, inplace=True)

#Creates outlier thresholds
def outlier_thresholds(dataframe, col_name, q1=0.01, q3=0.99):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

#Checks column for outlier values
def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

#Grabs and displays outliers
def grab_outliers(dataframe, col_name, index=False):
    low, up = outlier_thresholds(dataframe, col_name)

    if dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].shape[0] > 10:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].head())
    else:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))])

    if index:
        outlier_index = dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].index
        return outlier_index

#Deltes outliers from the dataframe
def remove_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    df_without_outliers = dataframe[~((dataframe[col_name] < low_limit) | (dataframe[col_name] > up_limit))]
    return df_without_outliers

#Used for creating new feature for categorize time ranges
def time_category(value):
    if value <= 15:
        return '15_minutes'
    elif value > 15 and value <= 30:
        return '30_minutes'
    elif value > 30 and value <= 60:
        return '1_hour'
    elif value > 60 and value <= 120:
        return '2_hours'
    elif value > 120 and value <= 180:
        return '3_hours'
    elif value > 180:
        return '3_plus_hours'

#Used for creating new feature to transform PT0H0M to minutes
def time_transformation(dataframe, time_cols):
    for col in time_cols:
        new_col = col + '_minutes'
        dataframe[new_col] = dataframe[col].apply(lambda x: parse_time_string(x))

#Categorizes calories
def calorie_categorize(calorie):
    if calorie <= 400:
        return 'Low-Calorie'
    elif calorie <= 600:
        return 'Moderate-Calorie'
    else:
        return 'High-Calorie'


# Function to convert string representation to a list
def convert_to_list(ingredient):
    # Remove 'c(', ')', double quotes, and split by ', ' to get individual items
    items = ingredient.replace('c(', '').replace(')', '').replace('"', '').split(', ')
    return items


# Function to convert string representation to a list and remove duplicates
def convert_to_unique_list(ingredients):
    all_ingredients = []
    for string in ingredients:
        # Remove 'c(', ')', double quotes, and split by ', ' to get individual items
        items = string.replace('c(', '').replace(')', '').replace('"', '').split(', ')
        all_ingredients.extend(items)
    # Create a set to eliminate duplicates, then convert it back to a list
    unique_ingredients = list(set(all_ingredients))
    return unique_ingredients

#Gets most common ingredients
def get_most_common_ingredients(ingredients, amount):
    all_ingredients = []
    for string in ingredients:
        # Remove 'c(', ')', double quotes, and split by ', ' to get individual items
        items = string.replace('c(', '').replace(')', '').replace('"', '').split(', ')
        all_ingredients.extend(items)

    # Count the occurrences of each ingredient
    ingredient_counter = Counter(all_ingredients)
    # Get the most common ingredients
    top_repeated = ingredient_counter.most_common(amount)
    return top_repeated

#This is used to create a new feature called weighted rating score which is based on rating and review count to use in recommendation
def weighted_rating_score(df, review_effect):
    df["reviewCountScaled"] = MinMaxScaler(feature_range=(1, 10)). \
        fit(df[["ReviewCount"]]). \
        transform(df[["ReviewCount"]])
    df['WeightedRatingScore'] = df["reviewCountScaled"] * (1 - review_effect) + df["AggregatedRating"] * review_effect
    return df

#Our data has 250K rows and to make the system work faster, We have filtered the data based on ingredients inputs and time range
def filter_by_user_input(dataframe, user_input,time_range):
    user_input_list = [item.strip() for item in user_input.split(',')]

    filtered_rows = []
    for index, row in dataframe.iterrows():
        ingredients = row['RecipeIngredientParts'].replace('c(', '').replace(')', '').replace('"', '').split(', ')
        if all(ingredient in ingredients for ingredient in user_input_list):
            filtered_rows.append(row)

    filtered_df = pd.DataFrame(filtered_rows)
    if time_range == '15_minutes':
        filtered_df=filtered_df[filtered_df['TotalTime_minutes']<=15]
    elif time_range == '30_minutes':
        filtered_df=filtered_df[filtered_df['TotalTime_minutes']<=30]
    elif time_range == '1_hour':
        filtered_df=filtered_df[filtered_df['TotalTime_minutes']<=60]
    elif time_range == '2_hours':
        filtered_df=filtered_df[filtered_df['TotalTime_minutes']<=120]
    elif time_range == '3_hours':
        filtered_df=filtered_df[filtered_df['TotalTime_minutes']<=180]
    else:
        filtered_df = filtered_df
    return filtered_df

#Combines Ingredients and Quantities
def combine_quantities_and_ingredients(row):
    ingredients = row['RecipeIngredientParts']
    quantities = row['RecipeIngredientQuantities']

    # Split the ingredients and quantities strings into lists
    ingredient_list = [ingredient.strip('"') for ingredient in ingredients.strip('c()').split(', ')]
    quantity_list = [quantity.strip('"') for quantity in quantities.strip('c()').split(', ')]

    # Combine quantities and ingredients into formatted strings
    combined = list(zip(quantity_list, ingredient_list))
    formatted_list = [f"{quantity} {ingredient}" for quantity, ingredient in combined]
    formatted_output = ', '.join(formatted_list)

    return formatted_output

#Recommender System Function
def recommendation_tfid(df, user_input, rating_affect):
    #Turns string input into list
    user_input = [user_input]
    #Calls count vectorizer
    vectorizer = CountVectorizer()
    #Creates ingredient matrix for inding similarity
    ingredients_matrix = vectorizer.fit_transform(df['RecipeIngredientParts'])
    #Turns ingredient list input into vectorizer
    user_vector = vectorizer.transform(user_input)

    #Finds similarities by using cosine similarity
    similarities = cosine_similarity(user_vector, ingredients_matrix)

    #Creates new dataframe with similarity scores
    indices = pd.Series(df.index, index=df['Name'])
    #Drops dublicated vales
    indices = indices[~indices.index.duplicated(keep='last')]

    df.reset_index(drop=True, inplace=True)
    #Creates a similiarity score dataframe with similarity scores and weighted rating score based on review rating and review count
    data = {'score': similarities[0], 'weightedrating': df['WeightedRatingScore']}
    similarity_scores = pd.DataFrame(data)

    #Creates new feature from similarty score and wighted rating score
    similarity_scores['weighted_score'] = similarity_scores['score'] * rating_affect + similarity_scores[
        'weightedrating'] * (1 - rating_affect)
    food_indices = similarity_scores.sort_values("weighted_score", ascending=False)[0:11].index
    #Returens similar recipes
    return df['Name'].iloc[food_indices]

#Fills missing values with PT0H0M to match them with the pattern
fill_missing(df_recipes, 'CookTime', 'PT0H0M')

#Categorizes Calore levels
df_recipes['Calorie_Level']=df_recipes['Calories'].apply(lambda calorie: calorie_categorize(calorie))

#Creates new feature of times in minutes format
time_cols=['PrepTime','CookTime','TotalTime']
time_transformation(df_recipes, time_cols)

#Checks and grabs outliers
check_outlier(df_recipes, 'TotalTime_minutes')
grab_outliers(df_recipes, 'TotalTime_minutes')

#Removes outliers
df_recipes=remove_outlier(df_recipes, 'TotalTime_minutes')
#Categorizes Time based on minutes
df_recipes['TimeCategory']=df_recipes['TotalTime_minutes'].apply(lambda x: time_category(x))

# Apply the formatting function to the 'Ingredients' column in the DataFrame
df_recipes['FormattedIngredients'] = df_recipes.apply(combine_quantities_and_ingredients, axis=1)

#Drops null rows cause our data is too big
df_recipes.dropna(inplace=True)

#Creates new feature based on rating and review counts
weighted_rating_score(df_recipes,0.8)


#We have used this dataframe in the streamlit cause otherwise it was too big.
df_recipes.to_csv('foods_cleaned.csv')


#Oneri Sistemi
#Gets ingredients and time range inputs from the user
ingredient_input = input("Enter ingredients separated by commas: ")
time_range = input("Enter time range: ")

# Filter the DataFrame based on user input
filtered_df=filter_by_user_input(df_recipes,ingredient_input,time_range)

#Runs the recommender system 
recommend_dish=recommendation_tfid(filtered_df,ingredient_input,0.8)



