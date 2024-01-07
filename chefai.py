import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from PIL import Image
import time


img = Image.open('logo.png')

st.set_page_config(
    page_title="ChefAi",
    page_icon=img,
    layout="wide",
    initial_sidebar_state="auto",

)


# Add a picture of food
# Center-align the image using st.image
st.image('main_image.jpeg', use_column_width=True)



df_recipes = pd.read_csv('foods_cleaned.csv')


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



def filter_by_user_input(dataframe, user_input, time_range):
    if isinstance(dataframe, pd.Series):
        dataframe = pd.DataFrame(dataframe)  # Convert Series to DataFrame

    user_input_list = [item.strip() for item in user_input.split(',')]

    filtered_rows = []
    for index, row in dataframe.iterrows():
        ingredients = row['RecipeIngredientParts'].replace('c(', '').replace(')', '').replace('"', '').split(', ')
        if all(ingredient in ingredients for ingredient in user_input_list):
            filtered_rows.append(row)

    filtered_df = pd.DataFrame(filtered_rows)

    if time_range == '15_minutes':
        filtered_df = filtered_df[filtered_df['TotalTime_minutes'] <= 15]
    elif time_range == '30_minutes':
        filtered_df = filtered_df[filtered_df['TotalTime_minutes'] <= 30]
    elif time_range == '1_hour':
        filtered_df = filtered_df[filtered_df['TotalTime_minutes'] <= 60]
    elif time_range == '2_hours':
        filtered_df = filtered_df[filtered_df['TotalTime_minutes'] <= 120]
    elif time_range == '3_hours':
        filtered_df = filtered_df[filtered_df['TotalTime_minutes'] <= 180]

    return filtered_df



def input_format_change(ingredient_input):
    s = ''
    for i in ingredient_input:
        if s != '':
            s = s + ', ' + i
        else:
            s = i
    return s

def recommendation_tfid(df, user_input, rating_affect):
    user_input = [user_input]
    vectorizer = CountVectorizer()
    ingredients_matrix = vectorizer.fit_transform(df['RecipeIngredientParts'])
    user_vector = vectorizer.transform(user_input)

    similarities = cosine_similarity(user_vector, ingredients_matrix)

    indices = pd.Series(df.index, index=df['Name'])
    indices = indices[~indices.index.duplicated(keep='last')]

    df.reset_index(drop=True, inplace=True)
    data = {'score': similarities[0], 'weightedrating': df['WeightedRatingScore']}
    similarity_scores = pd.DataFrame(data)

    similarity_scores['weighted_score'] = similarity_scores['score'] * rating_affect + similarity_scores[
        'weightedrating'] * (1 - rating_affect)
    food_indices = similarity_scores.sort_values("weighted_score", ascending=False)[0:5].index

    df = df[df['Name'].index.isin(food_indices)]
    return df

def update_progress_bar():
    progress_text = "Operation in progress. Please wait."
    my_bar = st.progress(0, text=progress_text)

    for percent_complete in range(100):
        time.sleep(0.01)
        my_bar.progress(percent_complete + 1, text=progress_text)
    time.sleep(1)
    my_bar.empty()



ingredient_list=convert_to_unique_list(df_recipes['RecipeIngredientParts'])
#ingredient_input = st.text_input("Enter ingredients separated by commas:")
ingredient_input=st.multiselect(label='Please Pick your Ingredients',options=ingredient_list)
time_range = st.selectbox(label="Time range", options=['',"15_minutes", "30_minutes", "1_hour", "2_hours", "3_hours", "3_plus_hours"])


ingredient_input=input_format_change(ingredient_input)

if st.button("Recommend"):

    update_progress_bar()

    if ingredient_input:
        filtered_df = filter_by_user_input(df_recipes, ingredient_input, time_range)
        recommended_dishes = recommendation_tfid(filtered_df, ingredient_input, 0.8)
        st.subheader("Recommended Dishes:")

        if not recommended_dishes.empty:
            # Create a dictionary to store whether the ingredients expander is open for each dish
            expanders_open = {}

            for idx, row in recommended_dishes.iterrows():
                title = row['Name']
                cleaned_ingredients = row['FormattedIngredients']
                calories = row['Calories']
                time = row['TotalTime_minutes']
                rating = row['AggregatedRating']
                description = row['Description']

                # Create an expander for each dish
                with st.expander(f"{title}", expanded=expanders_open.get(title, False)):
                    st.write(f" {description}")  # Display the Calories information
                    st.write(f"Calories: {calories}")  # Display the Calories information
                    st.write(f"Total Time: {time}")  # Display the Time information
                    st.write(f"Rating: {rating}")  # Display the Time information

                    # st.markdown(cleaned_ingredients)
                    # Split the ingredients string at the comma
                    ingredients_list = [ingredient.lstrip("'") for ingredient in cleaned_ingredients.split("', ")]

                    # Remove "for serving" from each ingredient
                    ingredients_list = [ingredient.replace('for serving', '') for ingredient in ingredients_list]

                    # Check if the first ingredient starts with "[" and remove it
                    if ingredients_list[0].startswith("['"):
                        ingredients_list[0] = ingredients_list[0][2:]

                    # Check if the last ingredient ends with ']'
                    if ingredients_list[-1].endswith("']"):
                        ingredients_list[-1] = ingredients_list[-1][:-2]

                    st.markdown('\n'.join([f"- {ingredient}" for ingredient in ingredients_list]))


        else:
            st.write("No recommended dishes found. Please try a different combination of ingredients.")


    else:
        st.warning("Please enter ingredients to get recommendations.")

st.sidebar.header("About This App")
st.sidebar.info("Welcome to the ChefAI System! This web app suggests dishes based on the ingredients you provide.")
st.sidebar.info("To get your recommended dishes, simply enter the ingredients you'd like to use and click on the 'Recommend' button.")
st.sidebar.info("Be creative!")
st.sidebar.info("We hope you find a delicious dish to enjoy.")

