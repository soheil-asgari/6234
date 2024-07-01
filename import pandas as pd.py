# %% [markdown]
# **Import necessary libraries**

# %%
import pandas as pd
import numpy as np
import requests
import json
import urllib.request
from bs4 import BeautifulSoup
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import csr_matrix

# %% [markdown]
# **Getting information about the events from the website's API**


# %%
def data_scrap(url: str):
    """
    Scrapes event information from the given URL.

    Parameters:
    url (str): The URL of the website to scrape.

    Returns:
    tuple: DataFrames containing event titles, dates, and locations.
    """
    url = url
    response = requests.get(url)

    if response.status_code == 200:

        soup = BeautifulSoup(response.text, "html.parser")

        h3_tags_title = soup.find_all("h3", class_="blog_post_title my-2")
        h3_tags_location = soup.find_all("div", class_="blog_post_title my-2")
        h3_tags_date = soup.find_all("div", class_="theater-date my-2")

        titles = []
        location = []
        date = []

        for title in h3_tags_title:
            if title.text.strip():
                titles.append(title.text.strip())

        df_title = pd.DataFrame({"Titles": titles})

        for loc in h3_tags_location:
            if loc.text.strip():
                location.append(loc.text.strip())

        df_location = pd.DataFrame({"Titles": location})

        for dt in h3_tags_date:
            if dt.text.strip():
                date.append(dt.text.strip())

        df_date = pd.DataFrame({"Titles": date})

    return df_title, df_date, df_location


# %% [markdown]
# **Getting information about user's buying and interaction history from the API**


# %%
def user_buy_interaction_from_api(buy_api: str, iter_api: str):
    """
    Retrieves user's buying and interaction history from the provided APIs.

    Parameters:
    buy_api (str): The API endpoint for buying history.
    iter_api (str): The API endpoint for interaction history.

    Returns:
    tuple: DataFrames containing interaction and buying history.
    """
    buy_link = buy_api
    iter_link = iter_api
    urllib.request.urlretrieve(iter_link, "log.xlsx")
    iter_history = pd.read_excel("log.xlsx")
    urllib.request.urlretrieve(buy_link, "visitor.xlsx")
    buy_history = pd.read_excel("visitor.xlsx")

    return iter_history, buy_history


# %% [markdown]
# **Data Preparation**


# %%
def prepare_interaction_data(interaction: pd.DataFrame):
    """
    Prepares interaction data by handling missing values and dropping unnecessary columns.

    Parameters:
    interaction (pd.DataFrame): DataFrame containing interaction data.

    Returns:
    pd.DataFrame: Processed interaction DataFrame.
    """
    interaction["بازدید"] = interaction["بازدید"].fillna("ffill")
    interaction["نام و نام خانوادگی"] = interaction["نام و نام خانوادگی"].fillna("none")
    interaction["شماره موبایل"] = interaction["شماره موبایل"].fillna("none")

    # Drop "صفحه اصلی" from "بازدید" columns
    interaction = interaction[interaction["بازدید"] != "صفحه اصلی"]

    return interaction


# %%
def merge_dataframes(
    df_title: pd.DataFrame,
    df_date: pd.DataFrame,
    df_location: pd.DataFrame,
    interaction: pd.DataFrame,
    buy_history: pd.DataFrame,
):
    """
    Merges multiple DataFrames into a single DataFrame.

    Parameters:
    df_title (pd.DataFrame): DataFrame containing event titles.
    df_date (pd.DataFrame): DataFrame containing event dates.
    df_location (pd.DataFrame): DataFrame containing event locations.
    interaction (pd.DataFrame): DataFrame containing interaction data.
    buy_history (pd.DataFrame): DataFrame containing buying history.

    Returns:
    pd.DataFrame: Merged DataFrame.
    """
    merge_df = pd.DataFrame(
        {
            "Titles": df_title["Titles"],
            "Location": df_location["Titles"],
            "Date": df_date["Titles"],
        }
    )

    merge_df = pd.concat(
        [merge_df["Titles"], interaction["بازدید"], buy_history["رویداد"]]
    ).reset_index()

    merge_df.columns = ["index", "Titles"]

    return merge_df


# %%
def preprocess_interaction_data(df: pd.DataFrame):
    """
    Preprocesses interaction data by encoding user names to user IDs.

    Parameters:
    df (pd.DataFrame): DataFrame containing interaction data.

    Returns:
    pd.DataFrame: Processed DataFrame.
    """
    le = LabelEncoder()
    df["userId"] = le.fit_transform(df["نام و نام خانوادگی"])

    return df


# %%
def vectorize_text(df: pd.DataFrame):
    """
    Vectorizes text data.

    Parameters:
    df (pd.DataFrame): DataFrame containing text data.

    Returns:
    pd.DataFrame: Vectorized DataFrame.
    """
    vectorized = CountVectorizer(token_pattern=r"(?u)\b\w+\b")
    X = vectorized.fit_transform(df["Titles"])

    feature_names = vectorized.get_feature_names_out()
    one_hot_df = pd.DataFrame(X.toarray(), columns=feature_names)

    dfs = pd.concat([df, one_hot_df], axis=1)
    dfs.drop(columns=["Titles"], inplace=True)

    return dfs


# %%
def create_interaction_matrix(df: pd.DataFrame):
    """
    Creates the interaction matrix.

    Parameters:
    df (pd.DataFrame): DataFrame containing interaction data.

    Returns:
    csr_matrix: Interaction matrix.
    dict: User mapper.
    dict: Item mapper.
    dict: Inverse user mapper.
    dict: Inverse item mapper.
    """
    M = df["userId"].nunique()
    N = df["بازدید"].nunique()

    user_mapper = dict(zip(np.unique(df["userId"]), list(range(M))))
    item_mapper = dict(zip(np.unique(df["بازدید"]), list(range(N))))
    user_inv_mapper = dict(zip(list(range(M)), np.unique(df["userId"])))
    item_inv_mapper = dict(zip(list(range(N)), np.unique(df["بازدید"])))
    user_index = [user_mapper[i] for i in df["userId"]]
    item_index = [item_mapper[i] for i in df["بازدید"]]

    X = csr_matrix((df["زمان تعامل(تانیه)"], (user_index, item_index)), shape=(M, N))

    return X, user_mapper, item_mapper, user_inv_mapper, item_inv_mapper


# %%
def find_similar_item(item_name, X, item_mapper, item_inv_mapper, k, metric="cosine"):
    """
    Finds similar items based on a given item.

    Parameters:
    item_name (str): Name of the item.
    X (csr_matrix): Interaction matrix.
    item_mapper (dict): Item mapper.
    item_inv_mapper (dict): Inverse item mapper.
    k (int): Number of similar items to find.
    metric (str): Similarity metric.

    Returns:
    list: List of similar item names.
    """
    X = X.T
    neighbors_ids = []

    item_index = item_mapper[item_name]
    item_vector = X[item_index]
    if isinstance(item_vector, (np.ndarray)):
        item_vector = item_vector.reshape(1, -1)

    # Use k+1 since kNN output includes the item of interest
    knn = NearestNeighbors(n_neighbors=k + 1, algorithm="brute", metric=metric)
    knn.fit(X)
    neighbors = knn.kneighbors(item_vector, return_distance=False)

    for i in range(0, k):
        n = neighbors.item(i)
        neighbors_ids.append(item_inv_mapper[n])

    neighbors_ids.pop(0)

    return neighbors_ids


# %%
def get_recommended_items(
    user_interaction, dfs, merged_df, interaction, n_recommendations=1
):
    """
    Recommends items to users based on their interaction history.

    Parameters:
    user_interaction (dict): Dictionary containing user interaction history.
    dfs (pd.DataFrame): DataFrame containing vectorized text data.
    merged_df (pd.DataFrame): Merged DataFrame containing event information.
    interaction (pd.DataFrame): DataFrame containing interaction data.
    n_recommendations (int): Number of recommendations to provide.

    Returns:
    dict: Dictionary containing recommended items for each user.
    """
    recommended_items = {}

    for user, interaction_id in user_interaction.items():
        # Find index corresponding to interaction ID
        idx = interaction[interaction["بازدید"] == interaction_id].index
        if len(idx) > 0:
            idx = idx[0]  # Take the first index if multiple interactions found
            iter_names = cosine_similarity(dfs, dfs)
            recommended = cosine_similarity(dfs[idx].reshape(1, -1), dfs).argsort()[
                :, ::-1
            ][:, 1 : n_recommendations + 1]
            recommended_items[user] = merged_df.iloc[recommended[0]]["Titles"].tolist()
        else:
            recommended_items[user] = []  # No interaction found for this user

    return recommended_items


# %% [markdown]
# **Usage Examples**

# %%
# Scraping data from the website
df_title, df_date, df_location = data_scrap("https://www.6234.ir/")

# Retrieving user interaction and buying history from APIs
interaction, buy_history = user_buy_interaction_from_api(
    "https://6234.ir/api/ticket?token=apiqazxcvbnm&ofDate=1402/08/20&toDate=1402/08/29",
    "https://6234.ir/api/log?token=apiqazxcvbnm&ofDate=1402/08/20&toDate=1402/12/29",
)

# Preparing interaction data
interaction = prepare_interaction_data(interaction)

# Merging all DataFrames
merged_df = merge_dataframes(df_title, df_date, df_location, interaction, buy_history)

# Preprocessing interaction data
df = preprocess_interaction_data(interaction)

# Vectorizing text data
dfs = vectorize_text(df_title)

# Creating interaction matrix
X, user_mapper, item_mapper, user_inv_mapper, item_inv_mapper = (
    create_interaction_matrix(df)
)

# Finding similar items
similar_items = find_similar_item(
    "کنسرت تست ( بدون انتخاب صندلی )", X, item_mapper, item_inv_mapper, k=1
)

# Getting recommended items for each user
users_phone = interaction["شماره موبایل"].unique()
user_interaction = {}
for phone in users_phone:
    user_interaction[phone] = interaction[interaction["شماره موبایل"] == phone][
        "بازدید"
    ].max()

recommended_items = get_recommended_items(user_interaction, dfs, merged_df, interaction)
recommended_items
