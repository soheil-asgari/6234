# %% [markdown]
# # Import Required libraries
# در این بخش، کتابخانه‌های مورد نیاز برای کار ما را وارد می‌کنیم.

# %%
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import csr_matrix

# %% [markdown]
# ## Scraping Functions
# در این بخش، توابع مورد نیاز برای گرفتن اطلاعات از وبسایت را تعریف می‌کنیم.


# %%
def scrape_data(url: str, tag: str, class_name: str):
    """
    Scrapes data from a webpage using BeautifulSoup.

    Args:
    url (str): The URL of the webpage to scrape.
    tag (str): The HTML tag to search for.
    class_name (str): The class name of the tag to search for.

    Returns:
    DataFrame: A DataFrame containing the scraped data.
    """
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, "html.parser")
        tags = soup.find_all(tag, class_=class_name)
        data = [tag.text.strip() for tag in tags if tag.text.strip()]
        df = pd.DataFrame({class_name: data})
        return df
    else:
        print("Failed to fetch data from the URL.")
        return None


# %% [markdown]
# ## Data Processing Functions
# در این بخش، توابعی برای پردازش داده‌ها و ایجاد ویژگی‌های جدید تعریف می‌کنیم.


# %%
def preprocess_text(df: pd.DataFrame):
    """
    Preprocesses text data by converting it to lowercase and removing punctuation.

    Args:
    df (DataFrame): DataFrame containing text data.

    Returns:
    DataFrame: Processed DataFrame.
    """
    df["Titles"] = df["Titles"].str.lower()
    df["Titles"] = df["Titles"].str.replace(r"[^\w\s]", "")
    return df


# %%
def encode_labels(df: pd.DataFrame):
    """
    Encodes categorical labels into numerical values using LabelEncoder.

    Args:
    df (DataFrame): DataFrame containing categorical labels.

    Returns:
    DataFrame: Encoded DataFrame.
    """
    le = LabelEncoder()
    df["Encoded_Labels"] = le.fit_transform(df["Labels"])
    return df


# %%
def vectorize_text(df: pd.DataFrame):
    """
    Vectorizes text data using CountVectorizer.

    Args:
    df (DataFrame): DataFrame containing text data.

    Returns:
    DataFrame: DataFrame with text vectorized.
    """
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(df["Titles"])
    feature_names = vectorizer.get_feature_names_out()
    vectorized_df = pd.DataFrame(X.toarray(), columns=feature_names)
    return vectorized_df


# %% [markdown]
# ## Main Function
# در این بخش، تابع اصلی برنامه را تعریف می‌کنیم که کارهای مورد نیاز را انجام می‌دهد.


# %%
def main():
    # Scraping data
    url = "https://www.6234.ir/"
    titles_df = scrape_data(url, "h3", "blog_post_title my-2")

    # Preprocessing text
    titles_df = preprocess_text(titles_df)

    # Vectorizing text
    vectorized_titles = vectorize_text(titles_df)

    # Printing vectorized titles
    print("Vectorized Titles:")
    print(vectorized_titles)


# %%
# Run the main function
if __name__ == "__main__":
    main()
