# %% [markdown]
# # Import necessary library

# %%
import os
import jdatetime
import pandas as pd
import numpy as np
import requests
import json
import urllib.request
from datetime import timedelta
from bs4 import BeautifulSoup
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix


# %%
class Recommender:
    def __init__(self):
        self.url = None
        self.df_title = None
        self.df_location = None
        self.df_date = None
        self.buy_history = None
        self.iter_history = None
        self.interaction = None
        self.merged_df = None
        self.dfs = None
        self.recommender = None
        self.event_df = None

    def data_scrap(self, url: str):
        """use site url to scrap necessary data

        Args:
            url (str): site address

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

            self.df_title = pd.DataFrame({"Titles": titles})
            df_title = self.df_title

            for loc in h3_tags_location:
                if loc.text.strip():
                    location.append(loc.text.strip())

            self.df_location = pd.DataFrame({"Titles": location})
            df_location = self.df_location

            for dt in h3_tags_date:
                if dt.text.strip():
                    date.append(dt.text.strip())

            self.df_date = pd.DataFrame({"Titles": date})
            df_date = self.df_date

            return df_title, df_location, df_date

    def user_buy_interaction_from_api(self, buy_api: str, iter_api: str):
        """use api to scrap buy and interaction users

        Args:
            buy_api (str): api to scrape user buy history
            iter_api (str): api to scrape user iter history
        """
        pd.options.mode.copy_on_write = True
        buy_link = buy_api
        iter_link = iter_api

        urllib.request.urlretrieve(iter_link, "log.xlsx")
        self.iter_history = pd.read_excel("log.xlsx")
        iter_history = pd.read_excel("log.xlsx")

        urllib.request.urlretrieve(buy_link, "visitor.xlsx")
        self.buy_history = pd.read_excel("visitor.xlsx")
        buy_history = pd.read_excel("visitor.xlsx")

        return iter_history, buy_history

    def generate_date_ranges(self, start_date, end_date):
        date_ranges = []
        current_start_date = start_date
        while current_start_date < end_date:
            current_end_date = current_start_date + timedelta(days=60)
            if current_end_date > end_date:
                current_end_date = end_date
            date_ranges.append((current_start_date, current_end_date))
            current_start_date = current_end_date
        return date_ranges

    def fetch_data_from_api_log(self, start_date, end_date):
        start_date = start_date.strftime("%Y/%m/%d")
        end_date = end_date.strftime("%Y/%m/%d")
        url = f"https://6234.ir/api/log?token=aiapiqazxcvbnm1403&ofDate={start_date}&toDate={end_date}"
        return url

    def interaction_auto(self):
        start_date_jalali = jdatetime.datetime.strptime(
            jdatetime.date(1402, 1, 1).strftime("%Y/%m/%d"), "%Y/%m/%d"
        ).date()
        end_date_jalali = jdatetime.datetime.strptime(
            jdatetime.datetime.now().strftime("%Y/%m/%d"), "%Y/%m/%d"
        ).date()
        date_ranges = self.generate_date_ranges(start_date_jalali, end_date_jalali)
        for start, end in date_ranges:
            urllib.request.urlretrieve(
                self.fetch_data_from_api_log(start, end), f"log{start.month}.xlsx"
            )

        df_api = {}
        for start, end in date_ranges:
            month = start.month
            df = pd.read_excel(f"log{month}.xlsx")

            if month in df_api:

                df_api[month] = pd.concat([df_api[month], df], ignore_index=True)
            else:
                df_api[month] = df

        combined_df = pd.concat(df_api.values(), ignore_index=True)

        return combined_df

    def fetch_data_from_api_buy(self, start_date, end_date):
        start_date = start_date.strftime("%Y/%m/%d")
        end_date = end_date.strftime("%Y/%m/%d")
        url = f"https://6234.ir/api/ticket?token=aiapiqazxcvbnm1403&ofDate={start_date}&toDate={end_date}"
        return url

    def buy_auto(self):
        start_date_jalali = jdatetime.datetime.strptime(
            jdatetime.date(1402, 1, 1).strftime("%Y/%m/%d"), "%Y/%m/%d"
        ).date()
        end_date_jalali = jdatetime.datetime.strptime(
            jdatetime.datetime.now().strftime("%Y/%m/%d"), "%Y/%m/%d"
        ).date()
        date_ranges = self.generate_date_ranges(start_date_jalali, end_date_jalali)
        for start, end in date_ranges:
            urllib.request.urlretrieve(
                self.fetch_data_from_api_buy(start, end), f"log{start.month}.xlsx"
            )

        df_api = {}
        for start, end in date_ranges:
            month = start.month
            df = pd.read_excel(f"log{month}.xlsx")

            if month in df_api:

                df_api[month] = pd.concat([df_api[month], df], ignore_index=True)
            else:
                df_api[month] = df

        combined_df = pd.concat(df_api.values(), ignore_index=True)

        return combined_df

    def preprocessing_interaction(self, interaction_df: pd.DataFrame):
        """preprocessing interaction data for use in model

        Args:
            interaction_df (pd.DataFrame): interaction pd from user_buy_interaction_from_api func

        Returns:
            interaction_df (pd.DataFrame): interaction_df
        """
        interaction_df["بازدید"] = interaction_df["بازدید"].fillna("ffill")
        interaction_df["نام و نام خانوادگی"] = interaction_df[
            "نام و نام خانوادگی"
        ].fillna("none")
        interaction_df["شماره موبایل"] = interaction_df["شماره موبایل"].fillna("none")
        interaction_df = interaction_df[interaction_df["بازدید"] != "صفحه اصلی"]

        le = LabelEncoder()
        interaction_df.loc[:, "userId"] = le.fit_transform(
            interaction_df["نام و نام خانوادگی"]
        )

        return interaction_df

    def event_api(self, api: str):
        event_link = api

        urllib.request.urlretrieve(event_link, "event.xlsx")
        self.iter_history = pd.read_excel("event.xlsx")
        event_df = pd.read_excel("event.xlsx")
        event_df["Titles"] = event_df["عنوان"]

        return event_df

    def merged_all_df(
        self,
        df_title: pd.DataFrame,
        df_location: pd.DataFrame,
        df_date: pd.DataFrame,
        df_interaction: pd.DataFrame,
        # df_buy_history: pd.DataFrame,
        event_df: pd.DataFrame,
    ):
        """merged all df to concat all titles under each other

        Args:
            df_title (pd.DataFrame): df_title scrape from data_scrap func output
            df_location (pd.DataFrame): df_location scrape from data_scrap func output
            df_date (pd.DataFrame): df_date scrape from data_scrap func output
            df_interaction (pd.DataFrame): df_interaction scrape from user_buy_interaction_from_api func output
            df_buy_history (pd.DataFrame): df_buy_history scrape from user_buy_interaction_from_api func output

        Returns:
            merged df: Pandas DataFrame
        """
        merge_df = pd.DataFrame(
            {
                "Titles": df_title["Titles"],
                "Location": df_location["Titles"],
                "Date": df_date["Titles"],
            }
        )

        merge_df = pd.concat(
            [
                merge_df["Titles"],
                df_interaction["بازدید"],
                # df_buy_history["رویداد"],
                event_df["Titles"],
            ]
        ).reset_index()

        merge_df.columns = ["index", "Titles"]

        return merge_df

    def list_to_string(self, row):
        return " ".join(row)

    def remove_excel(self, excel_list: list):
        for i in excel_list:
            os.remove(i)

    def preprocessing_merged_df(self, merged_df: pd.DataFrame):
        """preprocessing merged_df data for use in model

        Args:
            merged_df (pd.DataFrame): merged_df pd from merged_all_df func output

        Returns:
            merged_df: Pandas DataFrame
        """

        df_ohe = merged_df["Titles"].str.split(" ").reset_index().astype("str")
        df_ohe["Titles"] = df_ohe["Titles"].apply(self.list_to_string)

        le = LabelEncoder()
        merged_df["ohe"] = le.fit_transform(df_ohe["Titles"])

        self.merged_df = merged_df

        return merged_df

    def vectorized_text(self, df_title: pd.DataFrame):
        """vectorized_text for merged Convert a collection of text documents to a matrix of token counts

        Args:
            df_title (pd.DataFrame): use df_title from data_scrap func output

        Returns:
            X : array of shape (n_samples, n_features)
        """

        vectorized = CountVectorizer(token_pattern=r"(?u)\b\w+\b")
        X = vectorized.fit_transform(self.merged_df["Titles"])

        feature_names = vectorized.get_feature_names_out()
        one_hot_df = pd.DataFrame(X.toarray(), columns=feature_names)

        dfs = pd.concat([df_title, one_hot_df], axis=1)
        dfs.drop(columns=["Titles"], inplace=True)

        self.dfs = dfs
        return dfs

    def creat_X(self, interaction_df):
        """Compressed Sparse Row matrix.

        Args:
            iteraction_df (_type_): use preprocessing_interaction func output

        Returns:
            sparse matrix of type '<class 'numpy.float64'>
        """

        M = interaction_df["userId"].nunique()
        N = interaction_df["بازدید"].nunique()

        user_mapper = dict(zip(np.unique(interaction_df["userId"]), list(range(M))))
        item_mapper = dict(zip(np.unique(interaction_df["بازدید"]), list(range(N))))

        user_inv_mapper = dict(zip(list(range(M)), np.unique(interaction_df["userId"])))
        item_inv_mapper = dict(zip(list(range(N)), np.unique(interaction_df["بازدید"])))

        user_index = [user_mapper[i] for i in interaction_df["userId"]]
        item_indx = [item_mapper[i] for i in interaction_df["بازدید"]]

        X = csr_matrix(
            (interaction_df["زمان تعامل(تانیه)"], (user_index, item_indx)), shape=(M, N)
        )

        return X, user_mapper, item_mapper, user_inv_mapper, item_inv_mapper

    def cosine_similioraty(
        self,
        dfs: pd.DataFrame,
        event_df: pd.DataFrame,
        interaction_df: pd.DataFrame,
        idx: str,
        n_recommendations: int = 1,
    ):
        """Compute cosine similarity between samples in X and Y.

        Cosine similarity, or the cosine kernel, computes similarity as the normalized dot product of X and Y:

                Args:
                    dfs (pd.DataFrame): use vectorized_text func outputs
                    merged_df (pd.DataFrame): use preprocessing_merged_df func outputs
                    interaction_df (pd.DataFrame): use preprocessing_interaction func output
                    idx (str): idx of user interation and buy
                    n_recommendations (int, optional): Number of outgoing recommenders. Defaults to 1.

                Returns:
                    list: user best recommenders
        """
        cosine_sim = cosine_similarity(dfs, dfs)
        iter_idx = dict(zip(event_df["Titles"].unique(), list(event_df.index)))
        idx = iter_idx[idx]
        n_recommendations = n_recommendations
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1 : (n_recommendations + 1)]
        similar_item = [i[0] for i in sim_scores]
        recomended = event_df["Titles"].iloc[similar_item]
        recomended = recomended.to_list()

        return recomended

    def recomender_users(
        self,
        interaction_df: pd.DataFrame,
        dfs: pd.DataFrame,
        event_df: pd.DataFrame,
        n_recommendations=1,
    ):
        """use interaction_df, dfs, event_df to recommend best for each user

        Args:
            interaction_df (pd.DataFrame): output of preprocessing_interaction function
            dfs (pd.DataFrame): output of vectorized_text function
            event_df (pd.DataFrame): output of event_api function
            n_recommendations (int, optional): number of recommendations per user. Defaults to 1.

        Returns:
            dict: user(phone number) recommender
        """
        users_phone = interaction_df["شماره موبایل"].unique()
        user_recommendations = {}

        for phone_number in users_phone:
            user_data = (
                interaction_df[interaction_df["شماره موبایل"] == phone_number][
                    ["زمان تعامل(تانیه)", "بازدید"]
                ]
                .max()
                .reset_index()
                .T
            )
            user_data.columns = ["زمان تعامل(تانیه)", "بازدید"]
            user_data.drop(index="index", inplace=True)
            idx = user_data["بازدید"].tolist()[0]

            recommendations = self.cosine_similioraty(
                dfs,
                event_df,
                interaction_df,
                idx=idx,
                n_recommendations=n_recommendations,
            )

            # Remove duplicates from recommendations
            unique_recommendations = list(dict.fromkeys(recommendations))

            # Keep only the first n_recommendations unique recommendations
            final_recommendations = []
            seen = set()
            for rec in recommendations:
                if rec not in seen:
                    final_recommendations.append(rec)
                    seen.add(rec)
                if len(final_recommendations) == n_recommendations + 1:
                    break

            user_recommendations[phone_number] = final_recommendations

        return user_recommendations


# %%
recomender = Recommender()

# %%
interaction = recomender.interaction_auto()

# %%
buy_history = recomender.buy_auto()

# %%
df_title, df_location, df_date = recomender.data_scrap("https://www.6234.ir/")

# %%
event_df = recomender.event_api("https://6234.ir/api/event?token=aiapiqazxcvbnm1403")

# %%
interaction = recomender.preprocessing_interaction(interaction)

# %%
event_df_ = event_df[["Titles"]]
event_df_ = event_df_.dropna()
event = event_df_["Titles"]
filtered_interaction = interaction[interaction["بازدید"].isin(event.tolist())]
merged_df = recomender.merged_all_df(
    df_title, df_location, df_date, filtered_interaction, event_df
)

# %%
event_df_ = event_df[["Titles"]]
event_df_ = event_df_.dropna()

# %%
merged_df = recomender.preprocessing_merged_df(merged_df)
dfs = recomender.vectorized_text(merged_df)

# %%
X, user_mapper, item_mapper, user_inv_mapper, item_inv_mapper = recomender.creat_X(
    interaction,
)

# %%
recomender.recomender_users(filtered_interaction, dfs, merged_df, n_recommendations=4)

# %%
