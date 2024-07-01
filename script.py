import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors


class Recommender:
    def __init__(self):
        self.df_title = None
        self.df_location = None
        self.df_date = None
        self.interaction = None
        self.buy_history = None
        self.df = None
        self.df_ohe = None
        self.merge_df = None

    def data_scrap(self, url: str):
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

            for loc in h3_tags_location:
                if loc.text.strip():
                    location.append(loc.text.strip())

            self.df_location = pd.DataFrame({"Titles": location})

            for dt in h3_tags_date:
                if dt.text.strip():
                    date.append(dt.text.strip())

            self.df_date = pd.DataFrame({"Titles": date})

    def user_interaction_from_file(self, file_path: str):
        self.interaction = pd.read_excel(file_path)

    def user_buy_history_from_file(self, file_path: str):
        self.buy_history = pd.read_excel(file_path)

    def list_to_string(self, row):
        return " ".join(row)

    def merged_df(
        self,
    ):
        merge_df = pd.DataFrame(
            {
                "Titles": self.df_title["Titles"],
                "Location": self.df_location["Titles"],
                "Date": self.df_date["Titles"],
            }
        )
        merge_df = pd.concat(
            [merge_df["Titles"], self.interaction["بازدید"], self.buy_history["رویداد"]]
        ).reset_index()
        le = LabelEncoder()

        merge_df.columns = ["index", "Titles"]
        self.merge_df = merge_df

        merge_df = merge_df["Titles"].str.split(" ").reset_index().astype("str")

        merge_df["ohe"] = le.fit_transform(merge_df["Titles"])

        return merge_df

    def iteraction_pre(self):
        df = self.interaction
        le = LabelEncoder()
        df["userId"] = le.fit_transform(df["نام و نام خانوادگی"])
        self.df = df
        return df

    def creat_X(self, df):

        M = df["userId"].nunique()
        N = df["بازدید"].nunique()

        user_mapper = dict(zip(np.unique(df["userId"]), list(range(M))))
        item_mapper = dict(zip(np.unique(df["بازدید"]), list(range(N))))

        user_inv_mapper = dict(zip(list(range(M)), np.unique(df["userId"])))
        item_inv_mapper = dict(zip(list(range(N)), np.unique(df["بازدید"])))

        user_index = [user_mapper[i] for i in df["userId"]]
        item_indx = [item_mapper[i] for i in df["بازدید"]]

        X = csr_matrix((df["زمان تعامل(تانیه)"], (user_index, item_indx)), shape=(M, N))

        return X, user_mapper, item_mapper, user_inv_mapper, item_inv_mapper

    def find_similar_item(
        self, iter_name, X, item_mapper, item_inv_mapper, k, metrics="cosine"
    ):

        X = X.T
        neighbours_ids = []

        iter_ind = item_mapper[iter_name]
        iter_vec = X[iter_ind]
        if isinstance(iter_vec, (np.ndarray)):
            iter_vec = iter_vec.reshape(1, -1)
        # use k+1 since kNN output includes the user recommender of interest
        knn = NearestNeighbors(n_neighbors=k + 1, algorithm="brute", metric=metrics)
        knn.fit(X)
        neighbours = knn.kneighbors(iter_vec, return_distance=False)
        for i in range(0, k):
            n = neighbours.item(i)
            neighbours_ids.append(item_inv_mapper[n])
        neighbours_ids.pop(0)
        return neighbours_ids

    def vectorized_text(self, df):
        vectorized = CountVectorizer(token_pattern=r"(?u)\b\w+\b")
        X = vectorized.fit_transform(df["Titles"])

        feature_names = vectorized.get_feature_names_out()
        one_hot_df = pd.DataFrame(X.toarray(), columns=feature_names)

        dfs = pd.concat([df, one_hot_df], axis=1)
        dfs.drop(columns=["Titles"], inplace=True)

        return dfs

    def cosine_similioraty(
        self,
        dfs: pd.DataFrame,
        merge_pd,
        interaction,
        idx: str,
        n_recommendations: int = 1,
    ):
        cosine_sim = cosine_similarity(dfs, dfs)
        iter_idx = dict(zip(merge_pd["Titles"], list(interaction.index)))
        idx = iter_idx[idx]
        n_recommendations = n_recommendations
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1 : (n_recommendations + 1)]
        similar_item = [i[0] for i in sim_scores]
        recomended = merge_pd["Titles"].iloc[similar_item]
        recomended = recomended.to_list()

        return recomended

    def users_recommenders(self, dfs, merge_df, iteraction):
        user_iter = {}
        users = self.interaction["نام و نام خانوادگی"].unique()
        for i in users:
            user_it = (
                self.interaction[self.interaction["نام و نام خانوادگی"] == i][
                    ["زمان تعامل(تانیه)", "بازدید"]
                ]
                .max()
                .reset_index()
                .T
            )
            user_it.columns = ["زمان تعامل(تانیه)", "بازدید"]
            user_it.drop(index="index", inplace=True)
            user_it["بازدید"]
            idx = user_it["بازدید"].to_list()[0]
            names = i
            mobile = str(
                self.interaction[self.interaction["نام و نام خانوادگی"] == i][
                    "شماره موبایل"
                ].unique()[0]
            )
            mobile = mobile[:-2]
            iters = self.cosine_similioraty(
                dfs, merge_df, iteraction, idx=idx, n_recommendations=1
            )
            user_dict = {mobile: iters}
            user_iter.update(user_dict)

        return user_it


r = Recommender()
r.data_scrap(url="https://www.6234.ir/")
user_iteraction = r.user_interaction_from_file("./لاک بازدید- نمونه.xlsx")
user_buy = r.user_buy_history_from_file("./گزارش بلیط- نمونه.xlsx")
df = r.merged_df()
pre = r.iteraction_pre()
x = r.creat_X(pre)
df["Titles"] = df["Titles"].apply(r.list_to_string)
dfs = r.vectorized_text(r.df_title)
user = r.interaction["نام و نام خانوادگی"].unique()
interaction = r.interaction
merge_df = r.merge_df
print(merge_df)
recomender = r.users_recommenders(dfs, merge_df, interaction)
print(recomender)
