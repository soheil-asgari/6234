import pandas as pd
import requests
from bs4 import BeautifulSoup


def title_scrape(addres: str):
    """
    This function scrapes the title of a webpage.
    """
    url = addres
    response = requests.get(url)

    if response.status_code == 200:

        soup = BeautifulSoup(response.text, "html.parser")

        h3_tags = soup.find_all("h3", class_="blog_post_title my-2")

        titles = []

        for tag in h3_tags:
            if tag.text.strip():
                titles.append(tag.text.strip())

        df = pd.DataFrame({"Titles": titles})

        return df
    else:
        print(
            "مشکلی در دریافت اطلاعات از وب‌سایت رخ داده است. کد وضعیت:",
            response.status_code,
        )


def location_scrape(addres: str):
    """
    This function scrapes the location of a webpage.
    """
    url = "https://www.6234.ir/"

    response = requests.get(url)

    if response.status_code == 200:

        soup = BeautifulSoup(response.text, "html.parser")

        h3_tags = soup.find_all("div", class_="blog_post_title my-2")

        titles = []

        for tag in h3_tags:
            if tag.text.strip():
                titles.append(tag.text.strip())

        df_location = pd.DataFrame({"Titles": titles})

        return df_location
    else:
        print(
            "مشکلی در دریافت اطلاعات از وب‌سایت رخ داده است. کد وضعیت:",
            response.status_code,
        )


def date_scrape(addres: str):
    url = "https://www.6234.ir/"

    response = requests.get(url)

    if response.status_code == 200:

        soup = BeautifulSoup(response.text, "html.parser")

        h3_tags = soup.find_all("div", class_="theater-date my-2")

        titles = []

        for tag in h3_tags:
            if tag.text.strip():
                titles.append(tag.text.strip())

        df_date = pd.DataFrame({"Titles": titles})

        return df_date
    else:
        print(
            "مشکلی در دریافت اطلاعات از وب‌سایت رخ داده است. کد وضعیت:",
            response.status_code,
        )


def merge_df(df, df_location, df_date):
    merge_pd = pd.DataFrame(
        {
            "Titles": df["Titles"],
            "Location": df_location["Titles"],
            "Date": df_date["Titles"],
        }
    )

    return merge_pd
