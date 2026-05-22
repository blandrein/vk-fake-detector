import vk_api
import re
import numpy as np


class VKExtractor:

    def __init__(self, token):

        self.vk = vk_api.VkApi(
            token=token
        ).get_api()

    # =========================
    # GET USER
    # =========================
    def get_user(self, user_id):

        return self.vk.users.get(

            user_ids=user_id,

            fields=(

                "sex,site,nickname,about,interests,"
                "books,movies,music,quotes,tv,"
                "career,military,universities,"
                "schools,relatives,home_town,"
                "games,activities"

            )

        )[0]

    # =========================
    # GET POSTS
    # =========================
    def get_posts(self, user_id, count=100):

        try:

            return self.vk.wall.get(

                owner_id=user_id,
                count=count

            )["items"]


        except Exception:

            return []

    # =========================
    # ANALYZE POSTS
    # =========================
    def analyze_posts(self, posts):

        if not posts:

            return {

                "posts_count": 0,

                "avg_likes": 0,
                "avg_comments": 0,
                "avg_views": 0,

                "avg_text_length": 0,

                "links_ratio": 0,
                "hashtags_ratio": 0,
                "attachments_ratio": 0,
                "reposts_ratio": 0,

                "ads_ratio": 0,

                "posting_frequency_days": 0,

                "phone_numbers_ratio": 0,

                "avg_text_uniqueness": 0
            }

        total_likes = 0
        total_comments = 0
        total_views = 0

        links = 0
        hashtags = 0
        attachments = 0
        reposts = 0
        ads = 0
        phones = 0

        texts = []
        lengths = []

        timestamps = []

        for post in posts:

            text = post.get("text", "")

            texts.append(text)

            lengths.append(len(text))

            total_likes += post.get(
                "likes",
                {}
            ).get("count", 0)

            total_comments += post.get(
                "comments",
                {}
            ).get("count", 0)

            total_views += post.get(
                "views",
                {}
            ).get("count", 0)

            # LINKS
            if re.search(
                r"https?://",
                text
            ):
                links += 1

            # HASHTAGS
            if re.search(
                r"#\w+",
                text
            ):
                hashtags += 1

            # ATTACHMENTS
            if post.get("attachments"):
                attachments += 1

            # REPOSTS
            if "copy_history" in post:
                reposts += 1

            # PHONE NUMBERS
            if re.search(
                r"\+?\d[\d\s\-()]{7,}",
                text
            ):
                phones += 1

            # ADS
            if re.search(
                r"(реклама|bitcoin|crypto|ставк|заработ)",
                text.lower()
            ):
                ads += 1

            timestamps.append(
                post.get("date", 0)
            )

        n = len(posts)

        # =========================
        # POSTING FREQUENCY
        # =========================

        posting_frequency = 0

        if len(timestamps) > 1:

            timestamps.sort(reverse=True)

            diffs = [

                (timestamps[i] - timestamps[i + 1]) / 86400

                for i in range(
                    len(timestamps) - 1
                )
            ]

            posting_frequency = float(
                np.mean(diffs)
            )

        # =========================
        # UNIQUENESS
        # =========================

        unique_ratio = (

            len(set(texts)) / n

            if n else 0
        )

        return {

            "posts_count":
                n,

            "avg_likes":
                total_likes / n,

            "avg_comments":
                total_comments / n,

            "avg_views":
                total_views / n,

            "avg_text_length":
                float(np.mean(lengths)),

            "links_ratio":
                links / n,

            "hashtags_ratio":
                hashtags / n,

            "attachments_ratio":
                attachments / n,

            "reposts_ratio":
                reposts / n,

            "ads_ratio":
                ads / n,

            "posting_frequency_days":
                posting_frequency,

            "phone_numbers_ratio":
                phones / n,

            "avg_text_uniqueness":
                unique_ratio
        }

    # =========================
    # PROFILE FEATURES
    # =========================
    def extract_profile(self, u):

        return {

            "has_website":
                int(bool(u.get("site"))),

            "gender":
                u.get("sex", 0),

            "has_nickname":
                int(bool(u.get("nickname"))),

            "has_maiden_name":
                int(bool(u.get("maiden_name"))),

            "has_interests":
                int(bool(u.get("interests"))),

            "has_books":
                int(bool(u.get("books"))),

            "has_tv":
                int(bool(u.get("tv"))),

            "has_quotes":
                int(bool(u.get("quotes"))),

            "has_about":
                int(bool(u.get("about"))),

            "has_games":
                int(bool(u.get("games"))),

            "has_movies":
                int(bool(u.get("movies"))),

            "has_activities":
                int(bool(u.get("activities"))),

            "has_music":
                int(bool(u.get("music"))),

            "has_career":
                int(bool(u.get("career"))),

            "has_military_service":
                int(bool(u.get("military"))),

            "has_hometown":
                int(bool(u.get("home_town"))),

            "has_universities":
                int(bool(u.get("universities"))),

            "has_schools":
                int(bool(u.get("schools"))),

            "has_relatives":
                int(bool(u.get("relatives")))
        }

    # =========================
    # FULL EXTRACT
    # =========================
    def extract(self, user_id):

        user = self.get_user(user_id)

        posts = self.get_posts(user_id)

        profile = self.extract_profile(user)

        behavior = self.analyze_posts(posts)

        return {

            **profile,
            **behavior
        }


# =========================
# TEST
# =========================
if __name__ == "__main__":

    TOKEN = "vk1.a.Za-RRBVEk6DPMR5S_epMNERi70IPdc8-wmyKlrCTB0k6vX4J0O2mSaTYw-jn4Gbilb_g0wNDADx4NDV5YCii0mQWkO9mtbTVXjuNqFIValPxOdZfBFSPqFJbcKRNWZ7ffd4HAf0DpoHFXV7qZCrOa8d6uZ7YAA_pI2a0v8vPgAuucsfaT2ynbuvVE3MgD8vwNAjg3Cs4-JsnVrg9jtC_sA"

    extractor = VKExtractor(TOKEN)

    user_id = input("VK user id: ")

    features = extractor.extract(user_id)

    print("\n=== FEATURES ===\n")

    print("count:", len(features))

    for k, v in features.items():

        print(f"{k}: {v}")