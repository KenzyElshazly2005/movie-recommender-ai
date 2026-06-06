import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics.pairwise import cosine_similarity

# =========================
# 1. Load Data
# =========================

print("\n📦 Loading datasets...\n")

movies = pd.read_csv(r"movies.csv")
ratings = pd.read_csv(r"ratings.csv")

print("✔ Data Loaded Successfully!")

# =========================
# 2. Explore Data
# =========================

print("\n🎬 Movies Sample:")
print(movies.head())

print("\n⭐ Ratings Sample:")
print(ratings.head())

print("\n📌 Movies Info:")
print(movies.info())

print("\n📌 Ratings Info:")
print(ratings.info())

print("\n📊 Missing Values:")
print(movies.isnull().sum())
print(ratings.isnull().sum())

# =========================
# 3. Merge Data
# =========================

data = pd.merge(ratings, movies, on="movieId")

# =========================
# 4. Movie Statistics
# =========================

movie_stats = data.groupby("title").agg(
    average_rating=("rating", "mean"),
    number_of_ratings=("rating", "count")
)

# =========================
# 5. Visualizations
# =========================

print("\n📊 Creating Visualizations...")

# Chart 1
plt.figure(figsize=(10,6))

sns.countplot(x="rating", data=ratings)

plt.title("Ratings Distribution")
plt.xlabel("Rating")
plt.ylabel("Count")

plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# Chart 2
top_movies_count = ratings["movieId"].value_counts().head(10)

plt.figure(figsize=(12,6))

sns.barplot(
    x=top_movies_count.index.astype(str),
    y=top_movies_count.values
)

plt.title("Top 10 Most Rated Movies")
plt.xlabel("Movie ID")
plt.ylabel("Number of Ratings")

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Chart 3
top_rated = data.groupby("title")["rating"].mean()
top_rated = top_rated.sort_values(ascending=False).head(10)

plt.figure(figsize=(12,7))

sns.barplot(
    x=top_rated.values,
    y=top_rated.index
)

plt.title("Top 10 Highest Rated Movies")
plt.xlabel("Average Rating")
plt.ylabel("Movie Title")

plt.yticks(fontsize=10)
plt.tight_layout()
plt.show()

# Chart 4
movies["genres"] = movies["genres"].fillna("")

all_genres = movies["genres"].str.split("|").explode()

plt.figure(figsize=(12,7))

sns.countplot(
    y=all_genres,
    order=all_genres.value_counts().index
)

plt.title("Genre Distribution")
plt.xlabel("Count")
plt.ylabel("Genre")

plt.yticks(fontsize=10)
plt.tight_layout()
plt.show()

# Chart 5
genre_data = data.copy()
genre_data["genres"] = genre_data["genres"].str.split("|")
genre_data = genre_data.explode("genres")

genre_ratings = genre_data.groupby("genres")["rating"].mean().sort_values(ascending=False)

plt.figure(figsize=(12,7))

sns.barplot(
    x=genre_ratings.values,
    y=genre_ratings.index
)

plt.title("Average Rating Per Genre")
plt.xlabel("Average Rating")
plt.ylabel("Genre")

plt.yticks(fontsize=10)
plt.tight_layout()
plt.show()

# Chart 6
genre_count = all_genres.value_counts()

plt.figure(figsize=(12,7))

sns.barplot(
    x=genre_count.values,
    y=genre_count.index
)

plt.title("Number of Movies Per Genre")
plt.xlabel("Movies Count")
plt.ylabel("Genre")

plt.yticks(fontsize=10)
plt.tight_layout()
plt.show()

# =========================
# 6. Recommendation Model
# =========================

print("\n⚙ Building Recommendation Model...")

movie_matrix = data.pivot_table(
    index="movieId",
    columns="userId",
    values="rating"
)

movie_matrix = movie_matrix.fillna(0)

similarity = cosine_similarity(movie_matrix)

print("✔ Similarity Matrix Ready!")

# =========================
# 7. Search Function
# =========================

def search_movie(movie_name):

    result = movies[
        movies["title"].str.contains(
            movie_name,
            case=False,
            na=False,
            regex=False
        )
    ]

    return result

# =========================
# 8. Recommendation Function
# =========================

def recommend(movie_id, n=5):

    idx = list(movie_matrix.index).index(movie_id)

    scores = list(
        enumerate(similarity[idx])
    )

    scores = sorted(
        scores,
        key=lambda x: x[1],
        reverse=True
    )

    recommendations = []

    for score in scores[1:n+1]:

        recommended_movie_id = (
            movie_matrix.index[score[0]]
        )

        title = movies[
            movies["movieId"]
            == recommended_movie_id
        ]["title"].values[0]

        recommendations.append(title)

    return recommendations

# =========================
# 9. Top Movies Function
# =========================

def show_top_movies():

    print("\n🏆 TOP RATED MOVIES\n")

    top_movies = movie_stats[
        movie_stats["number_of_ratings"] >= 50
    ]

    top_movies = top_movies.sort_values(
        by="average_rating",
        ascending=False
    )

    print(top_movies.head(10))

# =========================
# 10. Main Program
# =========================

while True:

    print("\n" + "=" * 45)
    print("🎬 MOVIE RECOMMENDATION SYSTEM")
    print("=" * 45)

    print("\n1. Search Movie")
    print("2. Show Top Rated Movies")
    print("3. Exit")

    choice = input(
        "\nChoose option (1/2/3 or Search/Top/Exit): "
    ).strip().lower()

    # =====================

    if choice in ["1", "search"]:

        movie_name = input(
            "\nEnter movie name: "
        )

        results = search_movie(
            movie_name
        )

        if results.empty:

            print("\n❌ Movie not found!")
            continue

        print("\n🎬 Matching Movies:\n")

        for i, row in enumerate(
            results.head(10).itertuples(),
            start=1
        ):
            print(f"{i}. {row.title}")

        selected = input(
            "\nChoose movie number: "
        )

        try:

            selected = int(selected)

            movie = results.iloc[
                selected - 1
            ]

        except:

            print(
                "\n❌ Invalid Selection!"
            )

            continue

        title = movie["title"]
        genres = movie["genres"]
        movie_id = movie["movieId"]

        print("\n🎬 MOVIE DETAILS")
        print("-" * 40)

        print(f"Title : {title}")
        print(f"Genres: {genres}")

        if title in movie_stats.index:

            avg_rating = round(
                movie_stats.loc[
                    title
                ]["average_rating"],
                2
            )

            total_ratings = int(
                movie_stats.loc[
                    title
                ]["number_of_ratings"]
            )

            print(
                f"⭐ Average Rating : {avg_rating}"
            )

            print(
                f"👥 Number of Ratings : {total_ratings}"
            )

        answer = input(
            "\nDo you want recommendations? (yes/no): "
        ).strip().lower()

        if answer in ["yes", "y"]:

            recommendations = recommend(
                movie_id
            )

            print(
                "\n🎯 Recommended Movies\n"
            )

            for i, rec in enumerate(
                recommendations,
                start=1
            ):

                if rec in movie_stats.index:

                    rating = round(
                        movie_stats.loc[rec][
                            "average_rating"
                        ],
                        2
                    )

                    print(
                        f"{i}. {rec} ⭐ {rating}"
                    )

                else:

                    print(
                        f"{i}. {rec}"
                    )

    # =====================

    elif choice in ["2", "top"]:

        show_top_movies()

    # =====================

    elif choice in ["3", "exit"]:

        print("\n👋 Goodbye!")
        break

    # =====================

    else:

        print("\n❌ Invalid Choice!")

# =========================
# End Project
# =========================

print("\n🎉 Project Completed Successfully!")
