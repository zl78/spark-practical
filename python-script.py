from pyspark.sql import SparkSession
from pyspark.sql import functions as f
import math
import sys
import pandas as pd

spark = SparkSession.builder.appName("CS5052-Part1").getOrCreate()

ratings_path = sys.argv[1]

movies_path = sys.argv[2]

ratings = (
    spark.read.csv(
        path=ratings_path,
        sep=",",
        header=True,
        schema="userId INT, movieId INT, rating DOUBLE, timestamp INT",
    )
)

movies = (
    spark.read.csv(
        path=movies_path,
        sep=",",
        header=True,
        schema="movieId INT, title STRING, genres STRING",
    )
)

# Q2
def storeDataset(path_to_store):
    part1q2 = ratings.join(movies, ["movieId"], "left")
    part1q2.write.save(path_to_store + "csvFile.parquet", format="parquet")
    
# Q3    
def searchUserForMoviesAndGenre(userId):
    part1_question3 = ratings\
        .join(movies, ["movieId"])\
        .groupBy("userId")\
        .agg(
            f.count("*").alias("movie watched")
        )\
        .filter("userId = " + str(userId))
    part1_question3.show()

    part1_question3 = ratings\
        .join(movies, ["movieId"])\
        .withColumn("genres_array", f.split("genres", "\|"))\
        .withColumn("genre", f.explode("genres_array"))\
        .groupBy("userId")\
        .agg(
            f.count("*").alias("genre watched")
        )\
        .filter("userId = " + str(userId))
    part1_question3.show()

# Q4
def searchMovie(idTitle):
    part1_question4 = ratings\
        .join(movies, ["movieId"])\
        .withColumn("name", f.expr("substring(title, 1, length(title)-7)"))
    if is_number(idTitle):
        part1_question4\
            .groupBy("movieId")\
            .agg(
                f.count("*").alias("user_watched"),
                f.avg("rating")
            )\
            .filter("movieId = " + str(idTitle)).show()
    else:
        part1_question4\
            .groupBy("name")\
            .agg(
                f.count("*").alias("user_watched"),
                f.avg("rating")
            )\
            .filter(f.col("name") == idTitle).show()

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass

    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass

    return False

# Q5
def searchGenre(genreToSearch):
    part1_q5 = movies\
            .withColumn("genres_array", f.split("genres", "\|"))\
            # .withColumn("genre", f.explode("genres_array"))
    # for x in genreToSearch:
    part1_q5 = part1_q5.filter(f.array_contains(part1_q5.genres_array, genreToSearch))
    part1_q5.show(part1_q5.count(), False)

# Q6
def searchMoviesByYear(year):
    part1_question6 = movies\
            .withColumn("year", f.substring(f.col("title"), -5, 4))\
            .filter("year = " + str(year))
    part1_question6.show(part1_question6.count(), False)

# Q7
def topN_MoviesHighestRating(nMovies):
    part1Q7 = ratings\
        .join(movies, ["movieId"])\
        .groupBy("movieId", "title")\
        .agg(
            f.avg("rating").alias("average rating")
        )\
        .sort(f.col("average rating").desc())
    part1Q7.show(int(nMovies))

# Q8
def topN_MoviesHightestWatches(nMovies):
    part1_question8 = ratings\
        .join(movies, ["movieId"])\
        .groupBy("movieId","title")\
        .agg(
            f.count("*").alias("user_watched")
        )\
        .sort(f.col("user_watched").desc())
    part1_question8.show(int(nMovies))

# Part2 Q1
def findFavouriteGenreForUser(userId):
    part2q1 = movies\
            .withColumn("genres_array", f.split("genres", "\|"))\
            .withColumn("genre", f.explode("genres_array"))
    part2q1 = part2q1\
        .join(ratings, ["movieId"])\
        .filter(f.col("userId") == userId)\
        .groupBy("genre")\
        .agg(
            f.avg("rating").alias("averageRating")
        )\
        .sort(f.col("averageRating").desc())
    part2q1.show(1)

# Part2 Q2
def movieTastesCal():
    ratings_dict = {}
    for userId, movieId, rating in watchs_df.select('userId', 'movieId', 'rating').rdd.map(
            lambda x: (x.userId, x.movieId, x.rating)).collect():
        if userId not in ratings_dict.keys():
            ratings_dict[userId] = {}
        ratings_dict[userId][movieId] = int(rating)

def pearson_dis(rating1, rating2):
    """
    Calculate the pearson distance between two ratings
    """
    sum_xy = 0
    sum_x = 0
    sum_y = 0
    sum_x2 = 0
    sum_y2 = 0
    n = 0
    for key in rating1:
        if key in rating2:
            n += 1
            x = rating1[key]
            y = rating2[key]
            sum_xy += x * y
            sum_x += x
            sum_y += y
            sum_x2 += x ** 2
            sum_y2 += y ** 2
            
    if n == 0:
        return 0
    denominator = sqrt(sum_x2 - (sum_x ** 2 / n)) * sqrt(sum_y2 - (sum_y ** 2 / n))
    if denominator == 0:
        return 0
    else:
        return (sum_xy - (sum_x * sum_y) / n) / denominator

# Part 3
def computerNearestNeighbor(userid, users):
    """
    Give the userid, calculate the distance between other users and it, and sort them.
    """
    distances = []
    for user in users:
        if user != userid:
            distance = pearson_dis(users[user], users[userid])
            distances.append((distance, user))
            
    # Sort by distance 
    distances.sort()
    return distances

#  nearest/top K neighbors
def recommend(userid, users):
    
    nearest = computerNearestNeighbor(userid, users)[0][1]
    recommendations = []
    # Find movies that the nearest neighbor has watched but the user hasn’t watched, and calculate recommendations
    neighborRatings = users[nearest]
    userRatings = users[userid]
    for artist in neighborRatings:
        if not artist in userRatings:
            recommendations.append((artist, neighborRatings[artist]))
    results = sorted(recommendations, key=lambda artistTuple: artistTuple[1], reverse=True)
    df = {"UserID": [], "MovieID": [], "Rating": []}
    for result in results:
        df["UserID"].append(userid)
        df["MovieID"].append(result[0])
        df["Rating"].append(result[1])
    return pd.DataFrame(df)

def make_df(users):
    df = pd.DataFrame({"UserID":[],"MovieID":[],"Rating":[]})
    for user in users:
        df = pd.concat([df,recommend(user,users)],ignore_index=True)
    return df


print("Core features")
print("------Part 1------")
print("1: Store the dataset")
print("2: Search user by id, show the number of movies/genre that he/she has watched")
print("3: Search movie by id/title, show the average rating, the number of users that have watched the movie")
print("4: Search genre, show all movies in that genre")
print("5: Search movies by year")
print("6: List the top n movies with highest rating, ordered by the rating")
print("7: List the top n movies with the highest number of watches, ordered by the number of watches")
print("------Part 2------")
print("8: Find the favourite genre of a given user")
print("9: Compare the movie tastes of two users")
print("------Part 3------")
print("10: ")

inputNum = int(sys.argv[3])
if inputNum == 2:
    path = sys.argv[4]
    storeDataset(path)
elif inputNum == 3:
    userId = int(sys.argv[4])
    searchUserForMoviesAndGenre(userId)
elif inputNum == 4:
    idOrTitle = sys.argv[4]
    searchMovie(idOrTitle)
elif inputNum == 5:
    genreToSearch = sys.argv[4]
    searchGenre(genreToSearch)
elif inputNum == 6:
    movieByYear = sys.argv[4]
    searchMoviesByYear(movieByYear)
elif inputNum == 7:
    topNMoviesHighestRating = sys.argv[4]
    topN_MoviesHighestRating(int(topNMoviesHighestRating))
elif inputNum == 8:
    topNMoviesHightestWatchers = sys.argv[4]
    topN_MoviesHightestWatches(int(topNMoviesHightestWatchers))
elif inputNum == 9:
    userId = sys.argv[4]
    findFavouriteGenreForUser(int(userId))
elif inputNum == 10:
    userID1 = int(sys.argv[4])
    userID2 = int(sys.argv[5])
    ratings_dict = {}
    for userId, movieId, rating in watchs_df.select('userId', 'movieId', 'rating').rdd.map(
            lambda x: (x.userId, x.movieId, x.rating)).collect():
        if userId not in ratings_dict.keys():
            ratings_dict[userId] = {}
        ratings_dict[userId][movieId] = int(rating)
    print(pearson_dis(ratings_dict[userID1],ratings_dict[userID2]))
elif inputNum == 11:
    watchs_df = ratings.join(movies, ["movieId"], "left")
    ratings_dict = {}
    for userId, movieId, rating in watchs_df.select('userId', 'movieId', 'rating').rdd.map(
            lambda x: (x.userId, x.movieId, x.rating)).collect():
        if userId not in ratings_dict.keys():
            ratings_dict[userId] = {}
        ratings_dict[userId][movieId] = int(rating)
    df = make_df(ratings_dict)
    df.head()
    