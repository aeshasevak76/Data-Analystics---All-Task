import os, sys

# Force PySpark to use the current Python interpreter (venv)
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lower, regexp_replace, split, explode, count
from pyspark.sql.types import *
import pyspark.sql.functions as F

# 1. CREATE SPARK SESSION
spark = SparkSession.builder \
    .appName("BigData_Analysis_Project") \
    .getOrCreate()

print("\n Spark Session Started Successfully!\n")

# --------------------------------------------
# 2. CREATE SAMPLE DATA (NO CSV FILE NEEDED)
data = [
    ("Happy", "I love this product, it works great!"),
    ("Sad", "This is terrible, I'm very disappointed"),
    ("Neutral", "The item is okay, nothing special"),
    ("Happy", "Amazing experience, highly recommended"),
    ("Angry", "Worst purchase ever, total waste of money"),
    ("Happy", "Absolutely fantastic, exceeded expectations"),
    ("Neutral", "It does what it's supposed to do"),
    ("Sad", "Very disappointed with the quality"),
    ("Happy", "Best decision ever, love it so much"),
    ("Angry", "Complete disaster, never buying again"),
    ("Happy", "Wonderful product, very satisfied"),
    ("Neutral", "Average, could be better"),
    ("Sad", "Not what I expected at all"),
    ("Happy", "Excellent quality, highly impressed"),
    ("Angry", "Frustrating experience, poor service"),
]

schema = StructType([
    StructField("mood_type", StringType(), True),
    StructField("user_input", StringType(), True)
])

df = spark.createDataFrame(data, schema=schema)
print(" Sample Data Created Successfully!")
print(f"Total Rows: {df.count()}")
df.printSchema()

# --------------------------------------------
# 3. BASIC DATA EXPLORATION
print("\n SAMPLE DATA:")
df.show(10, truncate=False)

print("\n TOTAL ROWS:", df.count())
print("\n TOTAL COLUMNS:", len(df.columns))
print("\n COLUMN NAMES:", df.columns)

# --------------------------------------------
# 4. HANDLE MISSING VALUES
df_clean = df.na.drop(how="all")     # drop fully empty rows
df_clean = df_clean.fillna("Unknown")

print("\n Missing values handled successfully!")

# --------------------------------------------
# 5. BASIC STATISTICS (NUMERIC COLS)
numeric_cols = [c for c, t in df_clean.dtypes if t in ["int", "double", "float", "long"]]

if numeric_cols:
    print("\n BASIC STATISTICS:")
    df_clean.describe(numeric_cols).show()
else:
    print("\n⚠ No numeric columns found for statistics.")
# --------------------------------------------
# 6. GROUP-BY ANALYSIS
group_column = "mood_type"

if group_column in df_clean.columns:
    print("\n GROUP BY ANALYSIS:")
    df_clean.groupBy(group_column).count().orderBy("count", ascending=False).show()
else:
    print(f"\n⚠ Column '{group_column}' not found in dataset.")
# --------------------------------------------
# 7. TEXT ANALYSIS — MOST COMMON WORDS
text_column = "user_input"

if text_column in df_clean.columns:

    print("\n Extracting Top 20 Most Common Words...")

    text_df = (
        df_clean
        .withColumn(text_column, lower(col(text_column)))
        .withColumn(text_column, regexp_replace(text_column, "[^a-zA-Z ]", " "))
        .withColumn("word", explode(split(col(text_column), "\\s+")))
        .filter(col("word") != "")
    )
    top_words = (
        text_df.groupBy("word")
        .count()
        .orderBy("count", ascending=False)
        .limit(20)
    )
    print("\n TOP 20 WORDS:")
    top_words.show(truncate=False)
else:
    print(f"\n⚠ Column '{text_column}' not found for text analysis.")
# --------------------------------------------
# 8. SAVE OUTPUT TO FILES
df_clean.write.mode("overwrite").csv("clean_output_csv")
top_words.write.mode("overwrite").csv("top_words_output")
print("\n Output saved to folder:")
print(" - clean_output_csv/")
print(" - top_words_output/")
# --------------------------------------------
# 9. STOP SPARK SESSION
spark.stop()
print("\n Big Data Analysis Completed Successfully!")