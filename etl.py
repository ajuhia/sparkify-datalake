import configparser
from datetime import datetime
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col
from pyspark.sql.functions import year, month, dayofmonth, hour, weekofyear, date_format
from pyspark.sql import functions as F

config = configparser.ConfigParser()
config.read('dl.cfg')

os.environ['AWS_ACCESS_KEY_ID']=config['AWS']['AWS_ACCESS_KEY_ID']
os.environ['AWS_SECRET_ACCESS_KEY']=config['AWS']['AWS_SECRET_ACCESS_KEY']

def create_spark_session():
    """
    This function creates spark session.
    """
    spark = SparkSession \
        .builder \
        .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:2.7.0") \
        .getOrCreate()

    return spark


def process_song_data(spark, input_data, output_data):
    """
    This function performs following tasks:
    1. Reads song data from song dataset in s3, creates songs table and writes songs table to parquet files partitioned by year and artist.
    2. Creates artists table using above song data and writes artists table to parquet files.
    """
    # get filepath to song data file
    song_data = os.path.join(input_data, 'song_data/*/*/*/*.json')
    
    # read song data file
    df = spark.read.json(song_data)

    # extract columns to create songs table
    songs_table = df.select(['song_id','title','artist_id','year','duration'])
    
    # write songs table to parquet files partitioned by year and artist
    songs_table.write.mode("overwrite").parquet(os.path.join(output_data, 'songs'), partitionBy=['year', 'artist_id'])

    # extract columns to create artists table
    artists_table = df.select(['artist_id', 'artist_name', 'artist_location', 'artist_latitude', 'artist_longitude'])
    
    # write artists table to parquet files
    artists_table.write.mode("overwrite").parquet(os.path.join(output_data, 'artists'))


def process_log_data(spark, input_data, output_data):
    """
    This function performs following tasks:
    Reads log data from log dataset in s3, based on action: 'NextSong',  creates users,time, and songplays table and  writes these tables to paraquet files. Songsplays table is partitioned partitioned by year and artist.
    """
    # get filepath to log data file
    log_data = os.path.join(input_data, 'log_data/*/*/*.json')

    # read log data file
    df = spark.read.json(log_data)
    
    # filter by actions for song plays
    df = df.where(df.page == 'NextSong')

    # extract columns for users table    
    users_table = df.select(['userId', 'firstName', 'lastName', 'gender', 'level'])
    users_table = users_table.drop_duplicates(subset=['userId'])
    
    # write users table to parquet files
    users_table.write.mode("overwrite").parquet(os.path.join(output_data, 'users'))

    # create timestamp column from original timestamp column
    get_timestamp = udf(lambda x: datetime.fromtimestamp(x/1000).strftime('%Y-%m-%d'))
    df = df.withColumn('timestamp', get_timestamp('ts'))
    
    # create datetime column from original timestamp column
    get_datetime = udf(lambda x: str(datetime.fromtimestamp(int(x) / 1000)))
    df = df.withColumn('datetime', get_datetime(df.ts))
    
    
    # extract columns to create time table
    time_table = df.select(
                 F.col("timestamp").alias("start_time"),
                 F.hour("timestamp").alias('hour'),
                 F.dayofmonth("timestamp").alias('day'),
                 F.weekofyear("timestamp").alias('week'),
                 F.month("timestamp").alias('month'), 
                 F.year("timestamp").alias('year'), 
                 F.date_format(F.col("timestamp"), "E").alias("weekday")
            )
    
    # write time table to parquet files partitioned by year and month
    time_table.write.mode("overwrite").parquet(os.path.join(output_data, 'time'), partitionBy=['year', 'month'])
    
    # read in song data to use for songplays table
    song_df = spark.read.json(input_data + 'song_data/*/*/*/*.json')

    # extract columns from joined song and log datasets to create songplays table 
    df = df.alias('log_df')
    song_df    = song_df.alias('song_df')
    new_df  = df.join(song_df, col('log_df.artist') == col(
        'song_df.artist_name'), 'inner')
    
    songplays_table = new_df.select(
            col('log_df.datetime').alias('start_time'),
            col('log_df.userId').alias('user_id'),
            col('log_df.level').alias('level'),
            col('song_df.song_id').alias('song_id'),
            col('song_df.artist_id').alias('artist_id'),
            col('log_df.sessionId').alias('session_id'),
            col('log_df.location').alias('location'), 
            col('log_df.userAgent').alias('user_agent'),
            year('log_df.datetime').alias('year'),
            month('log_df.datetime').alias('month')) \
            .withColumn('songplay_id', F.monotonically_increasing_id())

    # write songplays table to parquet files partitioned by year and month
    songplays_table.write.mode("overwrite").parquet(os.path.join(output_data, 'songplays'), partitionBy=['year', 'month'])


def main():
    """
    This function:
    1. Creates a spark session.
    2. Calls process_song_data() and process_log_data() to read the song and log data from s3 and transform this data to create fact and dimension tables, which will then be written to parquet files.Finally parquet files will be loaded back to s3.
    """
    print("ETL process started!")
    spark = create_spark_session()
    
    input_data = "s3a://udacity-dend/"
    output_data = "s3://juhi-sparkify/"
    #output_data = "data/output/" #for testing only
    
    
    process_song_data(spark, input_data, output_data)    
    process_log_data(spark, input_data, output_data)

    print("ETL process completed!")

if __name__ == "__main__":
    main()
