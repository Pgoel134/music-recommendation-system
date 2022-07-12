import numpy as np
import pandas as pd
import recommenders as recommenders
import Recommenders as Recommenders

song_df_1 = pd.read_csv('triplets_file.csv')
print(song_df_1.head())

song_df_2 = pd.read_csv('song_data.csv')
print(song_df_2.head())

# combine both data
song_df = pd.merge(song_df_1, song_df_2.drop_duplicates(['song_id']), on='song_id', how='left')
print(song_df.head())

print(len(song_df_1), len(song_df_2))

print(len(song_df))

# creating new feature combining title and artist name
song_df['song'] = song_df['title']+' - '+song_df['artist_name']
print(song_df.head())

# taking top 10k samples for quick results
song_df = song_df.head(10000)

# cummulative sum of listen count of the songs
song_grouped = song_df.groupby(['song']).agg({'listen_count':'count'}).reset_index()
print(song_grouped.head())
grouped_sum = song_grouped['listen_count'].sum()
song_grouped['percentage'] = (song_grouped['listen_count'] / grouped_sum ) * 100
song_grouped.sort_values(['listen_count', 'song'], ascending=[0,1])

#1st recommendation by popularity of songs
pr = Recommenders.popularity_recommender_py()
pr.create(song_df, 'user_id', 'song')

# display the top 10 popular songs
print(pr.recommend(song_df['user_id'][5]))#1st user
print('\n')
print(pr.recommend(song_df['user_id'][100]))#2nd user
print('\n')

#recommendation through listening of similar songs
ir = Recommenders.item_similarity_recommender_py()
ir.create(song_df, 'user_id', 'song')
user_items = ir.get_user_items(song_df['user_id'][5])#history view

# display user songs history
for user_item in user_items:
    print(user_item)
# give song recommendation for that user
print(ir.recommend(song_df['user_id'][5]))
print('\n')

# give related songs based on the words
print(ir.get_similar_items(['Oliver James - Fleet Foxes', 'The End - Pearl Jam']))




