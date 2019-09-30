import pandas as pd 
import numpy as np
import requests 
import json
import os

class nflGameStats (object):

    def __init__(self, df):
        self.df = df

    

    def stats(self, url, key):
        r = requests.get(url).text
        json_obj = json.loads(r)
        h_stat = json_obj[key]['home']['stats']['team']
        a_stat = json_obj[key]['away']['stats']['team']
        # h_stat = np.array(list(h_stat.items()))
        # a_stat = np.array(list(a_stat.items()))
        return h_stat, a_stat
    
    # def away_stats(self, json_obj, key):
    #     a_stat = json_obj[key]['away']['stats']['team']
    #     return np.array(list(a_stat.items()))

    def h_winner(self, h_score, a_score):
        if h_score > a_score:
            return 1
        else:
            return 0

    def a_winner(self, h_score, a_score):
        if a_score > h_score:
            return 1
        else:
            return 0

    def clean_TOP(self):
        self.df.h_top = self.df.h_top.apply(lambda x: sum([a*b for a,b in zip([60, 1], map(int,x.split(':')))])/60)
        self.df.a_top = self.df.a_top.apply(lambda x: sum([a*b for a,b in zip([60, 1], map(int,x.split(':')))])/60)


    def operator(self):
       self.df['h_win'] = self.df.apply(lambda x: self.h_winner(x['home_score'], x['away_score']), axis=1)
       self.df['a_win'] = self.df.apply(lambda x: self.a_winner(x['home_score'], x['away_score']), axis=1)
       stat_cats = ['totfd', 'totyds', 'pyds', 'ryds', 'pen', 'penyds', 'trnovr', 'pt',
       'ptyds', 'ptavg', 'top']
       for stat_cat in stat_cats:
           self.df['h_' + stat_cat] = ""
           self.df['a_' + stat_cat] = ""
       print(self.df)
       for idx, row in self.df.iterrows():
            print(idx)
            for stat_cat in stat_cats:
                print(stat_cat)
                stats = self.stats(row['game_url'], str(row['game_id']))
                self.df.at[idx, 'h_' + stat_cat] = stats[0][stat_cat]
                self.df.at[idx, 'a_' + stat_cat] = stats[1][stat_cat]




if __name__ == '__main__':
    # DATA_DIRECTORY = os.path.join(os.path.split(os.getcwd())[0], 'data')
    # df18 = pd.read_csv('nflscrapR-data/games_data/regular_season/reg_games_2018.csv')
    # df17 = pd.read_csv('nflscrapR-data/games_data/regular_season/reg_games_2017.csv')
    
    # stats18 = nflGameStats(df18)
    # stats18.operator()
    # stats18.clean_TOP()
    # stats18.df.to_csv('data/stats_18.csv')

    # stats17 = nflGameStats(df17)
    # stats17.operator()
    # stats17.clean_TOP()
    # stats17.df.to_csv('data/stats_17.csv')

    # stats17 = pd.read_csv('data/stats_17.csv')
    # stats18 = pd.read_csv('data/stats_18.csv')