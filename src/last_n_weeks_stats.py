import pandas as pd 
import numpy as np
import requests 
import json
import os


class periodStats(object):

    def __init__(self, stats_df, empty_df, n):
        self.stats_df = stats_df
        self.empty_df = empty_df
        self.n = n

    def get_all_home_teams(self):
        self.home_teams = self.stats_df.groupby(['home_team', 'game_id', 'week', 'home_score', 'h_win', 'net_totfd', 'net_totyds', 'net_pyds', 'net_ryds', 'net_pen', 'net_penyds', 'net_trnovr', 'net_pt', 'net_ptyds', 'net_ptavg', 'net_top']).count()
        self.home_teams = self.home_teams.reset_index(drop=False)
        self.home_teams = self.home_teams.sort_values(by=['home_team', 'game_id'])
        self.home_teams = self.home_teams.rename(columns={"home_team": "team", "home_score": "score", 'net_totfd': 'totfd', 'net_totyds': 'totyds', 'net_pyds': 'pyds', 'net_ryds': 'ryds', 'net_pen': 'pen', 'net_penyds': 'penyds', 'net_trnovr': 'trnover', 'net_pt': 'pt', 'net_ptyds': 'ptyds', 'net_ptavg': 'ptavg', 'net_top': 'top'})
        self.home_teams = self.home_teams.drop(self.home_teams.columns[16:], axis =1)

    def get_all_away_teams(self):
        self.away_teams = self.stats_df.groupby(['away_team', 'game_id', 'week', 'away_score', 'h_win', 'net_totfd', 'net_totyds', 'net_pyds', 'net_ryds', 'net_pen', 'net_penyds', 'net_trnovr', 'net_pt', 'net_ptyds', 'net_ptavg', 'net_top']).count()
        self.away_teams = self.away_teams.reset_index(drop=False)
        self.away_teams = self.away_teams.sort_values(by=['away_team', 'game_id'])
        self.away_teams = self.away_teams.rename(columns={"away_team": "team", "away_score": "score", 'net_totfd': 'totfd', 'net_totyds': 'totyds', 'net_pyds': 'pyds', 'net_ryds': 'ryds', 'net_pen': 'pen', 'net_penyds': 'penyds', 'net_trnovr': 'trnover', 'net_pt': 'pt', 'net_ptyds': 'ptyds', 'net_ptavg': 'ptavg', 'net_top': 'top'})
        self.away_teams = self.away_teams.drop(self.away_teams.columns[16:], axis=1)

    def get_all_teams(self):
        frames = [self.home_teams, self.away_teams]
        self.all_teams = pd.concat(frames)
        self.all_teams = self.all_teams.sort_values(by=['team', 'game_id']).reset_index()
        self.all_teams = self.all_teams.drop(self.all_teams.columns[0], axis=1)

    def last_n_weeks(self, h_team, a_team, game_id):
        ## getting dataframes for home and away team
        h_df = self.all_teams[self.all_teams['team'] == h_team]
        a_df = self.all_teams[self.all_teams['team'] == a_team]
        
        ## specifying the index for the last n games
        h_end_index = h_df[h_df['game_id'] == game_id].index
        h_start_index = h_end_index - self.n
        a_end_index = a_df[a_df['game_id'] == game_id].index
        a_start_index = a_end_index - self.n
        
        ## creating dataframes for the last n games for home and away
        h_period_df = h_df[(h_df.index >= h_start_index[0]) & (h_df.index < h_end_index[0])]
        a_period_df = a_df[(a_df.index >= a_start_index[0]) & (a_df.index < a_end_index[0])]
        
        ## creating a one row df for home and away -> the mean of each stat
        h_agg_df = h_period_df.agg("mean").to_frame().transpose()
        a_agg_df = a_period_df.agg("mean").to_frame().transpose()
        
        ## cleaning the aggregate frames
        h_agg_df = h_agg_df.drop(h_agg_df.columns[[0, 1, 3]], axis =1)
        a_agg_df = a_agg_df.drop(a_agg_df.columns[[0, 1, 3]], axis =1)
        
        h_agg_df['h_team'] = h_df['team'].unique()[0]
        a_agg_df['a_team'] = a_df['team'].unique()[0]
        
        h_agg_df = h_agg_df.set_index('h_team')
        a_agg_df = a_agg_df.set_index('a_team')
        
        h_agg_df = h_agg_df.rename(columns={h_agg_df.columns[i]: 'h_'+ h_agg_df.columns[i] for i in range(len(h_agg_df.columns))})
        a_agg_df = a_agg_df.rename(columns={a_agg_df.columns[i]: 'a_'+ a_agg_df.columns[i] for i in range(len(h_agg_df.columns))})
        
        h_agg_df = h_agg_df.reset_index()
        a_agg_df = a_agg_df.reset_index()
        
        game_agg_df = h_agg_df.join(a_agg_df)
        # print(h_agg_df)
        for i in range(1, 13):
            game_agg_df['lastN_net_' + game_agg_df.columns[i].replace('h_', '')] = game_agg_df.iloc[:, i] - game_agg_df.iloc[:, i+13]
        return game_agg_df

    def add_period(self):
        for idx, row in self.empty_df.iterrows():
            print(idx)
            game_agg_df = self.last_n_weeks(row['home_team'], row['away_team'], row['game_id'])
            for col in game_agg_df.columns[27:]:
                self.empty_df.at[idx, col] = game_agg_df.loc[:, col][0]

    

if __name__ == '__main__':
    df = pd.read_csv('data/net_data_1718.csv')
    df = df.drop(df.columns[0], axis=1)


    empty_df = pd.read_csv('data/reg_games_stats_2018.csv')
    empty_df = empty_df.drop(empty_df.columns[0], axis=1)
    empty_df = empty_df.drop(empty_df.columns[6:7], axis=1)
    empty_df = empty_df.drop(empty_df.columns[10:], axis=1)


    stats_18 = periodStats(df, empty_df, 2)
    stats_18.get_all_home_teams()
    stats_18.get_all_away_teams()
    stats_18.get_all_teams()
    stats_18.add_period()

    stats_18.empty_df.to_csv('data/period_stats_18_2_net.csv')