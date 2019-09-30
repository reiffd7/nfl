import pandas as pd 
import numpy as np
import requests 
import json
import os


class combineYears(object):

    def __init__(self, df1, df2, p):
        self.df1 = df1
        self.df2 = df2
        self.p = p

    def combine_years(self):
        frames = [self.df1, self.df2]
        self.df3 = pd.concat(frames)
        self.df3 = self.df3.drop(self.df3.columns[0], axis=1).reset_index(drop=True)

    def add_nets(self):
        for item in self.df3.columns[10::2]:
            stat_term = item.replace('h_', '')
            self.df3['net_' + stat_term] = self.df3[item] - self.df3['a_'+stat_term]
       
  
    

if __name__ == '__main__':
    stats17 = pd.read_csv('data/stats_17.csv')
    stats18 = pd.read_csv('data/stats_18.csv')

    data18 = combineYears(stats17, stats18, 6)
    data18.combine_years()
    data18.add_nets()
    data18.df3.to_csv('data/net_data_1718.csv')