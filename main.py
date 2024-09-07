import pandas as pd
import random
import time
import requests
from bs4 import BeautifulSoup
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import pickle


from flask import *
from flask_bootstrap import Bootstrap
from flask_pymongo import PyMongo

import pandas as pd
pd.options.mode.chained_assignment = None

app = Flask(__name__)

app.secret_key = 'your_secret_key'

Bootstrap(app)
api_key = 'AIzaSyCqJBDynKiv3iPjc1q_S2JAbXkfBBkGi74'

fantasy_data = pd.read_csv("FantasyData21-23.csv")
fantasy_data_2023 = pd.read_csv("FullFantasyData2023.csv")

with open('model_qb.pkl', 'rb') as file:
    model_qb = pickle.load(file)
with open('model_rb.pkl', 'rb') as file:
    model_rb = pickle.load(file)
with open('model_wr.pkl', 'rb') as file:
    model_wr = pickle.load(file)

rookie_data = pd.read_csv("NFLRookieData.csv")
fantasy_positions = ['QB', 'RB', 'FB', 'WR', 'TE']
rookie_data = rookie_data[rookie_data['position'].isin(fantasy_positions)]

schedules = pd.read_csv("Schedules.csv")

win_percentage_data = pd.read_csv("WinPercentageData.csv")
win_percentage_data = win_percentage_data.replace(r'[^\w\s]|_', '', regex=True)

def update_opponent(opponent):
    if opponent[0] != '@':
        return 'v' + opponent
    return opponent

for col in schedules.columns[1:]:
    schedules[col] = schedules[col].apply(update_opponent)

complete_team_abbreviation_mapping = {
    "Cardinals": "ARI",
    "Falcons": "ATL",
    "Ravens": "BAL",
    "Bills": "BUF",
    "Panthers": "CAR",
    "Bears": "CHI",
    "Bengals": "CIN",
    "Browns": "CLE",
    "Cowboys": "DAL",
    "Broncos": "DEN",
    "Lions": "DET",
    "Packers": "GB",
    "Texans": "HOU",
    "Colts": "IND",
    "Jaguars": "JAX",
    "Chiefs": "KC",
    "Raiders": "LV",
    "Chargers": "LAC",
    "Rams": "LAR",
    "Dolphins": "MIA",
    "Vikings": "MIN",
    "Patriots": "NE",
    "Saints": "NO",
    "Giants": "NYG",
    "Jets": "NYJ",
    "Eagles": "PHI",
    "Steelers": "PIT",
    "49ers": "SF",
    "Seahawks": "SEA",
    "Buccaneers": "TB",
    "Titans": "TEN",
    "Commanders": "WSH"
}

fix_abv = {
    "SFO": "SF",
    "TAM": "TB",
    "GNB": "GB",
    "KAN": "KC",
    "LVR": "LV",
    "NOR": "NO",
    "WAS": "WSH",
    "NWE": "NE",
}

full_team_names_mapping = {
    "Arizona Cardinals": "ARI",
    "Atlanta Falcons": "ATL",
    "Baltimore Ravens": "BAL",
    "Buffalo Bills": "BUF",
    "Carolina Panthers": "CAR",
    "Chicago Bears": "CHI",
    "Cincinnati Bengals": "CIN",
    "Cleveland Browns": "CLE",
    "Dallas Cowboys": "DAL",
    "Denver Broncos": "DEN",
    "Detroit Lions": "DET",
    "Green Bay Packers": "GB",
    "Houston Texans": "HOU",
    "Indianapolis Colts": "IND",
    "Jacksonville Jaguars": "JAX",
    "Kansas City Chiefs": "KC",
    "Las Vegas Raiders": "LV",
    "Los Angeles Chargers": "LAC",
    "Los Angeles Rams": "LAR",
    "Miami Dolphins": "MIA",
    "Minnesota Vikings": "MIN",
    "New England Patriots": "NE",
    "New Orleans Saints": "NO",
    "New York Giants": "NYG",
    "New York Jets": "NYJ",
    "Philadelphia Eagles": "PHI",
    "Pittsburgh Steelers": "PIT",
    "San Francisco 49ers": "SF",
    "Seattle Seahawks": "SEA",
    "Tampa Bay Buccaneers": "TB",
    "Tennessee Titans": "TEN",
    "Washington Commanders": "WSH"
}

win_percentage_data.rename(columns={'Tm':'Team'}, inplace=True)
win_percentage_data['Team'] = win_percentage_data['Team'].map(full_team_names_mapping).fillna(win_percentage_data['Team'])

fantasy_data_2023.rename(columns={'FantPos':'position'}, inplace=True)
fantasy_data_2023.rename(columns={'Tm':'Team'}, inplace=True)
fantasy_data_2023 = fantasy_data_2023.replace(r'[^\w\s]|_', '', regex=True)
fantasy_data_2023['Team'] = fantasy_data_2023['Team'].map(fix_abv).fillna(fantasy_data_2023['Team'])
fantasy_data_2023 = fantasy_data_2023[fantasy_data_2023.Team != '2TM']
fantasy_data_2023 = fantasy_data_2023[fantasy_data_2023.Team != '3TM']
fantasy_data_2023['Injury'] = 0
fantasy_data_2023.dropna(subset=['PPR'], inplace=True)
fantasy_data_2023.fillna(0, inplace=True)
fantasy_data_2023.dropna(subset=['Team'], inplace=True)
fantasy_data_2023 = pd.merge(fantasy_data_2023, win_percentage_data, on=['Team', 'Year'], how='left')

fantasy_data_2023['PPG'] = fantasy_data_2023['PPR']/fantasy_data_2023['G']
fantasy_data_2023['Cmp/Gm'] = fantasy_data_2023['Cmp']/fantasy_data_2023['G']
fantasy_data_2023['PassYds/Gm'] = fantasy_data_2023['PassYds']/fantasy_data_2023['G']
fantasy_data_2023['PassTD/Gm'] = fantasy_data_2023['PassTD']/fantasy_data_2023['G']
fantasy_data_2023['Int/Gm'] = fantasy_data_2023['Int']/fantasy_data_2023['G']
fantasy_data_2023['PassAtt/Gm'] = fantasy_data_2023['PassAtt']/fantasy_data_2023['G']
fantasy_data_2023['RushAtt/Gm'] = fantasy_data_2023['RushAtt']/fantasy_data_2023['G']
fantasy_data_2023['RushYds/Gm'] = fantasy_data_2023['RushYds']/fantasy_data_2023['G']
fantasy_data_2023['RushTD/Gm'] = fantasy_data_2023['RushTD']/fantasy_data_2023['G']
fantasy_data_2023['Tgt/Gm'] = fantasy_data_2023['Tgt']/fantasy_data_2023['G']
fantasy_data_2023['Rec/Gm'] = fantasy_data_2023['Rec']/fantasy_data_2023['G']
fantasy_data_2023['RecYds/Gm'] = fantasy_data_2023['RecYds']/fantasy_data_2023['G']
fantasy_data_2023['RecTD/Gm'] = fantasy_data_2023['RecTD']/fantasy_data_2023['G']
fantasy_data_2023['Fmb/Gm'] = fantasy_data_2023['Fmb']/fantasy_data_2023['G']

fantasy_data['PPG'] = fantasy_data['PPR']/fantasy_data['G']
fantasy_data['Cmp/Gm'] = fantasy_data['Cmp']/fantasy_data['G']
fantasy_data['PassYds/Gm'] = fantasy_data['PassYds']/fantasy_data['G']
fantasy_data['PassTD/Gm'] = fantasy_data['PassTD']/fantasy_data['G']
fantasy_data['Int/Gm'] = fantasy_data['Int']/fantasy_data['G']
fantasy_data['PassAtt/Gm'] = fantasy_data['PassAtt']/fantasy_data['G']
fantasy_data['RushAtt/Gm'] = fantasy_data['RushAtt']/fantasy_data['G']
fantasy_data['RushYds/Gm'] = fantasy_data['RushYds']/fantasy_data['G']
fantasy_data['RushTD/Gm'] = fantasy_data['RushTD']/fantasy_data['G']
fantasy_data['Tgt/Gm'] = fantasy_data['Tgt']/fantasy_data['G']
fantasy_data['Rec/Gm'] = fantasy_data['Rec']/fantasy_data['G']
fantasy_data['RecYds/Gm'] = fantasy_data['RecYds']/fantasy_data['G']
fantasy_data['RecTD/Gm'] = fantasy_data['RecTD']/fantasy_data['G']
fantasy_data['Fmb/Gm'] = fantasy_data['Fmb']/fantasy_data['G']

fantasy_data['DevTrait'] = "None"
fantasy_data['DevSpeed'] = "N/A"

rookie_data['ProspectLevel'] = ""
rookie_data['GS'] = 17
rookie_data['PassYds/Gm'] = 0
rookie_data['PassTD/Gm'] = 0
rookie_data['PassAtt/Gm'] = 0
rookie_data['Cmp%'] = 0
rookie_data['RushYds/Gm'] = 0
rookie_data['RushTD/Gm'] = 0
rookie_data['Rec/Gm'] = 0
rookie_data['RecYds/Gm'] = 0
rookie_data['Y/A'] = 0
rookie_data['RushAtt/Gm'] = 0
rookie_data['RushAtt'] = 0
rookie_data['Tgt/Gm'] = 0
rookie_data['RecTD/Gm'] = 0
rookie_data['Y/R'] = 0
rookie_data['Year'] = 2023
rookie_data['DevTrait'] = "None"
rookie_data['DevSpeed'] = "N/A"

rookie_data['Team'] = rookie_data['Team'].map(fix_abv).fillna(rookie_data['Team'])

dev_levels = ['None', 'Slow', 'Medium', 'Fast', 'Superstar']
qb_dev_styles = ['Gunslinger', 'Improviser', 'Precision Passer', 'Escape Artist', 'Field General']
rb_dev_styles = ['Workhorse', 'Explosive Rusher', 'Backfield Reciever', 'Balanced Back', 'Unstoppable Force']
wr_dev_styles = ['Deep Threat', 'Playmaker', 'Redzone Threat', 'Balanced Reciever', 'Matchup Nightmare']
for index, player in rookie_data.iterrows():
    if int(player['Pick']) <= 15:
        rookie_data.loc[(rookie_data['Player'] == player['Player']), "ProspectLevel"] = "Top"
        dev_level = ""
        level_num = random.randint(0, 10)
        if level_num <= 1:
            dev_level = 'None'
        elif level_num <= 3:
            dev_level = 'Slow'
        elif level_num <= 5:
            dev_level = 'Medium'
        elif level_num <= 7:
            dev_level = 'Fast'
        else:
            dev_level = 'Superstar'
        rookie_data.loc[(rookie_data['Player'] == player['Player']), "DevSpeed"] = dev_level
        if player['position'] == 'QB':
            if dev_level != 'None':
                rookie_data.loc[(rookie_data['Player'] == player['Player']), "DevTrait"] = random.choice(qb_dev_styles)
            rookie_data.loc[(rookie_data['Player'] == player['Player']), "PassYds/Gm"] = 170
            rookie_data.loc[(rookie_data['Player'] == player['Player']), "PassTD/Gm"] = 1
            rookie_data.loc[(rookie_data['Player'] == player['Player']), "PassAtt/Gm"] = 25
            rookie_data.loc[(rookie_data['Player'] == player['Player']), "Cmp%"] = 0.6
            rookie_data.loc[(rookie_data['Player'] == player['Player']), "RushYds/Gm"] = 1
        elif player['position'] == 'RB' or player['position'] == 'FB':
            if dev_level != 'None':
                rookie_data.loc[(rookie_data['Player'] == player['Player']), "DevTrait"] = random.choice(rb_dev_styles)
            rookie_data.loc[(rookie_data['Player'] == player['Player']), "RushYds/Gm"] = 40
            rookie_data.loc[(rookie_data['Player'] == player['Player']), "RushTD/Gm"] = 0.2
            rookie_data.loc[(rookie_data['Player'] == player['Player']), "RushAtt/Gm"] = 10
            rookie_data.loc[(rookie_data['Player'] == player['Player']), "Rec/Gm"] = 1
            rookie_data.loc[(rookie_data['Player'] == player['Player']), "RecYds/Gm"] = 7
            rookie_data.loc[(rookie_data['Player'] == player['Player']), "Y/A"] = 3
            rookie_data.loc[(rookie_data['Player'] == player['Player']), "RushAtt"] = 170
        elif player['position'] == 'WR' or player['position'] == 'TE':
            if dev_level != 'None':
                rookie_data.loc[(rookie_data['Player'] == player['Player']), "DevTrait"] = random.choice(wr_dev_styles)
            rookie_data.loc[(rookie_data['Player'] == player['Player']), "Tgt/Gm"] = 6
            rookie_data.loc[(rookie_data['Player'] == player['Player']), "Rec/Gm"] = 4
            rookie_data.loc[(rookie_data['Player'] == player['Player']), "RecYds/Gm"] = 37
            rookie_data.loc[(rookie_data['Player'] == player['Player']), "RecTD/Gm"] = 0.2
            rookie_data.loc[(rookie_data['Player'] == player['Player']), "Y/R"] = 7
    elif int(player['Pick']) < 100:
        rookie_data.loc[(rookie_data['Player'] == player['Player']), "ProspectLevel"] = "Middle"
        dev_level = ""
        level_num = random.randint(0, 10)
        if level_num <= 2:
            dev_level = 'None'
        elif level_num <= 4:
            dev_level = 'Slow'
        elif level_num <= 6:
            dev_level = 'Medium'
        elif level_num <= 8:
            dev_level = 'Fast'
        else:
            dev_level = 'Superstar'
        rookie_data.loc[(rookie_data['Player'] == player['Player']), "DevSpeed"] = dev_level
        if player['position'] == 'RB' or player['position'] == 'FB':
            if dev_level != 'None':
                rookie_data.loc[(rookie_data['Player'] == player['Player']), "DevTrait"] = random.choice(rb_dev_styles)
            rookie_data.loc[(rookie_data['Player'] == player['Player']), "RushYds/Gm"] = 20
            rookie_data.loc[(rookie_data['Player'] == player['Player']), "RushTD/Gm"] = 0.1
            rookie_data.loc[(rookie_data['Player'] == player['Player']), "RushAtt/Gm"] = 5
            rookie_data.loc[(rookie_data['Player'] == player['Player']), "Rec/Gm"] = 0.5
            rookie_data.loc[(rookie_data['Player'] == player['Player']), "RecYds/Gm"] = 3.5
            rookie_data.loc[(rookie_data['Player'] == player['Player']), "Y/A"] = 1.5
            rookie_data.loc[(rookie_data['Player'] == player['Player']), "RushAtt"] = 85
        elif player['position'] == 'WR' or player['position'] == 'TE':
            if dev_level != 'None':
                rookie_data.loc[(rookie_data['Player'] == player['Player']), "DevTrait"] = random.choice(wr_dev_styles)
            rookie_data.loc[(rookie_data['Player'] == player['Player']), "Tgt/Gm"] = 3
            rookie_data.loc[(rookie_data['Player'] == player['Player']), "Rec/Gm"] = 2
            rookie_data.loc[(rookie_data['Player'] == player['Player']), "RecYds/Gm"] = 18.5
            rookie_data.loc[(rookie_data['Player'] == player['Player']), "RecTD/Gm"] = 0.1
            rookie_data.loc[(rookie_data['Player'] == player['Player']), "Y/R"] = 3.5
    else:
        rookie_data.loc[(rookie_data['Player'] == player['Player']), "ProspectLevel"] = "Low"
        dev_level = ""
        level_num = random.randint(0, 10)
        if level_num <= 5:
            dev_level = 'None'
        elif level_num <= 7:
            dev_level = 'Slow'
        elif level_num <= 8:
            dev_level = 'Medium'
        elif level_num <= 9:
            dev_level = 'Fast'
        else:
            dev_level = 'Superstar'
        rookie_data.loc[(rookie_data['Player'] == player['Player']), "DevSpeed"] = dev_level
        if player['position'] == 'RB' or player['position'] == 'FB':
            if dev_level != 'None':
                rookie_data.loc[(rookie_data['Player'] == player['Player']), "DevTrait"] = random.choice(rb_dev_styles)
            rookie_data.loc[(rookie_data['Player'] == player['Player']), "RushYds/Gm"] = 10
            rookie_data.loc[(rookie_data['Player'] == player['Player']), "RushTD/Gm"] = 0.05
            rookie_data.loc[(rookie_data['Player'] == player['Player']), "RushAtt/Gm"] = 2.5
            rookie_data.loc[(rookie_data['Player'] == player['Player']), "Rec/Gm"] = 0.25
            rookie_data.loc[(rookie_data['Player'] == player['Player']), "RecYds/Gm"] = 1.25
            rookie_data.loc[(rookie_data['Player'] == player['Player']), "Y/A"] = 0.75
            rookie_data.loc[(rookie_data['Player'] == player['Player']), "RushAtt"] = 43
        elif player['position'] == 'WR' or player['position'] == 'TE':
            if dev_level != 'None':
                rookie_data.loc[(rookie_data['Player'] == player['Player']), "DevTrait"] = random.choice(wr_dev_styles)
            rookie_data.loc[(rookie_data['Player'] == player['Player']), "Tgt/Gm"] = 1.5
            rookie_data.loc[(rookie_data['Player'] == player['Player']), "Rec/Gm"] = 1
            rookie_data.loc[(rookie_data['Player'] == player['Player']), "RecYds/Gm"] = 9.25
            rookie_data.loc[(rookie_data['Player'] == player['Player']), "RecTD/Gm"] = 0.05
            rookie_data.loc[(rookie_data['Player'] == player['Player']), "Y/R"] = 1.75

missing_columns = [col for col in fantasy_data_2023.columns if col not in rookie_data.columns]
for col in missing_columns:
    rookie_data[col] = 0

fantasy_data_2023 = pd.concat([fantasy_data_2023, rookie_data], ignore_index=True)

player_teams = pd.read_csv("NFLTeams - Sheet1.csv")
player_teams['Team'] = player_teams['Team'].map(complete_team_abbreviation_mapping)
player_teams.rename(columns={'Pos':'position'}, inplace=True)
player_teams['Player'] = player_teams['First Name'] + ' ' + player_teams['Last Name']

pass_defense = pd.read_csv("NFLPassDefenseRankings - Sheet1.csv")
pass_defense.rename(columns={'TEAM':'Team'}, inplace=True)
pass_defense['Team'] = pass_defense['Team'].map(complete_team_abbreviation_mapping)

run_defense = pd.read_csv("NFLRunDefenseRankings - Sheet1.csv")
run_defense.rename(columns={'TEAM':'Team'}, inplace=True)
run_defense['Team'] = run_defense['Team'].map(complete_team_abbreviation_mapping)

################################################

def determine_stars(ppg, injury):
    if injury > 0:
        return 0
    else:
        if ppg >= 19:
            return 5
        elif ppg >= 15:
            return 4
        elif ppg >=12:
            return 3
        elif ppg >=5:
            return 2
        else:
            return 1

fantasy_data_2023['StarRating'] = fantasy_data_2023.apply(lambda row: determine_stars(row['PPG'], row['Injury']), axis=1)


def count_star_teammates(df):
    star_columns = [f'{i}StarTeammates' for i in range(5, 0, -1)]
    for col in star_columns:
        df[col] = 0

    grouped = df.groupby(['Team', 'position', 'Year'])

    def count_teammates(group):
        for i in range(1, 6):
            star_col = f'{i}StarTeammates'
            group[star_col] = group.apply(lambda row: ((group['StarRating'] == i) & (group.index != row.name)).sum(),
                                          axis=1)
        return group

    df = grouped.apply(count_teammates).reset_index(drop=True)
    return df

fantasy_data_2023 = count_star_teammates(fantasy_data_2023)
fantasy_data_2023['GoodTeammates'] = fantasy_data_2023['5StarTeammates'] + fantasy_data_2023['4StarTeammates'] + fantasy_data_2023['3StarTeammates']

def calculate_wr_quality(wrte_df, year):
    wrte_df_year = wrte_df[wrte_df['Year'] == year]
    wrte_df_sorted = wrte_df_year.sort_values(by=['Team', 'StarRating'], ascending=[True, False])
    top_2_wrte = wrte_df_sorted.groupby('Team').head(2)
    wr_quality = top_2_wrte.groupby('Team')['StarRating'].mean().reset_index()
    wr_quality.columns = ['Team', 'wr_quality']
    return wr_quality

def add_wr_quality_to_qb_data(qb_df, wrte_df):
    years = qb_df['Year'].unique()
    result_df = pd.DataFrame()
    for year in years:
        wr_quality = calculate_wr_quality(wrte_df, year)
        qb_df_year = qb_df[qb_df['Year'] == year]
        qb_df_year = qb_df_year.merge(wr_quality, on='Team', how='left')
        result_df = pd.concat([result_df, qb_df_year])
    return result_df

wr_pos = ['WR', 'TE']
fantasy_data_2023 = add_wr_quality_to_qb_data(fantasy_data_2023, fantasy_data_2023[fantasy_data_2023['position'].isin(wr_pos)])

fantasy_data_2023['Cmp%'] = fantasy_data_2023['Cmp']/fantasy_data_2023['PassAtt']

#####################


qb_data = fantasy_data_2023[fantasy_data_2023['position'] == 'QB']
qb_data = qb_data[["Player", "position", "Team", "Year", "Age", "StarRating", "PPG", "PPR", "GoodTeammates", "RushYds/Gm", "PassYds/Gm", "PassTD/Gm", "PassAtt", "GS", "Cmp", "wr_quality", "Cmp%"]]
pos_to_keep = ['RB', 'FB']
rbfb_data = fantasy_data_2023[fantasy_data_2023['position'].isin(pos_to_keep)]
rbfb_data = rbfb_data[["Player", "position", "Team", "Year", "Age", "StarRating", "PPG", "PPR", "GoodTeammates", "RushYds/Gm", "RushTD/Gm", "Rec/Gm", "RecYds/Gm", "RushAtt/Gm", "Y/A", "RushAtt"]]
pos_to_keep = ['WR', 'TE']
wrte_data = fantasy_data_2023[fantasy_data_2023['position'].isin(pos_to_keep)]
wrte_data = wrte_data[["Player", "position", "Team", "Year", "Age", "StarRating", "PPG", "PPR", "GoodTeammates", "Tgt/Gm", "Rec/Gm", "RecYds/Gm", "RecTD/Gm", "Y/R"]]

features_qb = ["wr_quality", "GoodTeammates", "RushYds/Gm", "PassYds/Gm", "PassTD/Gm", "Cmp%", "PassAtt", "GS"]
qb_predictions = model_qb.predict(qb_data[qb_data['Year'] == 2023][features_qb])
qb_data.loc[qb_data['Year'] == 2023, 'PredAvgPts'] = qb_predictions
qb_data = qb_data.sort_values(by='PredAvgPts', ascending=False)

features_rb = ["GoodTeammates", "RushYds/Gm", "RushTD/Gm", "Rec/Gm", "RecYds/Gm", "RushAtt/Gm", "Y/A", "RushAtt"]
rb_predictions = model_rb.predict(rbfb_data[rbfb_data['Year'] == 2023][features_rb])
rbfb_data.loc[rbfb_data['Year'] == 2023, 'PredAvgPts'] = rb_predictions
rbfb_data = rbfb_data.sort_values(by='PredAvgPts', ascending=False)

features_wr = ["GoodTeammates", "Tgt/Gm", "Rec/Gm", "RecYds/Gm", "RecTD/Gm", "Y/R"]
wr_predictions = model_wr.predict(wrte_data[wrte_data['Year'] == 2023][features_wr])
wrte_data.loc[wrte_data['Year'] == 2023, 'PredAvgPts'] = wr_predictions
wrte_data = wrte_data.sort_values(by='PredAvgPts', ascending=False)

fantasy_data_2023 = fantasy_data_2023[fantasy_data_2023['Year'] == 2023]

fantasy_data_2023 = fantasy_data_2023.merge(qb_data[['Player', 'Year', 'PredAvgPts']], on=['Player', 'Year'], how='left')
fantasy_data_2023 = fantasy_data_2023.merge(rbfb_data[['Player', 'Year', 'PredAvgPts']], on=['Player', 'Year'], how='left')
fantasy_data_2023 = fantasy_data_2023.merge(wrte_data[['Player', 'Year', 'PredAvgPts']], on=['Player', 'Year'], how='left')

fantasy_data_2023['PredAvgPts'] = fantasy_data_2023['PredAvgPts'].fillna(fantasy_data_2023['PredAvgPts_x']).fillna(fantasy_data_2023['PredAvgPts_y'])
fantasy_data_2023 = fantasy_data_2023.drop(columns=['PredAvgPts_x', 'PredAvgPts_y'])

fantasy_data_2023["PredPts"] = fantasy_data_2023["PredAvgPts"]*17
fantasy_data_2023 = fantasy_data_2023.sort_values('PredPts', ascending=False)

fantasy_data_2023 = fantasy_data_2023.reset_index(drop=True)

fantasy_data_2023 = pd.merge(fantasy_data_2023, schedules, on=['Team'], how='left')

fantasy_data_2023['TotalPts'] = 0
fantasy_data_2023['AvgPts'] = 0
fantasy_data_2023['weekly_pred_pts'] = 0
fantasy_data_2023['GP'] = 0
fantasy_data_2023['GamesBoosted'] = 0
fantasy_data_2023['Starting'] = "No"
fantasy_data_2023['FantasyTeam'] = "FA"

global draft_board
draft_board = fantasy_data_2023.copy()

global current_week
current_week = 0

teams = [[], [], [], [], [], [], [], []]
teams_need = [["QB", "WR", "WR", "RB", "RB"], ["QB", "WR", "WR", "RB", "RB"], ["QB", "WR", "WR", "RB", "RB"], ["QB", "WR", "WR", "RB", "RB"], ["QB", "WR", "WR", "RB", "RB"], ["QB", "WR", "WR", "RB", "RB"], ["QB", "WR", "WR", "RB", "RB"], ["QB", "WR", "WR", "RB", "RB"]]
team_points = [0, 0, 0, 0, 0, 0, 0, 0]
team_weekly_points = [0, 0, 0, 0, 0, 0, 0, 0]
team_wins = [0, 0, 0, 0, 0, 0, 0, 0]
waiver_order = [7, 6, 5, 4, 3, 2, 1, 0]
waiver_claims = ["", "", "", "", "", "", "", ""]
draft_message = ""
drafted_players = []
html_table = draft_board.to_html()
cold_teams = ['NYG', 'NYJ', 'NE', 'BUF', 'GB', 'MIN', 'DEN']

user_team = 0


def get_gif_url(api_key, query):
    url = f"https://g.tenor.com/v1/search?q={query}&key={api_key}&limit=1"
    response = requests.get(url)
    if response.status_code == 200:
        gifs = response.json()['results']
        if gifs:
            return gifs[0]['media'][0]['gif']['url']
    return None


def round_robin_schedule(teams_by_num):
    if len(teams_by_num) % 2 != 0:
        teams_by_num.append(None)

    n = len(teams_by_num)
    schedule = []

    for round_num in range(n - 1):
        round_matches = []
        for i in range(n // 2):
            team1 = teams_by_num[i]
            team2 = teams_by_num[n - 1 - i]
            if team1 is not None and team2 is not None:
                round_matches.append([team1, team2])
        schedule.append(round_matches)
        # Rotate the teams for the next round
        teams_by_num.insert(1, teams_by_num.pop())

    return schedule


def redistribute_stats_for_player(df, player_name, GamesInjured):
    player = df[df['Player'] == player_name].iloc[0]
    team = player['Team']
    position = player['position']
    tgt_gm = player['Tgt/Gm']
    rec_gm = player['Rec/Gm']
    rush_att = player['RushAtt/Gm']
    rush_yds = rush_att * player['Y/A']
    pass_att = player['PassAtt/Gm']

    teammates = df[(df['Team'] == team) &
                   (df['position'] == position) &
                   (df['Player'] != player_name)]

    num_teammates = len(teammates)

    if num_teammates > 0:
        random_proportions = np.random.rand(num_teammates)
        random_proportions /= random_proportions.sum()
        if df.loc[teammates.index, 'GamesBoosted'].values[0] < GamesInjured+1:
            df.loc[teammates.index, 'GamesBoosted'] = GamesInjured+1
        df.loc[teammates.index, 'Tgt/Gm'] += tgt_gm * random_proportions
        df.loc[teammates.index, 'Rec/Gm'] += rec_gm * random_proportions
        df.loc[teammates.index, 'RushAtt/Gm'] += rush_att * random_proportions
        df.loc[teammates.index, 'RushYds/Gm'] += rush_yds * random_proportions
        df.loc[teammates.index, 'PassAtt/Gm'] += pass_att * random_proportions

        for i, proportion in zip(teammates.index, random_proportions):
            df.at[i, 'RecYds/Gm'] += rec_gm * proportion * df.at[i, 'Y/R']
        for i, proportion in zip(teammates.index, random_proportions):
            df.at[i, 'RushYds/Gm'] += rush_att * proportion * df.at[i, 'Y/A']
    return df

def check_boosts(df):
    for index, player in df.iterrows():
        if player['GamesBoosted'] <= 0:
            columns_to_update = ["Tgt/Gm", "Rec/Gm", "RushAtt/Gm", "RushYds/Gm", "PassAtt/Gm"]
            for column in columns_to_update:
                matching_value = fantasy_data.loc[
                    (fantasy_data['Player'] == player['Player']) &
                    (fantasy_data['Year'] == 2023),
                    column
                ]
            if not matching_value.empty:
                fantasy_data_2023.loc[
                    (fantasy_data_2023['Player'] == player['Player']) &
                    (fantasy_data_2023['Year'] == 2023),
                    "Tgt/Gm"
                ] = fantasy_data.loc[
                    (fantasy_data['Player'] == player['Player']) &
                    (fantasy_data['Year'] == 2023),
                    "Tgt/Gm"
                ].values[0]

                fantasy_data_2023.loc[
                    (fantasy_data_2023['Player'] == player['Player']) &
                    (fantasy_data_2023['Year'] == 2023),
                    "Rec/Gm"
                ] = fantasy_data.loc[
                    (fantasy_data['Player'] == player['Player']) &
                    (fantasy_data['Year'] == 2023),
                    "Rec/Gm"
                ].values[0]

                fantasy_data_2023.loc[
                    (fantasy_data_2023['Player'] == player['Player']) &
                    (fantasy_data_2023['Year'] == 2023),
                    "RushAtt/Gm"
                ] = fantasy_data.loc[
                    (fantasy_data['Player'] == player['Player']) &
                    (fantasy_data['Year'] == 2023),
                    "RushAtt/Gm"
                ].values[0]

                fantasy_data_2023.loc[
                    (fantasy_data_2023['Player'] == player['Player']) &
                    (fantasy_data_2023['Year'] == 2023),
                    "RushYds/Gm"
                ] = fantasy_data.loc[
                    (fantasy_data['Player'] == player['Player']) &
                    (fantasy_data['Year'] == 2023),
                    "RushYds/Gm"
                ].values[0]

                fantasy_data_2023.loc[
                    (fantasy_data_2023['Player'] == player['Player']) &
                    (fantasy_data_2023['Year'] == 2023),
                    "PassAtt/Gm"
                ] = fantasy_data.loc[
                    (fantasy_data['Player'] == player['Player']) &
                    (fantasy_data['Year'] == 2023),
                    "PassAtt/Gm"
                ].values[0]

        return df


teams_by_num = list(range(8))
schedule = round_robin_schedule(teams_by_num)

def upgrade_rookies():
    level_impact = 0
    for index, row in fantasy_data_2023.iterrows():
        if row['DevSpeed'] == 'Slow':
            level_impact = 1
        if row['DevSpeed'] == 'Medium':
            level_impact = 1.5
        if row['DevSpeed'] == 'Fast':
            level_impact = 2
        if row['DevSpeed'] == 'Superstar':
            level_impact = 3
        if row['DevTrait'] == 'Gunslinger':
            fantasy_data_2023.loc[(fantasy_data_2023['Player'] == row['Player']), "PassTD/Gm"] += random.uniform(1.1, 1.3*level_impact)*fantasy_data_2023.loc[(fantasy_data_2023['Player'] == row['Player']), "PassTD/Gm"].values[0]
            fantasy_data_2023.loc[(fantasy_data_2023['Player'] == row['Player']), "PassYds/Gm"] += random.uniform(1.1, 1.3*level_impact)*fantasy_data_2023.loc[(fantasy_data_2023['Player'] == row['Player']), "PassYds/Gm"].values[0]
            fantasy_data_2023.loc[(fantasy_data_2023['Player'] == row['Player']), "PassAtt"] += random.uniform(1.1, 1.3*level_impact)*fantasy_data_2023.loc[(fantasy_data_2023['Player'] == row['Player']), "PassAtt"].values[0]
        if row['DevTrait'] == 'Improviser':
            fantasy_data_2023.loc[(fantasy_data_2023['Player'] == row['Player']), "RushYds/Gm"] += random.uniform(1.1, 1.3*level_impact)*fantasy_data_2023.loc[(fantasy_data_2023['Player'] == row['Player']), "RushYds/Gm"].values[0]
        if row['DevTrait'] == 'Precision Passer':
            fantasy_data_2023.loc[(fantasy_data_2023['Player'] == row['Player']), "Cmp%"] += random.uniform(1.1, 1.3*level_impact)*fantasy_data_2023.loc[(fantasy_data_2023['Player'] == row['Player']), "Cmp%"].values[0]
            fantasy_data_2023.loc[(fantasy_data_2023['Player'] == row['Player']), "PassYds/Gm"] += random.uniform(1.1, 1.3)*fantasy_data_2023.loc[(fantasy_data_2023['Player'] == row['Player']), "PassYds/Gm"].values[0]
        if row['DevTrait'] == 'Escape Artist':
            fantasy_data_2023.loc[(fantasy_data_2023['Player'] == row['Player']), "RushYds/Gm"] += random.uniform(1.1, 1.3*level_impact)*fantasy_data_2023.loc[(fantasy_data_2023['Player'] == row['Player']), "RushYds/Gm"].values[0]
            fantasy_data_2023.loc[(fantasy_data_2023['Player'] == row['Player']), "PassYds/Gm"] += random.uniform(1.1, 1.3*level_impact)*fantasy_data_2023.loc[(fantasy_data_2023['Player'] == row['Player']), "PassYds/Gm"].values[0]
        if row['DevTrait'] == 'Field General':
            fantasy_data_2023.loc[(fantasy_data_2023['Player'] == row['Player']), "Cmp%"] += random.uniform(1.1, 1.3*level_impact)*fantasy_data_2023.loc[(fantasy_data_2023['Player'] == row['Player']), "Cmp%"].values[0]
            fantasy_data_2023.loc[(fantasy_data_2023['Player'] == row['Player']), "PassYds/Gm"] += random.uniform(1.1, 1.3*level_impact)*fantasy_data_2023.loc[(fantasy_data_2023['Player'] == row['Player']), "PassYds/Gm"].values[0]
            fantasy_data_2023.loc[(fantasy_data_2023['Player'] == row['Player']), "PassTD/Gm"] += random.uniform(1.1, 1.3*level_impact)*fantasy_data_2023.loc[(fantasy_data_2023['Player'] == row['Player']), "PassTD/Gm"].values[0]
            fantasy_data_2023.loc[(fantasy_data_2023['Player'] == row['Player']), "RushYds/Gm"] += random.uniform(1.1, 1.3*level_impact)*fantasy_data_2023.loc[(fantasy_data_2023['Player'] == row['Player']), "RushYds/Gm"].values[0]
            fantasy_data_2023.loc[(fantasy_data_2023['Player'] == row['Player']), "PassAtt"] += random.uniform(1.1, 1.3*level_impact)*fantasy_data_2023.loc[(fantasy_data_2023['Player'] == row['Player']), "PassAtt"].values[0]


@app.route('/', methods=['GET', 'POST'])
def home():
    return render_template('home.html')

@app.route('/after_draft', methods=['GET', 'POST'])
def home_page():
    global teams
    global current_week
    return render_template('after_draft.html', teams=teams, current_week=current_week)

@app.route('/ViewTeams', methods=['GET', 'POST'])
def view_teams():
    team_colors = {
        'ARI': {'primary': '#97233F', 'secondary': '#FFB612'},
        'ATL': {'primary': '#A71930', 'secondary': '#000000'},
        'BAL': {'primary': '#241773', 'secondary': '#9E7C0C'},
        'BUF': {'primary': '#00338D', 'secondary': '#C60C30'},
        'CAR': {'primary': '#0085CA', 'secondary': '#101820'},
        'CHI': {'primary': '#C83803', 'secondary': '#0B162A'},
        'CIN': {'primary': '#FB4F14', 'secondary': '#000000'},
        'CLE': {'primary': '#311D00', 'secondary': '#FF3C00'},
        'DAL': {'primary': '#041E42', 'secondary': '#869397'},
        'DEN': {'primary': '#FB4F14', 'secondary': '#002244'},
        'DET': {'primary': '#0076B6', 'secondary': '#B0B7BC'},
        'GB': {'primary': '#203731', 'secondary': '#FFB612'},
        'HOU': {'primary': '#03202F', 'secondary': '#A71930'},
        'IND': {'primary': '#002C5F', 'secondary': '#A5ACAF'},
        'JAX': {'primary': '#006778', 'secondary': '#D7A22A'},
        'KC': {'primary': '#E31837', 'secondary': '#FFB81C'},
        'LV': {'primary': '#A5ACAF', 'secondary': '#000000'},
        'LAC': {'primary': '#0073CF', 'secondary': '#FFC20E'},
        'LAR': {'primary': '#003594', 'secondary': '#FFA300'},
        'MIA': {'primary': '#008E97', 'secondary': '#FC4C02'},
        'MIN': {'primary': '#4F2683', 'secondary': '#FFC62F'},
        'NE': {'primary': '#002244', 'secondary': '#C60C30'},
        'NO': {'primary': '#D3BC8D', 'secondary': '#101820'},
        'NYG': {'primary': '#0B2265', 'secondary': '#A71930'},
        'NYJ': {'primary': '#125740', 'secondary': '#000000'},
        'PHI': {'primary': '#004C54', 'secondary': '#A5ACAF'},
        'PIT': {'primary': '#FFB612', 'secondary': '#101820'},
        'SF': {'primary': '#AA0000', 'secondary': '#B3995D'},
        'SEA': {'primary': '#002244', 'secondary': '#69BE28'},
        'TB': {'primary': '#D50A0A', 'secondary': '#FF7900'},
        'TEN': {'primary': '#4B92DB', 'secondary': '#C8102E'},
        'WSH': {'primary': '#773141', 'secondary': '#FFB612'}
    }
    players_dict = fantasy_data_2023.set_index('Player')['Team'].to_dict()
    return render_template('ViewTeams.html', teams=teams, team_colors=team_colors, players_dict=players_dict, fantasy_data_2023=fantasy_data_2023)

@app.route('/TierList', methods=['GET','POST'])
def view_tier_list():
    team_colors = {
        'ARI': {'primary': '#97233F', 'secondary': '#FFB612'},
        'ATL': {'primary': '#A71930', 'secondary': '#000000'},
        'BAL': {'primary': '#241773', 'secondary': '#9E7C0C'},
        'BUF': {'primary': '#00338D', 'secondary': '#C60C30'},
        'CAR': {'primary': '#0085CA', 'secondary': '#101820'},
        'CHI': {'primary': '#C83803', 'secondary': '#0B162A'},
        'CIN': {'primary': '#FB4F14', 'secondary': '#000000'},
        'CLE': {'primary': '#311D00', 'secondary': '#FF3C00'},
        'DAL': {'primary': '#041E42', 'secondary': '#869397'},
        'DEN': {'primary': '#FB4F14', 'secondary': '#002244'},
        'DET': {'primary': '#0076B6', 'secondary': '#B0B7BC'},
        'GB': {'primary': '#203731', 'secondary': '#FFB612'},
        'HOU': {'primary': '#03202F', 'secondary': '#A71930'},
        'IND': {'primary': '#002C5F', 'secondary': '#A5ACAF'},
        'JAX': {'primary': '#006778', 'secondary': '#D7A22A'},
        'KC': {'primary': '#E31837', 'secondary': '#FFB81C'},
        'LV': {'primary': '#A5ACAF', 'secondary': '#000000'},
        'LAC': {'primary': '#0073CF', 'secondary': '#FFC20E'},
        'LAR': {'primary': '#003594', 'secondary': '#FFA300'},
        'MIA': {'primary': '#008E97', 'secondary': '#FC4C02'},
        'MIN': {'primary': '#4F2683', 'secondary': '#FFC62F'},
        'NE': {'primary': '#002244', 'secondary': '#C60C30'},
        'NO': {'primary': '#D3BC8D', 'secondary': '#101820'},
        'NYG': {'primary': '#0B2265', 'secondary': '#A71930'},
        'NYJ': {'primary': '#125740', 'secondary': '#000000'},
        'PHI': {'primary': '#004C54', 'secondary': '#A5ACAF'},
        'PIT': {'primary': '#FFB612', 'secondary': '#101820'},
        'SF': {'primary': '#AA0000', 'secondary': '#B3995D'},
        'SEA': {'primary': '#002244', 'secondary': '#69BE28'},
        'TB': {'primary': '#D50A0A', 'secondary': '#FF7900'},
        'TEN': {'primary': '#4B92DB', 'secondary': '#C8102E'},
        'WSH': {'primary': '#773141', 'secondary': '#FFB612'}
    }
    players = {star: fantasy_data_2023[fantasy_data_2023['StarRating'] == star]['Player'].tolist() for star in range(1, 6)}
    teams = {row['Player']: row['Team'] for _, row in fantasy_data_2023.iterrows()}
    return render_template('TierList.html', players=players, team_colors=team_colors, teams=teams)


@app.route('/player/<player_name>')
def player_details(player_name):
    player_data = fantasy_data_2023[fantasy_data_2023['Player'] == player_name].iloc[0]
    player_image = fetch_player_image(player_name)
    fantasy_team = "FA"
    team_num = 0
    for team in teams:
        for player in team:
            if player == player_name:
                fantasy_team = team_num
        team_num+=1
    return render_template('player_details.html', player_name=player_name, player_image=player_image, total_pts=round(player_data['TotalPts'], 1), fantasy_team=fantasy_team, avg_points = player_data['AvgPts'])


def fetch_player_image(player_name):
    search_query = f'{player_name} headshot'
    search_url = f'https://www.bing.com/images/search?q={search_query}'
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    def get_image_src(soup):
        images = soup.find_all('img', {'class': 'mimg'})
        for image in images:
            src = image.get('src')
            if src and not src.startswith('data:'):
                return src
        return None
    for page in range(1, 4):
        response = requests.get(f"{search_url}&first={page * 35 + 1}",
                                headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')
        image_src = get_image_src(soup)
        if image_src:
            return image_src
    return None

@app.route('/WaiverClaims', methods=['GET', 'POST'])
def make_waiver_claim():
    global user_team
    global waiver_order
    global waiver_claims
    waiver_claims = ["", "", "", "", "", "", "", ""]
    if request.method == 'POST':
        player_name = request.form['player_name']
        player = draft_board[draft_board['Player'] == player_name].to_dict('records')
        i = waiver_order.index(user_team)
        if player:
            waiver_claims[i] = player_name
    auto_waiver_claim()
    print(waiver_claims)
    return render_template('WaiverClaims.html', table=draft_board[['Player', 'position', 'Team', 'PredPts', 'PredAvgPts', 'StarRating']].to_html())

def auto_waiver_claim():
    global waiver_order
    global waiver_claims
    global teams
    global draft_board
    draft_board = pd.merge(draft_board, fantasy_data_2023[['Player', 'AvgPts']], on='Player', how='left')
    draft_board = draft_board.drop('AvgPts_x', axis=1)
    draft_board['AvgPts'] = draft_board['AvgPts_y']
    draft_board = draft_board.drop('AvgPts_y', axis=1)
    team_num = 1
    for team in teams[1:]:
        team_min = 500
        weakest_position = ""
        for player in team:
            if fantasy_data_2023.loc[fantasy_data_2023['Player'] == player, "AvgPts"].values[0] < team_min:
                team_min = fantasy_data_2023.loc[fantasy_data_2023['Player'] == player, "AvgPts"].values[0]
                weakest_position = fantasy_data_2023.loc[fantasy_data_2023['Player'] == player, "position"].values[0]
        i = waiver_order.index(team_num)
        for index, row in draft_board.iterrows():
            if row['position'] == weakest_position and row['AvgPts'] < team_min:
                print(i)
                print(waiver_claims[i])
                waiver_claims[i] = row['Player']
                break
            waiver_claims[i] = ""
        team_num+=1

@app.route('/WaiverResults', methods=['GET', 'POST'])
def assign_waiver_claims():
    global waiver_order
    global waiver_claims
    global draft_board
    claim_num = 0
    successful_waivers = []
    for claim in waiver_claims:
        if (draft_board['Player'] == claim).any():
            teams[waiver_order[claim_num]].append(claim)
            fantasy_data_2023.loc[fantasy_data_2023['Player'] == claim, "FantasyTeam"] = waiver_order[claim_num]
            successful_waivers.append(draft_board['Player'])
            draft_board = draft_board[draft_board.Player != claim]
            team_num = waiver_order.pop(claim_num)
            claim_num-=1
            waiver_order.append(team_num)
            print(teams)
        claim_num+=1
    return render_template('WaiverResults.html', teams=teams, waiver_claims=waiver_claims)

@app.route('/LineupChanges', methods=['GET', 'POST'])
def set_lineup():
    global current_week
    qblist = []
    rblist = []
    wrlist = []
    fantasy_data_2023['Starting'] = "No"
    for index, row in fantasy_data_2023.iterrows():
        player = row['Player']
        pred_weekly_pts(player)
    auto_update_lineup()
    players = fantasy_data_2023[['Player', 'Wk'+str(current_week+1), 'position', 'weekly_pred_pts']].to_dict(orient='records')
    for player in teams[0]:
        if fantasy_data_2023.loc[fantasy_data_2023['Player'] == player, "position"].values[0] == "QB":
            qblist.append(player)
        if fantasy_data_2023.loc[fantasy_data_2023['Player'] == player, "position"].values[0] == "RB":
            rblist.append(player)
        if fantasy_data_2023.loc[fantasy_data_2023['Player'] == player, "position"].values[0] == "WR":
            wrlist.append(player)
    qb_players = [p for p in players if p['Player'] in qblist]
    rb_players = [p for p in players if p['Player'] in rblist]
    wr_players = [p for p in players if p['Player'] in wrlist]
    return render_template('LineupChanges.html', qblist=qblist, rblist=rblist, wrlist=wrlist, players=players, current_week='Wk'+str(current_week+1), qb_players=qb_players, rb_players=rb_players, wr_players=wr_players)

def pred_weekly_pts(player):
        weekly_pred_pts = round((fantasy_data_2023.loc[fantasy_data_2023['Player'] == player, "PredAvgPts"].values[0]), 1)
        opponent = str(fantasy_data_2023.loc[fantasy_data_2023['Player'] == player, 'Wk'+str(current_week+1)].values[0])[1:]
        if opponent != 'YE':
            if fantasy_data_2023.loc[fantasy_data_2023['Player'] == player, "position"].values[0] == 'RB':
                defense_ranking = run_defense.loc[run_defense['Team'] == opponent, 'Rank'].values[0]
            else:
                defense_ranking = pass_defense.loc[pass_defense['Team'] == opponent, 'Rank'].values[0]
            if defense_ranking < 16:
                fantasy_data_2023.loc[fantasy_data_2023['Player'] == player, "weekly_pred_pts"].values[0]
            else:
                fantasy_data_2023.loc[fantasy_data_2023['Player'] == player, "weekly_pred_pts"].values[0] -= random.randint(0, 3)
        else:
            fantasy_data_2023.loc[fantasy_data_2023['Player'] == player, "weekly_pred_pts"].values[0] = 0
        fantasy_data_2023.loc[fantasy_data_2023['Player'] == player, "weekly_pred_pts"] = weekly_pred_pts

def auto_update_lineup():
    global current_week
    for team in teams[1:]:
        qblist = []
        rblist = []
        wrlist = []
        for player in team:
            opponent = fantasy_data_2023.loc[fantasy_data_2023['Player'] == player, 'Wk'+str(current_week+1)].values[0]
            if fantasy_data_2023.loc[fantasy_data_2023['Player'] == player, "position"].values[0] == "QB":
                qblist.append(player)
            if fantasy_data_2023.loc[fantasy_data_2023['Player'] == player, "position"].values[0] == "RB":
                rblist.append(player)
            if fantasy_data_2023.loc[fantasy_data_2023['Player'] == player, "position"].values[0] == "WR":
                wrlist.append(player)
        pred_list = []
        for qb in qblist:
            pred_list.append(fantasy_data_2023.loc[fantasy_data_2023['Player'] == qb, "AvgPts"].values[0])
        max_index = pred_list.index(max(pred_list))
        fantasy_data_2023.loc[fantasy_data_2023['Player'] == qblist[max_index], "Starting"] = 'Yes'

        pred_list = []
        for rb in rblist:
            pred_list.append(fantasy_data_2023.loc[fantasy_data_2023['Player'] == rb, "AvgPts"].values[0])

        for x in range(2):
            max_index = pred_list.index(max(pred_list))
            fantasy_data_2023.loc[fantasy_data_2023['Player'] == rblist[max_index], "Starting"] = 'Yes'
            pred_list[max_index] = -float('inf')

        pred_list = []
        for wr in wrlist:
            pred_list.append(fantasy_data_2023.loc[fantasy_data_2023['Player'] == wr, "AvgPts"].values[0])

        for x in range(2):
            max_index = pred_list.index(max(pred_list))
            fantasy_data_2023.loc[fantasy_data_2023['Player'] == wrlist[max_index], "Starting"] = 'Yes'
            pred_list[max_index] = -float('inf')


@app.route('/submit-lineup', methods=['POST'])
def submit_lineup():
    selected_qb = request.form.get('qb')
    selected_rb1 = request.form.get('rb1')
    selected_rb2 = request.form.get('rb2')
    selected_wr1 = request.form.get('wr1')
    selected_wr2 = request.form.get('wr2')

    fantasy_data_2023.loc[fantasy_data_2023['Player'] == selected_qb, 'Starting'] = "Yes"
    fantasy_data_2023.loc[fantasy_data_2023['Player'] == selected_rb1, 'Starting'] = "Yes"
    fantasy_data_2023.loc[fantasy_data_2023['Player'] == selected_rb2, 'Starting'] = "Yes"
    fantasy_data_2023.loc[fantasy_data_2023['Player'] == selected_wr1, 'Starting'] = "Yes"
    fantasy_data_2023.loc[fantasy_data_2023['Player'] == selected_wr2, 'Starting'] = "Yes"

    return redirect('/WeeklyStats')

@app.route('/WeeklyStats', methods=['GET', 'POST'])
def sim_week():
    global teams
    global team_points
    global current_week
    global fantasy_data_2023
    global team_weekly_points
    global schedule
    current_week+=1
    fantasy_data_2023['GP'] += 1
    fantasy_data_2023['Injury']-=1
    fantasy_data_2023['GamesBoosted'] -= 1
    team_num = 0
    team_weekly_points = [0, 0, 0, 0, 0, 0, 0, 0]
    weather_note = "Good conditions"
    weather_impact = 0
    new_injury = None

    fantasy_data_2023[str("Wk" + str(current_week))+"Pts"] = 0
    if current_week > 1:
        fantasy_data_2023['StarRating'] = fantasy_data_2023.apply(
            lambda row: determine_stars(row['AvgPts'], row['Injury']), axis=1)
    fantasy_data_2023 = check_boosts(fantasy_data_2023)
    fantasy_data_2023 = count_star_teammates(fantasy_data_2023)
    fantasy_data_2023['GoodTeammates'] = fantasy_data_2023['5StarTeammates'] + fantasy_data_2023['4StarTeammates'] + \
                                         fantasy_data_2023['3StarTeammates']
    for index, row in fantasy_data_2023.iterrows():
        player = row['Player']
        sim_player_points(player)

    dfteam_0 = pd.DataFrame()
    for team in teams:
        for player in team:
            if fantasy_data_2023.loc[fantasy_data_2023['Player'] == player, "Starting"].values[0] == "Yes":
                team_points[team_num]+=round(fantasy_data_2023.loc[fantasy_data_2023['Player'] == player, str("Wk" + str(current_week)+"Pts")].values[0], 1)
                team_weekly_points[team_num]+=round(fantasy_data_2023.loc[fantasy_data_2023['Player'] == player, str("Wk" + str(current_week)+"Pts")].values[0], 1)
            player_row = fantasy_data_2023.loc[fantasy_data_2023['Player'] == player]
            if team_num == 0:
                dfteam_0 = pd.concat([dfteam_0, player_row], ignore_index=True)
        team_num+=1
    dfteam_0 = dfteam_0[['Player', str("Wk" + str(current_week)+"Pts"), 'TotalPts', 'Injury', 'GameNotes', 'Starting']]
    dfteam_0_html = dfteam_0.to_html(classes='table table-striped', index=False)
    players_points = fantasy_data_2023[fantasy_data_2023['Starting'] == 'Yes'].groupby('FantasyTeam').apply(
        lambda x: x[['Player', 'Wk' + str(current_week) + 'Pts']].to_dict(orient='records')).to_dict()
    update_winners()
    fantasy_data_2023 = fantasy_data_2023.sort_values(by='Wk' + str(current_week) + 'Pts', ascending=False)
    upgrade_rookies()
    #print(fantasy_data_2023[['Player', 'RushYds/Gm', 'RushTD/Gm', 'RecYds/Gm', 'RecTD/Gm', 'Rec/Gm']].head(20))
    return render_template('WeeklyStats.html', table=fantasy_data_2023[['Player', 'Team', 'position', str("Wk" + str(current_week)+"Pts"), 'TotalPts', 'Injury', 'GameNotes', 'Starting']].to_html(), dfteam_0=dfteam_0_html, team_wins=team_wins, current_week=str(current_week), intcurrent_week=int(current_week), matchups=schedule[current_week-1], team_points=team_points,team_weekly_points=team_weekly_points, players_points=players_points)

def update_winners():
    global schedule
    global current_week
    global team_points
    global team_weekly_points
    global team_wins
    current_week_schedule = schedule[current_week-1]
    for matchup in current_week_schedule:
        points = []
        for team in matchup:
            points.append(team_weekly_points[int(team)])
        max_points = max(points)
        max_index = points.index(max_points)
        winning_team = matchup[max_index]
        team_wins[int(winning_team)] += 1


def sim_player_points(player):
    global cold_teams
    global current_week
    global fantasy_data_2023
    global features_qb
    global features_rb
    global features_wr
    team_abbreviations = [
        "ARI", "ATL", "BAL", "BUF", "CAR", "CHI", "CIN", "CLE", "DAL", "DEN",
        "DET", "GB", "HOU", "IND", "JAX", "KC", "LV", "LAC", "LAR", "MIA",
        "MIN", "NE", "NO", "NYG", "NYJ", "PHI", "PIT", "SF", "SEA", "TB",
        "TEN", "WSH"
    ]
    strong_home = ['SEA', 'GB', 'BAL', 'SF', 'BUF', 'MIA', 'DET']
    opponent = (fantasy_data_2023.loc[fantasy_data_2023['Player'] == player, str("Wk" + str(current_week))].values[0])[1:]
    player_team = fantasy_data_2023.loc[fantasy_data_2023['Player'] == player, "Team"].values[0]
    weather_impact = 0
    home_impact = 0
    game_notes = fantasy_data_2023.loc[fantasy_data_2023['Player'] == player, str("Wk" + str(current_week))].values[0]
    if opponent not in team_abbreviations:
        points_scored = 0
        return points_scored
    #Home Field Advantage
    if (fantasy_data_2023.loc[fantasy_data_2023['Player'] == player, str("Wk" + str(current_week))].values[0])[0] == '@':
        if opponent in strong_home:
            home_impact = random.randint(-10, 0)
        else:
            home_impact = random.randint(-5, 0)
    else:
        if player_team in strong_home:
            home_impact = random.randint(0, 10)
        else:
            home_impact = random.randint(0, 5)
    #Cold Teams
    if (fantasy_data_2023.loc[fantasy_data_2023['Player'] == player, str("Wk" + str(current_week))].values[0])[0] == '@':
        if opponent in cold_teams:
            weather_impact = random.randint(0, 5)
            weather_note = ", Cold weather"
            game_notes += weather_note
    else:
        if player_team in cold_teams:
            weather_impact = random.randint(0, 5)
            weather_note = ", Cold weather"
            game_notes += weather_note
    if fantasy_data_2023.loc[fantasy_data_2023['Player'] == player, "position"].values[0] == "RB":
        opponent_ranking = run_defense.loc[run_defense['Team'] == str(opponent), "Rank"].values[0]
    else:
        opponent_ranking = pass_defense.loc[pass_defense['Team'] == str(opponent), "Rank"].values[0]
        weather_impact = -weather_impact
    #points_scored = round(((fantasy_data_2023.loc[fantasy_data_2023['Player'] == player, "PredPts"].values[0] / 17) * (
          #      1 - (opponent_ranking / 100)) + random.randint(-10, 10) + weather_impact + home_impact), 1)
    if fantasy_data_2023.loc[fantasy_data_2023['Player'] == player, "position"].values[0] == 'QB':
        if opponent_ranking <= 10:
            fantasy_data_2023.loc[fantasy_data_2023['Player'] == player, "Cmp%"]*=random.uniform(0.7, 1)
            fantasy_data_2023.loc[fantasy_data_2023['Player'] == player, "PassYds/Gm"] *= random.uniform(0.7, 1)
            fantasy_data_2023.loc[fantasy_data_2023['Player'] == player, "PassTD/Gm"] *= random.uniform(0.7, 1)
        elif opponent_ranking >= 20:
            fantasy_data_2023.loc[fantasy_data_2023['Player'] == player, "Cmp%"]*=random.uniform(1, 1.2)
            fantasy_data_2023.loc[fantasy_data_2023['Player'] == player, "PassYds/Gm"] *= random.uniform(1, 1.2)
            fantasy_data_2023.loc[fantasy_data_2023['Player'] == player, "PassTD/Gm"] *= random.uniform(1, 1.2)
        fantasy_data_2023[features_qb] = fantasy_data_2023[features_qb].round(0)
        points_scored = model_qb.predict(fantasy_data_2023[fantasy_data_2023['Player'] == player][features_qb])[0]
    elif fantasy_data_2023.loc[fantasy_data_2023['Player'] == player, "position"].values[0] == 'RB' or fantasy_data_2023.loc[fantasy_data_2023['Player'] == player, "position"].values[0] == 'FB':
        if opponent_ranking <= 10:
            fantasy_data_2023.loc[fantasy_data_2023['Player'] == player, "Y/A"]*=random.uniform(0.7, 1)
            fantasy_data_2023.loc[fantasy_data_2023['Player'] == player, "RushYds/Gm"] *= random.uniform(0.7, 1)
            fantasy_data_2023.loc[fantasy_data_2023['Player'] == player, "RushTD/Gm"] *= random.uniform(0.7, 1)
        elif opponent_ranking >= 20:
            fantasy_data_2023.loc[fantasy_data_2023['Player'] == player, "Y/A"]*=random.uniform(1, 1.2)
            fantasy_data_2023.loc[fantasy_data_2023['Player'] == player, "RushYds/Gm"] *= random.uniform(1, 1.2)
            fantasy_data_2023.loc[fantasy_data_2023['Player'] == player, "RushTD/Gm"] *= random.uniform(1, 1.2)
        fantasy_data_2023[features_rb] = fantasy_data_2023[features_rb].round(0)
        points_scored = model_rb.predict(fantasy_data_2023[fantasy_data_2023['Player'] == player][features_rb])[0]
    elif fantasy_data_2023.loc[fantasy_data_2023['Player'] == player, "position"].values[0] == 'WR' or fantasy_data_2023.loc[fantasy_data_2023['Player'] == player, "position"].values[0] == 'TE':
        if opponent_ranking <= 10:
            fantasy_data_2023.loc[fantasy_data_2023['Player'] == player, "Y/R"]*=random.uniform(0.7, 1)
            fantasy_data_2023.loc[fantasy_data_2023['Player'] == player, "RecYds/Gm"] *= random.uniform(0.7, 1)
            fantasy_data_2023.loc[fantasy_data_2023['Player'] == player, "RecTD/Gm"] *= random.uniform(0.7, 1)
        elif opponent_ranking >= 20:
            fantasy_data_2023.loc[fantasy_data_2023['Player'] == player, "Y/R"]*=random.uniform(1, 1.2)
            fantasy_data_2023.loc[fantasy_data_2023['Player'] == player, "RecYds/Gm"] *= random.uniform(1, 1.2)
            fantasy_data_2023.loc[fantasy_data_2023['Player'] == player, "RecTD/Gm"] *= random.uniform(1, 1.2)
        fantasy_data_2023[features_wr] = fantasy_data_2023[features_wr].round(0)
        points_scored = model_wr.predict(fantasy_data_2023[fantasy_data_2023['Player'] == player][features_wr])[0]

    #Injuries
    if fantasy_data_2023.loc[fantasy_data_2023['Player'] == player, "Injury"].values[0] > 0:
        points_scored = 0
        fantasy_data_2023.loc[fantasy_data_2023['Player'] == player, "GP"]-=1
    if fantasy_data_2023.loc[fantasy_data_2023['Player'] == player, "PredAvgPts"].values[0] < 5:
        injury_chance = random.randint(0, 50)
    else:
        injury_chance = random.randint(0, 20)
    if injury_chance < 2:
        fantasy_data_2023.loc[fantasy_data_2023['Player'] == player, "StarRating"] = 0
        if random.randint(0, 10) <= 1:
            fantasy_data_2023.loc[fantasy_data_2023['Player'] == player, 'Injury'] = 18
            fantasy_data_2023 = redistribute_stats_for_player(fantasy_data_2023, player, 18)
            new_injury = "Season ending injury"
        elif random.randint(0, 10) <= 3:
            fantasy_data_2023.loc[fantasy_data_2023['Player'] == player, 'Injury'] = 10
            fantasy_data_2023 = redistribute_stats_for_player(fantasy_data_2023, player, 10)
            new_injury = "Severe injury"
        elif random.randint(0, 10) <= 5:
            fantasy_data_2023.loc[fantasy_data_2023['Player'] == player, 'Injury'] = 5
            fantasy_data_2023 = redistribute_stats_for_player(fantasy_data_2023, player, 5)
            new_injury = "Moderate injury"
        elif random.randint(0, 10) <= 7:
            fantasy_data_2023.loc[fantasy_data_2023['Player'] == player, 'Injury'] = 3
            fantasy_data_2023 = redistribute_stats_for_player(fantasy_data_2023, player, 3)
            new_injury = "Mild injury"
        else:
            fantasy_data_2023.loc[fantasy_data_2023['Player'] == player, 'Injury'] = 1
            fantasy_data_2023 = redistribute_stats_for_player(fantasy_data_2023, player, 1)
            new_injury = "Day-to-day injury"
        game_notes += (", " + new_injury)

    else:
        fantasy_data_2023.loc[fantasy_data_2023['Player'] == player, "StarRating"] = determine_stars(
            fantasy_data_2023.loc[fantasy_data_2023['Player'] == player, "AvgPts"].values[0], fantasy_data_2023.loc[fantasy_data_2023['Player'] == player, "Injury"].values[0])

    #set_qb_points(fantasy_data_2023)
    fantasy_data_2023.loc[fantasy_data_2023['Player'] == player, "TotalPts"] += round(points_scored, 1)
    fantasy_data_2023.loc[fantasy_data_2023['Player'] == player, str("Wk" + str(current_week))+"Pts"] = round(points_scored, 1)
    fantasy_data_2023.loc[fantasy_data_2023['Player'] == player, "GameNotes"] = game_notes
    fantasy_data_2023.loc[fantasy_data_2023['Player'] == player, "AvgPts"] = round(((fantasy_data_2023.loc[
        fantasy_data_2023['Player'] == player, "TotalPts"].values[0]) / (fantasy_data_2023.loc[
        fantasy_data_2023['Player'] == player, "GP"].values[0])), 1)
    return points_scored


def set_qb_points(fantasy_data_2023):
    global current_week
    grouped = fantasy_data_2023.groupby('Team')

    for team, group in grouped:
        qbs = group[group['position'] == 'QB']

        if not qbs.empty:
            max_qb = qbs.loc[qbs['PredPts'].idxmax()]

            fantasy_data_2023.loc[(fantasy_data_2023['Team'] == team) & (fantasy_data_2023['position'] == 'QB') & (
                        fantasy_data_2023['Player'] != max_qb['Player']), str("Wk" + str(current_week))+"Pts"] = 0

    return fantasy_data_2023

@app.route('/draft', methods=['GET', 'POST'])
def draft():
    global draft_board
    global teams
    global draft_message
    global drafted_players
    global gif_url
    gif_url=""
    if len(drafted_players) >= 40:
        return redirect('/after_draft')
    if request.method == 'POST':
        player_name = request.form['player_name']
        player = draft_board[draft_board['Player'] == player_name].to_dict('records')
        if player:
            teams[user_team].append(player_name)
            drafted_players.append(player_name)
            fantasy_data_2023.loc[fantasy_data_2023['Player'] == player_name, "FantasyTeam"] = user_team
            draft_board = draft_board[draft_board.Player != player_name]
            draft_message = f"Team {user_team} drafted {player_name}"
            gif_url = get_gif_url(api_key, player_name)
            # Redirect to the same route to handle auto-draft for computer teams
            return redirect(url_for('auto_draft_step'))

    return render_template('draft.html', table=draft_board[['Player', 'position', 'Team', 'PredPts', 'PredAvgPts', 'StarRating', 'DevTrait', 'DevSpeed']].to_html(), team=teams[user_team], draft_message=draft_message, drafted_players=drafted_players, gif_url=gif_url)

@app.route('/auto_draft_step')
def auto_draft_step():
    global teams
    global draft_message
    global drafted_players
    global gif_url

    for team in range(1, len(teams)):  # Loop over computer teams
        auto_draft(team)
    if len(drafted_players) >= 40:
        return redirect('/after_draft')
    return redirect(url_for('draft'))


def user_draft():
  global draft_board
  global teams
  if request.method == 'POST':
    p_name = input("Name: ")
    player_name = request.form['player_name']
    player = draft_board[draft_board['Player'] == player_name].to_dict('records')
    if player:
      teams[user_team].append(player_name)
      drafted_players.append(player_name)
      draft_board = draft_board[draft_board.Player != player_name]
      draft_message = f"Team {user_team} drafted {player_name}"
    return player_name

def auto_draft(team):
  global draft_board
  chosen = False
  rank = 0
  while chosen == False:
    if draft_board.iloc[rank]["position"] in teams_need[team]:
        if len(drafted_players) < 15 and draft_board.iloc[rank]["position"] == 'QB':
            pass
        else:
          teams[team].append(draft_board.iloc[rank]["Player"])
          draft_message = f"Team {team} drafted {draft_board.iloc[rank]['Player']}"
          drafted_players.append(draft_board.iloc[rank]["Player"])
          fantasy_data_2023.loc[fantasy_data_2023['Player'] == draft_board.iloc[rank]["Player"], "FantasyTeam"] = team
          teams_need[team].remove(draft_board.iloc[rank]["position"])
          draft_board = draft_board[draft_board.Player != draft_board.iloc[rank]["Player"]]
          chosen = True
    rank += 1

if __name__ == '__main__':
    app.run(debug=True)