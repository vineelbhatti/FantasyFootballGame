import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import pickle

fantasy_data_24 = pd.read_csv("/Users/vineel/PycharmProjects/FantasyFootballGame/player_stats_2024.csv")
fantasy_data_23 = pd.read_csv("/Users/vineel/PycharmProjects/FantasyFootballGame/player_stats_2023.csv")
fantasy_data_22 = pd.read_csv("/Users/vineel/PycharmProjects/FantasyFootballGame/player_stats_2022.csv")
fantasy_data_21 = pd.read_csv("/Users/vineel/PycharmProjects/FantasyFootballGame/player_stats_2021.csv")
fantasy_data_20 = pd.read_csv("/Users/vineel/PycharmProjects/FantasyFootballGame/player_stats_2020.csv")
fantasy_data_24 = fantasy_data_24[fantasy_data_24['player_name'].isin(fantasy_data_23['player_name'])]

fantasy_data = pd.concat([fantasy_data_24, fantasy_data_23, fantasy_data_21, fantasy_data_21, fantasy_data_20])

team_data = pd.read_csv("/Users/vineel/PycharmProjects/FantasyFootballGame/NFLDefenseData.csv")

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

team_data['Team'] = team_data['Team'].map(full_team_names_mapping)
team_data.rename(columns={'Team':'opponent_team'}, inplace=True)

fix_abv = {
    "LA": "LAR",
    "WAS": "WSH"
}
fantasy_data['opponent_team'] = fantasy_data['opponent_team'].map(fix_abv).fillna(fantasy_data['opponent_team'])

fantasy_data = pd.merge(fantasy_data, team_data, on=['opponent_team'], how='left')
fantasy_data.rename(columns={'recent_team':'Team', 'completions':'Cmp', 'attempts':'PassAtt', 'passing_yards':'PassYds', 'passing_tds':'PassTD', 'interceptions':'Int',
                          'carries':'RushAtt', 'rushing_yards':'RushYds', 'rushing_tds':'RushTD', 'rushing_fumbles':'Fmb', 'receptions':'Rec', 'targets':'Tgt',
                          'receiving_yards':'RecYds', 'receiving_tds':'RecTD'}, inplace=True)

qb_data = fantasy_data[fantasy_data['position'] == 'QB']
qb_data['Cmp%'] = qb_data['Cmp'] / qb_data['PassAtt']
qb_data['Cmp%'] = qb_data['Cmp%'].fillna(0)

pos_to_keep = ['RB', 'FB']
rbfb_data = fantasy_data[fantasy_data['position'].isin(pos_to_keep)]

pos_to_keep = ['WR', 'TE']
wrte_data = fantasy_data[fantasy_data['position'].isin(pos_to_keep)]

############################

features_qb = ["PYds/G", "PassAtt", "Cmp%", "RushAtt"]
target = ["PassYds", "PassTD", "Int", "RushYds", "RushTD"]

train_data_qb = qb_data[qb_data['season'] != 2024]
test_data_qb = qb_data[qb_data['season'] == 2024]

X_train_qb = train_data_qb[features_qb]
y_train_qb = train_data_qb[target]
X_test_qb = test_data_qb[features_qb]
y_test_qb = test_data_qb[target]

model_qb = MultiOutputRegressor(RandomForestRegressor())

model_qb.fit(X_train_qb, y_train_qb)

y_pred_qb = model_qb.predict(X_test_qb)

#############################################

#These features are for RB
features_rb = ["RYds/G", "RushAtt", "Tgt", "rushing_epa", "racr"]
target = ["RushYds", "RushTD", "Fmb", "Rec", "RecYds", "RecTD"]

train_data_rb = rbfb_data[rbfb_data['season'] != 2024]
test_data_rb = rbfb_data[rbfb_data['season'] == 2024]

X_train_rb = train_data_rb[features_rb]
y_train_rb = train_data_rb[target]
X_test_rb = test_data_rb[features_rb]
y_test_rb = test_data_rb[target]

model_rb = MultiOutputRegressor(RandomForestRegressor())

model_rb.fit(X_train_rb, y_train_rb)

y_pred_rb = model_rb.predict(X_test_rb)

#############################################

#These features are for WR
features_wr = ["PYds/G", "Tgt", "receiving_epa", "pacr"]
target = ["RecYds", "RecTD", "Rec"]

train_data_wr = wrte_data[wrte_data['season'] != 2024]
test_data_wr = wrte_data[wrte_data['season'] == 2024]

X_train_wr = train_data_wr[features_wr]
y_train_wr = train_data_wr[target]
X_test_wr = test_data_wr[features_wr]
y_test_wr = test_data_wr[target]

model_wr = MultiOutputRegressor(RandomForestRegressor())

model_wr.fit(X_train_wr, y_train_wr)

y_pred_wr = model_wr.predict(X_test_wr)

##################################

with open('new_model_qb.pkl', 'wb') as file:
    pickle.dump(model_qb, file)
with open('new_model_rb.pkl', 'wb') as file:
    pickle.dump(model_rb, file)
with open('new_model_wr.pkl', 'wb') as file:
    pickle.dump(model_wr, file)

with open('new_model_qb.pkl', 'rb') as file:
    loaded_model_qb = pickle.load(file)