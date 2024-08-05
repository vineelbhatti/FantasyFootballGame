import pandas as pd
import random
import time
import requests

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score

from flask import *
from flask_bootstrap import Bootstrap
from flask_pymongo import PyMongo

app = Flask(__name__)

GIPHY_API_KEY = 'Ctxh2h5PyG5btxvRF3LoBeTjj5nmrnQJ'

fantasy_data_2023 = pd.read_csv("FantasyData2023.csv")

full_df = pd.read_csv("adp_merged_7_17.csv")
df_stats = pd.read_csv("fantasy_merged_7_17.csv")
full_df.rename(columns={'name':'Player'}, inplace=True)
df_allyears = pd.merge(full_df, df_stats, on = ["Player", "Year"])
df_allyears.rename(columns={'team':'Team'}, inplace=True)
win = pd.read_csv("Team Win Percentage - Sheet1 (2).csv")
df_win = pd.merge(df_allyears, win, on = ["Team", "Year"])

df_win["TotalYds"] = df_win["RecYds"] + df_win["RushYds"] + df_win["Yds"]
df_win["TotalTD"] = df_win["RecTD"] + df_win["RushTD"] + df_win["TD"]

enc = LabelEncoder()
enc.fit(df_win['position'])
df_win['position_enc'] = enc.transform(df_win['position'])

df_2022 = df_win[df_win['Year'] == 2022]
fantasy_data_2023 = fantasy_data_2023[fantasy_data_2023['Player'].isin(df_2022['Player'])]
df_2022 = df_2022[df_2022['Player'].isin(fantasy_data_2023['Player'])]
fantasy_data_2023 = pd.merge(fantasy_data_2023,df_2022[['position','Player']],on='Player', how='left')

X = df_2022[["position_enc", "adp", "WinP", "Age", "TotalYds", "TotalTD", ]]
Y = fantasy_data_2023["Pts"]

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=20)
rf_model = RandomForestRegressor()
rf_model.fit(X_train, y_train)

y_predictions = rf_model.predict(X_test)
fantasy_data_2023["PredPts"] = rf_model.predict(X)
fantasy_data_2023 = fantasy_data_2023.sort_values('PredPts', ascending=False)

fantasy_data_2023 = fantasy_data_2023.reset_index(drop=True)

fantasy_data_2023['TD'] = fantasy_data_2023['TD'] = fantasy_data_2023['TD.1'] + fantasy_data_2023['TD.2'] + fantasy_data_2023['TD.3']
fantasy_data_2023 = fantasy_data_2023.drop(columns=['TD.1', 'TD.2', 'TD.3'])

fantasy_data_2023["Yds"] = fantasy_data_2023["Yds"].replace(',','', regex=True).astype(int)
fantasy_data_2023["Yds.1"] = fantasy_data_2023["Yds.1"].replace(',','', regex=True).astype(int)
fantasy_data_2023["Yds.2"] = fantasy_data_2023["Yds.2"].replace(',','', regex=True).astype(int)

fantasy_data_2023["Yds"] = pd.to_numeric(fantasy_data_2023["Yds"])
fantasy_data_2023["Yds.1"] = pd.to_numeric(fantasy_data_2023["Yds.1"])
fantasy_data_2023["Yds.2"] = pd.to_numeric(fantasy_data_2023["Yds.2"])

fantasy_data_2023['Yds'] = fantasy_data_2023['Yds'] = fantasy_data_2023['Yds.1'] + fantasy_data_2023['Yds.2']
fantasy_data_2023 = fantasy_data_2023.drop(columns=['Yds.1', 'Yds.2', 'Att', 'Cmp', 'Int', '2Pt', 'Att.1', '2Pt.1', '2Pt.2', 'Rec', 'FL'])

global draft_board
draft_board = fantasy_data_2023.copy()

teams = [[], [], []]
teams_need = [["QB", "WR", "WR", "RB", "RB"], ["QB", "WR", "WR", "RB", "RB"], ["QB", "WR", "WR", "RB", "RB"]]
draft_message = ""
drafted_players = []
html_table = draft_board.to_html()

user_team = 0

def search_giphy(player_name):
  search_url = f'https://api.giphy.com/v1/gifs/search?api_key={GIPHY_API_KEY}&q={player_name}&limit=1'
  response = requests.get(search_url)
  data = response.json()
  if data['data']:
    gif_url = data['data'][0]['images']['original']['url']
    return gif_url
  return None


@app.route('/', methods=['GET', 'POST'])
def home():
  return render_template('home.html')

@app.route('/draft', methods=['GET', 'POST'])
def draft():
  print('draft called')
  for team in range(3):
    print(team)
    if team == user_team:
      user_draft()
    else:
      auto_draft(team)
    print(teams)
  return render_template('draft.html', table=html_table, team=teams[user_team], draft_message=draft_message, drafted_players=drafted_players)

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
      gif_url = search_giphy(player_name)
    return player_name

def auto_draft(team):
  global draft_board
  chosen = False
  rank = 0
  while chosen == False:
    if draft_board.iloc[rank]["position"] in teams_need[team]:
      teams[team].append(draft_board.iloc[rank]["Player"])
      draft_message = f"Team {team} drafted {draft_board.iloc[rank]['Player']}"
      gif_url = search_giphy(draft_board.iloc[rank]["Player"])
      drafted_players.append(draft_board.iloc[rank])
      teams_need[team].remove(draft_board.iloc[rank]["position"])
      draft_board = draft_board.iloc[rank + 1:]
      chosen = True
    rank += 1

if __name__ == '__main__':
    app.run(debug=True)