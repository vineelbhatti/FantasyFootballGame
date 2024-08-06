import pandas as pd
import random
import time
import requests
from bs4 import BeautifulSoup

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


from flask import *
from flask_bootstrap import Bootstrap
from flask_pymongo import PyMongo

import pandas as pd
pd.options.mode.chained_assignment = None

app = Flask(__name__)

app.secret_key = 'your_secret_key'

Bootstrap(app)
api_key = 'AIzaSyCqJBDynKiv3iPjc1q_S2JAbXkfBBkGi74'

fantasy_data_2023 = pd.read_csv("FantasyData21-23.csv")

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

fantasy_data_2023.fillna(0, inplace=True)

def determine_stars(ppg):
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


fantasy_data_2023['StarRating'] = fantasy_data_2023['PPG'].apply(determine_stars)

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

#print(fantasy_data_2023[['Player', 'position', 'StarRating', '5StarTeammates', '4StarTeammates']].head(20))

#####################
qb_data = fantasy_data_2023[fantasy_data_2023['position'] == 'QB']
qb_data = qb_data[["Player", "GS", "position", "StarRating", "GoodTeammates", '5StarTeammates', '4StarTeammates', '3StarTeammates', '2StarTeammates', '1StarTeammates', "Team", "Cmp", "PassAtt", "Year", "Age", "PPR", "Y/A", "VBD", "RushYds/Gm", "PassYds/Gm", "PassTD/Gm", "RushTD/Gm", "Fmb/Gm", "Cmp/Gm", "PassAtt/Gm", "Int/Gm", "RushAtt/Gm", "W-L%", "PF", "SRS", "OSRS"]]
pos_to_keep = ['RB', 'FB']
rbfb_data = fantasy_data_2023[fantasy_data_2023['position'].isin(pos_to_keep)]
rbfb_data = rbfb_data[["Player", "position", "StarRating", "GoodTeammates", '5StarTeammates', '4StarTeammates', '3StarTeammates', '2StarTeammates', '1StarTeammates', "Team", "Year", "Age", "PPR", "Y/A", "RushYds/Gm", "RushAtt", "Tgt/Gm", "Rec/Gm", "RushTD/Gm", "RecYds/Gm", "RecTD/Gm", "Fmb/Gm", "RushAtt/Gm", "W-L%", "PF", "SRS", "OSRS"]]
pos_to_keep = ['WR', 'TE']
wrte_data = fantasy_data_2023[fantasy_data_2023['position'].isin(pos_to_keep)]
wrte_data = wrte_data[["Player", "position", "StarRating", "GoodTeammates", '5StarTeammates', '4StarTeammates', '3StarTeammates', '2StarTeammates', '1StarTeammates', "Team", "Year", "Age", "PPR", "Tgt/Gm", "Rec/Gm", "RecYds/Gm", "RecTD/Gm", "Fmb/Gm", "Y/R", "W-L%", "PF", "SRS", "OSRS"]]

#print(qb_data[['Player', 'position']].head(50))

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

qb_data = add_wr_quality_to_qb_data(qb_data, wrte_data)

qb_data['Cmp%'] = qb_data['Cmp']/qb_data['PassAtt']

#These features are for QB
features_qb = ['wr_quality', "RushYds/Gm", "PassYds/Gm", "PassTD/Gm", "Cmp%", "PassAtt", "GS"]
#features_qb += [f"{stat_qb}_diff_previous_year" for stat_qb in features_qb]
target = "PPR"

train_data_qb = qb_data[qb_data['Year'] != 2023]
test_data_qb = qb_data[qb_data['Year'] == 2023]

X_train_qb = train_data_qb[features_qb]
y_train_qb = train_data_qb[target]
X_test_qb = test_data_qb[features_qb]
y_test_qb = test_data_qb[target]

model_qb = RandomForestRegressor()

model_qb.fit(X_train_qb, y_train_qb)

y_pred_qb = model_qb.predict(X_test_qb)

mae = mean_absolute_error(y_test_qb, y_pred_qb)
mse = mean_squared_error(y_test_qb, y_pred_qb)
#r2 = r2_score(y_test, y_pred)

print(f'QB Mean Absolute Error: {mae}')

df_2023_predictions_qb = pd.DataFrame()
df_2023_predictions_qb['y_test'] = y_test_qb
df_2023_predictions_qb['y_pred'] = y_pred_qb

X_2023 = qb_data[qb_data['Year'] == 2023][features_qb]
y_2024_pred_qb = model_qb.predict(X_2023)
qb_data.loc[qb_data['Year'] == 2023, 'PredPts'] = y_2024_pred_qb

#print(df_2023_predictions_qb.head(10))
qb_data = qb_data.sort_values('PredPts', ascending=False)
#print(qb_data.head(10))
print(features_qb)
print(model_qb.feature_importances_)

#############################################

#These features are for RB
features_rb = ["GoodTeammates", "RushYds/Gm", "RushTD/Gm", "Rec/Gm", "RecYds/Gm", "RushAtt/Gm", "Y/A", "RushAtt"]
#features_rb += [f"{stat_rb}_diff_previous_year" for stat_rb in features_rb]
#features_rb += ["Age"]
target = "PPR"

train_data_rb = rbfb_data[rbfb_data['Year'] != 2023]
test_data_rb = rbfb_data[rbfb_data['Year'] == 2023]

X_train_rb = train_data_rb[features_rb]
y_train_rb = train_data_rb[target]
X_test_rb = test_data_rb[features_rb]
y_test_rb = test_data_rb[target]

model_rb = RandomForestRegressor()

model_rb.fit(X_train_rb, y_train_rb)

y_pred_rb = model_rb.predict(X_test_rb)

mae_rb = mean_absolute_error(y_test_rb, y_pred_rb)

print(f'RB Mean Absolute Error: {mae_rb}')

df_2023_predictions_rb = pd.DataFrame()
df_2023_predictions_rb['y_test_rb'] = y_test_rb
df_2023_predictions_rb['y_pred_rb'] = y_pred_rb

X_2023_rb = rbfb_data[rbfb_data['Year'] == 2023][features_rb]
y_2024_pred_rb = model_rb.predict(X_2023_rb)
rbfb_data.loc[rbfb_data['Year'] == 2023, 'PredPts'] = y_2024_pred_rb

rbfb_data = rbfb_data.sort_values('PredPts', ascending=False)
#print(rbfb_data.head(10))
#print(features_rb)
#print(model_rb.feature_importances_)

#############################################

#These features are for WR
features_wr = ["GoodTeammates", "Tgt/Gm", "Rec/Gm", "RecYds/Gm", "RecTD/Gm", "Y/R"]
#features_wr += [f"{stat_wr}_diff_previous_year" for stat_wr in features_wr]
features_wr += ["Age"]
target = "PPR"

train_data_wr = wrte_data[wrte_data['Year'] != 2023]
test_data_wr = wrte_data[wrte_data['Year'] == 2023]

X_train_wr = train_data_wr[features_wr]
y_train_wr = train_data_wr[target]
X_test_wr = test_data_wr[features_wr]
y_test_wr = test_data_wr[target]

model_wr = RandomForestRegressor()

model_wr.fit(X_train_wr, y_train_wr)

y_pred_wr = model_wr.predict(X_test_wr)

mae_wr = mean_absolute_error(y_test_wr, y_pred_wr)
mse_wr = mean_squared_error(y_test_wr, y_pred_wr)
#r2 = r2_score(y_test, y_pred)

print(f'WR Mean Absolute Error: {mae_wr}')
#print(f'WR Mean Squared Error: {mse_wr}')
#print(f'QB R^2 Score: {r2}')

df_2023_predictions_wr = pd.DataFrame()
df_2023_predictions_wr['y_test_wr'] = y_test_wr
df_2023_predictions_wr['y_pred_wr'] = y_pred_wr

X_2023_wr = wrte_data[wrte_data['Year'] == 2023][features_wr]
y_2024_pred_wr = model_wr.predict(X_2023_wr)
wrte_data.loc[wrte_data['Year'] == 2023, 'PredPts'] = y_2024_pred_wr

#print(df_2023_predictions_wr.head(10))
wrte_data = wrte_data.sort_values('PredPts', ascending=False)
#print(wrte_data[['Player', 'Year', 'PredPts', 'GoodTeammates']].head(10))
#print(features_wr)
#print(model_wr.feature_importances_)

#print(fantasy_data_2023[['PPR']].describe())

fantasy_data_2023 = fantasy_data_2023[fantasy_data_2023['Year'] == 2023]

fantasy_data_2023 = fantasy_data_2023.merge(qb_data[['Player', 'Year', 'PredPts']], on=['Player', 'Year'], how='left')
fantasy_data_2023 = fantasy_data_2023.merge(rbfb_data[['Player', 'Year', 'PredPts']], on=['Player', 'Year'], how='left')
fantasy_data_2023 = fantasy_data_2023.merge(wrte_data[['Player', 'Year', 'PredPts']], on=['Player', 'Year'], how='left')

fantasy_data_2023['PredPts'] = fantasy_data_2023['PredPts'].fillna(fantasy_data_2023['PredPts_x']).fillna(fantasy_data_2023['PredPts_y'])
fantasy_data_2023 = fantasy_data_2023.drop(columns=['PredPts_x', 'PredPts_y'])

fantasy_data_2023["PredAvgPts"] = fantasy_data_2023["PredPts"]/17
fantasy_data_2023 = fantasy_data_2023.sort_values('PredPts', ascending=False)

fantasy_data_2023 = fantasy_data_2023.reset_index(drop=True)

#print(fantasy_data_2023[['Player', 'position', 'StarRating', 'PredPts']].head(20))

fantasy_data_2023 = pd.merge(fantasy_data_2023, schedules, on=['Team'], how='left')

fantasy_data_2023['Injury'] = 0
fantasy_data_2023['TotalPts'] = 0
fantasy_data_2023['weekly_pred_pts'] = 0
fantasy_data_2023['Starting'] = "No"
fantasy_data_2023['FantasyTeam'] = "FA"


global draft_board
draft_board = fantasy_data_2023.copy()

global current_week
current_week = 0

teams = [[], [], [], [], [], [], [], []]
teams_need = [["QB", "WR", "WR", "RB", "RB"], ["QB", "WR", "WR", "RB", "RB"], ["QB", "WR", "WR", "RB", "RB"], ["QB", "WR", "WR", "RB", "RB"], ["QB", "WR", "WR", "RB", "RB"], ["QB", "WR", "WR", "RB", "RB"], ["QB", "WR", "WR", "RB", "RB"], ["QB", "WR", "WR", "RB", "RB"]]
team_points = [0, 0, 0, 0, 0, 0, 0, 0]
team_wins = [0, 0, 0, 0, 0, 0, 0, 0]
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
        teams_by_num.append(None)  # Add a dummy team if the number of teams is odd

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


# List of teams as integers from 0 to 7
teams_by_num = list(range(8))
schedule = round_robin_schedule(teams_by_num)


@app.route('/', methods=['GET', 'POST'])
def home():
    return render_template('home.html')

@app.route('/after_draft', methods=['GET', 'POST'])
def show_teams():
    return render_template('after_draft.html', teams=teams)

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
    return render_template('player_details.html', player_name=player_name, player_image=player_image, total_pts=player_data['TotalPts'], fantasy_team=fantasy_team)


def fetch_player_image(player_name):
    search_query = f'{player_name} headshot'
    search_url = f'https://www.bing.com/images/search?q={search_query}'
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    response = requests.get(search_url, headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')
    image = soup.find('img', {'class': 'mimg'})
    if image:
        return image['src']
    else:
        return None

@app.route('/WaiverClaims', methods=['GET', 'POST'])
def waiver_claims():
    return render_template('WaiverClaims.html')

@app.route('/LineupChanges', methods=['GET', 'POST'])
def set_lineup():
    global current_week
    qblist = []
    rblist = []
    wrlist = []
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
        weekly_pred_pts = round(((fantasy_data_2023.loc[fantasy_data_2023['Player'] == player, "PredPts"].values[0]) / 17), 1)
        opponent = str(fantasy_data_2023.loc[fantasy_data_2023['Player'] == player, 'Wk'+str(current_week+1)].values[0])[1:]
        if fantasy_data_2023.loc[fantasy_data_2023['Player'] == player, "position"].values[0] == 'RB':
            defense_ranking = run_defense.loc[run_defense['Team'] == opponent, 'Rank'].values[0]
        else:
            defense_ranking = pass_defense.loc[pass_defense['Team'] == opponent, 'Rank'].values[0]
        if defense_ranking < 16:
            fantasy_data_2023.loc[fantasy_data_2023['Player'] == player, "weekly_pred_pts"].values[0]
        else:
            fantasy_data_2023.loc[fantasy_data_2023['Player'] == player, "weekly_pred_pts"].values[0] -= random.randint(0, 3)
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
            pred_list.append(fantasy_data_2023.loc[fantasy_data_2023['Player'] == qb, "weekly_pred_pts"].values[0])
        for index in range(len(pred_list)-1):
            if pred_list[index] == max(pred_list):
                fantasy_data_2023.loc[fantasy_data_2023['Player'] == qblist[index], "Starting"] = 'Yes'
        pred_list = []
        for rb in rblist:
            pred_list.append(fantasy_data_2023.loc[fantasy_data_2023['Player'] == rb, "weekly_pred_pts"].values[0])

        for x in range(2):
            max_index = pred_list.index(max(pred_list))
            fantasy_data_2023.loc[fantasy_data_2023['Player'] == rblist[max_index], "Starting"] = 'Yes'
            pred_list[max_index] = -float('inf')

        pred_list = []
        for wr in wrlist:
            pred_list.append(fantasy_data_2023.loc[fantasy_data_2023['Player'] == wr, "weekly_pred_pts"].values[0])

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
    global schedule
    current_week+=1
    fantasy_data_2023['Injury']-=1
    team_num = 0
    opponent_list = [[],[],[]]
    weather_note = "Good conditions"
    weather_impact = 0
    new_injury = None

    fantasy_data_2023[str("Wk" + str(current_week))+"Pts"] = 0
    for index, row in fantasy_data_2023.iterrows():
        player = row['Player']
        sim_player_points(player)

    dfteam_0 = pd.DataFrame()
    for team in teams:
        for player in team:
            #print(fantasy_data_2023.loc[fantasy_data_2023['Player'] == player, "Starting"].values[0] == "Yes")
            if fantasy_data_2023.loc[fantasy_data_2023['Player'] == player, "Starting"].values[0] == "Yes":
                team_points[team_num]+=round(fantasy_data_2023.loc[fantasy_data_2023['Player'] == player, str("Wk" + str(current_week)+"Pts")].values[0], 1)
            player_row = fantasy_data_2023.loc[fantasy_data_2023['Player'] == player]
            if team_num == 0:
                dfteam_0 = pd.concat([dfteam_0, player_row], ignore_index=True)
        team_num+=1
    dfteam_0 = dfteam_0[['Player', str("Wk" + str(current_week)+"Pts"), 'TotalPts', 'Injury', 'GameNotes', 'Starting']]
    dfteam_0_html = dfteam_0.to_html(classes='table table-striped', index=False)
    players_points = fantasy_data_2023.groupby('FantasyTeam').apply(
        lambda x: x[['Player', 'Wk' + str(current_week) + 'Pts']].to_dict(orient='records')).to_dict()
    update_winners()
    return render_template('WeeklyStats.html', table=fantasy_data_2023[['Player', str("Wk" + str(current_week)+"Pts"), 'TotalPts', 'Injury', 'GameNotes', 'Starting']].to_html(), dfteam_0=dfteam_0_html, team_wins=team_wins, current_week=str(current_week), matchups=schedule[current_week-1], team_points=team_points, players_points=players_points)

def update_winners():
    global schedule
    global current_week
    global team_points
    global team_wins
    current_week_schedule = schedule[current_week-1]
    for matchup in current_week_schedule:
        points = []
        for team in matchup:
            points.append(team_points[int(team)])
        print(points)
        max_points = max(points)
        print(max_points)
        max_index = points.index(max_points)
        winning_team = matchup[max_index]
        team_wins[int(winning_team)] += 1


def sim_player_points(player):
    global cold_teams
    global current_week
    global fantasy_data_2023
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
    points_scored = round(((fantasy_data_2023.loc[fantasy_data_2023['Player'] == player, "PredPts"].values[0] / 17) * (
                1 - (opponent_ranking / 100)) + random.randint(-10, 10) + weather_impact + home_impact), 1)
    #Injuries
    if fantasy_data_2023.loc[fantasy_data_2023['Player'] == player, "Injury"].values[0] > 0:
        points_scored = 0
    injury_chance = random.randint(0, 10)
    if injury_chance < 2:
        if random.randint(0, 10) <= 1:
            fantasy_data_2023.loc[fantasy_data_2023['Player'] == player, 'Injury'] = 18
            new_injury = "Season ending injury"
        elif random.randint(0, 10) <= 3:
            fantasy_data_2023.loc[fantasy_data_2023['Player'] == player, 'Injury'] = 10
            new_injury = "Severe injury"
        elif random.randint(0, 10) <= 5:
            fantasy_data_2023.loc[fantasy_data_2023['Player'] == player, 'Injury'] = 5
            new_injury = "Moderate injury"
        elif random.randint(0, 10) <= 7:
            fantasy_data_2023.loc[fantasy_data_2023['Player'] == player, 'Injury'] = 3
            new_injury = "Mild injury"
        else:
            fantasy_data_2023.loc[fantasy_data_2023['Player'] == player, 'Injury'] = 1
            new_injury = "Day-to-day injury"
        game_notes += (", " + new_injury)
    set_qb_points(fantasy_data_2023)
    fantasy_data_2023.loc[fantasy_data_2023['Player'] == player, "TotalPts"] += points_scored
    fantasy_data_2023.loc[fantasy_data_2023['Player'] == player, str("Wk" + str(current_week))+"Pts"] = round(points_scored, 1)
    fantasy_data_2023.loc[fantasy_data_2023['Player'] == player, "GameNotes"] = game_notes
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

    return render_template('draft.html', table=draft_board[['Player', 'position', 'Team', 'PredPts', 'PredAvgPts', 'StarRating']].to_html(), team=teams[user_team], draft_message=draft_message, drafted_players=drafted_players, gif_url=gif_url)

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
          draft_board = draft_board.iloc[rank + 1:]
          chosen = True
    rank += 1

if __name__ == '__main__':
    app.run(debug=True)
