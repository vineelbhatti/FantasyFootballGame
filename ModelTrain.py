import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import pickle

fantasy_data = pd.read_csv("/Users/vineel/PycharmProjects/FantasyFootballGame/FantasyData21-23.csv")

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

fantasy_data.rename(columns={'FantPos':'position'}, inplace=True)
fantasy_data.rename(columns={'Tm':'Team'}, inplace=True)
fantasy_data = fantasy_data.replace(r'[^\w\s]|_', '', regex=True)
fantasy_data['Team'] = fantasy_data['Team'].map(fix_abv).fillna(fantasy_data['Team'])
fantasy_data = fantasy_data[fantasy_data.Team != '2TM']
fantasy_data = fantasy_data[fantasy_data.Team != '3TM']
fantasy_data.dropna(subset=['PPR'], inplace=True)
fantasy_data.fillna(0, inplace=True)
fantasy_data.dropna(subset=['Team'], inplace=True)

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

fantasy_data['StarRating'] = fantasy_data['PPG'].apply(determine_stars)

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

fantasy_data = count_star_teammates(fantasy_data)
fantasy_data['GoodTeammates'] = fantasy_data['5StarTeammates'] + fantasy_data['4StarTeammates'] + fantasy_data['3StarTeammates']

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
fantasy_data = add_wr_quality_to_qb_data(fantasy_data, fantasy_data[fantasy_data['position'].isin(wr_pos)])

fantasy_data['Cmp%'] = fantasy_data['Cmp']/fantasy_data['PassAtt']


qb_data = fantasy_data[fantasy_data['position'] == 'QB']
qb_data = qb_data[["Player", "position", "Team", "Year", "Age", "StarRating", "PPG", "PPR", "GoodTeammates", "RushYds/Gm", "PassYds/Gm", "PassTD/Gm", "PassAtt", "GS", "Cmp", "wr_quality", "Cmp%"]]
pos_to_keep = ['RB', 'FB']
rbfb_data = fantasy_data[fantasy_data['position'].isin(pos_to_keep)]
rbfb_data = rbfb_data[["Player", "position", "Team", "Year", "Age", "StarRating", "PPG", "PPR", "GoodTeammates", "RushYds/Gm", "RushTD/Gm", "Rec/Gm", "RecYds/Gm", "RushAtt/Gm", "Y/A", "RushAtt"]]
pos_to_keep = ['WR', 'TE']
wrte_data = fantasy_data[fantasy_data['position'].isin(pos_to_keep)]
wrte_data = wrte_data[["Player", "position", "Team", "Year", "Age", "StarRating", "PPG", "PPR", "GoodTeammates", "Tgt/Gm", "Rec/Gm", "RecYds/Gm", "RecTD/Gm", "Y/R"]]

features_qb = ["wr_quality", "GoodTeammates", "RushYds/Gm", "PassYds/Gm", "PassTD/Gm", "Cmp%", "PassAtt", "GS"]
target = "PPG"

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

print(f'QB Mean Absolute Error: {mae}')

#############################################

#These features are for RB
features_rb = ["GoodTeammates", "RushYds/Gm", "RushTD/Gm", "Rec/Gm", "RecYds/Gm", "RushAtt/Gm", "Y/A", "RushAtt"]
target = "PPG"

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

#############################################

#These features are for WR
features_wr = ["GoodTeammates", "Tgt/Gm", "Rec/Gm", "RecYds/Gm", "RecTD/Gm", "Y/R"]
target = "PPG"

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

print(f'WR Mean Absolute Error: {mae_wr}')

with open('model_qb.pkl', 'wb') as file:
    pickle.dump(model_qb, file)
with open('model_rb.pkl', 'wb') as file:
    pickle.dump(model_rb, file)
with open('model_wr.pkl', 'wb') as file:
    pickle.dump(model_wr, file)

with open('model_qb.pkl', 'rb') as file:
    loaded_model_qb = pickle.load(file)