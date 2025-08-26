# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 16:33:12 2024
football player projections from fbref data
@author: Subramanya.Ganti
"""
#%% initialize
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import requests
from bs4 import BeautifulSoup
import time
from itertools import combinations
from sklearn import linear_model
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import poisson
from scipy.stats import skellam
from io import StringIO
from datetime import datetime

from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

path = "C:/Users/Subramanya.Ganti/Downloads/Sports/football"
#path = "C:/Users/uttam/Desktop/Sports/football"
valid_leagues = ['serie a','bundesliga','premier league','la liga','ligue un',
                 'championship','liga portugal','eredivisie','serie b','belgian pro league',
                 'brazilian serie a','mls','liga mx',
                 'champions league','europa league','conference league']

proj_year = 2026
standard = 'premier league'

#%% functions
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36'
}

def league_mapping(code):
    league_code = {
        9: 'Premier-League', 11: 'Serie-A', 12: 'La-Liga', 13: 'Ligue-1', 20: 'Bundesliga',
        10: 'Championship', 18: 'Seire-B',
        23: 'Eredivisie', 32: 'Primeira-Liga', 37: 'Belgian-Pro-League', 
        24: 'Serie-A', 31: 'Liga-MX', 21: 'Liga-Profesional-Argentina', 22: 'Major-League-Soccer',
        676: 'UEFA-Euro', 685: 'Copa-America', 1: 'World-Cup', 
        8: 'Champions-League', 19: 'Europa-League', 882: 'Conference-League', 14:'Copa-Libertadores'
        }
    try:
        league = league_code[code]
    except KeyError:
        print("unknown league was selected, premier league chosen as default")
        code = 9
        league = 'Premier-League'
    return code,league

def code_mapping(league):
    code_league = {
        'serie a':11,'bundesliga':20,'premier league':9,'la liga':12,'ligue un':13,
        'championship':10,'liga portugal':32,'eredivisie':23,'serie b':18,'belgian pro league':37,
        'brazilian serie a':24,'mls':22,'liga mx':31,
        'champions league':8,'europa league':19,'conference league':882
        }
    try:
        code = code_league[league]
    except KeyError:
        print("unknown code was selected, premier league chosen as default")
        code = 9
        league = 'premier league'
    return league,code

def fbref_league_fixtures(season,code):
    code,league = league_mapping(code)
    if(code in [24,21,22,14]):
        table = pd.read_html(f'https://fbref.com/en/comps/{code}/{season}/{season}-{league}-Stats')
    else:
        table = pd.read_html(f'https://fbref.com/en/comps/{code}/{season}-{season+1}/{season}-{season+1}-{league}-Stats')
    return table

def opp_touches_error(start,end,code):
    all_ot = []
    season = start
    while(season <= end):
        time.sleep(6.01)
        print(code,season)
        table = fbref_league_fixtures(season,code)
        try:
            ot = table[19]
            ot.columns = ot.columns.droplevel(0)
            ot = ot[['90s','Squad','Touches']]
        except KeyError:
            try:
                ot = table[21]
                ot.columns = ot.columns.droplevel(0)
                ot = ot[['90s','Squad','Touches']]
            except KeyError:
                try:
                    ot = table[27]
                    ot.columns = ot.columns.droplevel(0)
                    ot = ot[['90s','Squad','Touches']]
                except KeyError:
                    ot = table[29]
                    ot.columns = ot.columns.droplevel(0)
                    ot = ot[['90s','Squad','Touches']]
        ot['Season'] = season
        ot['Squad'] = ot['Squad'].str.replace('vs ','')
        all_ot.append(ot)
        season += 1
    all_ot = pd.concat(all_ot)
    return all_ot

def fbref_team_ids(season,code):
    code,league = league_mapping(code)
    if(code in [24,21,22]): #leagues start in winter
        url = f'https://fbref.com/en/comps/{code}/{season}/{season}-{league}-Stats'
    else: #leagues start in summer
        url = f'https://fbref.com/en/comps/{code}/{season}-{season+1}/{season}-{season+1}-{league}-Stats'
    #take care to verify why this bypass is needed
    data  = requests.get(url,verify=False,headers=headers).text
    #data  = requests.get(url).text
    soup = BeautifulSoup(data,"html.parser")
    if(code in [8,19,882,14,676,685,1]):
        #continental leagues
        links = BeautifulSoup(data,"html.parser").select('table a')
        urls = [link['href'] for link in links]
        urls = list(set(urls))
        urls = [item for item in urls if 'squads' in item]
    else:
        #country leagues
        links = BeautifulSoup(data,"html.parser").select('th a')
        urls = [link['href'] for link in links]
        urls = list(set(urls))
    #print(urls)
    urls = pd.DataFrame(urls, columns=['links'])
    urls['team'] = urls['links'].str.split("/").str[-1]
    urls['team'] = urls['team'].str.replace('-Stats','')
    urls['code'] = urls['links'].str.split("/").str[3]
    urls['season'] = season
    urls = urls[['team','code','season']]
    return urls

def player_stats(club,code,season,league_code):
    if(league_code in [8,19,882,14,676,685,1]):
        #ref = pd.read_html(f'https://fbref.com/en/squads/{code}/{season}-{season+1}/c{league_code}/{club}-Stats')
        url = f'https://fbref.com/en/squads/{code}/{season}-{season+1}/c{league_code}/{club}-Stats'
    elif(league_code in [24,21,22]): #leagues start in winter
        #ref = pd.read_html(f'https://fbref.com/en/squads/{code}/{season}/{club}-Stats')
        url = f'https://fbref.com/en/squads/{code}/{season}/{club}-Stats'
    else: #leagues start in summer
        #ref = pd.read_html(f'https://fbref.com/en/squads/{code}/{season}-{season+1}/{club}-Stats')
        url = f'https://fbref.com/en/squads/{code}/{season}-{season+1}/{club}-Stats'
    
    #url = 'https://fbref.com/en/squads/943e8050/2023-2024/9/Burnley-Stats-Premier-League'
    
    data  = requests.get(url,verify=False,headers=headers).text
    #data  = requests.get(url).text
    soup = BeautifulSoup(data,"html.parser")
    tables = soup.find_all('table')
    
    ref = []
    for i, table in enumerate(tables):
        try:
            # pd.read_html can directly parse a table element converted to string
            df = pd.read_html(StringIO(str(table)))[0]
            ref.append(df)
            #print(f"Table {i+1} successfully converted to DataFrame.")
        except Exception as e:
            print(f"{club},{season},Could not convert Table {i+1} to DataFrame: {e}")

    basic = ref[0]
    basic.columns = basic.columns.droplevel(0)
    basic = basic[['Player', 'Nation', 'Pos', 'Age', 'MP', 'Min', '90s','Gls','PK','PKatt','xG','npxG']]
    basic.columns = ['Player', 'Nation', 'Pos', 'Age', 'MP', 'Min', '90s', 'Gls', 'Gls/90','PK', 'PKatt', 'xG', 'xG/90', 'npxG', 'npxG/90']
    basic = basic[['Player', 'Nation', 'Pos', 'Age', 'MP', 'Min', '90s','Gls','PK','PKatt','xG','npxG']]
    keeper_basic = ref[2]
    keeper_basic.columns = keeper_basic.columns.droplevel(0)
    keeper_basic = keeper_basic[['Player', 'Nation', 'Pos', 'Age','SoTA','Saves','Save%','PKatt', 'PKA', 'PKsv', 'PKm']]
    keeper_basic.columns = ['Player', 'Nation', 'Pos', 'Age','SoTA','Saves','Save%','PKSave%','PKatt', 'PKA', 'PKsv', 'PKm']
    keeper_adv = ref[3]
    keeper_adv.columns = keeper_adv.columns.droplevel(0)
    keeper_adv = keeper_adv[['Player', 'Nation', 'Pos', 'Age', 'OG', 'PSxG', 'PSxG/SoT', 'PSxG+/-']]
    shots = ref[4]
    shots.columns = shots.columns.droplevel(0)
    shots = shots[['Player', 'Nation', 'Pos', 'Age','Sh', 'SoT']]
    passing = ref[5]
    passing.columns = passing.columns.droplevel(0)
    passing.columns = ['Player', 'Nation', 'Pos', 'Age', '90s', 'TotCmp', 'TotAtt', 'TotCmp%', 'TotDist', 'PrgDist',
           'Cmp', 'Att', 'Cmp%', 'Cmp', 'Att', 'Cmp%', 'Cmp', 'Att', 'Cmp%', 'Ast',
           'xAG', 'xA', 'A-xAG', 'KP', '1/3', 'PPA', 'CrsPA', 'PrgP','Matches']
    passing = passing[['Player', 'Nation', 'Pos', 'Age','TotCmp', 'TotAtt', 'TotCmp%', 'TotDist', 'PrgDist','Ast','xAG', 'xA','KP','PrgP']]
    #check from here
    pass_types = ref[6]
    pass_types.columns = pass_types.columns.droplevel(0)
    pass_types = pass_types[['Player', 'Nation', 'Pos', 'Age','Live', 'Dead', 'FK','TB', 'Sw', 'Crs', 'TI', 'CK']]
    offball = ref[8]
    offball.columns = offball.columns.droplevel(0)
    offball.columns = ['Player', 'Nation', 'Pos', 'Age', '90s', 'Tkl', 'TklW', 'Def 3rd', 'Mid 3rd', 'Att 3rd',
           'cTkl', 'cAtt', 'cTkl%', 'Lost', 'Blocks', 'blkSh', 'blkPass', 'Int', 'Tkl+Int',
           'Clr', 'Err','Matches']
    offball = offball[['Player', 'Nation', 'Pos', 'Age','Tkl', 'TklW','Blocks', 'blkSh', 'blkPass', 'Int','Clr', 'Err']]
    home_touches = ref[9]
    home_touches.columns = home_touches.columns.droplevel(0)
    home_touches = home_touches[['Player', 'Nation', 'Pos', 'Age','Touches','Carries','PrgC','TotDist','PrgDist','Rec','PrgR']]
    discipline = ref[11]
    discipline.columns = discipline.columns.droplevel(0)
    discipline = discipline[['Player', 'Nation', 'Pos', 'Age','CrdY', 'CrdR', '2CrdY', 'Fls', 'Fld','Recov', 'Won', 'Lost']]
    discipline.columns = ['Player', 'Nation', 'Pos', 'Age','CrdY', 'CrdR', '2CrdY', 'Fls', 'Fld','Recov', 'headWon', 'headLost']
    minutes = ref[10]
    minutes.columns = minutes.columns.droplevel(0)
    minutes = minutes[['Player', 'Nation', 'Pos', 'Age','Starts', 'Mn/Start', 'Subs', 'Mn/Sub', 'unSub']]
    dfs = [basic, keeper_basic, keeper_adv, shots, passing, pass_types, offball, home_touches, discipline, minutes]
    merged_df = dfs[0]
    for i in range(1, len(dfs)):
        merged_df = pd.merge(merged_df, dfs[i], on=['Player', 'Nation', 'Pos', 'Age'], how='left')
    merged_df['o_Touches'] = merged_df.loc[merged_df['Player']=='Opponent Total','Touches'].sum()/11
    #games =  merged_df.loc[merged_df['Player']=='Opponent Total','90s'].sum()
    games =  merged_df.loc[merged_df['Pos']=='GK','MP'].sum()
    merged_df['o_Touches'] = merged_df['o_Touches'] * merged_df['90s'] / games
    merged_df = merged_df.dropna(subset=['Nation','Min'])
    merged_df['season'] = season
    merged_df['club'] = club
    #post reading tables from beautifu soup, player age is a string of YY-DDD
    try:
        pyears = merged_df['Age'].str.split('-').str[0].astype(int)
        pdays = merged_df['Age'].str.split('-').str[1].astype(int)
        #check these for every new season run
        if(league_code in [21,22,24]): delta = (datetime.now() - datetime(proj_year-1, 2, 1)).days
        else : delta = (datetime.now() - datetime(proj_year-1, 8, 1)).days
        pdays -= delta
        pdays = pdays.apply(lambda x: -1 if x < 0 else 0)
        merged_df['Age']  = pyears + pdays
    except AttributeError:
        merged_df['Age'] #its an integer, so no issue
    return merged_df
    
def team_stats(init_season,end_season,code):
    season = init_season; final = []
    while(season < end_season+1):
        time.sleep(6.01); print(season)
        ref = fbref_league_fixtures(season,code)
        basic = ref[0]
        basic = basic[['Squad', 'MP', 'GF', 'GA','xG', 'xGA' ,'Pts']]
        keeper_basic = ref[4]
        keeper_basic.columns = keeper_basic.columns.droplevel(0)
        keeper_basic = keeper_basic[['Squad','Save%','PKatt', 'PKA', 'PKsv', 'PKm']]
        keeper_basic.columns = ['Squad','Save%','PKSave%','PKatt', 'PKA', 'PKsv', 'PKm']
        keeper_adv = ref[6]
        keeper_adv.columns = keeper_adv.columns.droplevel(0)
        keeper_adv = keeper_adv[['Squad', 'OG', 'PSxG', 'PSxG/SoT', 'PSxG+/-']]
        shots = ref[8]
        shots.columns = shots.columns.droplevel(0)
        shots = shots[['Squad','Sh', 'SoT']]
        passing = ref[10]
        passing.columns = passing.columns.droplevel(0)
        passing.columns = ['Squad', '# Pl', '90s', 'TotCmp', 'TotAtt', 'TotCmp%', 'TotDist', 'PrgDist',
               'Cmp', 'Att', 'Cmp%', 'Cmp', 'Att', 'Cmp%', 'Cmp', 'Att', 'Cmp%', 'Ast',
               'xAG', 'xA', 'A-xAG', 'KP', '1/3', 'PPA', 'CrsPA', 'PrgP']
        passing = passing[['Squad','TotCmp', 'TotAtt', 'TotCmp%', 'TotDist', 'PrgDist','Ast','xAG', 'xA','PrgP','KP']]
        pass_types = ref[12]
        pass_types.columns = pass_types.columns.droplevel(0)
        offball = ref[16]
        offball.columns = offball.columns.droplevel(0)
        offball.columns = ['Squad', '# Pl', '90s', 'Tkl', 'TklW', 'Def 3rd', 'Mid 3rd', 'Att 3rd',
               'cTkl', 'cAtt', 'cTkl%', 'Lost', 'Blocks', 'blkSh', 'blkPass', 'Int', 'Tkl+Int',
               'Clr', 'Err']
        offball = offball[['Squad','Tkl', 'TklW','Blocks', 'blkSh', 'blkPass', 'Int','Clr', 'Err']]
        home_touches = ref[18]
        home_touches.columns = home_touches.columns.droplevel(0)
        home_touches = home_touches[['Squad','Touches','Carries','PrgC','TotDist','PrgDist','Rec','PrgR']]
        opp_touches = ref[19]
        opp_touches.columns = opp_touches.columns.droplevel(0)
        opp_touches = opp_touches[['Squad','Touches','Carries','PrgC','TotDist','PrgDist','Rec','PrgR']]
        opp_touches.columns = ['Squad', 'o_Touches', 'o_Carries','o_PrgC', 'o_TotDist', 'o_PrgDist', 'o_Rec', 'o_PrgR']
        opp_touches['Squad'] = opp_touches['Squad'].str.replace('vs ','')
        discipline = ref[22]
        discipline.columns = discipline.columns.droplevel(0)
        discipline = discipline[['Squad','CrdY', 'CrdR', '2CrdY', 'Fls', 'Fld','Recov', 'Won', 'Lost']]
        discipline.columns = ['Squad','CrdY', 'CrdR', '2CrdY', 'Fls', 'Fld','Recov', 'headWon', 'headLost']
        
        dfs = [basic, keeper_basic, keeper_adv, shots, passing, pass_types, offball, home_touches, opp_touches, discipline]
        merged_df = dfs[0]
        for i in range(1, len(dfs)):
            merged_df = pd.merge(merged_df, dfs[i], on='Squad', how='inner')
        
        merged_df['season'] = season
        final.append(merged_df)
        season += 1
    
    final = pd.concat(final)
    return final

def aggregate_stats(df,player_yes):
    if(player_yes == 1):
        df['90s'] = df['Min']/90
        analysis = df[['Player','Nation','Pos','club','Age','season','90s','Touches','o_Touches','Save%','PKsv%','Goals%','PKatt%','Sh','SoT','PKcon%',
                       'TotAtt','TotCmp%','PrgP','Assist%','CC','Carries','PrgC','Tkl', 'TklW','blkSh', 'blkPass', 'Int', 'Clr','Err','Fls', 'Fld',
                       'Age_copy','Min%','Starts','Subs','unSub','Touch%','Mn/Start','Mn/Sub','CrdY','CrdR']]
    else: 
        analysis = df[['Squad','season','MP','Pts','GF', 'GA','xG','xGA','Ast','Touches','o_Touches','Save%','Sh','TotAtt','TotCmp%','PrgP',
                       'Carries','PrgC','Tkl', 'TklW','blkSh', 'blkPass', 'Int', 'Clr','Err','Fls','Fld']]
    
    analysis['PrgP'] = analysis['PrgP']/analysis['TotAtt']
    analysis['PrgC'] = analysis['PrgC']/analysis['Carries']
    analysis['Sh'] = analysis['Sh']/analysis['Touches']
    analysis['TotAtt'] = analysis['TotAtt']/analysis['Touches']
    #analysis['KP'] = analysis['KP']/analysis['Touches']
    analysis['Carries'] = analysis['Carries']/analysis['Touches']   
    analysis['blkSh'] = analysis['blkSh']/analysis['o_Touches']
    analysis['blkPass'] = analysis['blkPass']/analysis['o_Touches']
    analysis['Int'] = analysis['Int']/analysis['o_Touches']
    analysis['Clr'] = analysis['Clr']/analysis['o_Touches']
    analysis['Err'] = analysis['Err']/analysis['o_Touches']
    analysis['Fls'] = analysis['Fls']/analysis['o_Touches']
    analysis['Fld'] = analysis['Fld']/analysis['Touches']
    if(player_yes == 1):
        analysis['Tkl'] = analysis['Tkl']/analysis['o_Touches']
        analysis['SoT'] = analysis['SoT']/analysis['Touches']
        analysis['CC'] = analysis['CC']/analysis['Touches']
        analysis['CrdY'] = analysis['CrdY']/analysis['o_Touches']
        analysis['CrdR'] = analysis['CrdR']/analysis['o_Touches']
        analysis['Touches'] = analysis['Touches']/analysis['90s']
        analysis['o_Touches'] = analysis['o_Touches']/analysis['90s']
    else:
        analysis['TklW'] = 100*analysis['TklW']/analysis['Tkl']
        analysis['Tkl'] = analysis['Tkl']/analysis['o_Touches']
        analysis['Touches'] = analysis['Touches']/analysis['MP']
        analysis['o_Touches'] = analysis['o_Touches']/analysis['MP']
        analysis['Pts'] = analysis['Pts']/analysis['MP']
        analysis['GF'] = analysis['GF']/analysis['MP']
        analysis['GA'] = analysis['GA']/analysis['MP']
        analysis['Ast'] = analysis['Ast']/analysis['MP']
        analysis['xG'] = analysis['xG']/analysis['MP']
        analysis['xGA'] = analysis['xGA']/analysis['MP']
        analysis['GD'] = analysis['GF'] - analysis['GA']
        analysis['xGD'] = analysis['xG'] - analysis['xGA']
        analysis['pace'] = analysis['Touches'] + analysis['o_Touches']
        analysis['dominance'] = analysis['Touches']/analysis['o_Touches']
        analysis.drop(['Touches','o_Touches'], axis=1, inplace=True)
    return analysis

def regression(df, a, target):
    df.reset_index(drop=True, inplace=True)
    analysis = aggregate_stats(df,0)    
    # Generate some synthetic data
    y = analysis[target]
    X = analysis[['Save%','Sh','TotAtt','TotCmp%','PrgP','Carries','PrgC','Tkl','TklW','blkSh','blkPass','Int','Clr','Err','Fls','Fld','dominance','pace']]
    #normalization
    y_mean = y.mean(); y_std = y.std()
    X_mean = X.mean(); X_std = X.std()
    #print("y_mean",y_mean,"y_stdev",y_std)
    #print("X_mean",X_mean,"X_stdev",X_std)
    y=(y-y.mean())/y.std()
    X=(X-X.mean())/X.std()
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    # Initialize and fit the Lasso model
    # 'alpha' is the regularization parameter (strength of the penalty)
    lasso_model = Lasso(alpha=a) 
    lasso_model.fit(X_train, y_train)
    # Print the coefficients
    print("Lasso Coefficients:", lasso_model.coef_)
    # Make predictions
    y_pred = lasso_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print("Mean Squared Error (MSE):", mse)
    print("R-squared (R2):", r2)
    
    analysis['pred'] = lasso_model.predict(X)
    analysis['pred'] = (analysis['pred']*y_std) + y_mean
    s = analysis[['Squad','season',target,'pred']]
    
    data = pd.DataFrame(columns=['variable','weight','mean','stdev'])
    X_mean = pd.DataFrame(X_mean)
    X_mean.reset_index(inplace=True)
    data['variable'] = X_mean['index']
    data['mean'] = X_mean[0]
    X_std = pd.DataFrame(X_std)
    X_std.reset_index(inplace=True)
    data['stdev'] = X_std[0]
    data.loc[len(data)] = ['pred',np.nan,y_mean,y_std]
    data['weight'] = np.append(lasso_model.coef_,[np.nan])
    
    return s,analysis,lasso_model,data

def multi_leagues(read):
    if(read == 0):
        t_stats_1 = team_stats(2017,2024,9)
        t_stats_2 = team_stats(2017,2024,11)
        t_stats_3 = team_stats(2017,2024,12)
        t_stats_4 = team_stats(2017,2024,13)
        t_stats_5 = team_stats(2017,2024,20)
        df = pd.concat([t_stats_1,t_stats_2,t_stats_3,t_stats_4,t_stats_5])
    else:
        df = pd.read_excel(f'{path}/fbref/big 5 leagues raw.xlsx','Sheet1')
        df = df.drop('index', axis=1)
    return df

def multi_team_links(start,end,code):
    raw = []
    season = start
    while(season<=end):
        links = fbref_team_ids(season,code)
        for l in links['team'].index:
            time.sleep(6.01)
            t = links.iloc[l]['team']
            c = links.iloc[l]['code']
            s = links.iloc[l]['season']
            print(s,t)
            try:
                raw_stats = player_stats(t,c,s,code)
                raw.append(raw_stats)
            except ValueError:
                print("player stats dont exist for the above club")
        season += 1
    raw = pd.concat(raw)
    return raw

def new_season_data(s):
    for l in valid_leagues: #['brazilian serie a','mls']:
        print(l)
        l,c = code_mapping(l)
        try:
            league_current = multi_team_links(s,s,c)
            league = pd.read_excel(f'{path}/fbref/{l}.xlsx','Sheet1')
            league = league.drop('Unnamed: 0', axis=1)
            league = league[league['season']<s]
            league = pd.concat([league,league_current])
            with pd.ExcelWriter(f'{path}/fbref/{l}.xlsx', engine='openpyxl', mode='a', if_sheet_exists='overlay') as writer:
                league.to_excel(writer, sheet_name='Sheet1', index=True)
        except ValueError:
            print(l,"skipped because there is no data")

def extract_player_data(convert,target):
    df_all = []
    exceptions = pd.read_excel(f'{path}/calibration.xlsx','exceptions')
    exceptions['yob'] = proj_year - exceptions['Age'] - 1  
    name_changes = pd.read_excel(f'{path}/calibration.xlsx','name changes')
    for l in valid_leagues:
        df = pd.read_excel(f'{path}/fbref/{l}.xlsx','Sheet1')
        df = df.drop('Unnamed: 0', axis=1)
        df['CC'] = df['Ast'] + df['KP']
        df = df[['Player','club','Nation','Pos','Age','season','Min','Touches','o_Touches','Save%','Sh','TotAtt','TotCmp%','PrgP','Carries','PrgC',
                 'Tkl', 'TklW','blkSh', 'blkPass', 'Int', 'Clr','Err','Fls', 'Fld','Starts', 'Mn/Start', 'Subs', 'Mn/Sub', 'unSub', 'Gls', 'PK',
                 'PKatt_x','SoT','Ast','CC', 'PKatt_y', 'PKA', 'PKsv', 'PKm','CrdY','CrdR']]
        df['yob'] = df['season'] - df['Age']
        if(l in ['brazilian serie a','mls']): 
            df['yob'] -= 1
            df = df.merge(exceptions, left_on=['Player','Nation','yob'], right_on=['Player','Nation','yob'], how='left')
            df.loc[df['Age_y'].notna(), 'yob'] += 1
            df = df.rename(columns={'Age_x': 'Age', 'season_x': 'season'})
            df = df.drop(columns=['Age_y','season_y'])
        df = df.drop_duplicates(subset=['Player', 'club', 'Nation', 'Pos', 'Age', 'season'], keep='first')
        
        df['MP_GK'] = 0
        df.loc[df['Pos']=='GK','MP_GK'] = 1 
        df['MP_GK'] *= df['Starts']
        df['npG'] = df['Gls'] - df['PK']
        club_gp = df.pivot_table(values=['MP_GK','Touches','npG','Ast','PKatt_x'],index=['club','season'],aggfunc='sum')
        df = df.merge(club_gp,left_on=['club','season'],right_on=['club','season'],how='left')
        df = df.merge(name_changes,left_on=['Player','Nation','yob','Pos'],right_on=['Player','Nation','yob','Pos'],how='left')
        df.loc[df['new_name'].notna(), 'Player'] = df['new_name']
        df.loc[df['Pos'] == 'GK', 'Save%'] = df.loc[df['Pos'] == 'GK', 'Save%'].fillna(0)
        df['Touch%'] = (df['Touches_x']/(df['Min']/90))/(df['Touches_y']/df['MP_GK_y'])
        df['Goals%'] = (df['npG_x']/(df['Min']/90))/(df['npG_y']/df['MP_GK_y'])
        df['Assist%'] = (df['Ast_x']/(df['Min']/90))/(df['Ast_y']/df['MP_GK_y'])
        df['PKatt%'] = (df['PKatt_x_x']/(df['Min']/90))/(df['PKatt_x_y']/df['MP_GK_y'])
        #df['PKsv%'] = df['PKsv']/df['PKatt_y']
        df.rename(columns={'Touches_x': 'Touches'}, inplace=True)
        
        if(convert == 1):
            factors = league_conversion_factors(1)
            f = np.exp(factors.loc[l] - factors.loc[target])
            df[f.index] = df[f.index] * f.values
            #use the PrgP factor for Assists and chances created
            df['CC'] *= f['PrgP']
            df['SoT'] *= f['Sh']
            
        df_all.append(df)    
    return df_all,valid_leagues
    
def league_conversion_factors(read_file):
    if(read_file == 1):
        all_eqn = pd.read_excel(f'{path}/calibration.xlsx','conversions',index_col=0)
    else:
        df,leagues = extract_player_data(0,'')
        combos = list(combinations(range(0,len(df)), 2))
        categories = ['Touches', 'o_Touches', 'Save%', 'Sh', 'TotAtt', 'TotCmp%', 'PrgP', 'Carries',
                      'PrgC', 'Tkl', 'TklW', 'blkSh', 'blkPass', 'Int', 'Clr', 'Err', 'Fls', 'Fld', 'CrdY', 'CrdR']
        all_eqn = [['category'] + leagues]
        
        for ch in categories:
            eqn = pd.DataFrame(columns=range(0,len(df)), index=range(0,len(combos)))
            eqn[f'{ch} log factor'] = 0.0
            eqn[f'{ch} Mins'] = 0.0
            r = 0
            for c in combos:
                from_df = df[c[0]]
                from_df['season+1'] = from_df['season'] + 1
                #from_df[from_df['Min']>=450]
                to_df = df[c[1]]
                to_df['season+1'] = to_df['season'] + 1
                #to_df[to_df['Min']>=450]
                if(ch == 'Save%'):
                    from_df = from_df[from_df['Pos']=='GK']
                    to_df = to_df[to_df['Pos']=='GK']
                
                df_from_to = to_df.merge(from_df, left_on=['Player','Nation','yob','season'], right_on=['Player','Nation','yob','season+1'])
                #print("from",c[0],"to",c[1],(df_from_to[f'{ch}_x'].sum()/df_from_to['Min_x'].sum())/(df_from_to[f'{ch}_y'].sum()/df_from_to['Min_y'].sum()))
                df_to_from = from_df.merge(to_df, left_on=['Player','Nation','yob','season'], right_on=['Player','Nation','yob','season+1'])
                #print("from",c[1],"to",c[0],(df_to_from[f'{ch}_x'].sum()/df_to_from['Min_x'].sum())/(df_to_from[f'{ch}_y'].sum()/df_to_from['Min_y'].sum()))
                
                eqn.loc[r,c[0]] = 1
                eqn.loc[r,c[1]] = -1
                #if((df_from_to['Min_x'].sum()<1000)|(df_from_to['Min_y'].sum()<1000)|(df_to_from['Min_x'].sum()<1000)|(df_to_from['Min_y'].sum()<1000)):
                #    factor = np.nan
                #else:
                if(ch in ['Save%','TotCmp%']):
                    factor = np.log(((df_from_to[f'{ch}_x']*df_from_to['Min_x']).sum()/df_from_to['Min_x'].sum())/((df_from_to[f'{ch}_y']*df_from_to['Min_y']).sum()/df_from_to['Min_y'].sum())) -\
                         np.log(((df_to_from[f'{ch}_x']*df_to_from['Min_x']).sum()/df_to_from['Min_x'].sum())/((df_to_from[f'{ch}_y']*df_to_from['Min_y']).sum()/df_to_from['Min_y'].sum())) 
                elif(ch in ['Touches','o_Touches','Sh', 'TotAtt', 'PrgP', 'Carries', 'PrgC', 
                            'Tkl', 'TklW', 'blkSh', 'blkPass', 'Int', 'Clr', 'Err', 'Fls', 'Fld', 'CrdY', 'CrdR']):
                    from_df_team = pd.pivot_table(from_df,values=[ch,'Min'],index=['club','season','season+1'],aggfunc="sum")
                    from_df_team = from_df_team.reset_index()
                    to_df_team = pd.pivot_table(to_df,values=[ch,'Min'],index=['club','season','season+1'],aggfunc="sum")
                    to_df_team = to_df_team.reset_index()
                    
                    df_from_to_club = to_df_team.merge(from_df_team, left_on=['club','season'], right_on=['club','season+1'])
                    df_to_from_club = from_df_team.merge(to_df_team, left_on=['club','season'], right_on=['club','season+1'])
                    
                    factor = np.log((df_from_to[f'{ch}_x'].sum()/df_from_to['Min_x'].sum())/(df_from_to[f'{ch}_y'].sum()/df_from_to['Min_y'].sum())) -\
                         np.log((df_to_from[f'{ch}_x'].sum()/df_to_from['Min_x'].sum())/(df_to_from[f'{ch}_y'].sum()/df_to_from['Min_y'].sum()))
                         
                    factor += 0.5*(np.log((df_from_to_club[f'{ch}_x'].sum()/df_from_to_club['Min_x'].sum())/(df_from_to_club[f'{ch}_y'].sum()/df_from_to_club['Min_y'].sum())) -\
                         np.log((df_to_from_club[f'{ch}_x'].sum()/df_to_from_club['Min_x'].sum())/(df_to_from_club[f'{ch}_y'].sum()/df_to_from_club['Min_y'].sum())))
                else:
                    factor = np.log((df_from_to[f'{ch}_x'].sum()/df_from_to['Min_x'].sum())/(df_from_to[f'{ch}_y'].sum()/df_from_to['Min_y'].sum())) -\
                         np.log((df_to_from[f'{ch}_x'].sum()/df_to_from['Min_x'].sum())/(df_to_from[f'{ch}_y'].sum()/df_to_from['Min_y'].sum())) 
                
                factor = factor/2
                tot_mins_sample = df_from_to['Min_x'].sum() + df_from_to['Min_y'].sum() + df_to_from['Min_y'].sum() + df_to_from['Min_x'].sum()
                #factor *= tot_mins_sample/(tot_mins_sample + 100000)
                eqn.loc[r,f'{ch} log factor'] = factor
                eqn.loc[r,f'{ch} Mins'] = tot_mins_sample
                r+=1
                
                #eqn.loc[r,c[0]] = -1
                #eqn.loc[r,c[1]] = 1
                #eqn.loc[r,f'{ch} log factor'] = np.log((df_to_from[f'{ch}_x'].sum()/df_to_from['Min_x'].sum())/(df_to_from[f'{ch}_y'].sum()/df_to_from['Min_y'].sum())) 
                #r+=1
            
            eqn[list(range(0,len(df)))] = eqn[list(range(0,len(df)))].fillna(0) #.infer_objects(copy=False)
            eqn.replace([np.inf, -np.inf], np.nan, inplace=True)
            eqn = eqn[eqn[f'{ch} log factor'].notna()]
            
            regr = linear_model.LinearRegression(fit_intercept=False)
            regr.fit(eqn[list(range(0,len(df)))], eqn[f'{ch} log factor'], sample_weight=eqn[f'{ch} Mins'])
            #all_eqn.append(eqn)
            #all_eqn.loc[r0] = list(regr.coef_) + [ch]
            coef_list = list(regr.coef_)
            minimum_element = min(coef_list)
            coef_list = [element - minimum_element for element in coef_list]
            print(ch,coef_list)
            all_eqn.append([ch] + coef_list)
        
        all_eqn = pd.DataFrame(all_eqn)
        all_eqn.columns = all_eqn.iloc[0];all_eqn = all_eqn.drop(0)
        all_eqn = all_eqn.T
        all_eqn.columns = all_eqn.iloc[0]
        all_eqn = all_eqn.drop('category')
        all_eqn = all_eqn.apply(pd.to_numeric, errors='ignore')
        #assumption, not enough gk transfers across leagues
        #all_eqn['Save%'] = all_eqn['Sh']/2
    return all_eqn

def aging_analysis(read_file):
    if(read_file == 1):
        df_all = pd.read_excel(f'{path}/calibration.xlsx','aging')
    else:
        df,leagues = extract_player_data(1,standard)
        df = pd.concat(df)
        df = df.reset_index(drop=True)
        df = df[df['Pos']!='GK']
        df = df[df['Min']>450]
        categories = ['Touches', 'o_Touches', 'Sh', 'TotAtt', 'PrgP', 'Carries',
                      'PrgC', 'Tkl', 'TklW', 'blkSh', 'blkPass', 'Int', 'Clr', 'Err', 'Fls', 'Fld','CrdY', 'CrdR']
        df_all = []
        for key in categories:
            df2 = df[['Age', 'season', 'Min',key]]
            df2[f'{key}_Min'] = df[key]/df2['Min']
            df2['Age2'] = df['Age']*df2['Age']
            df2 = df2[[f'{key}_Min','Age','Age2','Min']]
            df2 = df2.reset_index(drop=True)
            df2 = df2.dropna()
            
            model = smf.mixedlm(f"{key}_Min ~ Age + Age2", data=df2, groups="Min")
            result = model.fit()
            params = result.params
            #print(result)
            #print(result.params)
            aging = pd.DataFrame(columns=['Age','Age2',key])
            aging['Age'] = range(1,50)
            aging['Age2'] = aging['Age']*aging['Age']
            aging[key] = params['Intercept'] + params['Age']*aging['Age'] + params['Age2']*aging['Age2']
            df_all.append(aging)
        
        df_all = pd.concat(df_all, axis=1)
        df_all = df_all.drop(columns=['Age','Age2'])
        df_all['Age'] = range(1,50)
        df_all['Save%'] = df_all['o_Touches']
        df_all['TotCmp%'] = df_all['TotAtt']
        df_all = df_all[['Age','Touches', 'o_Touches', 'Save%', 'Sh', 'TotAtt', 'TotCmp%', 'PrgP', 'Carries',
                      'PrgC', 'Tkl', 'TklW', 'blkSh', 'blkPass', 'Int', 'Clr', 'Err', 'Fls', 'Fld','CrdY', 'CrdR']]
    return df_all

def keep_unique_substrings(row):
    substrings = row.split(',')  # Split the string into substrings
    unique_substrings = list(dict.fromkeys(substrings))  # Remove duplicates while preserving order
    return ','.join(unique_substrings)

def mean_reversion():
    df,leagues = extract_player_data(1,standard)
    df = pd.concat(df)
    df = df.reset_index(drop=True)
    df['Mn/Sub'] *= df['Subs']/(df['Subs']+df['unSub'])
    
    coeffs = pd.read_excel(f'{path}/calibration.xlsx','model coefficients')
    coeffs = coeffs.drop('Unnamed: 0', axis=1)
    
    df['TotCmp%'] = df['TotCmp%'].fillna(coeffs.loc[coeffs['variable']=='TotCmp%','mean'].sum())
    df[['Touches','Sh','TotAtt','PrgP','Carries','PrgC','Tkl','TklW','blkSh',
        'blkPass', 'Int', 'Clr','Err','Fls', 'Fld','Touch%', 'Goals%', 'Assist%',
        'PKatt%','PK','PKatt_x_x', 'SoT', 'CC','PKsv','PKatt_y','CrdY','CrdR','Mn/Start','Mn/Sub']] = df[['Touches','Sh','TotAtt','PrgP','Carries','PrgC','Tkl','TklW',
                                                                                 'blkSh', 'blkPass', 'Int', 'Clr','Err','Fls', 'Fld','Touch%', 
                                                                                 'Goals%', 'Assist%','PKatt%', 'PK','PKatt_x_x', 'SoT', 'CC',
                                                                                 'PKsv','PKatt_y','CrdY','CrdR','Mn/Start','Mn/Sub']].fillna(0)
                                                                                                    
    #df['PKsv%'] = df['PKsv']/df['PKatt_y']
    df1 = df.pivot_table(values=['Min', 'Touches', 'o_Touches', 'Sh', 'TotAtt', 'PrgP', 'Carries', 'PrgC', 'Tkl', 'TklW', 
                                 'blkSh', 'blkPass', 'Int', 'Clr', 'Err', 'Fls', 'Fld', 'Starts', 'Subs', 
                                 'unSub','MP_GK_y', 'SoT','CC', 'PK', 'PKatt_x_x','PKsv','PKatt_y','CrdY','CrdR'],
                        index = ['Player', 'club', 'Nation', 'Age', 'season','yob'],
                        aggfunc="sum")
    df2 = df.pivot_table(values=['Save%', 'TotCmp%','Touch%','Goals%', 'Assist%','PKatt%','Mn/Start','Mn/Sub'],
                        index = ['Player', 'club', 'Nation', 'Age', 'season','yob'],
                        aggfunc=lambda rows: np.average(rows, weights=df.loc[rows.index, 'Min']))
    df_pos = df.pivot_table(values=['Pos'], index=['Player', 'club', 'Nation', 'Age', 'season','yob'], aggfunc=lambda x: ','.join(x.unique()))
    df1 = df1.reset_index()
    df2 = df2.reset_index()
    df_pos = df_pos.reset_index()
    df = df1.merge(df2)
    df = df.merge(df_pos)
    
    df['Min%'] = (df['Min']/90)/df['MP_GK_y']
    df['Starts'] = df['Starts']/df['MP_GK_y']
    df['Subs'] = df['Subs']/df['MP_GK_y']
    df['unSub'] = df['unSub']/df['MP_GK_y']
    
    pt = df.pivot_table(values=['Min%','Starts','Subs','unSub','Mn/Start','Mn/Sub'],
                        index = ['Player','Nation','club','season','yob','Pos'],
                        aggfunc=lambda rows: np.average(rows, weights=df.loc[rows.index, 'MP_GK_y']))
    pt = pt.reset_index()
    pt['weight'] = pow(2/3,proj_year-pt['season'])
    pt['weight2'] = pow(2/3,proj_year-pt['season']) * pt['Min%']
    df['weight'] = pow(2/3,proj_year-df['season'])
    df['weight2'] = pow(2/3,proj_year-df['season']) * df['Min']
    df_agg = df.pivot_table(values=['Min','Touches','o_Touches','Sh','TotAtt','PrgP','Carries','PrgC',
                                    'Tkl', 'TklW','blkSh', 'blkPass', 'Int', 'Clr','Err','Fls', 'Fld',
                                    'SoT', 'CC', 'PK', 'PKatt_x_x','PKsv','PKatt_y','CrdY','CrdR'],
                              index=['Player','Nation','yob'], 
                              aggfunc=lambda rows: np.average(rows, weights=df.loc[rows.index, 'weight']))
    df_agg2 = df.pivot_table(values=['Save%','TotCmp%','Touch%','Goals%','Assist%','PKatt%'],
                              index=['Player','Nation','yob'], 
                              aggfunc=lambda rows: np.average(rows, weights=df.loc[rows.index, 'weight2']))
    pos = df.pivot_table(values=['Pos'], index=['Player','Nation','yob'], aggfunc=lambda x: ','.join(x.unique()))
    team = df.pivot_table(values=['club'], index=['Player','Nation','yob'], columns=['season'], aggfunc=lambda x: ','.join(x.unique()))
    team.columns = team.columns.droplevel(0)
    age = df.pivot_table(values=['Age','season'], index=['Player','Nation','yob'], aggfunc='max')
    #review this
    pt2 = pt.copy()
    pt = pt.pivot_table(values=['Min%','Starts','Subs','unSub'], 
                        index=['Player','Nation','yob'], 
                        aggfunc=lambda rows: np.average(rows, weights=pt.loc[rows.index, 'weight']))
    pt2 = pt2[pt2['season']>2017]
    pt2[['Mn/Start','Mn/Sub']] = pt2[['Mn/Start','Mn/Sub']].fillna(0)
    pt2 = pt2.pivot_table(values=['Mn/Start','Mn/Sub'], 
                        index=['Player','Nation','yob'], 
                        aggfunc=lambda rows: np.average(rows, weights=pt2.loc[rows.index, 'weight2']))
    avg = pd.read_excel(f'{path}/fbref/{standard}.xlsx','Sheet1')
    avg['CC'] = avg['Ast'] + avg['KP']
    avg_save_pct = 100*avg['Saves'].sum()/avg['SoTA'].sum() 
    avg_pk_save_pct = 100*avg['PKsv'].sum()/avg['PKatt_y'].sum()
    avg_tklw = 100*avg['TklW'].sum()/avg['Tkl'].sum()
    avg_tot_cmp = 100*avg['TotCmp'].sum()/avg['TotAtt'].sum()
    avg_pk_con = 100*avg['PK'].sum()/avg['PKatt_x'].sum()
    avg = avg[['Min','Touches','o_Touches','Save%','Sh','TotAtt','PrgP','Carries','PrgC',
               'Tkl','blkSh', 'blkPass', 'Int', 'Clr','Err','Fls', 'Fld', 'CC','SoT','CrdY','CrdR']].sum()
    #avg = df[['Touches','o_Touches','Save%','Sh','TotAtt','TotCmp%','PrgP','Carries','PrgC',
    #         'Tkl', 'TklW','blkSh', 'blkPass', 'Int', 'Clr','Err','Fls', 'Fld','Touch%']].sum()
    for x in avg.index:
        if(x != 'Min'): avg[x] = 600 * avg[x].sum()/avg['Min'].sum()
    #percentage stats need to be fixed
    avg['Save%'] = avg_save_pct
    avg['PKsv%'] = avg_pk_save_pct
    avg['TotCmp%'] = avg_tot_cmp
    avg['TklW'] = avg_tklw
    avg['PKcon%'] = avg_pk_con
    avg['Touch%'] = 1/11
    avg['Goals%'] = 1/11
    avg['Assist%'] = 1/11
    avg['Mn/Start'] = pt2['Mn/Start'].mean()
    avg['Mn/Sub'] = pt2['Mn/Sub'].mean()
    avg = avg.drop(labels=['Min'])
    
    pos = pos.reset_index()
    team = team.reset_index()
    team = team[['Player','Nation','yob',proj_year-1]]
    team = team.rename(columns={proj_year-1: 'club'})
    age = age.reset_index()
    age['Age_copy'] = age['Age']
    age['Age'] = age['Age'] + proj_year - age['season']
    age['season'] = proj_year
    df_agg = df_agg.reset_index()
    df_agg2 = df_agg2.reset_index()
    #df_agg2['PKsv%'] *= 100 
    pt = pt.reset_index()
    pt2 = pt2.reset_index()
    df_agg = df_agg2.merge(df_agg, left_on=['Player','Nation','yob'], right_on=['Player','Nation','yob'])
    df_agg = pos.merge(df_agg, left_on=['Player','Nation','yob'], right_on=['Player','Nation','yob'])
    df_agg = team.merge(df_agg, left_on=['Player','Nation','yob'], right_on=['Player','Nation','yob'])
    df_agg = age.merge(df_agg, left_on=['Player','Nation','yob'], right_on=['Player','Nation','yob'])
    df_agg = pt.merge(df_agg, left_on=['Player','Nation','yob'], right_on=['Player','Nation','yob'])
    df_agg = pt2.merge(df_agg, left_on=['Player','Nation','yob'], right_on=['Player','Nation','yob'])
    df_agg['TklW'] = 100*df_agg['TklW']/df_agg['Tkl']
    df_agg['TklW'] = df_agg['TklW'].fillna(0)
    df_agg['PKcon%'] = 100*df_agg['PK']/df_agg['PKatt_x_x']
    df_agg['PKsv%'] = 100*df_agg['PKsv']/df_agg['PKatt_y']
    df_agg['PKsv%'] = df_agg.loc[df_agg['Pos'] == 'GK', 'PKsv%'].fillna(avg_pk_save_pct)
    
    for x in avg.index:
        if(x not in ['Save%','TotCmp%','TklW','Touch%','PKsv%','PKcon%','Goals%','Assist%','Mn/Start','Mn/Sub']):  df_agg[x] = df_agg[x] + avg[x]
        elif(x in ['PKsv%','PKcon%']): df_agg[x] = (df_agg[x]*df_agg['Min'] + avg[x]*6000)/(df_agg['Min']+6000)
        else: df_agg[x] = (df_agg[x]*df_agg['Min'] + avg[x]*600)/(df_agg['Min']+600)
    df_agg['Min'] = df_agg['Min'] + 600
    
    #VERIFY FROM HERE, ALSO ADD CHANCES CREATED TO THE LEAGUE CONVERSIONS AND AGING
    df_agg = aggregate_stats(df_agg,1)
    df_agg['90s'] = df_agg['90s'] - (600/90)
    df_agg['Pos'] = df_agg['Pos'].str.replace(' ','')
    df_agg['Pos'] = df_agg['Pos'].apply(keep_unique_substrings)
    df_agg['club'] = df_agg['club'].str.replace('-',' ')
    
    aging = aging_analysis(1)
    #copies based on existing curves
    aging['SoT'] = aging['Sh']
    aging['CC'] = aging['PrgP']
    aging['PKsv%'] = aging['Save%']
    projections_copy = df_agg.merge(aging, left_on='Age_copy', right_on='Age')
    projections_copy = projections_copy.merge(aging, left_on='Age_x', right_on='Age')
    
    for v in ['Touches','o_Touches','Save%','Sh','TotAtt','TotCmp%','PrgP','Carries','PrgC',
             'Tkl', 'TklW','blkSh', 'blkPass', 'Int', 'Clr','Err','Fls', 'Fld', 'SoT','CC', 'PKsv%','CrdY','CrdR']:
        projections_copy[f'{v}_x'] *= projections_copy[v] / projections_copy[f'{v}_y']
        
    projections_copy = projections_copy[['Player', 'Nation', 'Pos', 'club', 'Age_x', 'season', 'Min%', 'Mn/Start', 'Mn/Sub',
           'Touch%','Touches_x', 'o_Touches_x', 'Save%_x', 'PKsv%_x', 'Goals%', 'PKatt%', 'Sh_x', 'SoT_x', 'PKcon%', 
           'TotAtt_x', 'TotCmp%_x', 'PrgP_x', 'Assist%', 'CC_x', 'Carries_x', 'PrgC_x', 'Tkl_x', 'TklW_x', 'blkSh_x', 
           'blkPass_x', 'Int_x', 'Clr_x', 'Err_x', 'Fls_x', 'Fld_x', 'CrdY_x','CrdR_x']]
    projections_copy.columns = ['Player', 'Nation', 'Pos', 'club', 'Age', 'season', 'p(90/G)', 'Mn/Start', 'Mn/Sub',
           'Touch%','Touches', 'o_Touches', 'Save%', 'PKsv%', 'Goals%', 'PKatt%', 'Sh', 'SoT', 'PKcon%', 'TotAtt', 'TotCmp%', 
           'PrgP', 'Assist%', 'CC', 'Carries', 'PrgC', 'Tkl', 'TklW', 'blkSh', 'blkPass', 'Int', 'Clr', 'Err', 'Fls', 'Fld','CrdY','CrdR']
    return projections_copy

def lineup_projection(team,custom_lineups,custom_mins,return_all_stats):
    #team='Leeds United'; custom_mins=1; custom_lineups=1
    df = pd.read_excel(f'{path}/projections.xlsx','Sheet1')
    df = df.drop('Column1', axis=1)
    df['TklW'] = df['TklW'].fillna(60.58)
    squads = pd.read_excel(f'{path}/projections.xlsx','squads')
    squads = squads.drop('Column1', axis=1)
    df = pd.merge(df, squads, on=['Player','Nation','Pos','Age'], how='left')
    if(custom_lineups == 1): df['club_x'] = df['club_y']
    if(custom_mins == 1): df['p(90/G)_x'] = df['p(90/G)_y']
    df.rename(columns={'club_x': 'club', 'p(90/G)_x': 'p(90/G)'}, inplace=True)
    df.drop(['club_y','p(90/G)_y'], axis=1, inplace=True) 
    df['p(90/G)'] = df['p(90/G)'].fillna(0)
    
    coeffs = pd.read_excel(f'{path}/calibration.xlsx','model coefficients')
    coeffs = coeffs.drop('Unnamed: 0', axis=1)
    gf = pd.read_excel(f'{path}/calibration.xlsx','GF')
    gf = gf.drop('Unnamed: 0', axis=1)
    ga = pd.read_excel(f'{path}/calibration.xlsx','GA')
    ga = ga.drop('Unnamed: 0', axis=1)
    
    df_team = df[df['club']==team]
    df_team[['Start','Sub']] = df_team[['Start','Sub']].fillna(0)
    if(df_team['p(90/G)'].sum() == 0): df_team['p(90/G)'] = (df_team['Start']*df_team['Mn/Start'] + df_team['Sub']*df_team['Mn/Sub'])/90
    
    keepers = df_team[df_team['Pos']=='GK']    
    keepers = keepers.sort_values(by='p(90/G)', ascending=False)
    keepers['rank'] = list(range(1,len(keepers)+1))
    outfielders = df_team[df_team['Pos']!='GK']
    outfielders = outfielders.sort_values(by='p(90/G)', ascending=False)
    outfielders['rank'] = list(range(1,len(outfielders)+1))
    
    exp = 1.0
    while((keepers['p(90/G)'].sum() < 0.99) or (keepers['p(90/G)'].sum() > 1.01)):
        if(keepers['p(90/G)'].sum() < 0.99):
            keepers['p(90/G)'] *= pow(exp,keepers['rank'])
            exp += 0.0001
        else:
            keepers['p(90/G)'] *= pow(exp,keepers['rank'])
            exp -= 0.0001
        keepers['p(90/G)'] = keepers['p(90/G)'].clip(upper=1)
    
    exp = 1.0
    while((outfielders['p(90/G)'].sum() <= 9.9) or (outfielders['p(90/G)'].sum() >= 10.1)):
        if(outfielders['p(90/G)'].sum() <= 9.9):
            outfielders['p(90/G)'] *= pow(exp,outfielders['rank'])
            exp += 0.0001
        elif(outfielders['p(90/G)'].sum() >= 10.1):
            outfielders['p(90/G)'] *= pow(exp,outfielders['rank'])
            exp -= 0.0001
        outfielders['p(90/G)'] = outfielders['p(90/G)'].clip(upper=1)
        #print(outfielders['p(90/G)'].sum())
    
    keepers['p(90/G)'] = keepers['p(90/G)']/keepers['p(90/G)'].sum()
    outfielders['p(90/G)'] = 10*outfielders['p(90/G)']/outfielders['p(90/G)'].sum()
    touches = (outfielders['p(90/G)']*outfielders['Touches']).sum() + (keepers['p(90/G)']*keepers['Touches']).sum()
    opp_touches = (outfielders['p(90/G)']*outfielders['o_Touches']).sum() + (keepers['p(90/G)']*keepers['o_Touches']).sum()
    touch_pct = (outfielders['p(90/G)']*outfielders['Touch%']).sum() + (keepers['p(90/G)']*keepers['Touch%']).sum()
    outfielders['o_Touches'] =  opp_touches/11
    keepers['o_Touches'] =  opp_touches/11
    outfielders['Touch%'] /=  touch_pct
    keepers['Touch%'] /=  touch_pct
    outfielders['Touches'] =  touches * outfielders['Touch%']
    keepers['Touches'] =  touches * keepers['Touch%']
    
    measure = []
    for m in ['Touches','o_Touches','Save%','Sh','TotAtt','TotCmp%','PrgP','Carries','PrgC','Tkl','TklW',
              'blkSh','blkPass','Int', 'Clr', 'Err', 'Fls', 'Fld']:
        if(m in ['Touches','o_Touches','Save%']):
            measure.append((outfielders[m] * outfielders['p(90/G)']).sum() + (keepers[m] * keepers['p(90/G)']).sum())
        elif(m in ['TotCmp%']):
            measure.append(((outfielders[m] * outfielders['TotAtt'] * outfielders['p(90/G)'] * outfielders['Touches']).sum() + (keepers[m] * keepers['TotAtt'] * keepers['p(90/G)'] * keepers['Touches']).sum())/measure[0])
        elif(m in ['TklW']):
            measure.append(((outfielders[m] * outfielders['Tkl'] * outfielders['p(90/G)'] * outfielders['o_Touches']).sum() + (keepers[m] * keepers['Tkl'] * keepers['p(90/G)'] * keepers['o_Touches']).sum())/measure[1])
        elif(m in ['Tkl','blkSh','blkPass','Int', 'Clr', 'Err', 'Fls']):
            measure.append(((outfielders[m] * outfielders['p(90/G)'] * outfielders['o_Touches']).sum() + (keepers[m] * keepers['p(90/G)'] * keepers['o_Touches']).sum())/measure[1])
        elif(m in ['Sh','TotAtt','PrgP','Carries','PrgC','Fld']):
            measure.append(((outfielders[m] * outfielders['p(90/G)'] * outfielders['Touches']).sum() + (keepers[m] * keepers['p(90/G)'] * keepers['Touches']).sum())/measure[0])
    
    measure[5]/= measure[4]
    measure[10]/= measure[9]
    #adding dominance and pace which are derived from touches and opp touches
    measure.append(measure[0]/measure[1])
    measure.append(measure[0]+measure[1])
    
    pts = (np.array(measure[2:]) - np.array(coeffs['mean'].to_list()[:-1])) / np.array(coeffs['stdev'].to_list()[:-1])
    pts *= np.array(coeffs['weight'].to_list()[:-1])
    pts = sum(pts) * coeffs.loc[coeffs['variable']=='pred','stdev'].sum()  + coeffs.loc[coeffs['variable']=='pred','mean'].sum()
    gf_t = (np.array(measure[2:]) - np.array(gf['mean'].to_list()[:-1])) / np.array(gf['stdev'].to_list()[:-1])
    gf_t *= np.array(gf['weight'].to_list()[:-1])
    gf_t = sum(gf_t) * gf.loc[gf['variable']=='pred','stdev'].sum()  + gf.loc[gf['variable']=='pred','mean'].sum()
    ga_t = (np.array(measure[2:]) - np.array(ga['mean'].to_list()[:-1])) / np.array(ga['stdev'].to_list()[:-1])
    ga_t *= np.array(ga['weight'].to_list()[:-1])
    ga_t = sum(ga_t) * ga.loc[ga['variable']=='pred','stdev'].sum()  + ga.loc[ga['variable']=='pred','mean'].sum()
    #print(team,pts)
    #print(outfielders[outfielders['p(90/G)']>0.01][['Player','p(90/G)']])
    if(return_all_stats == 1): return measure,keepers,outfielders
    else: return (pts,gf_t,ga_t)
    
def league_projections(league,custom_lineups,custom_mins):    
    table = [['Team','Points','GF/90','GA/90']]
    team_list = pd.read_excel(f'{path}/projections.xlsx','teams')
    team_list = list(team_list[league])
    for t in team_list:
        pts,gf,ga = lineup_projection(t,custom_lineups,custom_mins,0)
        table.append([t,pts,gf,ga])
    table = pd.DataFrame(table)
    table.columns = table.iloc[0];table = table.drop(0)
    table = table.apply(pd.to_numeric, errors='ignore')
    
    coeffs = pd.read_excel(f'{path}/calibration.xlsx','model coefficients')
    coeffs = coeffs.drop('Unnamed: 0', axis=1)
    lg_avg = coeffs.loc[coeffs['variable']=='pred','mean'].sum()
    print('projection to expectation delta',lg_avg/table['Points'].mean())
    table['Points'] *= (lg_avg/table['Points'].mean())
    table['GD/90'] = table['GF/90'] - table['GA/90']
    return table

def h2h(t1,t2,custom_lineups,custom_mins):
    #t1='Liverpool';t2='Newcastle United';custom_lineups=1;custom_mins=1
    m1,k1,o1 = lineup_projection(t1,custom_lineups,custom_mins,1)
    m2,k2,o2 = lineup_projection(t2,custom_lineups,custom_mins,1)
    
    gf = pd.read_excel(f'{path}/calibration.xlsx','GF')
    gf = gf.drop('Unnamed: 0', axis=1)
    ga = pd.read_excel(f'{path}/calibration.xlsx','GA')
    ga = ga.drop('Unnamed: 0', axis=1)
    ast = pd.read_excel(f'{path}/calibration.xlsx','Ast')
    ast = ast.drop('Unnamed: 0', axis=1)
    
    gf1 = (np.array(m1[2:]) - np.array(gf['mean'].to_list()[:-1])) / np.array(gf['stdev'].to_list()[:-1])
    gf1 *= np.array(gf['weight'].to_list()[:-1])
    gf1 = sum(gf1) * gf.loc[gf['variable']=='pred','stdev'].sum()  + gf.loc[gf['variable']=='pred','mean'].sum()  
    ga1 = (np.array(m1[2:]) - np.array(ga['mean'].to_list()[:-1])) / np.array(ga['stdev'].to_list()[:-1])
    ga1 *= np.array(ga['weight'].to_list()[:-1])
    ga1 = sum(ga1) * ga.loc[ga['variable']=='pred','stdev'].sum()  + ga.loc[ga['variable']=='pred','mean'].sum()
    
    gf2 = (np.array(m2[2:]) - np.array(gf['mean'].to_list()[:-1])) / np.array(gf['stdev'].to_list()[:-1])
    gf2 *= np.array(gf['weight'].to_list()[:-1])
    gf2 = sum(gf2) * gf.loc[gf['variable']=='pred','stdev'].sum()  + gf.loc[gf['variable']=='pred','mean'].sum()  
    ga2 = (np.array(m2[2:]) - np.array(ga['mean'].to_list()[:-1])) / np.array(ga['stdev'].to_list()[:-1])
    ga2 *= np.array(ga['weight'].to_list()[:-1])
    ga2 = sum(ga2) * ga.loc[ga['variable']=='pred','stdev'].sum()  + ga.loc[ga['variable']=='pred','mean'].sum()
    
    a1 = (np.array(m1[2:]) - np.array(ast['mean'].to_list()[:-1])) / np.array(ast['stdev'].to_list()[:-1])
    a1 *= np.array(ast['weight'].to_list()[:-1])
    a1 = sum(a1) * ast.loc[ast['variable']=='pred','stdev'].sum()  + ast.loc[ast['variable']=='pred','mean'].sum()
    a2 = (np.array(m1[2:]) - np.array(ast['mean'].to_list()[:-1])) / np.array(ast['stdev'].to_list()[:-1])
    a2 *= np.array(ast['weight'].to_list()[:-1])
    a2 = sum(a2) * ast.loc[ast['variable']=='pred','stdev'].sum()  + ast.loc[ast['variable']=='pred','mean'].sum()
    
    avg_goals = (gf.loc[gf['variable']=='pred','mean'].mean() + ga.loc[gf['variable']=='pred','mean'].mean())/2
    avg = pd.read_excel(f'{path}/fbref/{standard}.xlsx','Sheet1')
    avg = avg.pivot_table(values=['Gls','Starts'], index=['season'], aggfunc='sum')
    avg['G_per_game'] = avg['Gls']/(avg['Starts']/11)
    avg = avg.reset_index()
    avg['weight'] = pow(2/3,avg['season'].max()-avg['season'])
    avg['weight'] *= avg['Starts']
    avg['weight'] /= sum(avg['weight'])
    avg = (avg['weight']*avg['G_per_game']).sum()
    home_adv = 0.1 # research this in detail
    lg_avg_touches = 625 #check if this is 600 or 625
    
    t1_g = (avg/avg_goals) * gf1 * ga2 / avg_goals
    t2_g = (avg/avg_goals) * gf2 * ga1 / avg_goals
    delta_g = home_adv * (t1_g + t2_g)/2
    t1_g += delta_g
    t2_g -= delta_g
    
    t1_a = a1 * (t1_g/gf1)
    t2_a = a2 * (t2_g/gf2)
    
    t1_cs = poisson.pmf(0, t2_g)
    t2_cs = poisson.pmf(0, t1_g)
      
    t1_touches = m1[0]*m2[1]/lg_avg_touches
    t2_touches = m2[0]*m1[1]/lg_avg_touches
    delta_touches = home_adv * (t1_touches + t2_touches)/2
    t1_touches += delta_touches
    t2_touches -= delta_touches
    
    k1['Touches'] *= t1_touches/m1[0]
    k1['o_Touches'] *= t2_touches/m1[1]
    o1['Touches'] *= t1_touches/m1[0]
    o1['o_Touches'] *= t2_touches/m1[1]
    k2['Touches'] *= t2_touches/m2[0]
    k2['o_Touches'] *= t1_touches/m2[1]
    o2['Touches'] *= t2_touches/m2[0]
    o2['o_Touches'] *= t1_touches/m2[1]
    
    match_df = pd.concat([k1, k2, o1, o2], ignore_index=True)
    match_df = match_df[match_df['p(90/G)']>0]
    
    t1_gs = (match_df.loc[match_df['club']==t1,'Goals%']*match_df.loc[match_df['club']==t1,'p(90/G)']).sum()
    match_df.loc[match_df['club']==t1,'Goals%'] /= t1_gs
    t2_gs = (match_df.loc[match_df['club']==t2,'Goals%']*match_df.loc[match_df['club']==t2,'p(90/G)']).sum()
    match_df.loc[match_df['club']==t2,'Goals%'] /= t2_gs
    t1_pen = (match_df.loc[match_df['club']==t1,'PKatt%']*match_df.loc[match_df['club']==t1,'p(90/G)']).sum()
    match_df.loc[match_df['club']==t1,'PKatt%'] /= t1_pen
    t2_pen = (match_df.loc[match_df['club']==t2,'PKatt%']*match_df.loc[match_df['club']==t2,'p(90/G)']).sum()
    match_df.loc[match_df['club']==t2,'PKatt%'] /= t2_pen
    t1_as = (match_df.loc[match_df['club']==t1,'Assist%']*match_df.loc[match_df['club']==t1,'p(90/G)']).sum()
    match_df.loc[match_df['club']==t1,'Assist%'] /= t1_as
    t2_as = (match_df.loc[match_df['club']==t2,'Assist%']*match_df.loc[match_df['club']==t2,'p(90/G)']).sum()
    match_df.loc[match_df['club']==t2,'Assist%'] /= t2_as
    
    for c in match_df.columns:
        if(c in ['Sh','TotAtt','PrgP', 'Carries', 'PrgC','Fld','CC','SoT']):
            match_df[c] *=  match_df['Touches'] * match_df['p(90/G)']
        elif(c in ['Tkl','blkSh','blkPass', 'Int', 'Clr', 'Err', 'Fls','CrdY','CrdR']):
            match_df[c] *=  match_df['o_Touches'] * match_df['p(90/G)']
        elif(c == 'TotCmp%'):
            match_df[c] *= match_df['TotAtt']/100
        elif(c == 'TklW'):
            match_df[c] *= match_df['Tkl']/100
    
    match_df.loc[match_df['club']==t1,'npG'] = match_df['Goals%'] * match_df['p(90/G)'] * t1_g * 0.9
    match_df.loc[match_df['club']==t2,'npG'] = match_df['Goals%'] * match_df['p(90/G)'] * t2_g * 0.9
    match_df.loc[match_df['club']==t1,'pG'] = match_df['PKatt%'] * match_df['p(90/G)'] * t1_g * 0.1 #10% of goals are pens, research this
    match_df.loc[match_df['club']==t2,'pG'] = match_df['PKatt%'] * match_df['p(90/G)'] * t2_g * 0.1
    match_df.loc[match_df['club']==t1,'A'] = match_df['Assist%'] * match_df['p(90/G)'] * t1_a
    match_df.loc[match_df['club']==t2,'A'] = match_df['Assist%'] * match_df['p(90/G)'] * t2_a
    match_df.loc[match_df['club']==t1,'CS%'] = t1_cs
    match_df.loc[match_df['club']==t2,'CS%'] = t2_cs
    match_df.loc[match_df['club']==t1,'GC'] = t2_g
    match_df.loc[match_df['club']==t2,'GC'] = t1_g
    
    match_df['CBIT'] = match_df['TklW']+match_df['blkSh']+match_df['blkPass']+match_df['Int']+match_df['Clr']
    saves_t2 = match_df.loc[match_df['club']==t1,'SoT'].sum() - t1_g
    saves_t1 = match_df.loc[match_df['club']==t2,'SoT'].sum() - t2_g
    match_df.loc[(match_df['club']==t1)&(match_df['Pos']=='GK'),'Saves'] = match_df.loc[(match_df['club']==t1)&(match_df['Pos']=='GK'),'p(90/G)'] * saves_t1
    match_df.loc[(match_df['club']==t2)&(match_df['Pos']=='GK'),'Saves'] = match_df.loc[(match_df['club']==t2)&(match_df['Pos']=='GK'),'p(90/G)'] * saves_t2
    match_df['pMiss'] = match_df['pG']*(100-match_df['PKcon%'])/match_df['PKcon%']
    #fix this formula
    match_df.loc[(match_df['club']==t1)&(match_df['Pos']=='GK'),'pSaves'] = match_df.loc[(match_df['club']==t1)&(match_df['Pos']=='GK'),'p(90/G)'] * match_df.loc[(match_df['club']==t1)&(match_df['Pos']=='GK'),'PKsv%'] * t2_g * 0.1/100
    match_df.loc[(match_df['club']==t2)&(match_df['Pos']=='GK'),'pSaves'] = match_df.loc[(match_df['club']==t2)&(match_df['Pos']=='GK'),'p(90/G)'] * match_df.loc[(match_df['club']==t2)&(match_df['Pos']=='GK'),'PKsv%'] * t1_g * 0.1/100
    match_df[['pMiss','Saves','pSaves']] = match_df[['pMiss','Saves','pSaves']].fillna(0)
    
    draw = skellam.pmf(0,t1_g,t2_g)
    t1_win = 1-skellam.cdf(0,t1_g,t2_g)
    t2_win = skellam.cdf(-1,t1_g,t2_g)
    print(t1,round(t1_win,3),"draw",round(draw,3),t2,round(t2_win,3))
    print(t1,'goals',round(t1_g,2),'CS%',round(100*t1_cs,2))
    print(t2,'goals',round(t2_g,2),'CS%',round(100*t2_cs,2))
    
    #foul percentage
    match_df.loc[match_df['club']==t1,'Fls'] /=  match_df.loc[match_df['club']==t1,'Fls'].sum()
    match_df.loc[match_df['club']==t2,'Fls'] /=  match_df.loc[match_df['club']==t2,'Fls'].sum()
    match_df['Fls_Pen'] = match_df['Fls'] * match_df['GC']/10   
    #win loss
    match_df.loc[match_df['club']==t1,'win'] = t1_win
    match_df.loc[match_df['club']==t2,'win'] = t2_win
    match_df.loc[match_df['club']==t1,'loss'] = t2_win
    match_df.loc[match_df['club']==t2,'loss'] = t1_win
    #mapping pos
    match_df = position_mapping(match_df)
    match_df['FT_Pos'] = match_df['FT_Pos'].fillna(match_df['mapped_Pos'])
    
    match_df = match_df[['Player','Nation','FT_Pos','club','Age','p(90/G)','npG','pG','pMiss','A','SoT','CC','TotCmp%','GC','CS%','Saves','pSaves','TklW','Int','CBIT','CrdY','CrdR','Fls_Pen','win','loss']]
    match_df = fantasy_points(match_df)
    match_df['Mins'] = match_df['p(90/G)'] * 90
    match_df = match_df[['Player','club','FT_Pos','Mins','Points']]
    return match_df

def fantasy_points(df):
    #appearance and >60 mins
    df['Points'] = (1-poisson.cdf(25, 90*df['p(90/G)'])) + (1-poisson.cdf(60, 90*df['p(90/G)']))
    #assists, pen miss, cards (OG, causing pens and fks leading to goals missing)
    df['Points'] += 3*df['A'] - 2*df['pMiss'] -df['CrdY'] - 3*df['CrdR'] - 2*df['Fls_Pen']
    #win loss points (+/- 0.3)
    df['Points'] += 0.3*(df['win']-df['loss']) 
    #goals
    df['Points'] += np.where((df['FT_Pos']=='GK')|(df['FT_Pos']=='DF'), 6, np.where((df['FT_Pos']=='MF'), 5, 4)) * (df['npG']+df['pG'])
    #clean sheet
    df['Points'] += (1-poisson.cdf(60, 90*df['p(90/G)']))*df['CS%']*np.where((df['FT_Pos']=='GK')|(df['FT_Pos']=='DF'), 4, np.where((df['FT_Pos']=='MF'), 1, 0))
    #2 goals conceded
    df['Points'] += -0.5*np.where((df['FT_Pos']=='GK')|(df['FT_Pos']=='DF'), 1,0)*df['p(90/G)']*df['GC']
    #saves and pen saves
    df['Points'] += 0.5*df['Saves'] + 5*df['pSaves']
    #Shot on target
    df['Points'] += np.where((df['FT_Pos']=='GK')|(df['FT_Pos']=='DF'), .6, np.where((df['FT_Pos']=='MF'), .4, .4)) * df['SoT']
    #played full match
    df['Points'] += np.where((df['FT_Pos']=='MF')|(df['FT_Pos']=='FW'), 1,0) * df['p(90/G)']
    #sort descending
    df = df.sort_values(by='Points', ascending=False)
    return df

def position_mapping(df):
    mapping = {'DF,MF': 'DF', 'GK': 'GK', 'DF': 'DF', 'MF': 'MF', 'FW': 'FW', 
               'FW,MF': 'MF', 'DF,FW': 'DF', 'MF,FW': 'FW', 'DF,MF,FW':'DF',
               'FW,DF,MF':'DF','FW,MF,DF':'FW','MF,FW,DF':'MF','MF,DF':'MF',
               'MF,DF,FW':'MF','FW,DF':'DF','DF,FW,MF':'DF','GK,FW,MF':'FW', 
               'FW,MF,GK':'FW', 'MF,GK':'MF', 'GK,MF':'MF', 'FW,GK':'FW' }
    df['mapped_Pos'] = df['Pos'].map(mapping)
    return df

#%% extract data
#extract team stats for multiple leagues and years
#t_stats = multi_leagues(0)
#extract player stats for multiple leagues
#player_stats_raw = multi_team_links(2021,2024,21)
new_season_data(proj_year-1)
#ote = opp_touches_error(2017,2024,9)

#%% analyze
#team regression analysis
#t_stats = multi_leagues(1)
#summary,t_stats_reg,lasso,coeffs = regression(t_stats,0.001,'Pts') # target variables can be - GF, GA, xG, xGA, GD, xGD, Pts
#player projection data
factors = league_conversion_factors(0) #0 to generate them, 1 to read from the file
aging = aging_analysis(0) #0 to generate them, 1 to read from the file

#%% generate player projections
projections = mean_reversion()
#to identify players whose data has been duplicated due to yob mismatch
duplicates = projections.pivot_table(values=['season'], index=['Player','Nation','Age'], aggfunc='count')
duplicates = duplicates[duplicates['season']>1]

#%% points projections
#lineup_projection('Chelsea',0,0,0) #team, custom lineups, custom mins
#table = league_projections(standard,1,1) #team, custom lineups, custom mins
points = h2h('Newcastle United','Liverpool',1,1) #home team, away team, custom lineups, custom mins
