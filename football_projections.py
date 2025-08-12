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

from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

path = "C:/Users/Subramanya.Ganti/Downloads/Sports/football"
#path = "C:/Users/uttam/Desktop/Sports/football"
valid_leagues = ['serie a','bundesliga','premier league','la liga','ligue un',
                 'championship','liga portugal','eredivisie','serie b','belgian pro league',
                 'brazilian serie a','mls','liga mx',
                 'champions league','europa league','conference league']

#%% functions
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
    data  = requests.get(url,verify=False).text
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
        ref = pd.read_html(f'https://fbref.com/en/squads/{code}/{season}-{season+1}/c{league_code}/{club}-Stats')
    elif(league_code in [24,21,22]): #leagues start in winter
        ref = pd.read_html(f'https://fbref.com/en/squads/{code}/{season}/{club}-Stats')
    else: #leagues start in summer
        ref = pd.read_html(f'https://fbref.com/en/squads/{code}/{season}-{season+1}/{club}-Stats')
    
    #ref = pd.read_html('https://fbref.com/en/squads/943e8050/2023-2024/9/Burnley-Stats-Premier-League')
    
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
    #p_stats_agg = aggregate_stats(merged_df,1)
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
        analysis = df[['Player','Nation','Pos','club','Age','season','90s','Touches','o_Touches','Save%','Sh','TotAtt','TotCmp%','PrgP',
                       'Carries','PrgC','Tkl', 'TklW','blkSh', 'blkPass', 'Int', 'Clr','Err','Fls', 'Fld','Age_copy',
                       'Min%','Starts','Subs','unSub','Touch%']]
    else: 
        analysis = df[['Squad','season','MP','Pts','GF', 'GA','xG','xGA','Touches','o_Touches','Save%','Sh','TotAtt','TotCmp%','PrgP',
                       'Carries','PrgC','Tkl', 'TklW','blkSh', 'blkPass', 'Int', 'Clr','Err','Fls', 'Fld']]
    
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
        analysis['xG'] = analysis['xG']/analysis['MP']
        analysis['xGA'] = analysis['xGA']/analysis['MP']
        analysis['GD'] = analysis['GF'] - analysis['GA']
        analysis['xGD'] = analysis['xG'] - analysis['xGA']
        analysis['pace'] = analysis['Touches'] + analysis['o_Touches']
        analysis['dominance'] = analysis['Touches']/analysis['o_Touches']
        analysis.drop(['Touches','o_Touches'], axis=1, inplace=True)
    return analysis

def regression(df, a):
    df.reset_index(drop=True, inplace=True)
    analysis = aggregate_stats(df,0)    
    # Generate some synthetic data
    y = analysis['Pts']
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
    s = analysis[['Squad','season','Pts','GD','xGD','pred']]
    
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
        df = pd.read_excel(f'{path}/fbref.xlsx','big 5 leagues raw')
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

def extract_player_data(convert,target,proj_year):
    df_all = []
    exceptions = pd.read_excel(f'{path}/calibration.xlsx','exceptions')
    exceptions['yob'] = proj_year - exceptions['Age'] - 1  
    name_changes = pd.read_excel(f'{path}/calibration.xlsx','name changes')
    for l in valid_leagues:
        df = pd.read_excel(f'{path}/fbref.xlsx',l)
        df = df.drop('Unnamed: 0', axis=1)
        df = df[['Player','club','Nation','Pos','Age','season','Min','Touches','o_Touches','Save%','Sh','TotAtt','TotCmp%','PrgP','Carries','PrgC',
                 'Tkl', 'TklW','blkSh', 'blkPass', 'Int', 'Clr','Err','Fls', 'Fld','Starts', 'Mn/Start', 'Subs', 'Mn/Sub', 'unSub']]
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
        club_gp = df.pivot_table(values=['MP_GK','Touches'],index=['club','season'],aggfunc='sum')
        df = df.merge(club_gp,left_on=['club','season'],right_on=['club','season'],how='left')
        df = df.merge(name_changes,left_on=['Player','Nation','yob','Pos'],right_on=['Player','Nation','yob','Pos'],how='left')
        df.loc[df['new_name'].notna(), 'Player'] = df['new_name']
        df.loc[df['Pos'] == 'GK', 'Save%'] = df.loc[df['Pos'] == 'GK', 'Save%'].fillna(0)
        df['Touch%'] = (df['Touches_x']/(df['Min']/90))/(df['Touches_y']/df['MP_GK_y'])
        df.rename(columns={'Touches_x': 'Touches'}, inplace=True)
        
        if(convert == 1):
            factors = league_conversion_factors(1,proj_year)
            f = np.exp(factors.loc[l] - factors.loc[target])
            df[f.index] = df[f.index] * f.values
            
        df_all.append(df)    
    return df_all,valid_leagues
    
def league_conversion_factors(read_file,proj_year):
    if(read_file == 1):
        all_eqn = pd.read_excel(f'{path}/calibration.xlsx','conversions',index_col=0)
    else:
        df,leagues = extract_player_data(0,'',proj_year)
        combos = list(combinations(range(0,len(df)), 2))
        categories = ['Touches', 'o_Touches', 'Save%', 'Sh', 'TotAtt', 'TotCmp%', 'PrgP', 'Carries',
                      'PrgC', 'Tkl', 'TklW', 'blkSh', 'blkPass', 'Int', 'Clr', 'Err', 'Fls', 'Fld']
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
        #all_eqn['Save%'] = all_eqn['Sh']/4
    return all_eqn

def aging_analysis(read_file,proj_year):
    if(read_file == 1):
        df_all = pd.read_excel(f'{path}/calibration.xlsx','aging')
    else:
        df,leagues = extract_player_data(1,'premier league',proj_year)
        df = pd.concat(df)
        df = df.reset_index(drop=True)
        df = df[df['Pos']!='GK']
        df = df[df['Min']>450]
        categories = ['Touches', 'o_Touches', 'Sh', 'TotAtt', 'PrgP', 'Carries',
                      'PrgC', 'Tkl', 'TklW', 'blkSh', 'blkPass', 'Int', 'Clr', 'Err', 'Fls', 'Fld']
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
                      'PrgC', 'Tkl', 'TklW', 'blkSh', 'blkPass', 'Int', 'Clr', 'Err', 'Fls', 'Fld']]
    return df_all

def keep_unique_substrings(row):
    substrings = row.split(',')  # Split the string into substrings
    unique_substrings = list(dict.fromkeys(substrings))  # Remove duplicates while preserving order
    return ','.join(unique_substrings)

def mean_reversion(proj_year,standard):
    #proj_year = 2025; standard = 'premier league'
    df,leagues = extract_player_data(1,standard,proj_year)
    df = pd.concat(df)
    df = df.reset_index(drop=True)
    
    coeffs = pd.read_excel(f'{path}/calibration.xlsx','model coefficients')
    coeffs = coeffs.drop('Unnamed: 0', axis=1)
    
    df['TotCmp%'] = df['TotCmp%'].fillna(coeffs.loc[coeffs['variable']=='TotCmp%','mean'].sum())
    df[['Touches','Sh','TotAtt','PrgP','Carries','PrgC','Tkl','TklW',
        'blkSh', 'blkPass', 'Int', 'Clr','Err','Fls', 'Fld','Touch%']] = df[['Touches','Sh','TotAtt','PrgP','Carries','PrgC',
                                                                    'Tkl','TklW','blkSh', 'blkPass', 'Int', 'Clr',
                                                                    'Err','Fls', 'Fld','Touch%']].fillna(0)
    
    df1 = df.pivot_table(values=['Min', 'Touches', 'o_Touches', 'Sh', 'TotAtt', 'PrgP', 
                                'Carries', 'PrgC', 'Tkl', 'TklW', 'blkSh', 'blkPass', 'Int', 'Clr', 'Err', 
                                'Fls', 'Fld', 'Starts', 'Mn/Start', 'Subs', 'Mn/Sub', 'unSub','MP_GK_y'],
                        index = ['Player', 'club', 'Nation', 'Pos', 'Age', 'season','yob'],
                        aggfunc="sum")
    df2 = df.pivot_table(values=['Save%', 'TotCmp%','Touch%'],
                        index = ['Player', 'club', 'Nation', 'Pos', 'Age', 'season','yob'],
                        aggfunc=lambda rows: np.average(rows, weights=df.loc[rows.index, 'Min']))
    df1 = df1.reset_index()
    df2 = df2.reset_index()
    df = df1.merge(df2)
    
    df['Min%'] = (df['Min']/90)/df['MP_GK_y']
    df['Starts'] = df['Starts']/df['MP_GK_y']
    df['Subs'] = df['Subs']/df['MP_GK_y']
    df['unSub'] = df['unSub']/df['MP_GK_y']
    
    pt = df.pivot_table(values=['Min%','Starts','Subs','unSub'],
                        index = ['Player','Nation','club','season','yob','Pos'],
                        aggfunc=lambda rows: np.average(rows, weights=df.loc[rows.index, 'MP_GK_y']))
    pt = pt.reset_index()
    pt['weight'] = pow(2/3,proj_year-pt['season'])
    
    df['weight'] = pow(2/3,proj_year-df['season'])
    df['weight2'] = pow(2/3,proj_year-df['season']) * df['Min']
    df_agg = df.pivot_table(values=['Min','Touches','o_Touches','Sh','TotAtt','PrgP','Carries','PrgC',
                                'Tkl', 'TklW','blkSh', 'blkPass', 'Int', 'Clr','Err','Fls', 'Fld'],
                              index=['Player','Nation','yob'], 
                              aggfunc=lambda rows: np.average(rows, weights=df.loc[rows.index, 'weight']))
    df_agg2 = df.pivot_table(values=['Save%','TotCmp%','Touch%'],
                              index=['Player','Nation','yob'], 
                              aggfunc=lambda rows: np.average(rows, weights=df.loc[rows.index, 'weight2']))
    pos = df.pivot_table(values=['Pos'], index=['Player','Nation','yob'], aggfunc=lambda x: ','.join(x.unique()))
    team = df.pivot_table(values=['club'], index=['Player','Nation','yob'], columns=['season'], aggfunc=lambda x: ','.join(x.unique()))
    team.columns = team.columns.droplevel(0)
    age = df.pivot_table(values=['Age','season'], index=['Player','Nation','yob'], aggfunc='max')
    #review this
    pt = pt.pivot_table(values=['Min%','Starts','Subs','unSub'], 
                        index=['Player','Nation','yob'], 
                        aggfunc=lambda rows: np.average(rows, weights=pt.loc[rows.index, 'weight']))
    avg = pd.read_excel(f'{path}/fbref.xlsx',standard)
    avg = avg[['Min','Touches','o_Touches','Save%','Sh','TotAtt','PrgP','Carries','PrgC',
               'Tkl','blkSh', 'blkPass', 'Int', 'Clr','Err','Fls', 'Fld']].sum()
    #avg = df[['Touches','o_Touches','Save%','Sh','TotAtt','TotCmp%','PrgP','Carries','PrgC',
    #         'Tkl', 'TklW','blkSh', 'blkPass', 'Int', 'Clr','Err','Fls', 'Fld','Touch%']].sum()
    for x in avg.index:
        if(x != 'Min'): avg[x] = 600 * avg[x].sum()/avg['Min'].sum()
    #percentage stats need to be fixed
    avg['Save%'] = coeffs.loc[coeffs['variable']=='Save%','mean'].sum()
    avg['TotCmp%'] = coeffs.loc[coeffs['variable']=='TotCmp%','mean'].sum()
    avg['TklW'] = coeffs.loc[coeffs['variable']=='TklW','mean'].sum()
    avg['Touch%'] = 1/11
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
    pt = pt.reset_index()
    df_agg = df_agg2.merge(df_agg, left_on=['Player','Nation','yob'], right_on=['Player','Nation','yob'])
    df_agg = pos.merge(df_agg, left_on=['Player','Nation','yob'], right_on=['Player','Nation','yob'])
    df_agg = team.merge(df_agg, left_on=['Player','Nation','yob'], right_on=['Player','Nation','yob'])
    df_agg = age.merge(df_agg, left_on=['Player','Nation','yob'], right_on=['Player','Nation','yob'])
    df_agg = pt.merge(df_agg, left_on=['Player','Nation','yob'], right_on=['Player','Nation','yob'])
    df_agg['TklW'] = 100*df_agg['TklW']/df_agg['Tkl']
    for x in avg.index:
        if(x not in ['Save%','TotCmp%','TklW','Touch%']):  df_agg[x] = df_agg[x] + avg[x]
        else: df_agg[x] = (df_agg[x]*df_agg['Min'] + avg[x]*600)/(df_agg['Min']+600)
    df_agg['Min'] = df_agg['Min'] + 600
    df_agg = aggregate_stats(df_agg,1)
    df_agg['90s'] = df_agg['90s'] - (600/90)
    df_agg['Pos'] = df_agg['Pos'].str.replace(' ','')
    df_agg['Pos'] = df_agg['Pos'].apply(keep_unique_substrings)
    df_agg['club'] = df_agg['club'].str.replace('-',' ')
    
    aging = aging_analysis(1,proj_year)
    projections_copy = df_agg.merge(aging, left_on='Age_copy', right_on='Age')
    projections_copy = projections_copy.merge(aging, left_on='Age_x', right_on='Age')
    
    for v in ['Touches','o_Touches','Save%','Sh','TotAtt','TotCmp%','PrgP','Carries','PrgC',
             'Tkl', 'TklW','blkSh', 'blkPass', 'Int', 'Clr','Err','Fls', 'Fld']:
        projections_copy[f'{v}_x'] *= projections_copy[v] / projections_copy[f'{v}_y']
        
    projections_copy = projections_copy[['Player', 'Nation', 'Pos', 'club', 'Age_x', 'season', 'Min%','Starts','Subs','unSub',
           'Touch%','Touches_x', 'o_Touches_x', 'Save%_x', 'Sh_x', 'TotAtt_x', 'TotCmp%_x', 'PrgP_x', 'Carries_x', 'PrgC_x', 
           'Tkl_x', 'TklW_x', 'blkSh_x', 'blkPass_x', 'Int_x', 'Clr_x', 'Err_x', 'Fls_x', 'Fld_x']]
    projections_copy.columns = ['Player', 'Nation', 'Pos', 'club', 'Age', 'season', 'Min/G','p(start)','p(sub)','p(unSub)',
           'Touch%','Touches', 'o_Touches', 'Save%', 'Sh', 'TotAtt', 'TotCmp%', 'PrgP', 'Carries', 'PrgC', 
           'Tkl', 'TklW', 'blkSh', 'blkPass', 'Int', 'Clr', 'Err', 'Fls', 'Fld']
    return projections_copy

def lineup_projection(team,custom_lineups,custom_mins):
    #team='Liverpool'; custom_mins=1; custom_lineups=1
    df = pd.read_excel(f'{path}/projections.xlsx','Sheet1')
    df = df.drop('Column1', axis=1)
    squads = pd.read_excel(f'{path}/projections.xlsx','squads')
    squads = squads.drop('Column1', axis=1)
    df = pd.merge(df, squads, on=['Player','Nation','Pos','Age'], how='left')
    if(custom_lineups == 1): df['club_x'] = df['club_y']
    if(custom_mins == 1): df['Min/G_x'] = df['Min/G_y']
    df.rename(columns={'club_x': 'club', 'Min/G_x': 'Min/G'}, inplace=True)
    df.drop(['club_y','Min/G_y'], axis=1, inplace=True) 
    df['Min/G'] = df['Min/G'].fillna(0)
    
    coeffs = pd.read_excel(f'{path}/calibration.xlsx','model coefficients')
    coeffs = coeffs.drop('Unnamed: 0', axis=1)
    
    df_team = df[df['club']==team]
    keepers = df_team[df_team['Pos']=='GK']    
    keepers = keepers.sort_values(by='Min/G', ascending=False)
    keepers['rank'] = list(range(1,len(keepers)+1))
    outfielders = df_team[df_team['Pos']!='GK']
    outfielders = outfielders.sort_values(by='Min/G', ascending=False)
    outfielders['rank'] = list(range(1,len(outfielders)+1))
    
    exp = 1.0
    while((keepers['Min/G'].sum() < 0.99) or (keepers['Min/G'].sum() > 1.01)):
        if(keepers['Min/G'].sum() < 0.99):
            keepers['Min/G'] *= pow(exp,keepers['rank'])
            exp += 0.0001
        else:
            keepers['Min/G'] *= pow(exp,keepers['rank'])
            exp -= 0.0001
        keepers['Min/G'] = keepers['Min/G'].clip(upper=1)
    
    exp = 1.0
    while((outfielders['Min/G'].sum() <= 9.9) or (outfielders['Min/G'].sum() >= 10.1)):
        if(outfielders['Min/G'].sum() <= 9.9):
            outfielders['Min/G'] *= pow(exp,outfielders['rank'])
            exp += 0.0001
        elif(outfielders['Min/G'].sum() >= 10.1):
            outfielders['Min/G'] *= pow(exp,outfielders['rank'])
            exp -= 0.0001
        outfielders['Min/G'] = outfielders['Min/G'].clip(upper=1)
        #print(outfielders['Min/G'].sum())
    
    keepers['Min/G'] = keepers['Min/G']/keepers['Min/G'].sum()
    outfielders['Min/G'] = 10*outfielders['Min/G']/outfielders['Min/G'].sum()
    touches = (outfielders['Min/G']*outfielders['Touches']).sum() + (keepers['Min/G']*keepers['Touches']).sum()
    opp_touches = (outfielders['Min/G']*outfielders['o_Touches']).sum() + (keepers['Min/G']*keepers['o_Touches']).sum()
    touch_pct = (outfielders['Min/G']*outfielders['Touch%']).sum() + (keepers['Min/G']*keepers['Touch%']).sum()
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
            measure.append((outfielders[m] * outfielders['Min/G']).sum() + (keepers[m] * keepers['Min/G']).sum())
        if(m in ['TotCmp%','TklW']):
            measure.append(((outfielders[m] * outfielders['Min/G']).sum() + (keepers[m] * keepers['Min/G']).sum())/11)
        elif(m in ['Tkl','blkSh','blkPass','Int', 'Clr', 'Err', 'Fls']):
            measure.append(((outfielders[m] * outfielders['Min/G'] * outfielders['o_Touches']).sum() + (keepers[m] * keepers['Min/G'] * keepers['o_Touches']).sum())/measure[1])
        elif(m in ['Sh','TotAtt','PrgP','Carries','PrgC','Fld']):
            measure.append(((outfielders[m] * outfielders['Min/G'] * outfielders['Touches']).sum() + (keepers[m] * keepers['Min/G'] * keepers['Touches']).sum())/measure[0])
    
    measure.append(measure[0]/measure[1])
    measure.append(measure[0]+measure[1])
    pts = (np.array(measure[2:]) - np.array(coeffs['mean'].to_list()[:-1])) / np.array(coeffs['stdev'].to_list()[:-1])
    pts *= np.array(coeffs['weight'].to_list()[:-1])
    pts = sum(pts) * coeffs.loc[coeffs['variable']=='pred','stdev'].sum()  + coeffs.loc[coeffs['variable']=='pred','mean'].sum()
    #print(team,pts)
    #print(outfielders[outfielders['Min/G']>0.01][['Player','Min/G']])
    return pts
    
def league_projections(league,custom_lineups,custom_mins):
    table = [['Team','Points']]
    team_list = pd.read_excel(f'{path}/projections.xlsx','teams')
    team_list = list(team_list[league])
    for t in team_list:
        pts = lineup_projection(t,custom_lineups,custom_mins)
        table.append([t,pts])
    table = pd.DataFrame(table)
    table.columns = table.iloc[0];table = table.drop(0)
    table = table.apply(pd.to_numeric, errors='ignore')
    
    coeffs = pd.read_excel(f'{path}/calibration.xlsx','model coefficients')
    coeffs = coeffs.drop('Unnamed: 0', axis=1)
    lg_avg = coeffs.loc[coeffs['variable']=='pred','mean'].sum()
    table['Points'] *= (lg_avg/table['Points'].mean())
    return table

#%% extract data
#extract team stats for multiple leagues and years
#t_stats = multi_leagues(0)
#extract player stats for multiple leagues
player_stats_raw = multi_team_links(2021,2024,882)
#ote = opp_touches_error(2017,2024,9)

#%% analyze
#team regression analysis
#t_stats = multi_leagues(1)
#summary,t_stats_reg,lasso,coeffs = regression(t_stats,0.001)
#player projection data
factors = league_conversion_factors(0,2025)
aging = aging_analysis(1,2025)

#%% generate player projections
projections = mean_reversion(2025,'premier league')
#to identify players whose data has been duplicated due to yob mismatch
duplicates = projections.pivot_table(values=['season'], index=['Player','Nation','Age'], aggfunc='count')
duplicates = duplicates[duplicates['season']>1]

#%% points projections
#lineup_projection('Chelsea',0,0) #team, custom lineups, custom mins
table = league_projections('premier league',1,1) #team, custom lineups, custom mins