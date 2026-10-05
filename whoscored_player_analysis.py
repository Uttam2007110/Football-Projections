# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 15:24:55 2026

@author: Subramanya.Ganti
"""

#%% imports
import numpy as np
import pandas as pd
from itertools import combinations
from sklearn import linear_model
import statsmodels.api as sm
from patsy import dmatrix
pd.options.mode.chained_assignment = None

path = 'C:/Users/Subramanya.Ganti/Downloads/Sports/football/whoscored'
#path = 'C:/Users/uttam/Desktop/Sports/football/whoscored'

season = 2027
project_league = 'Italy'

#%% functions
def player_stats_extract(league):
    print(league)
    team_stats = pd.read_excel(f'{path}/{league}_teams.xlsx','Sheet1')
    team_stats['passesAgainst'] = (1-team_stats['possession']) * (team_stats['passTotal']/team_stats['possession'])
    
    full_player_stats = pd.read_excel(f'{path}/{league}_players.xlsx','Sheet1')
    full_player_stats = full_player_stats.merge(team_stats[['team','season','P','shotsConcededPerGame','GA','passesAgainst']], 
                                                on=['team','season'], how='left')
    
    full_player_stats['def_activity'] = full_player_stats['Shots_blocked'] + full_player_stats['clearanceTotal'] +\
                                        full_player_stats['interceptionAll'] + full_player_stats['tackleTotalAttempted']
    
    bcit_pivot = full_player_stats.pivot_table(index=['team','season'],values=['def_activity','goalTotal','assist','passTotal'],aggfunc='sum')
    bcit_pivot = bcit_pivot.reset_index()
    bcit_pivot = bcit_pivot.rename(columns={'def_activity': 'def_activity_team', 'passTotal':'passTotal_team'})
    
    full_player_stats = full_player_stats.merge(bcit_pivot, on=['team','season'], how='left')
    full_player_stats['inv_raw'] = (full_player_stats['def_activity']*full_player_stats['P']*90)/\
                                    (full_player_stats['def_activity_team']*full_player_stats['minsPlayed'])
    full_player_stats['inv_adj'] = 1/(1+full_player_stats['inv_raw']/.055)
    full_player_stats['sha_raw'] = full_player_stats['inv_adj']*full_player_stats['shotsConcededPerGame']*full_player_stats['minsPlayed']/90
    sha_raw_pivot = full_player_stats.pivot_table(index=['team','season'],values='sha_raw',aggfunc='sum')
    sha_raw_pivot = sha_raw_pivot.reset_index()
    full_player_stats = full_player_stats.merge(sha_raw_pivot, on=['team','season'], how='left')
    full_player_stats['shotsAgainstTotal'] = full_player_stats['P']*full_player_stats['shotsConcededPerGame']*\
                                             full_player_stats['sha_raw_x']/full_player_stats['sha_raw_y']
    full_player_stats['ShA'] = 90*full_player_stats['shotsAgainstTotal']/full_player_stats['minsPlayed']
    full_player_stats['GA_player'] = full_player_stats['GA']*(full_player_stats['minsPlayed']/90)
    
    full_player_stats['inv_rawp'] = (full_player_stats['passTotal']*full_player_stats['P']*90)/\
                                    (full_player_stats['passTotal_team']*full_player_stats['minsPlayed'])
    full_player_stats['inv_adjp'] = 1/(1+full_player_stats['inv_rawp']/.055)
    full_player_stats['sha_rawp'] = full_player_stats['inv_adjp']*full_player_stats['passesAgainst']*full_player_stats['minsPlayed']/(90*full_player_stats['P'])
    p_sha_raw_pivot = full_player_stats.pivot_table(index=['team','season'],values='sha_rawp',aggfunc='sum')
    p_sha_raw_pivot = p_sha_raw_pivot.reset_index()
    full_player_stats = full_player_stats.merge(p_sha_raw_pivot, on=['team','season'], how='left')
    full_player_stats['passesAgainstTotal'] = full_player_stats['passesAgainst']*\
                                             full_player_stats['sha_rawp_x']/full_player_stats['sha_rawp_y']
    full_player_stats['PssA'] = 90*full_player_stats['passesAgainstTotal']/full_player_stats['minsPlayed']
    
    full_player_stats['Sh'] = 90*full_player_stats['shotsTotal']/full_player_stats['minsPlayed']
    full_player_stats['Blk'] = 90*(full_player_stats['Shots_blocked']+full_player_stats['Cross_blocked']+full_player_stats['outfielderBlockedPass'])/full_player_stats['minsPlayed']
    full_player_stats['YC'] = 90*full_player_stats['yellowCard']/full_player_stats['minsPlayed']
    full_player_stats['RC'] = 90*full_player_stats['redCard']/full_player_stats['minsPlayed']
    full_player_stats['Clr'] = 90*full_player_stats['clearanceTotal']/full_player_stats['minsPlayed']
    full_player_stats['Drb'] = 90*full_player_stats['dribbleTotal']/full_player_stats['minsPlayed']
    full_player_stats['FlsW'] = 90*full_player_stats['foulGiven']/full_player_stats['minsPlayed']
    full_player_stats['Fls'] = 90*full_player_stats['foulCommitted']/full_player_stats['minsPlayed']
    full_player_stats['Int'] = 90*full_player_stats['interceptionAll']/full_player_stats['minsPlayed']
    full_player_stats['Pss'] = 90*full_player_stats['passTotal']/full_player_stats['minsPlayed']
    full_player_stats['KP'] = 90*full_player_stats['keyPassesTotal']/full_player_stats['minsPlayed']
    full_player_stats['KP/P'] = full_player_stats['keyPassesTotal']/full_player_stats['passTotal']
    full_player_stats['TO%'] = 90*(full_player_stats['turnover']+full_player_stats['dispossessed'])/full_player_stats['minsPlayed']
    full_player_stats['Int'] = 90*full_player_stats['interceptionAll']/full_player_stats['minsPlayed']
    full_player_stats['Off'] = 90*full_player_stats['offsideGiven']/full_player_stats['minsPlayed']
    full_player_stats['Tkl'] = 90*full_player_stats['tackleTotalAttempted']/full_player_stats['minsPlayed']
    full_player_stats['Head'] = 90*full_player_stats['duelAerialTotal']/full_player_stats['minsPlayed']
    full_player_stats['G%'] = full_player_stats['goalTotal_x']/(full_player_stats['goalTotal_y']*full_player_stats['minsPlayed']/(90*full_player_stats['P']))
    full_player_stats['A%'] = full_player_stats['assist_x']/(full_player_stats['assist_y']*full_player_stats['minsPlayed']/(90*full_player_stats['P']))
    full_player_stats['Save%'] = full_player_stats['saveTotal']/(full_player_stats['GA_player']+full_player_stats['saveTotal'])
    
    full_player_stats = full_player_stats.drop(columns=['shotsConcededPerGame','def_activity','def_activity_team','inv_raw','inv_adj',
                                                        'sha_raw_x','sha_raw_y','inv_rawp','inv_adjp','sha_rawp_x','sha_rawp_y',
                                                        'goalTotal_y','assist_y','GA_player'])
    #basic data cleaning
    full_player_stats['KP/P'] = full_player_stats['KP/P'].fillna(0)
    full_player_stats['aerial_win%'] = full_player_stats['aerial_win%'].fillna(0)
    full_player_stats['dribble_win%'] = full_player_stats['dribble_win%'].fillna(0)
    full_player_stats['long_success%'] = full_player_stats['long_success%'].fillna(0)
    full_player_stats['short_success%'] = full_player_stats['short_success%'].fillna(0)
    full_player_stats['long_bias'] = full_player_stats['long_bias'].fillna(0.25)
    full_player_stats['shots_target%'] = full_player_stats['shots_target%'].fillna(0)
    full_player_stats['shots_blocked%'] = full_player_stats['shots_blocked%'].fillna(0)
    full_player_stats['tackle_success%'] = full_player_stats['tackle_success%'].fillna(0)
    full_player_stats['KP/P'] = full_player_stats['KP/P'].clip(upper=1)
    full_player_stats['G%'] = full_player_stats['G%'].clip(upper=1)
    full_player_stats['A%'] = full_player_stats['A%'].clip(upper=1)
    full_player_stats['Save%'] = full_player_stats['Save%'].clip(upper=100)
    full_player_stats['long_bias'] = full_player_stats['long_bias'].clip(upper=1)
    return full_player_stats,team_stats

def stabilization_rate(full_player_stats):
    stable = full_player_stats[['playerId', 'name', 'age', 'season', 'height', 'weight', 'positionText', 'team', 'tournamentName',
                                'apps', 'minsPlayed', 'MPG', 'ShA', 'Sh', 'Blk', 'YC', 'RC', 'Clr', 'Drb', 'FlsW', 'Fls', 'Int', 'Pss',
                                'KP', 'KP/P', 'TO%', 'Off', 'Tkl', 'G%', 'A%', 'Save%','dribble_win%','long_success%', 'short_success%',
                                'long_bias','shots_target%', 'shots_blocked%','tackle_success%','Head','aerial_win%']]
    stable2 = stable.copy()
    stable2['season'] += 1 
    stable = stable.merge(stable2, on=['playerId','season'],how='left')
    stable = stable.dropna(subset=['name_y'])
    stable = stable[(stable['minsPlayed_x']>1000)&(stable['minsPlayed_y']>1000)]
    
    for c in ['MPG', 'ShA', 'Sh', 'Blk', 'YC', 'RC', 'Clr', 'Drb', 'FlsW', 'Fls', 'Int', 'Pss', 'KP', 'KP/P', 'TO%', 'Off', 'Tkl', 'G%','A%', 'Head',
              'Save%','dribble_win%','long_success%','short_success%','long_bias','shots_target%','shots_blocked%','tackle_success%','aerial_win%']:
        if(c=='Save%'): 
            print(c,"yoy correlation",round(stable.loc[stable['positionText_x']=='Goalkeeper',f'{c}_x'].corr(stable.loc[stable['positionText_y']=='Goalkeeper',f'{c}_y'])**2,2))
        else: 
            print(c,"yoy correlation",round(stable[f'{c}_x'].corr(stable[f'{c}_y'])**2,2))

def regression_player(full_player_stats,league_strength,season):
    target_league = 'England_1' #verify if this needs to be hardcoded
    stat_list = ['age','play','MPG','ShA','Sh','Blk','YC','RC','Clr','Drb','FlsW','Fls','Int','Pss','KP','KP/P','TO%','Off','Tkl',
                 'G%','A%','Save%','dribble_win%','long_success%','short_success%','long_bias','shots_target%','shots_blocked%',
                 'tackle_success%','PssA','Head','aerial_win%']
    stat_reg_mins = [1,0,6000,2000,1000,2000,6000,15000,2000,2000,2000,2000,2000,2000,2000,2000,1000,2000,2000,
                     3000,5000,12000,9000,3000,2000,1000,9000,9000,6000,2000,2000,3000]
    
    data = full_player_stats.copy()
    data['play'] = data['apps']/data['P']
    data['s_weight'] = 5*np.exp((data['season']-season)/2)
    data['s_weight2'] = np.exp(data['season']-season)
    
    #apply league strength conversions on data
    adjusted_stats = [stat for stat in stat_list if stat not in ['age','play','MPG']]
    league_strength_by_row = league_strength.loc[data["label"],adjusted_stats].to_numpy()
    target_league_strength = league_strength.loc[target_league,adjusted_stats].to_numpy()
    data[adjusted_stats] = data[adjusted_stats].to_numpy() * np.exp(league_strength_by_row - target_league_strength)
    
    reg_mins = pd.Series(stat_reg_mins,index=stat_list,dtype=float)    
    data["minsPlayed"] = pd.to_numeric(data["minsPlayed"], errors="coerce")
    data["s_weight"] = pd.to_numeric(data["s_weight"], errors="coerce")
    
    for stat in stat_list:  data[stat] = pd.to_numeric(data[stat], errors="coerce")
    data["effective_minutes"] = (data["minsPlayed"] * data["s_weight"])
    
    regression_values = data.loc[data['minsPlayed']>500,stat_list].quantile(0.5)
    #not all stats mean revert to 50th percentile
    regression_values['Save%'] = data.loc[(data['minsPlayed']>500)&(data['positionText']=='Goalkeeper'),'Save%'].quantile(0.5)
    regression_values['ShA'] = data.loc[data['minsPlayed']>500,'ShA'].quantile(0.5)
    regression_values['YC'] = data.loc[data['minsPlayed']>500,'YC'].mean()
    regression_values['RC'] = data.loc[data['minsPlayed']>500,'RC'].mean()
    regression_values['G%'] = data.loc[data['minsPlayed']>500,'G%'].mean()
    regression_values['A%'] = data.loc[data['minsPlayed']>500,'A%'].mean()
    
    identity_columns = ["playerId","name","height","weight","positionText"]
    latest_player_data = (data.sort_values("season").groupby("playerId", as_index=False).tail(1)[identity_columns].set_index("playerId"))    
    
    output_rows = []
    for player_id, player_rows in data.groupby("playerId", sort=False):
        values = player_rows[stat_list].to_numpy(dtype=float)
        effective_minute_weights = player_rows["effective_minutes"].to_numpy(dtype=float)
        row_weights = np.broadcast_to(effective_minute_weights[:, None], values.shape).copy()
        row_weights[:, stat_list.index("play")] = player_rows["s_weight2"].to_numpy(dtype=float)
    
        valid = np.isfinite(values)
        effective_minutes_by_stat = (valid * row_weights).sum(axis=0)
        weighted_sums = np.where(valid, values * row_weights, 0.0).sum(axis=0)
        raw_values = np.divide(weighted_sums, effective_minutes_by_stat, out=np.full(len(stat_list), np.nan), where=effective_minutes_by_stat > 0)
        has_observations = np.isfinite(raw_values)
    
        adjusted_values = (weighted_sums + regression_values.to_numpy() * reg_mins.to_numpy()) / (effective_minutes_by_stat + reg_mins.to_numpy())
        adjusted_values[~has_observations] = np.nan
    
        row = {"playerId": player_id, "total_effective_minutes": effective_minute_weights.sum()}
        row.update({f"raw_{stat}": value for stat, value in zip(stat_list, raw_values)})
        row.update({stat: value for stat, value in zip(stat_list, adjusted_values)})
        output_rows.append(row)

    result = pd.DataFrame(output_rows).set_index("playerId")
    result = latest_player_data.join(result, how="inner").reset_index()
    result['season'] = season
    result = result[['playerId','name','height','weight','positionText','season']+stat_list]
    result['Save%'] = np.where(result['positionText']=='Goalkeeper',result['Save%'],0)
    return result

def league_conversions(df_full,season):
    #season = 2027    
    #df,leagues = player_stats_extract_all() #list of dataframes for each league
    
    df = []
    leagues = df_full['label'].drop_duplicates().to_list()
    for l in leagues:
        df.append(df_full[df_full['label']==l])
    
    combos = list(combinations(range(0,len(df)), 2))
    all_eqn = [['category'] + leagues]
    
    categories = ['ShA','Sh','Blk','YC','RC','Clr','Drb','FlsW','Fls','Int','Pss','KP','KP/P','TO%','Off','Tkl','Head','G%','A%','Save%',
                 'dribble_win%','long_success%','short_success%','long_bias','shots_target%','shots_blocked%','tackle_success%','aerial_win%','PssA']
    
    for ch in categories:
        #ch = 'Save%'
        eqn = pd.DataFrame(columns=range(0,len(df)), index=range(0,len(combos)))
        eqn[f'{ch} log factor'] = 0.0
        eqn[f'{ch} Mins'] = 0.0
        r = 0
        
        for c in combos:
            from_df = df[c[0]] #pl1
            to_df = df[c[1]] #pl2
            if(ch == 'Save%'):
                from_df = from_df[from_df['positionText']=='Goalkeeper']
                to_df = to_df[to_df['positionText']=='Goalkeeper']
                
            df_from_to = to_df.merge(from_df, left_on=['playerId'], right_on=['playerId'])
            df_from_to['c_weight'] = np.exp((df_from_to['season_x']-season)/3)*np.exp((df_from_to['season_y']-season)/3)
            
            eqn.loc[r,c[0]] = 1
            eqn.loc[r,c[1]] = -1
            
            factor = (df_from_to['c_weight']*(df_from_to[f'{ch}_x']-df_from_to[f'{ch}_y'])).sum()/df_from_to['c_weight'].sum()
            factor = np.log((factor/from_df[f'{ch}'].mean()) + 1)
        
            eqn.loc[r,f'{ch} log factor'] = factor
            eqn.loc[r,f'{ch} Mins'] = (df_from_to['c_weight']*df_from_to['minsPlayed_x']*df_from_to['minsPlayed_y']).sum()
            #print(ch,round(factor,2))
            r+=1
            
        eqn[list(range(0,len(df)))] = eqn[list(range(0,len(df)))].fillna(0.0) #.infer_objects(copy=False)
        eqn.replace([np.inf, -np.inf], np.nan, inplace=True)
        eqn = eqn[eqn[f'{ch} log factor'].notna()]
        
        regr = linear_model.LinearRegression(fit_intercept=False)
        regr.fit(eqn[list(range(0,len(df)))], eqn[f'{ch} log factor'], sample_weight=eqn[f'{ch} Mins'])
        #all_eqn.append(eqn)
        #all_eqn.loc[r0] = list(regr.coef_) + [ch]
        coef_list = list(regr.coef_)
        minimum_element = min(coef_list)
        coef_list = [element - minimum_element for element in coef_list]
        #print()
        #print(ch,coef_list)
        all_eqn.append([ch] + coef_list)
        
    all_eqn = pd.DataFrame(all_eqn)
    all_eqn.columns = all_eqn.iloc[0];all_eqn = all_eqn.drop(0)
    all_eqn = all_eqn.T
    all_eqn.columns = all_eqn.iloc[0]
    all_eqn = all_eqn.drop('category')
    all_eqn = all_eqn.apply(pd.to_numeric, errors='ignore')    
    return all_eqn

def team_subclusters(team_stats_list):
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    team_stats = pd.concat(team_stats_list)
    team_stats = team_stats[['team','league','season','P','GD','possession']]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(team_stats[['GD','possession']])
    kmeans = KMeans(n_clusters=3, init='k-means++', n_init=50, random_state=42, algorithm="lloyd")
    kmeans.fit(X_scaled)
    
    #relabeling to ensure order is consistent
    raw_labels = kmeans.fit_predict(X_scaled)
    centroids = kmeans.cluster_centers_
    centroid_scores = centroids.mean(axis=1)
    cluster_order = np.argsort(-centroid_scores)
    label_map = np.empty(kmeans.n_clusters, dtype=int)
    label_map[cluster_order] = np.arange(kmeans.n_clusters)
    
    team_stats['cluster'] = label_map[raw_labels] #kmeans.labels_
    team_stats['label'] = team_stats['league']+"_"+team_stats['cluster'].astype(str)
    team_stats = team_stats[['team','season','label','P']]
    return team_stats

def player_age_correction(player_stats,season,offseason,refresh_teams):
    player_stats = player_stats.sort_values(by=['playerId', 'season'], ascending=[True, False])
    bio = player_stats[['playerId','name','season','team','age']].drop_duplicates(subset=['playerId'], keep='first')
    bio = bio.drop_duplicates(subset=['playerId'], keep='last')
    bio['age'] += season - bio['season']
    
    if(offseason == 1): bio['team'] = np.where(bio['season']!=season-1,np.nan,bio['team'])
    else: bio['team'] = np.where(bio['season']!=season,np.nan,bio['team'])
    
    old_bio = pd.read_excel(f'{path}/player_bio.xlsx','Sheet1')
    old_bio = old_bio.drop(columns=['Unnamed: 0'])
    if(refresh_teams == 1):
        bio = bio[['playerId','name','age','team']]
        bio = bio.merge(old_bio[['playerId','start','sub']], how='left')
    else:
        bio = bio[['playerId','name','age']]
        bio = bio.merge(old_bio[['playerId','team','start','sub']], how='left')
        
    bio = bio[['playerId','name','age','team','start','sub']]
    return bio

def aging_effects(player_stats):
    categories = ['ShA','Sh','Blk','YC','RC','Clr','Drb','FlsW','Fls','Int','Pss','KP','KP/P','TO%','Off','Tkl','Head','G%','A%','Save%',
                 'dribble_win%','long_success%','short_success%','long_bias','shots_target%','shots_blocked%','tackle_success%','aerial_win%','PssA']
    
    min_minutes, reference_age, spline_df = 500, 27, 5
    data = player_stats.copy()
    data[['age','minsPlayed'] + categories] = data[['age','minsPlayed'] + categories].apply(pd.to_numeric, errors='coerce')
    data = data.dropna(subset=['playerId','age','minsPlayed'])
    data = data[data['minsPlayed'] >= min_minutes]
    data = data[data['playerId'].isin(data.groupby('playerId')['age'].nunique().loc[lambda x: x >= 2].index)].copy()
    
    ages = np.sort(data['age'].unique())
    reference_age = ages[np.abs(ages - reference_age).argmin()]
    basis = dmatrix(f'bs(age, df={spline_df}, degree=3, include_intercept=False)', data, return_type='dataframe')
    basis.columns = [f'age_{i}' for i in range(basis.shape[1])]
    data = pd.concat([data.reset_index(drop=True), basis.reset_index(drop=True)], axis=1)
    basis_columns = basis.columns.tolist()
    
    coverage = data.groupby('age').agg(players=('playerId','nunique'), observations=('playerId','size'), minutes=('minsPlayed','sum')).reset_index()
    age_curves = []
    
    for stat in categories:
        d = data[['playerId','age','minsPlayed',stat] + basis_columns].dropna().copy()
        
        for column in [stat] + basis_columns:
            d[f'{column}_within'] = d[column] - (
                d[column].mul(d['minsPlayed']).groupby(d['playerId']).transform('sum')
                / d['minsPlayed'].groupby(d['playerId']).transform('sum')
            )
        
        model = sm.WLS(d[f'{stat}_within'], d[[f'{column}_within' for column in basis_columns]], weights=d['minsPlayed']).fit()
        reference_values = d.loc[d['age'].eq(reference_age), [stat,'minsPlayed']]
        reference_value = np.average(reference_values[stat], weights=reference_values['minsPlayed'])
        
        age_basis = dmatrix(f'bs(age, df={spline_df}, degree=3, include_intercept=False)', {'age': ages}, return_type='dataframe')
        reference_basis = dmatrix(f'bs(age, df={spline_df}, degree=3, include_intercept=False)', {'age': [reference_age]}, return_type='dataframe')
        
        fitted_values = reference_value + (age_basis.to_numpy() - reference_basis.to_numpy()) @ model.params.to_numpy()
        fitted_values = np.clip(fitted_values, 1e-8, None)
        
        curve = pd.DataFrame({'stat': stat, 'age': ages, 'fitted_value': fitted_values})
        curve['next_fitted_value'] = curve['fitted_value'].shift(-1)
        curve['age_multiplier'] = (curve['next_fitted_value'] / curve['fitted_value']).clip(0, 90)
        curve['r_squared_within'] = model.rsquared
        
        age_curves.append(curve)
    
    age_curves = pd.concat(age_curves, ignore_index=True).merge(coverage, on='age', how='left')

    age_delta_models = {}
    for stat, d in age_curves.groupby('stat'):
        d = d[['age','age_multiplier','observations']].dropna().copy()
        d = d[(d['age_multiplier'] >= 0) & (d['age_multiplier'] <= 90) & (d['observations'] > 0)]
        d['age_sq'] = d['age'] ** 2
        
        model = sm.WLS(
            d['age_multiplier'],
            sm.add_constant(d[['age','age_sq']]),
            weights=d['observations'],
        ).fit()
        
        age_delta_models[stat] = model.params
        
    return age_delta_models

def age_delta(age_delta_models, stat, age_x, age_y):
    if stat not in age_delta_models:  return 1.0
    if age_x == age_y: return 1.0
    
    params = age_delta_models[stat]
    
    def one_year_factor(age):
        factor = (params['const'] + params['age'] * age + params['age_sq'] * age ** 2)        
        return np.clip(factor, 0, 90)
    
    if age_y > age_x:
        ages = np.arange(age_x, age_y)
        return np.prod([one_year_factor(age) for age in ages])
    
    ages = np.arange(age_y, age_x)
    reverse_factor = np.prod([one_year_factor(age) for age in ages])
    
    return 1 / reverse_factor if reverse_factor > 0 else 0.0

def player_stats_extract_all():
    df = []; df_teams = []
    leagues = ['England','Spain','Germany','Italy','France','Netherlands','Turkey','Portugal','Belgium','Scotland','Russia',
               'Germany2','England2']
    for l in leagues:
        pl,tl = player_stats_extract(l)
        pl['league'] = l
        tl['league'] = l
        df.append(pl)
        df_teams.append(tl)
        
    team_labels = team_subclusters(df_teams)
    df_players = pd.concat(df)
    df_players = df_players.merge(team_labels[['team','season','label']], on=['team','season'], how='left')   
    df_players['age'] = np.where(df_players['season']<2027,df_players['age']+df_players['season']-2026,df_players['age'])
    
    #position text fixes
    df_players['positionText'] = df_players['positionText'].str.replace("Substitute","Midfielder")
    df_players['positionText'] = df_players['positionText'].str.replace("midfielder","Midfielder")
    df_players['positionText'] = df_players['positionText'].str.replace("defender","Defender")
    df_players['positionText'] = df_players['positionText'].str.replace("forward","Forward")
    df_players['positionText'] = df_players['positionText'].str.replace("goalkeeper","Goalkeeper")
    return df_players,team_labels

def regression_player_full(full_player_stats,league_strength,season):
    gk = full_player_stats[full_player_stats['positionText']=='Goalkeeper']
    df = full_player_stats[full_player_stats['positionText']=='Defender']
    md = full_player_stats[full_player_stats['positionText']=='Midfielder']
    fw = full_player_stats[full_player_stats['positionText']=='Forward']
    
    gk_reg = regression_player(gk,league_strength,season)
    df_reg = regression_player(df,league_strength,season)
    md_reg = regression_player(md,league_strength,season)
    fw_reg = regression_player(fw,league_strength,season)
    
    reg_stats = pd.concat([gk_reg,df_reg,md_reg,fw_reg])
    return reg_stats

def relevant_team_label(target_league,team_clusters,season):
    tc = team_clusters.copy()
    if(target_league == 'Germany2'):
        tc['alternate'] = np.where(tc['label'].str.contains('Germany_'),'Germany2_0',pd.NA)
    elif(target_league == 'England2'):
        tc['alternate'] = np.where(tc['label'].str.contains('England_'),'England2_0',pd.NA)
    elif(target_league == 'Germany'):
        tc['alternate'] = np.where(tc['label'].str.contains('Germany2_'),'Germany_2',pd.NA)
    elif(target_league == 'England'):
        tc['alternate'] = np.where(tc['label'].str.contains('England2_'),'England_2',pd.NA)
    else:
        tc['alternate'] = pd.NA
    
    teams_list = tc[tc['label'].str.contains(f"{target_league}_")]
    teams_list = teams_list[teams_list['season']==season]['team'].drop_duplicates().to_list()
    
    tc = tc[tc['team'].isin(teams_list)]
    tc.reset_index(drop=True, inplace=True)
    tc['label'] = np.where(tc['alternate'].notna(),tc['alternate'],tc['label'])
    full_range = pd.DataFrame({"season": range(tc["season"].min(), tc["season"].max() + 1)})
    full_range['team'] = [teams_list] * len(full_range)
    full_range = full_range.explode('team').reset_index(drop=True)
    tc = full_range.merge(tc, on=['team','season'], how='left')
    tc['label'] = tc['label'].fillna(f'{target_league}_2')
    tc['P'] = tc['P'].fillna(30)
    #tc = team_clusters[team_clusters["label"].str.contains(f"{target_league}_")]
    tc['weight'] = np.exp(tc['season']-season) * tc['P'] / 0.75
    tc['num'] = (tc['label'].str[-1:]).astype(int)
    
    tc_pivot = tc.pivot_table(index='team',values='num',aggfunc=lambda x: np.average(x, weights=tc.loc[x.index, 'weight']))
    tc_season = tc.pivot_table(index='team',values='season',aggfunc='max')
    tc_pivot['season'] = tc_season
    tc_pivot = tc_pivot.reset_index()       
    tc_pivot['label'] = target_league + "_" + round(tc_pivot['num'],0).astype(int).astype(str)
    tc_pivot = tc_pivot[['team','label','num']]
    return tc_pivot

def projections_target_league(target_league,regressed_stats,player_bio,league_strength,stat_aging,team_clusters,season):
    available_stats = ['ShA','Sh','Blk','YC','RC','Clr','Drb','FlsW','Fls','Int','Pss','KP','KP/P','TO%','Off','Tkl','Head','G%','A%','Save%',
                 'dribble_win%','long_success%','short_success%','long_bias','shots_target%','shots_blocked%','tackle_success%','aerial_win%','PssA']
    
    regressed_stats = regressed_stats.merge(player_bio[['playerId','age','team','start','sub']], on=['playerId'], how='left')
    age_adjusted_projection = regressed_stats.copy()
    
    for stat in available_stats:
        factors = np.array([age_delta(stat_aging,stat,age_x,age_y) for age_x, age_y in zip(age_adjusted_projection['age_x'],age_adjusted_projection['age_y'])])
        age_adjusted_projection[stat] = (pd.to_numeric(age_adjusted_projection[stat], errors='coerce')* factors)
        
    tc = relevant_team_label(target_league,team_clusters,season)
    
    age_adjusted_projection = age_adjusted_projection.merge(tc, on='team', how='left')
    age_adjusted_projection = age_adjusted_projection[~age_adjusted_projection['label'].isna()]
    #apply league strength conversions on data, reverse of the previous
    league_strength_by_row = league_strength.loc[age_adjusted_projection["label"],available_stats].to_numpy()
    target_league_strength = league_strength.loc['England_1',available_stats].to_numpy()
    age_adjusted_projection[available_stats] = age_adjusted_projection[available_stats].to_numpy() *\
                                                np.exp(target_league_strength-league_strength_by_row)
                                                
    return age_adjusted_projection,tc

def calibrate_to_mean(predictions, target_mean):
    #Calibrate a complete prediction cohort to a known mean on sqrt scale.
    predictions = np.asarray(predictions, dtype=float)
    if predictions.size == 0: return predictions
    if not np.isfinite(target_mean) or target_mean < 0: raise ValueError("target_mean must be a finite, non-negative value")

    mu = np.sqrt(np.clip(predictions, 0, None))
    discriminant = mu.mean()**2 - np.mean(mu**2) + target_mean
    if discriminant < 0: raise ValueError("The requested mean cannot be reached with a non-negative sqrt-scale shift for this prediction cohort")
    shift = -mu.mean() + np.sqrt(discriminant)
    calibrated = (mu + shift)**2
    if (mu + shift < 0).any(): raise ValueError("The requested mean requires negative values on the sqrt scale")
    return calibrated

def team_stats_regresion(league,target):
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.compose import TransformedTargetRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, WhiteKernel

    if(target == 'Pts'):
        variables = ['GF','GA','GD']
    else:
        variables = ['possession','Pace','long_success%','short_success%','long_bias','Sh','shots_target%','shots_blocked%',
                     'KP','TO%','Drb','dribble_win%', 'FlsW', 'FlsC', 'Tkl','tackle_success%', 'Int','Blk','Clr', 'Off',
                     'Head', 'aerial_win%', 'YC', 'RC', 'ShA','Save%']

    analysis = pd.read_excel(f'{path}/{league}_teams.xlsx','Sheet2')
    analysis['ShA'] *= analysis['P']
    #X = analysis[variables]
    #y = analysis[target]
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train = analysis[variables]
    y_train = analysis[target]

    # alpha chosen by season-grouped CV, not random folds
    kernel = RBF(length_scale=1.0) + WhiteKernel(noise_level=1.0)
    regressor = make_pipeline(StandardScaler(),GaussianProcessRegressor(kernel=kernel, normalize_y=True, random_state=42))
    reg_model = TransformedTargetRegressor(regressor=regressor,func=np.sqrt, inverse_func=np.square) # GF/GA/Pts only — NOT GD
    reg_model.fit(X_train, y_train)

    predictions = reg_model.predict(X_train)
    mse = mean_squared_error(y_train, predictions)
    train_r2 = r2_score(y_train, reg_model.predict(X_train))
    
    #avg goals in season
    analysis['weight'] = np.exp(analysis['season']-analysis['season'].max())*analysis['P']
    target_mean = (analysis[target]*analysis['weight']).sum()/analysis['weight'].sum()
    
    #print(target,"rmse is",mse**0.5)
    #print(target,"train R^2 is",train_r2)
    #print()
    return reg_model,target_mean #!!! should be league average value

def mins_adjustment(df,game_level):
    variables = ['possession','Pace','long_success%','short_success%','long_bias','Sh','shots_target%','shots_blocked%',
                 'KP','TO%','Drb','dribble_win%', 'FlsW', 'FlsC', 'Tkl','tackle_success%', 'Int','Blk','Clr', 'Off',
                 'Head', 'aerial_win%', 'YC', 'RC', 'ShA','Save%']
    
    if(game_level == 1):
        starter_keeper = df[(df['positionText']=='Goalkeeper')&(df['start']==1)]
        sub_keeper = df[(df['positionText']=='Goalkeeper')&(df['sub']==1)]
        starters = df[(df['positionText']!='Goalkeeper')&(df['start']==1)]
        subs = df[(df['positionText']!='Goalkeeper')&(df['sub']==1)]
        
        starter_keeper_mins = starter_keeper['MPG'].sum()
        if(starter_keeper_mins>90): 
            starter_keeper_mins = 90
            starter_keeper['MPG'] *= 90/starter_keeper['MPG'].sum()
            
        sub_keeper['MPG'] *= (90-starter_keeper_mins)/sub_keeper['MPG'].sum()
        subs['MPG'] = (100-sub_keeper['MPG'].sum())*subs['play']/subs['play'].sum()
    
        while(starters['MPG'].sum()>(900-starter_keeper_mins+1) or starters['MPG'].sum()<(900-starter_keeper_mins-1)):
            starters['MPG'] *= (900-starter_keeper_mins)/starters['MPG'].sum()
            starters['MPG'] = starters['MPG'].clip(upper=89)
            
        starters['MPG'] *= (900-starter_keeper_mins)/starters['MPG'].sum()
        team_df = pd.concat([starter_keeper,sub_keeper,starters,subs])
        team_df = team_df.sort_values(by=['MPG'], ascending=[False])
    else:
        keepers = df[df['positionText']=='Goalkeeper']
        others = df[df['positionText']!='Goalkeeper']
        keepers['MPG'] *= keepers['play']
        keepers['MPG'] *= 90/keepers['MPG'].sum()
        others['MPG'] *= others['play']
        others['MPG'] *= 900/others['MPG'].sum()
        team_df = pd.concat([keepers,others])
        team_df = team_df.sort_values(by=['MPG'], ascending=[False])
    
    #percentage based stats need to be recalculated
    team_df['long_att%'] = team_df['long_bias']/(team_df['long_bias']+1)
    team_df['aerial_win'] = team_df['aerial_win%'] * team_df['Head']
    team_df['dribble_win'] = team_df['dribble_win%'] * team_df['Drb']
    team_df['long_success'] = team_df['long_success%'] * team_df['Pss'] * team_df['long_att%']
    team_df['short_success'] = team_df['short_success%'] * team_df['Pss'] * (1-team_df['long_att%'])
    team_df['long_att'] = team_df['long_att%'] * team_df['Pss']
    team_df['short_att'] = (1-team_df['long_att%']) * team_df['Pss']
    team_df['shots_target'] = team_df['shots_target%'] * team_df['Sh']
    team_df['shots_blocked'] = team_df['shots_blocked%'] * team_df['Sh']
    team_df['tackle_success'] = team_df['tackle_success%'] * team_df['Tkl']
    
    totals = team_df.select_dtypes(include='number').mul(team_df['MPG']/90, axis=0).sum()
    totals['aerial_win%'] = totals['aerial_win']/totals['Head']
    totals['dribble_win%'] = totals['dribble_win']/totals['Drb']
    totals['long_success%'] = totals['long_success']/totals['long_att']
    totals['short_success%'] = totals['short_success']/totals['short_att']
    totals['long_bias'] = totals['long_att']/totals['short_att']
    totals['shots_target%'] = totals['shots_target']/totals['Sh']
    totals['shots_blocked%'] = totals['shots_blocked']/totals['Sh']
    totals['tackle_success%'] = totals['tackle_success']/totals['Tkl']
    
    totals['Pace'] = totals['Pss'] + totals['PssA']
    totals['Sh'] = totals['Sh']/totals['Pss']
    totals['ShA'] = totals['ShA']/totals['PssA']
    totals['Drb'] = totals['Drb']/totals['Pss']
    totals['FlsW'] = totals['FlsW']/totals['Pss']
    totals['FlsC'] = totals['Fls']/totals['PssA']
    totals['Tkl'] = totals['Tkl']/totals['PssA']
    totals['Int'] = totals['Int']/totals['PssA']
    totals['Blk'] = totals['Blk']/totals['PssA']
    totals['Clr'] = totals['Clr']/totals['Pss']
    totals['TO%'] = totals['TO%']/totals['Pss']
    totals['Off'] = totals['Off']/totals['PssA']
    totals['YC'] = totals['YC']/totals['PssA']
    totals['RC'] = totals['RC']/totals['PssA']
    totals['KP'] = totals['KP']/totals['Pss']
    totals['Head'] = totals['Head']/totals['PssA']
    totals['possession'] = totals['Pss']/(totals['Pss']+totals['PssA'])
    totals = totals.loc[totals.index.isin(variables)]
    totals = totals.loc[variables]
    return team_df,totals

def league_table(target_league,df):
    totals_df = []; team_list = []
    for t in df['team'].drop_duplicates():
        dft,totals_t = mins_adjustment(df[df['team']==t],0)
        totals_t = totals_t.to_frame().T
        totals_df.append(totals_t)
        team_list.append(t)
        
    totals_df = pd.concat(totals_df)
    #model calibration for league
    model_gf,lg_avg_gf = team_stats_regresion(target_league,'GF')
    model_ga,lg_avg_ga = team_stats_regresion(target_league,'GA')
    model_pts,lg_avg_pts = team_stats_regresion(target_league,'Pts')
    #predictions
    gf = model_gf.predict(totals_df)
    ga = model_ga.predict(totals_df)
    gf_factor = lg_avg_gf/gf.mean()
    ga_factor = lg_avg_ga/ga.mean()
    totals_df['GF'] = gf * gf_factor
    totals_df['GA'] = ga * ga_factor
    totals_df['GD'] = totals_df['GF'] - totals_df['GA']
    pts = model_pts.predict(totals_df[['GF','GA','GD']])
    pts_factor = lg_avg_pts/pts.mean()
    totals_df['Pts'] = pts * pts_factor
    totals_df['teams'] = team_list
    totals_df = totals_df[['teams','GF','GA','GD','Pts']]
    return totals_df,gf_factor,ga_factor,pts_factor

def game_engine(target_league,home,away,df): 
    standings,gf_factor,ga_factor,pts_factor = league_table(project_league,df)
    
    df_home = df[df['team']==home]
    df_away = df[df['team']==away]
    #mins adjustments - starters 900, subs 90 (verify this)
    df_home,home_totals = mins_adjustment(df_home,1)
    df_away,away_totals = mins_adjustment(df_away,1)
    home_totals = home_totals.to_frame().T
    away_totals = away_totals.to_frame().T
    #model calibration for league
    model_gf,lg_avg_gf = team_stats_regresion(target_league,'GF')
    model_ga,lg_avg_ga = team_stats_regresion(target_league,'GA')
    #game level predictions
    home_gf = model_gf.predict(home_totals) * gf_factor
    home_ga = model_ga.predict(home_totals) * ga_factor
    away_gf = model_gf.predict(away_totals) * gf_factor
    away_ga = model_ga.predict(away_totals) * ga_factor
    #print(home_gf,home_ga,away_gf,away_ga,gf_factor,ga_factor)
    game_gf = home_gf * away_ga / lg_avg_ga
    game_ga = home_ga * away_gf / lg_avg_ga
    game_gf = game_gf.sum()
    game_ga = game_ga.sum()

    print(home,round(game_gf,2),"-",round(game_ga,2),away)
    return standings

#%% calls
#full_player_stats,team_stats = player_stats_extract('Italy')
#stabilization_rate(full_player_stats)

player_stats, team_clusters = player_stats_extract_all()

player_bio = player_age_correction(player_stats,season,0,1) #offseason rosters, pick latest team

league_strength = league_conversions(player_stats,season)

stat_aging = aging_effects(player_stats)

regressed_stats = regression_player_full(player_stats,league_strength,season)

#%% league forecasts
if(False): #read_files == 1
    player_bio = pd.read_excel(f'{path}/player_bio.xlsx','Sheet1')
    player_bio[['start','sub']] = player_bio[['start','sub']].fillna(0)
    player_bio = player_bio.drop(columns=['Unnamed: 0'])

regressed_stats_adj,tc = projections_target_league(project_league,regressed_stats,player_bio,league_strength,stat_aging,team_clusters,season)

#%% game level results
standings,_,_,_ = league_table(project_league,regressed_stats_adj)
#standings = game_engine(project_league,'Inter','AC Milan',regressed_stats_adj)