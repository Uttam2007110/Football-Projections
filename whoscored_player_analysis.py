# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 15:24:55 2026
game projections using plyer level data from whoscored
@author: Subramanya.Ganti
"""

#%% imports
import numpy as np
import pandas as pd
from itertools import combinations
from sklearn import linear_model
import statsmodels.api as sm
from patsy import dmatrix
from scipy.stats import poisson
from scipy.stats import skellam
pd.options.mode.chained_assignment = None
np.seterr(divide='ignore', invalid='ignore')

path = 'C:/Users/Subramanya.Ganti/Downloads/Sports/football/whoscored'
#path = 'C:/Users/uttam/Desktop/Sports/football/whoscored'

#%% functions
def player_stats_extract(league,season):
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
    
    #apply season filter
    team_stats = team_stats[team_stats['season']<=season]
    full_player_stats = full_player_stats[full_player_stats['season']<=season]
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
    stat_reg_mins = [1,0,500,2000,1000,2000,6000,15000,2000,2000,2000,2000,2000,2000,2000,2000,1000,2000,2000,
                     3000,5000,12000,9000,3000,2000,1000,9000,9000,6000,2000,2000,3000]
    
    data = full_player_stats.copy()
    data['play'] = data['apps']/data['P']
    data['s_weight'] = 5*np.exp((data['season']-season)/2)
    data['s_weight2'] = np.exp(data['season']-season)
    
    #apply league strength conversions on data
    adjusted_stats = [stat for stat in stat_list if stat not in ['age']] #,'play','MPG'
    league_strength_by_row = league_strength.loc[data["label"],adjusted_stats].to_numpy()
    target_league_strength = league_strength.loc[target_league,adjusted_stats].to_numpy()
    #always e^(from - to)
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
    
    df_full['play'] = df_full['apps']/df_full['P']
    df = []
    leagues = df_full['label'].drop_duplicates().to_list()
    for l in leagues:
        df.append(df_full[df_full['label']==l])
    
    combos = list(combinations(range(0,len(df)), 2))
    all_eqn = [['category'] + leagues]
    
    categories = ['ShA','Sh','Blk','YC','RC','Clr','Drb','FlsW','Fls','Int','Pss','KP','KP/P','TO%','Off','Tkl','Head','G%','A%','Save%',
                 'dribble_win%','long_success%','short_success%','long_bias','shots_target%','shots_blocked%',
                 'tackle_success%','aerial_win%','PssA','MPG','play']
    
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
            
        eqn[list(range(0,len(df)))] = eqn[list(range(0,len(df)))].fillna(0.0).infer_objects(copy=False)
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
        print(ch,"done")
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

def player_transfers(player_stats,season):
    multi_team = player_stats[player_stats['season']==season]
    multi_team = multi_team[multi_team.duplicated(subset=['playerId'], keep=False)]
    multi_team = multi_team[['playerId','name','team','minsPlayed']]
    return multi_team

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
    full_transfer_list = player_transfers(_player_stats,season)
    full_transfer_list = full_transfer_list.sort_values(by=['minsPlayed'], ascending=[False])
    transfer_list = full_transfer_list.drop_duplicates(subset=['playerId'])
    bio = bio.merge(transfer_list[['playerId','team']], on='playerId', how='left')
    bio['team'] = np.where(~bio['team_y'].isna(),bio['team_y'],bio['team_x'])
    bio = bio[['playerId','name','age','team','start','sub']]
    
    full_transfer_list = full_transfer_list.groupby(["playerId", "name"], sort=False)["team"].agg(lambda s: s.dropna().drop_duplicates().tolist())
    full_transfer_list = pd.DataFrame(full_transfer_list.tolist(), index=full_transfer_list.index)
    full_transfer_list.columns = [f"team{i + 1}" for i in range(full_transfer_list.shape[1])]
    full_transfer_list = full_transfer_list.reset_index()
    return bio,full_transfer_list

def aging_effects(player_stats):
    categories = ['ShA','Sh','Blk','YC','RC','Clr','Drb','FlsW','Fls','Int','Pss','KP','KP/P','TO%','Off','Tkl','Head','G%','A%','Save%',
                 'dribble_win%','long_success%','short_success%','long_bias','shots_target%','shots_blocked%',
                 'tackle_success%','aerial_win%','PssA','MPG','play']
    
    min_minutes, reference_age, spline_df = 500, 27, 5
    data = player_stats.copy()
    data['play'] = data['apps']/data['P']
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
    print("Aging done")
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

def player_stats_extract_all(season):
    df = []; df_teams = []
    leagues = ['England','Spain','Germany','Italy','France','Netherlands','Turkey','Portugal','Belgium','Scotland','Russia',
               'Germany2','England2','Brazil','USA','England3','England4']
    for l in leagues:
        pl,tl = player_stats_extract(l,season)
        pl['league'] = l
        tl['league'] = l
        df.append(pl)
        df_teams.append(tl)
     
    print("Player stats extracted")
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
    print("Team clusters found")
    return df_players,team_labels

def regression_player_full(full_player_stats,league_strength,season):
    from sklearn.mixture import GaussianMixture
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    """
    gk = full_player_stats[full_player_stats['positionText']=='Goalkeeper']
    df = full_player_stats[full_player_stats['positionText']=='Defender']
    md = full_player_stats[full_player_stats['positionText']=='Midfielder']
    fw = full_player_stats[full_player_stats['positionText']=='Forward']
    gk_reg = regression_player(gk,league_strength,season)
    df_reg = regression_player(df,league_strength,season)
    md_reg = regression_player(md,league_strength,season)
    fw_reg = regression_player(fw,league_strength,season)    
    reg_stats = pd.concat([gk_reg,df_reg,md_reg,fw_reg])
    """
    categories = ['ShA','Sh','Blk','YC','RC','Clr','Drb','FlsW','Fls','Int','Pss','KP','KP/P','TO%','Off','Tkl','Head',
                  'G%','A%','Save%','dribble_win%','long_success%','short_success%','long_bias','shots_target%','shots_blocked%',
                  'tackle_success%','aerial_win%','PssA']

    #usually 8, sometimes 7
    def best_gmm(data, min_k=4, max_k=11, max_rows=4000, patience=2, random_state=0):
        """Choose the cluster count by BIC (linear in n) on a subsample, stopping once BIC keeps worsening."""
        rng = np.random.default_rng(random_state)
        sample = data[rng.choice(len(data), max_rows, replace=False)] if len(data) > max_rows else data
        best_model, best_bic, worse = None, np.inf, 0
        for k in range(min_k, min(max_k, len(sample) - 1) + 1):
            model = GaussianMixture(n_components=k, covariance_type='full', reg_covar=1e-4, random_state=random_state).fit(sample)
            bic = model.bic(sample)
            if(bic < best_bic): 
                best_model, best_bic, worse = model, bic, 0
            else:
                worse += 1
                if(worse >= patience): break
        return best_model

    # goalkeepers are kept out of the outfield clustering entirely and always form cluster 0
    gk_mask = (full_player_stats['positionText']=='Goalkeeper').to_numpy()
    train_mask = (~gk_mask) & (full_player_stats['minsPlayed']>500).to_numpy()
    outfield_categories = [c for c in categories if c!='Save%']

    imputer = SimpleImputer(strategy='median')
    scaler = StandardScaler()
    pca = PCA(n_components=10, random_state=0)
    train_x = pca.fit_transform(scaler.fit_transform(imputer.fit_transform(full_player_stats.loc[train_mask, outfield_categories])))
    gmm = best_gmm(train_x)
    print(f"regression_player_full: selected {gmm.n_components} outfield clusters via BIC")

    outfield_x = pca.transform(scaler.transform(imputer.transform(full_player_stats.loc[~gk_mask, outfield_categories])))
    full_player_stats['cluster'] = 0
    full_player_stats.loc[~gk_mask,'cluster'] = gmm.predict(outfield_x) + 1

    # each player gets the cluster where they logged the most minutes across seasons
    player_minutes = full_player_stats[~gk_mask].assign(minsPlayed=lambda x: x['minsPlayed'].fillna(0)).groupby(['playerId','cluster'], sort=False)['minsPlayed'].sum().reset_index()
    player_labels = player_minutes.loc[player_minutes.groupby('playerId')['minsPlayed'].idxmax(), ['playerId','cluster']]
    full_player_stats = full_player_stats.merge(player_labels, on='playerId', how='left')
    full_player_stats['cluster_y'] = np.where(gk_mask, 0, full_player_stats['cluster_y'])
    
    reg_stats = [];
    for c in full_player_stats['cluster_y'].drop_duplicates():
        cn = full_player_stats[full_player_stats['cluster_y']==c]
        cluster = regression_player(cn,league_strength,season)
        cluster['cluster'] = c
        reg_stats.append(cluster)
    
    reg_stats = pd.concat(reg_stats)
    
    return reg_stats

def relevant_team_label(target_league,team_clusters,season):
    tc = team_clusters.copy()
    if(target_league == 'Germany2'):
        tc['alternate'] = np.where(tc['label'].str.contains('Germany_'),'Germany2_0',pd.NA)
    elif(target_league == 'England2'):
        tc['alternate'] = np.where(tc['label'].str.contains('England_'),'England2_0',pd.NA)
        tc['alternate'] = np.where(tc['label'].str.contains('England3_'),'England2_2',tc['alternate'])
        tc['alternate'] = np.where(tc['label'].str.contains('England4_'),'England2_2',tc['alternate'])
    elif(target_league == 'England3'):
        tc['alternate'] = np.where(tc['label'].str.contains('England_'),'England3_0',pd.NA)
        tc['alternate'] = np.where(tc['label'].str.contains('England2_'),'England3_0',tc['alternate'])
        tc['alternate'] = np.where(tc['label'].str.contains('England4_'),'England3_2',tc['alternate'])
    elif(target_league == 'England4'):
        tc['alternate'] = np.where(tc['label'].str.contains('England_'),'England4_0',pd.NA)
        tc['alternate'] = np.where(tc['label'].str.contains('England2_'),'England4_0',tc['alternate'])
        tc['alternate'] = np.where(tc['label'].str.contains('England3_'),'England4_0',tc['alternate'])
    elif(target_league == 'Germany'):
        tc['alternate'] = np.where(tc['label'].str.contains('Germany2_'),'Germany_2',pd.NA)
    elif(target_league == 'England'):
        tc['alternate'] = np.where(tc['label'].str.contains('England2_'),'England_2',pd.NA)
        tc['alternate'] = np.where(tc['label'].str.contains('England3_'),'England_2',tc['alternate'])
        tc['alternate'] = np.where(tc['label'].str.contains('England4_'),'England_2',tc['alternate'])
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
    available_stats = ['ShA','Sh','Blk','YC','RC','Clr','Drb','FlsW','Fls','Int','Pss','KP','KP/P','TO%','Off','Tkl','Head','G%','A%',
                       'Save%','dribble_win%','long_success%','short_success%','long_bias','shots_target%','shots_blocked%',
                       'tackle_success%','aerial_win%','PssA','MPG','play']
    
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
    #points deductions not in the scope of model training
    analysis['Pts'] = 3*analysis['W'] + analysis['D']
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
    return reg_model,target_mean

def player_target_credits(players, training_teams, model, target):
    """Evaluate players and replacements pinned to 90 minutes.

    Absolute credits retain original minutes and reconstruct team predictions;
    position-centred ratings use impacts per actual post-rescaling minute.
    """
    VARIABLES = ['possession','Pace','long_success%','short_success%','long_bias','Sh','shots_target%','shots_blocked%',
                 'KP','TO%','Drb','dribble_win%', 'FlsW', 'FlsC', 'Tkl','tackle_success%', 'Int','Blk','Clr', 'Off',
                 'Head', 'aerial_win%', 'YC', 'RC', 'ShA','Save%']

    if(players.empty):
        raise ValueError("Player dataframe is empty")
    if(players[['playerId','team','positionText']].isna().any().any()):
        raise ValueError("Every player row requires playerId, team and positionText")

    reference_stats = training_teams[VARIABLES].mean()
    if(not np.isfinite(reference_stats.to_numpy(dtype=float)).all()):
        raise ValueError("Training reference contains NaN or infinity")

    def features_from_totals(totals):
        features = totals_to_features(pd.DataFrame(totals, columns=_TEAM_COUNT_COLUMNS))[VARIABLES].astype(float)
        if(not np.isfinite(features.to_numpy()).all()):
            invalid = features.columns[~np.isfinite(features.to_numpy()).all(axis=0)].tolist()
            raise ValueError(f"{team}: invalid aggregated inputs: {invalid}")
        return features

    def pinned_minutes(minutes, keeper, evaluated_mask):
        # Pin the evaluated player to 90 and rescale teammates to the remaining 90/900 budgets.
        pinned = minutes.copy()
        weights = pinned[evaluated_mask]
        pinned[evaluated_mask] = 90 * weights/weights.sum()
        for keeper_group, total in [(True, 90), (False, 900)]:
            group = keeper == keeper_group
            remaining = group & ~evaluated_mask
            budget = total - pinned[group & evaluated_mask].sum()
            remaining_minutes = pinned[remaining].sum()
            if(budget < -1e-10 or (budget > 1e-10 and remaining_minutes <= 0)):
                raise ValueError(f"{team}: insufficient teammates to pin evaluated player to 90 minutes")
            if(remaining_minutes > 0):
                pinned[remaining] *= max(budget, 0)/remaining_minutes
        if(not np.isclose(pinned[evaluated_mask].sum(), 90, rtol=1e-9, atol=1e-10)):
            raise ValueError(f"{team}: evaluated player minutes were not pinned to 90")
        return pinned

    records, team_records = [], []

    for team, raw in players.groupby('team', sort=False):
        raw = raw.copy()
        if(not np.isfinite(raw[['MPG','play']].to_numpy(dtype=float)).all()):
            raise ValueError(f"{team}: MPG or play contains NaN or infinity")
        if((raw[['MPG','play']] < 0).any().any()):
            raise ValueError(f"{team}: MPG and play must be non-negative")
        raw = raw.loc[(raw['MPG'] * raw['play']) > 0].copy()
        keepers = raw['positionText'].eq('Goalkeeper')
        if(raw.empty or not keepers.any() or keepers.all()):
            raise ValueError(f"{team}: positive goalkeeper and outfield minutes are required")

        actual, _ = mins_adjustment(raw.copy(), 0)
        actual = actual.reset_index(drop=True)
        minutes = actual['MPG'].to_numpy(dtype=float)
        if(not np.isfinite(minutes).all() or (minutes <= 0).any()):
            raise ValueError(f"{team}: invalid adjusted player minutes")
        total_minutes = minutes.sum()
        reference = actual.copy()

        # Construct per-90 reference counts that aggregate to the mean team inputs.
        own_passes = reference_stats['Pace'] * reference_stats['possession']
        against_passes = reference_stats['Pace'] * (1-reference_stats['possession'])
        factor = 90/total_minutes
        reference['Pss'] = own_passes * factor
        reference['PssA'] = against_passes * factor

        own_rates = ['Sh','Drb','FlsW','Clr','TO%','KP']
        against_rates = ['ShA','Tkl','Int','Blk','Off','YC','RC','Head']
        for stat in own_rates:
            reference[stat] = reference_stats[stat] * own_passes * factor
        for stat in against_rates:
            reference[stat] = reference_stats[stat] * against_passes * factor
        reference['Fls'] = reference_stats['FlsC'] * against_passes * factor

        percentages = ['long_success%','short_success%','long_bias','shots_target%','shots_blocked%',
                       'dribble_win%','tackle_success%','aerial_win%']
        for stat in percentages:
            reference[stat] = reference_stats[stat]

        keepers = reference['positionText'].eq('Goalkeeper')
        keeper_minutes = reference.loc[keepers, 'MPG'].sum()
        if(keeper_minutes <= 0):
            raise ValueError(f"No goalkeeper minutes for {team}")
        reference['Save%'] = 0.0
        reference.loc[keepers, 'Save%'] = reference_stats['Save%'] * 90/keeper_minutes

        # Per-90 counts for actual (A) and reference (R) rows; minutes are applied by matrix product.
        A = add_derived_counts(actual.copy())[_TEAM_COUNT_COLUMNS].to_numpy(dtype=float)
        R = add_derived_counts(reference.copy())[_TEAM_COUNT_COLUMNS].to_numpy(dtype=float)

        reference_inputs = features_from_totals((minutes @ R)[None, :]/90)
        if(not np.allclose(reference_inputs.iloc[0], reference_stats, rtol=1e-6, atol=1e-10)):
            raise ValueError(f"Reference aggregation does not reproduce training means for {team}")

        player_ids = actual['playerId'].drop_duplicates().to_numpy()
        masks = np.vstack([actual['playerId'].eq(pid).to_numpy() for pid in player_ids])
        player_minutes = (masks * minutes).sum(axis=1)
        minute_share = player_minutes/player_minutes.sum()

        # Pinned minutes are identical across the 4 contexts of a player (same MPG/positionText rows).
        keeper_rows = actual['positionText'].eq('Goalkeeper').to_numpy()
        pinned = np.vstack([pinned_minutes(minutes, keeper_rows, mask) for mask in masks])
        swap = (pinned * masks) @ (A - R)
        reference_totals, actual_totals = pinned @ R, pinned @ A
        # Per player: reference_with_player, reference, actual, actual_without_player.
        contexts = np.stack([reference_totals + swap, reference_totals,
                             actual_totals, actual_totals - swap], axis=1).reshape(-1, len(_TEAM_COUNT_COLUMNS))
        all_totals = np.vstack([minutes @ R, minutes @ A, contexts])/90
        X_batch = features_from_totals(all_totals)
        evaluation_minutes = np.full((len(player_ids), 4), 90.0)
        
        predictions = np.asarray(model.predict(X_batch), dtype=float)
        if(predictions.ndim != 1 or len(predictions) != 2 + 4*len(player_ids)):
            raise ValueError(f"{team}: expected one prediction per hypothetical team, got {predictions.shape}")
        if(not np.isfinite(predictions).all()):
            raise ValueError(f"{team}: model predictions contain NaN or infinity")
        
        reference_value, team_prediction = predictions[:2]
        player_predictions = predictions[2:].reshape(len(player_ids), 4)
        
        # Express 90-minute evaluation impacts at each player's original minutes.
        standalone = (player_predictions[:, 0] - player_predictions[:, 1]) * (
            player_minutes/evaluation_minutes[:, 0])
        
        # Player's effect in their actual team.
        actual_context = (player_predictions[:, 2] - player_predictions[:, 3]) * (
            player_minutes/evaluation_minutes[:, 2])
        
        # Approximate Shapley using the two endpoint contexts.
        approximate_marginal = 0.5 * (standalone + actual_context)
        
        # Reconcile the approximation so credits sum exactly to the team prediction.
        baseline_credit = reference_value * minute_share
        unadjusted_credit = baseline_credit + approximate_marginal
        allocation_residual = team_prediction - unadjusted_credit.sum()
        residual_credit = minute_share * allocation_residual
        allocated_credit = unadjusted_credit + residual_credit

        # Remove floating-point summation residue only.
        anchor = np.argmax(player_minutes)
        roundoff = team_prediction - allocated_credit.sum()
        allocated_credit[anchor] += roundoff
        residual_credit[anchor] += roundoff

        marginal = allocated_credit - baseline_credit
        standalone_credit = baseline_credit + standalone
        reconstructed = allocated_credit.sum()
        if(not np.isclose(reconstructed, team_prediction, rtol=1e-9, atol=1e-10)):
            raise ValueError(f"{team}: reconstructed={reconstructed}, predicted={team_prediction}")

        for i, pid in enumerate(player_ids):
            position_minutes = actual.loc[masks[i]].groupby('positionText', sort=False)['MPG'].sum()
            for position, mins in position_minutes.items():
                share = mins/player_minutes[i]
                records.append({'playerId': pid, 'team': team, 'positionText': position, 'minutes': mins,
                                'actual_credit': allocated_credit[i]*share, 'above_reference': marginal[i]*share,
                                'residual_credit': residual_credit[i]*share,
                                'standalone_credit': standalone_credit[i]*share,
                                'standalone_impact': standalone[i]*share})

        team_records.append({'team': team, f'Pred_{target}': team_prediction,
                             f'Reconstructed_{target}': reconstructed,
                             'allocation_residual': allocation_residual})

    detail = pd.DataFrame(records)
    if(detail.empty):
        raise ValueError("No players to evaluate")

    position_groups = detail.groupby('positionText', sort=False)
    position_minutes = position_groups['minutes'].transform('sum')
    for credit, prefix in [('actual_credit', target), ('standalone_credit', f'{target}_standalone')]:
        position_mean = position_groups[credit].transform('sum')/position_minutes
        detail[f'{prefix}_position_mean'] = position_mean
        detail[f'{prefix}_position_adjusted_per_min'] = detail[credit]/detail['minutes'] - position_mean
        detail[f'{prefix}_position_adjusted_credit'] = detail[credit] - detail['minutes']*position_mean

    # One output row per unique playerId, including players appearing for multiple teams.
    result = detail.groupby('playerId', sort=False)[['minutes','actual_credit','above_reference',
                                                    'residual_credit','standalone_credit','standalone_impact']].sum()
    result[f'{target}_per_min'] = result['actual_credit']/result['minutes']
    result[f'{target}_above_ref_per_min'] = result['above_reference']/result['minutes']
    result[f'{target}_standalone_per_min'] = result['standalone_credit']/result['minutes']
    result[f'{target}_standalone_impact_per_min'] = result['standalone_impact']/result['minutes']
    for prefix in [target, f'{target}_standalone']:
        adjusted = detail.groupby('playerId', sort=False)[f'{prefix}_position_adjusted_credit'].sum()
        result[f'{prefix}_position_adjusted_per_min'] = adjusted/result['minutes']
    return result, detail, pd.DataFrame(team_records)

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
        #new method
        key = 'positionText' #positionText or cluster
        df['raw_MPG'] = df['MPG']*df['play']
        df_pivot = df.pivot_table(index=key,values='raw_MPG',aggfunc='sum')
        df_pivot = df_pivot.reset_index()
        if(key=='cluster'): df_pivot[df_pivot[key]!=0]
        else: df_pivot = df_pivot[df_pivot[key]!='Goalkeeper']
        df_pivot['MPG'] = 900*df_pivot['raw_MPG']/df_pivot['raw_MPG'].sum()
        
        df_pivot['MPG'] = df_pivot['MPG'].clip(lower=90)
        df_pivot['MPG'] = 900*df_pivot['MPG']/df_pivot['MPG'].sum()
        #play probability changes based on injury status
        try:
            df['play'] *= df['status']
            df = df[df['play']>0]
        except KeyError:
            #for finding player avg values status wont be in the df, assume 1
            df
        #keepers, old logic is ok
        keepers = df[df['positionText']=='Goalkeeper']
        keepers['play_rank'] = keepers['play'].rank(ascending=False, method='dense')
        keepers['MPG'] *= keepers['play'] * pow(1/4,keepers['play_rank']-1) #keeper mins need to be decayed more aggressively
        keepers['MPG'] *= 90/keepers['MPG'].sum()
        #outfielders mins logic changed
        team_df = keepers.copy()
        for p in df_pivot[key].drop_duplicates():
            others = df[df[key]==p] #df[df['positionText']!='Goalkeeper']
            p_target_mins = df_pivot.loc[df_pivot[key]==p,'MPG'].sum()
            others['play_rank'] = (others['play']*others['MPG']).rank(ascending=False, method='dense')
            others['MPG'] *= others['play'] * pow(.975,others['play_rank']-1) #outfielders mins need to be decayed less aggressively
            #print(p,p_target_mins,others)
            if(p_target_mins>len(others)*85): p_target_mins = len(others)*85
            exp = 1
            while((others['MPG'].sum()<=p_target_mins-1 or others['MPG'].sum()>=p_target_mins+1) and (len(others)>0)):
                others['MPG'] *= np.pow(exp,others['play_rank']) #p_target_mins/others['MPG'].sum()
                others['MPG'] = others['MPG'].clip(upper=85) #!!!review this
                if(others['MPG'].sum()>p_target_mins+1): exp-= 0.001
                if(others['MPG'].sum()<p_target_mins-1): exp+= 0.001
            #final adjustment to make it 900
            others['MPG'] *= p_target_mins/others['MPG'].sum()
            team_df = pd.concat([team_df,others]) #pd.concat([keepers,others])
        
        team_df = team_df.sort_values(by=['MPG'], ascending=[False])
    
    #percentage based stats need to be recalculated
    team_df = add_derived_counts(team_df)
    totals = team_df[_TEAM_COUNT_COLUMNS].mul(team_df['MPG']/90, axis=0).sum()
    totals = totals_to_features(totals.to_frame().T).iloc[0].rename(None)
    totals = totals.loc[variables]
    return team_df,totals

# Per-90 player counts that are summed (MPG/90 weighted) to form team totals.
_TEAM_COUNT_COLUMNS = ['Pss','PssA','Sh','ShA','Drb','FlsW','Fls','Tkl','Int','Blk','Clr','TO%','Off','YC','RC',
                      'KP','Head','Save%','aerial_win','dribble_win','long_success','short_success','long_att',
                      'short_att','shots_target','shots_blocked','tackle_success']

def add_derived_counts(team_df):
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
    return team_df

def totals_to_features(totals):
    """Convert summed team counts (one row per team/context) into model inputs."""
    t = totals
    out = pd.DataFrame(index=t.index)
    out['possession'] = t['Pss']/(t['Pss']+t['PssA'])
    out['Pace'] = t['Pss'] + t['PssA']
    out['long_success%'] = t['long_success']/t['long_att']
    out['short_success%'] = t['short_success']/t['short_att']
    out['long_bias'] = t['long_att']/t['short_att']
    out['Sh'] = t['Sh']/t['Pss']
    out['shots_target%'] = t['shots_target']/t['Sh']
    out['shots_blocked%'] = t['shots_blocked']/t['Sh']
    out['KP'] = t['KP']/t['Pss']
    out['TO%'] = t['TO%']/t['Pss']
    out['Drb'] = t['Drb']/t['Pss']
    out['dribble_win%'] = t['dribble_win']/t['Drb']
    out['FlsW'] = t['FlsW']/t['Pss']
    out['FlsC'] = t['Fls']/t['PssA']
    out['Tkl'] = t['Tkl']/t['PssA']
    out['tackle_success%'] = t['tackle_success']/t['Tkl']
    out['Int'] = t['Int']/t['PssA']
    out['Blk'] = t['Blk']/t['PssA']
    out['Clr'] = t['Clr']/t['Pss']
    out['Off'] = t['Off']/t['PssA']
    out['Head'] = t['Head']/t['PssA']
    out['aerial_win%'] = t['aerial_win']/t['Head']
    out['YC'] = t['YC']/t['PssA']
    out['RC'] = t['RC']/t['PssA']
    out['ShA'] = t['ShA']/t['PssA']
    out['Save%'] = t['Save%']
    return out

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
    totals_df = totals_df.sort_values(by=['Pts'], ascending=[False])
    totals_df = totals_df.round({'GF':2, 'GA':2, 'GD':2, 'Pts':2})
    return totals_df,gf_factor,ga_factor,pts_factor

def player_valuation(target_league,league_player_stats):
    #standings,gf_factor,ga_factor,pts_factor = league_table(target_league,league_player_stats)
    #model calibration for league
    model_gf,lg_avg_gf = team_stats_regresion(target_league,'GF')
    model_ga,lg_avg_ga = team_stats_regresion(target_league,'GA')
    #team stats
    training_teams = pd.read_excel(f'{path}/{target_league}_teams.xlsx','Sheet2')
    training_teams['ShA'] *= training_teams['P']
    #player valuation
    gf_players, gf_detail, gf_teams = player_target_credits(league_player_stats, training_teams, model_gf, 'GF')
    ga_players, ga_detail, ga_teams = player_target_credits(league_player_stats, training_teams, model_ga, 'GA')
    
    credits_df = gf_players[['minutes','GF_per_min','GF_standalone_per_min','GF_position_adjusted_per_min','GF_standalone_position_adjusted_per_min']].join(
                 ga_players[['GA_per_min','GA_standalone_per_min','GA_position_adjusted_per_min','GA_standalone_position_adjusted_per_min']])
    credits_df['GD_per_min'] = credits_df['GF_per_min'] - credits_df['GA_per_min']
    credits_df['GD_standalone_per_min'] = credits_df['GF_standalone_per_min'] - credits_df['GA_standalone_per_min']
    credits_df['GD_position_adjusted_per_min'] = credits_df['GF_position_adjusted_per_min'] - credits_df['GA_position_adjusted_per_min']
    credits_df['GD_standalone_position_adjusted_per_min'] = credits_df['GF_standalone_position_adjusted_per_min'] - credits_df['GA_standalone_position_adjusted_per_min']
    credits_df['GF90'] = credits_df['GF_standalone_position_adjusted_per_min']*90
    credits_df['GA90'] = credits_df['GA_standalone_position_adjusted_per_min']*(-90)
    credits_df = credits_df.reset_index()
    credits_df['GD90'] = credits_df['GD_standalone_position_adjusted_per_min'] * 90
    credits_df['GD90_absolute'] = credits_df['GD_per_min'] * 90
    credits_df['GD90_avg_absolute'] = credits_df['GD_standalone_per_min'] * 90
    credits_df = credits_df.merge(league_player_stats[['playerId','name','team','positionText']], on='playerId', how='left')
    credits_df = credits_df[['playerId','name', 'team', 'positionText', 'minutes', 'GF90', 'GA90', 'GD90']]
    credits_df = credits_df.round({'GF90': 3, 'GA90': 3, 'GD90':3, 'minutes':1})
    return credits_df

def average_team_profiles(league_player_stats):
    counts = ['Pss','PssA','Sh','ShA','Drb','FlsW','Fls','Clr','TO%','KP','Tkl','Int','Blk','Off','YC','RC','Head','Save%']
    rate_bases = {'aerial_win%': 'Head', 'dribble_win%': 'Drb', 'shots_target%': 'Sh',
                  'shots_blocked%': 'Sh', 'tackle_success%': 'Tkl'}
    missing = [c for c in counts + list(rate_bases) + ['long_success%','short_success%','long_bias','MPG','play','positionText']
               if c not in league_player_stats.columns]
    if(missing):
        raise ValueError(f"league_player_stats is missing columns: {missing}")

    df = league_player_stats.copy()
    df['_w'] = (df['MPG']*df['play']).clip(lower=0).fillna(0)
    df['_long_att%'] = df['long_bias']/(df['long_bias']+1)

    def weighted(values, weights):
        valid = values.notna() & weights.notna() & (weights > 0)
        return (values[valid]*weights[valid]).sum()/weights[valid].sum() if valid.any() else np.nan

    profiles = {}
    for position, group in df.groupby('positionText', sort=False):
        w = group['_w'] if group['_w'].sum() > 0 else pd.Series(1.0, index=group.index)
        profile = {c: weighted(group[c], w) for c in counts}
        for rate, base in rate_bases.items():
            profile[rate] = weighted(group[rate], w*group[base])
        long_att = weighted(group['_long_att%'], w*group['Pss'])
        profile['long_bias'] = long_att/(1-long_att)
        profile['long_success%'] = weighted(group['long_success%'], w*group['Pss']*group['_long_att%'])
        profile['short_success%'] = weighted(group['short_success%'], w*group['Pss']*(1-group['_long_att%']))
        profiles[position] = profile
    profiles = pd.DataFrame(profiles).T
    profiles.index.name = 'positionText'
    profiles['Save%'] = np.where(profiles.index == 'Goalkeeper', profiles['Save%'], 0)
    return profiles

def player_valuation_average_team(target_league, league_player_stats, formation=None):
    """Return equal-90-minute, position-centred contributions and raw team predictions.

    Contributions are centred on the returned positionText after combining player
    rows, giving each player equal weight. Season minutes remain output metadata;
    contributions remain unrounded to preserve zero positional means.
    """
    if(formation is None):
        formation = {'Goalkeeper': 1, 'Defender': 4, 'Midfielder': 3, 'Forward': 3}
    if(formation.get('Goalkeeper', 0) != 1 or sum(formation.values()) != 11):
        raise ValueError("formation must contain exactly 1 Goalkeeper and 11 players in total")

    model_gf,lg_avg_gf = team_stats_regresion(target_league,'GF')
    model_ga,lg_avg_ga = team_stats_regresion(target_league,'GA')

    profiles = average_team_profiles(league_player_stats)
    absent = [p for p in formation if p not in profiles.index]
    if(absent):
        raise ValueError(f"No players available to build the average {absent}")
    stat_columns = profiles.columns.tolist()

    base_team = pd.concat([profiles.loc[[p]*n] for p, n in formation.items() if n > 0]).reset_index()
    base_team['MPG'] = 90.0
    base_team['play'] = 1.0

    players = league_player_stats[['playerId','name','team','positionText','MPG','play'] + stat_columns].copy()
    unknown = players.loc[~players['positionText'].isin([p for p, n in formation.items() if n > 0]), 'positionText'].unique()
    if(len(unknown)):
        raise ValueError(f"positionText values not present in the formation: {list(unknown)}")
    players['Save%'] = np.where(players['positionText'] == 'Goalkeeper', players['Save%'], 0)

    # Every lineup player (evaluated or average) plays 90 minutes, so team totals are plain sums
    # of per-90 counts; no minutes allocation (mins_adjustment) is needed.
    base_counts = add_derived_counts(base_team.copy())[_TEAM_COUNT_COLUMNS].astype(float)
    player_counts = add_derived_counts(players[stat_columns].astype(float))[_TEAM_COUNT_COLUMNS].to_numpy()
    position_counts = add_derived_counts(profiles[stat_columns].astype(float).copy())[_TEAM_COUNT_COLUMNS]
    base_totals = base_counts.sum().to_numpy()
    # Swap one average player of the same positionText for the evaluated player.
    swapped_totals = base_totals - position_counts.loc[players['positionText']].to_numpy() + player_counts
    features = totals_to_features(pd.DataFrame(np.vstack([base_totals, swapped_totals]),
                                               columns=_TEAM_COUNT_COLUMNS)).astype(float)
    if(not np.isfinite(features.to_numpy()).all()):
        bad = features.columns[~np.isfinite(features.to_numpy()).all(axis=0)].tolist()
        raise ValueError(f"Non-finite aggregated team inputs: {bad}")

    gf = np.asarray(model_gf.predict(features), dtype=float)
    ga = np.asarray(model_ga.predict(features), dtype=float)
    # scale so that the average team scores and concedes the league average
    gf *= lg_avg_gf/gf[0]
    ga *= lg_avg_ga/ga[0]

    players['GF90'] = gf[1:] - gf[0]
    players['GA90'] = -(ga[1:] - ga[0])
    players['GD90'] = players['GF90'] + players['GA90']
    players['team_GF'] = gf[1:]
    players['team_GA'] = ga[1:]

    # players listed under several rows (e.g. positions) are combined by expected minutes
    players['_w'] = (players['MPG']*players['play']).clip(lower=0).fillna(0)
    players['_w'] = np.where(players.groupby('playerId')['_w'].transform('sum') > 0, players['_w'], 1)
    for c in ['GF90','GA90','GD90','team_GF','team_GA']:
        players[c] *= players['_w']
    value = players.groupby('playerId', sort=False).agg(
        name=('name','first'), team=('team','first'), positionText=('positionText','first'),
        minutes=('_w','sum'), GF90=('GF90','sum'), GA90=('GA90','sum'), GD90=('GD90','sum'),
        team_GF=('team_GF','sum'), team_GA=('team_GA','sum')).reset_index()
    for c in ['GF90','GA90','GD90','team_GF','team_GA']:
        value[c] /= value['minutes']
    value['minutes'] = players.groupby('playerId', sort=False).apply(lambda x: (x['MPG']*x['play']).sum()).to_numpy()
    for c in ['GF90','GA90']:
        # Equal 90-minute weights reduce to the arithmetic mean within each position.
        position_mean = value.groupby('positionText')[c].transform('mean')
        value[c] -= position_mean
    value['GD90'] = value['GF90'] + value['GA90']
    value = value.sort_values('GD90', ascending=False)
    value = value.round({'team_GF': 3, 'team_GA': 3})
    print(f"average team: GF {lg_avg_gf:.3f}, GA {lg_avg_ga:.3f} per game")
    return value

def game_engine(gw,target_league,df,regstats,custom_lineups):
    player_df = []
    summary = [['Home Goals','Home','Home win%','Draw%','Away win%','Away','Away Goals']]
    #standings
    standings,gf_factor,ga_factor,pts_factor = league_table(target_league,df)
    #extract fixtures
    fixtures = pd.read_excel(f'{path}/schedule.xlsx',target_league)
    fixtures = fixtures[fixtures['GW']==gw]
    home_teams = fixtures['Home'].to_list()
    away_teams = fixtures['Away'].to_list()
    #model calibration for league
    model_gf,lg_avg_gf = team_stats_regresion(target_league,'GF')
    model_ga,lg_avg_ga = team_stats_regresion(target_league,'GA')
    #home advantage, last years numbers
    if(target_league == 'Italy'): home_f=1.07; away_f=.93
    elif(target_league == 'Germany'): home_f=1.084; away_f=.915
    elif(target_league == 'Germany2'): home_f=1.09; away_f=.91
    elif(target_league == 'England'): home_f=1.09; away_f=.91
    elif(target_league == 'Spain'): home_f=1.13; away_f=.87
    elif(target_league == 'France'): home_f=1.13; away_f=.87
    elif(target_league == 'England2'): home_f=1.11; away_f=.89
    elif(target_league == 'Turkey'): home_f=1.15; away_f=.85
    elif(target_league == 'Netherlands'): home_f=1.12; away_f=.88
    elif(target_league == 'Portugal'): home_f=1.08; away_f=.92
    else: home_f=1.1; away_f=.9

    for t in range(0,len(home_teams)):
        df_home = df[df['team']==home_teams[t]]
        df_away = df[df['team']==away_teams[t]]
        #mins adjustments - starters 900, subs 90 (verify this)
        df_home,home_totals = mins_adjustment(df_home,custom_lineups)
        df_away,away_totals = mins_adjustment(df_away,custom_lineups)
        home_totals = home_totals.to_frame().T
        away_totals = away_totals.to_frame().T
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
        #apply home advantage
        game_gf*= home_f
        game_ga*= away_f
        #Dixon-Coles approach for summary stats
        rho = -0.13 #!!!verify
        lam, mu = float(game_gf), float(game_ga)
        tau = np.array([
            1 - lam * mu * rho,   # 0-0
            1 + lam * rho,        # 0-1
            1 + mu * rho,         # 1-0
            1 - rho,              # 1-1
        ])
        if(not np.isfinite(tau).all() or (tau < 0).any()):
            #default back to simple skellam
            t1_cs = poisson.pmf(0, game_ga)
            t2_cs = poisson.pmf(0, game_gf)
            draw = skellam.pmf(0,game_gf,game_ga) * 100
            t1_win = (1-skellam.cdf(0,game_gf,game_ga)) * 100
            t2_win = (skellam.cdf(-1,game_gf,game_ga)) * 100
        else:
            t1_cs = poisson.pmf(0, mu)
            t2_cs = poisson.pmf(0, lam)
            correction = rho * poisson.pmf(1, lam) * poisson.pmf(1, mu)
            draw = (skellam.pmf(0, lam, mu) - 2 * correction) * 100
            t1_win = (1-skellam.cdf(0, lam, mu) + correction) * 100
            t2_win = (skellam.cdf(-1, lam, mu) + correction) * 100
        #prints
        print(home_teams[t],round(t1_win,2),"draw",round(draw,2),away_teams[t],round(t2_win,2))
        print(home_teams[t],'goals',round(game_gf,2),'CS%',round(100*t1_cs,2))
        print(away_teams[t],'goals',round(game_ga,2),'CS%',round(100*t2_cs,2))
        print()
        player_df.append(df_home)
        player_df.append(df_away)
        summary.append([round(game_gf,2),home_teams[t],round(t1_win,4),round(draw,4),round(t2_win,4),away_teams[t],round(game_ga,2)])
    
    player_df = pd.concat(player_df)
    player_df = player_df[['name','team','positionText','MPG']]
    summary = pd.DataFrame(summary)
    summary.columns = summary.iloc[0];summary = summary.drop(0)
    summary = summary.apply(pd.to_numeric, errors='ignore')
    summary = summary.round({'Home win%':1, 'Draw%':1, 'Away win%':1})
    #for league avg player value use regstats and not the translated df, although this ignores aging
    regstats = regstats[regstats['playerId'].isin(df['playerId'])]
    regstats = regstats.merge(df[['playerId','positionText','team']], on='playerId', how='left')
    regstats = regstats.rename(columns={'positionText_y':'positionText'})
    #player value
    player_val = player_valuation(target_league,df)
    player_val_avg_team = player_valuation_average_team(target_league,regstats) #instead of df
    player_val_avg_team = player_val_avg_team[['playerId','GF90','GA90','GD90']].rename(columns={'GF90':'GF90_avg_team','GA90':'GA90_avg_team','GD90':'GD90_avg_team'})
    player_val = player_val.merge(player_val_avg_team, on='playerId', how='outer')
    output_columns = ['playerId', 'name', 'team', 'positionText', 'minutes', 'GF90', 'GA90', 'GD90', 'GD90_avg_team']
    player_val = player_val.dropna(subset=output_columns)
    # Merging can change the population and position labels used for centring.
    for c in ['GF90_avg_team','GA90_avg_team']: player_val[c] -= player_val.groupby('positionText')[c].transform('mean')
    player_val['GD90_avg_team'] = player_val['GF90_avg_team'] + player_val['GA90_avg_team']
    player_val = player_val[output_columns]
    player_val = player_val.round({'GD90_avg_team': 3})
    return standings,summary,player_val

def player_availability(target_league,league_player_stats,ignore_flag):
    injuries_df = pd.read_excel(f'{path}/injuries.xlsx',target_league)
    injuries_df['status'] = 0
    injuries_df["dt"] = pd.to_datetime(injuries_df["Expected Return"],errors='coerce')
    #nearest sunday found, how would this need to be changed for midweeks?
    today = pd.Timestamp.now().normalize()
    days_to_sunday = (6 - today.weekday()) if today.weekday() < 3 else -(today.weekday() + 1) % 7
    nearest_sunday = today + pd.Timedelta(days=days_to_sunday)
    #status changes based on nearest sunday
    injuries_df["status"] = np.where(injuries_df["dt"] <= nearest_sunday, 0.75, injuries_df["status"])
    injuries_df["status"] = np.where(injuries_df["dt"].isna(), 0.5, injuries_df["status"])
    injuries_df = injuries_df.drop_duplicates(subset='Player')
    
    #removing accents from names
    def clean_string_key(series):
        return (
            series.astype(str)
            .str.normalize('NFKD')               # Separates characters from their accents
            .str.encode('ascii', errors='ignore')# Removes the standalone accents
            .str.decode('utf-8')                 # Converts back to a clean string
            #.str.lower()                        # Makes it case-insensitive
            .str.strip()                         # Removes any accidental whitespace
        )
    league_player_stats['clean_name'] = clean_string_key(league_player_stats['name'])
    injuries_df['clean_name'] = clean_string_key(injuries_df['Player'])
    #rename players to match
    injuries_df.loc[injuries_df['clean_name']=='Daniel Ballard','clean_name'] = 'Dan Ballard'
    injuries_df.loc[injuries_df['clean_name']=='Mykhaylo Mudryk','clean_name'] = 'Mykhailo Mudryk'
    injuries_df.loc[injuries_df['clean_name']=='Valentino Livramento','clean_name'] = 'Tino Livramento'
    injuries_df.loc[injuries_df['clean_name']=='Franz-Ethan Meichtry','clean_name'] = 'Ethan Meichtry'
    injuries_df.loc[injuries_df['clean_name']=='A. Buongiorno','clean_name'] = 'Allesio Buongiorno'
    injuries_df.loc[injuries_df['clean_name']=='Gleison Bremer','clean_name'] = 'Bremer'
    injuries_df.loc[injuries_df['clean_name']=='K. Thuram-Ulien','clean_name'] = 'Khephren Thuram'
    injuries_df.loc[injuries_df['clean_name']=='Miguel Angel Sierra','clean_name'] = 'Miguel Sierra'
    injuries_df.loc[injuries_df['clean_name']=='P. Aubameyang','clean_name'] = 'Pierre-Emerick Aubameyang'
    injuries_df.loc[injuries_df['clean_name']=='A. Claude-Maurice','clean_name'] = 'Alexis Claude-Maurice'
    injuries_df.loc[injuries_df['clean_name']=='C. Baumgartner','clean_name'] = 'Christoph Baumgartner'
    injuries_df.loc[injuries_df['clean_name']=='G. Konstantelias','clean_name'] = 'Giannis Konstantelias'
    injuries_df.loc[injuries_df['clean_name']=='K. Karetsas','clean_name'] = 'Konstantinos Karetsas'
    injuries_df.loc[injuries_df['clean_name']=='K. Kvaratskhelia','clean_name'] = 'Khvicha Kvaratskhelia'
    injuries_df.loc[injuries_df['clean_name']=='J. Strand Larsen','clean_name'] = 'Jrgen Strand Larsen'
    injuries_df.loc[injuries_df['clean_name']=='M. Fernandez-Pardo','clean_name'] = 'Matias Fernandez-Pardo'
    injuries_df.loc[injuries_df['clean_name']=='Vitaliy Mykolenko','clean_name'] = 'Vitalii Mykolenko'
    injuries_df.loc[injuries_df['clean_name']=='I. Ruiz de','clean_name'] = 'Inigo Ruiz de Galarreta'
    #merge
    league_player_stats = league_player_stats.merge(injuries_df[['clean_name','status']], on='clean_name', how='outer')
    league_player_stats['status'] = league_player_stats['status'].fillna(1)
    #mismatched names
    name_mismatch = league_player_stats[league_player_stats['team'].isna()]
    name_mismatch = name_mismatch['clean_name'].to_list()
    league_player_stats = league_player_stats[~league_player_stats['team'].isna()]
    
    #manual position chnages for corner cases
    league_player_stats.loc[league_player_stats['playerId']==300447,'positionText'] = 'Forward' #Raphinha
    league_player_stats.loc[league_player_stats['playerId']==498798,'positionText'] = 'Forward' #Gonzalo García
    league_player_stats.loc[league_player_stats['playerId']==429556,'positionText'] = 'Forward' #Emersonn
    
    if(ignore_flag == 1): league_player_stats['status'] = 1   
    return league_player_stats,name_mismatch,injuries_df

def correct_player_positions(league_player_stats,threshold):
    #threshold = 0.2
    outfield = league_player_stats['positionText'].ne('Goalkeeper')
    counts = pd.crosstab(league_player_stats.loc[outfield, 'cluster'], league_player_stats.loc[outfield, 'positionText'])
    share = counts.div(counts.sum(axis=1), axis=0)
    dominant = counts.idxmax(axis=1)
    
    key = pd.MultiIndex.from_frame(league_player_stats[['cluster','positionText']])
    pos_share = pd.Series(share.stack().reindex(key).to_numpy(), index=league_player_stats.index)
    change = outfield & pos_share.lt(threshold)
    league_player_stats.loc[change, 'positionText'] = league_player_stats.loc[change, 'cluster'].map(dominant)
    return league_player_stats

#%% calls
#stabilization_rate(full_player_stats)
#full_player_stats,team_stats = player_stats_extract('Italy',2027)

season = 2027

_player_stats, _team_clusters = player_stats_extract_all(season)

player_bio,transfers = player_age_correction(_player_stats,season,0,1) #offseason rosters, pick latest team

_league_strength = league_conversions(_player_stats,season)

_stat_aging = aging_effects(_player_stats)

_regressed_stats = regression_player_full(_player_stats,_league_strength,season)

#%% custom lineups
player_bio = pd.read_excel(f'{path}/player_bio.xlsx','Sheet1')
player_bio[['start','sub']] = player_bio[['start','sub']].fillna(0)
player_bio = player_bio.drop(columns=['Unnamed: 0'])

#%% league forecasts
project_league = 'England'
league_player_stats,tc = projections_target_league(project_league,_regressed_stats,player_bio,_league_strength,_stat_aging,_team_clusters,season)

#%% correct player positions and apply injury news
league_player_stats = correct_player_positions(league_player_stats,1) #threshold to change player position for a subcluster
league_player_stats,unmatched_names,injuries_df = player_availability(project_league,league_player_stats,0) #1 to ignore injuries

#%% game level results
#standings,_,_,_ = league_table(project_league,league_player_stats)

#player_value = player_valuation(project_league,league_player_stats)
#player_value_avg_team = player_valuation_average_team(project_league,league_player_stats)

standings,summary,player_value = game_engine(10,project_league,league_player_stats,_regressed_stats,0) #gameweek, use custom lineups
