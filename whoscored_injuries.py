# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 17:16:21 2026
player injury status extracted from sportsgambler.com
@author: uttam
"""

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
import time

#path = 'C:/Users/Subramanya.Ganti/Downloads/Sports/football/whoscored'
path = 'C:/Users/uttam/Desktop/Sports/football/whoscored'

#%% functions to extract data
def player_injury_info(s):
    print(s)
    url = f"https://www.sportsgambler.com/injuries/football/{s}/"
    headers = {"User-Agent": "Mozilla/5.0"}    
    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")    
    injury_rows = soup.find_all("div", class_="inj-row")    
    data = []
    
    for row in injury_rows:
        container = row.find("div", class_="inj-container")
        if container is None: continue
    
        team_heading = row.find_previous("h3", class_="injuries-title")
        team_name = team_heading.get_text(strip=True) if team_heading else None
    
        def get_text(parent, class_name):
            element = parent.find("span", class_=class_name.split())
            return element.get_text(strip=True) if element else None
    
        hidden = row.find("div", class_="injury-hidden-content")
    
        def get_hidden_value(label):
            if hidden is None: return None
    
            for span in hidden.find_all("span", recursive=False):
                bold = span.find("span", class_="bold-text")
                if bold and label in bold.get_text(strip=True):
                    return span.get_text(" ", strip=True).replace(bold.get_text(strip=True), "", 1).strip()
            return None
    
        data.append({
            "Team": team_name,
            "Player": get_text(container, "inj-player"),
            "Position": get_hidden_value("Position:"),
            #"Matches": get_hidden_value("Matches:"),
            #"Goals": get_hidden_value("Goals:"),
            #"Assists": get_hidden_value("Assists:"),
            "Injury": get_text(container, "inj-info"),
            "Expected Return": get_hidden_value("Expected Return:")
        })
    
    df = pd.DataFrame(data)
    time.sleep(10)
    return df

def update_file():
    england = player_injury_info('england-premier-league')
    england2 = player_injury_info('england-championship')
    france = player_injury_info('france-ligue-1')
    italy = player_injury_info('italy-serie-a')
    germany = player_injury_info('germany-bundesliga')
    spain = player_injury_info('spain-la-liga')
    netherlands = player_injury_info('netherlands-eredivisie')
    portugal = player_injury_info('portugal-primeira-liga')
    
    with pd.ExcelWriter(f'{path}/injuries.xlsx', engine='openpyxl') as writer:
        england.to_excel(writer, index=False, sheet_name='England')
        france.to_excel(writer, index=False, sheet_name='France')
        germany.to_excel(writer, index=False, sheet_name='Germany')
        spain.to_excel(writer, index=False, sheet_name='Spain')
        italy.to_excel(writer, index=False, sheet_name='Italy')
        netherlands.to_excel(writer, index=False, sheet_name='Netherlands')
        portugal.to_excel(writer, index=False, sheet_name='Portugal')
        england2.to_excel(writer, index=False, sheet_name='England2')

#%% run command
update_file()
