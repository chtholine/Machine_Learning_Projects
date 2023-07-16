import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

player_data = pd.read_csv('player_data.csv')
players = pd.read_csv('Players.csv')
season_stats = pd.read_csv('Seasons_Stats.csv')

# new player_data feature
player_data['weight_kg'] = player_data['weight'].astype(float) * 0.453592  # Перетворення ваги з фунтів у кілограми

# new players feature
players['age'] = pd.to_datetime(players['born']).apply(lambda x: (pd.to_datetime('today').year - x.year))  # Розрахунок віку гравців

# new season_stats feature
season_stats['PTS_per_game'] = season_stats['PTS'] / season_stats['G']  # Розрахунок середньої кількості очок на гру

print(player_data.head())
print(players.head())
print(season_stats.head())

# player_data positions visualization
sns.countplot(x='position', data=player_data)
plt.title('Player Positions')
plt.xlabel('Position')
plt.ylabel('Count')
plt.show()

# players height visualization
plt.figure(figsize=(10, 6))
sns.histplot(players['height'], bins=20)
plt.title('Player Heights')
plt.xlabel('Height (cm)')
plt.ylabel('Count')
plt.show()

# season_stats visualization
plt.figure(figsize=(10, 6))
sns.scatterplot(x='PTS', y='AST', data=season_stats)
plt.title('Points vs Assists')
plt.xlabel('Points')
plt.ylabel('Assists')
plt.show()