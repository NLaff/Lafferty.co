### Board Game Analysis with Pandas

I've been taking some Python classes through Data Camp. They're designed around teaching Python with data analysis applications, which is exactly what I was looking for. I wanted to apply some of the techniques I had learned on a data set that  was interesting to me, so I started browsing on Kaggle. I found [this](https://www.kaggle.com/mrpantherson/board-game-data), which is aggregated data from Board Game Geek. 

I'll do my best to explain what is going on, mostly to reinforce the lessons with myself. This is a raw notebook file, so some of the graphs do not explain the data well, some are ugly, and some are missing axis labels. 

Importing some packages:


```python
import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline # Display graphs inline
sns.set() # Default to Seaborn's graphical style
```

Load in the data:

```python
df = pd.read_csv('bgg_db_2017_03.csv', encoding='latin1')
```

Look at the first 3 rows in the data. I truncated lots of the columns so the table was more visible online.

```python
df.head(3)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>rank</th>
      <th>bgg_url</th>
      <th>game_id</th>
      <th>names</th>
      <th>min_players</th>
      <th>max_players</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>https://boardgamegeek.com/boardgame/161936/pan...</td>
      <td>161936</td>
      <td>Pandemic Legacy: Season 1</td>
      <td>2</td>
      <td>4</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>https://boardgamegeek.com/boardgame/182028/thr...</td>
      <td>182028</td>
      <td>Through the Ages: A New Story of Civilization</td>
      <td>2</td>
      <td>4</td>
    </tr>
    </tbody>
</table>
</div>


Plotting a histogram on the number of votes with 20 bins. This is awful looking, and is missing axis labels.

```python
plt.hist(df['num_votes'], bins=20)
```


![png](Board%20Game%20Analysis_files/Board%20Game%20Analysis_4_1.png)



Creating a new data frame with trimmed values, that contains the games with the 3 most ratings. This is also terrible way to display this data, but I wanted to try a swarm plot. 

```python
df2 = df[df['num_votes'] > 1000] # trim small values
df2 = df2.nlargest(3,'avg_rating')
sns.swarmplot(y='num_votes',x='names',data=df2)
```


![png](Board%20Game%20Analysis_files/Board%20Game%20Analysis_5_1.png)

Creating a new data frame that contains the name and the number of votes, sorted by number of votes descending.


```python
df3 = df2[['names','num_votes']].sort_values(by='num_votes',ascending=False)
df3 = df3.set_index('names')
df3
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>num_votes</th>
    </tr>
    <tr>
      <th>names</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Catan</th>
      <td>66826</td>
    </tr>
    <tr>
      <th>Carcassonne</th>
      <td>66218</td>
    </tr>
    <tr>
      <th>Pandemic</th>
      <td>61298</td>
    </tr>
    <tr>
      <th>Dominion</th>
      <td>55245</td>
    </tr>
    <tr>
      <th>7 Wonders</th>
      <td>50815</td>
    </tr>
    <tr>
      <th>Agricola</th>
      <td>47754</td>
    </tr>
    <tr>
      <th>Ticket to Ride</th>
      <td>47654</td>
    </tr>
    <tr>
      <th>Puerto Rico</th>
      <td>47307</td>
    </tr>
    <tr>
      <th>Small World</th>
      <td>42566</td>
    </tr>
    </tbody>
</table>
</div>


Creating a horizontal bar plot of the 20 most rated games using Seaborn's default color scheme. This was mainly to get practice plotting graphs with Seaborn

```python
df2 = df2.nlargest(20,'num_votes')
ax = sns.barplot(x='num_votes',y='names',data=df2)
sns.plt.title('Most Rated Games')
ax.set(xlabel='Number of Votes')
ax.set(ylabel='Game')
plt.xticks(rotation=45)
plt.show()
```


![png](Board%20Game%20Analysis_files/Board%20Game%20Analysis_8_0.png)


Creating another horizontal barplot, this time with the standard matplotlib plotting methods.

```python
df3['num_votes'][:20].plot(kind='barh')
plt.xlabel('Number of Ratings')
plt.ylabel('Game')
plt.title("Most Rated Games")
```




    <matplotlib.text.Text at 0x112175a10>




![png](Board%20Game%20Analysis_files/Board%20Game%20Analysis_9_1.png)

Slicing the data set into a new data frame that contains only the columns I care about, and sorting it descending by average rating. Again, truncated for brevity. 


```python
best_rated = df2[['names','avg_rating','num_votes','weight']].sort_values(by='avg_rating',ascending=False)
best_rated = best_rated.set_index('names')
best_rated
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>avg_rating</th>
      <th>num_votes</th>
      <th>weight</th>
    </tr>
    <tr>
      <th>names</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Gloomhaven</th>
      <td>9.23599</td>
      <td>2481</td>
      <td>3.7151</td>
    </tr>
    <tr>
      <th>Kingdom Death: Monster</th>
      <td>8.96679</td>
      <td>1868</td>
      <td>4.2085</td>
    </tr>
    <tr>
      <th>Through the Ages: A New Story of Civilization</th>
      <td>8.76347</td>
      <td>6481</td>
      <td>4.2953</td>
    </tr>
    <tr>
      <th>Pandemic Legacy: Season 1</th>
      <td>8.66514</td>
      <td>16385</td>
      <td>2.8067</td>
    </tr>
    <tr>
      <th>Star Wars: Rebellion</th>
      <td>8.55971</td>
      <td>7581</td>
      <td>3.5600</td>
    </tr>
    <tr>
      <th>Arkham Horror: The Card Game</th>
      <td>8.42524</td>
      <td>4175</td>
      <td>3.0252</td>
    </tr>
    <tr>
      <th>Mechs vs. Minions</th>
      <td>8.41385</td>
      <td>4175</td>
      <td>2.4598</td>
    </tr>
    <tr>
      <th>War of the Ring (Second Edition)</th>
      <td>8.38386</td>
      <td>6127</td>
      <td>3.9896</td>
    </tr>
    <tr>
      <th>Terraforming Mars</th>
      <td>8.36395</td>
      <td>6506</td>
      <td>3.2618</td>
    </tr>
    </tbody>
</table>
<p>1566 rows × 3 columns</p>
</div>



Plotting out the worst rated games, and how many votes they had. People really didn't like Zombies!!!


```python
worst_rated = df2[['names','avg_rating','num_votes']].sort_values(by='avg_rating',ascending=True)
worst_rated = worst_rated.set_index('names')
worst_rated['num_votes'][:20].plot(kind='barh')
```


![png](Board%20Game%20Analysis_files/Board%20Game%20Analysis_13_1.png)


Slicing the original data frame on just Zombies!!!

```python
df[df['names'] == 'Zombies!!!']
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>rank</th>
      <th>bgg_url</th>
      <th>game_id</th>
      <th>names</th>
      <th>min_players</th>
      <th>max_players</th>
      <th>avg_time</th>
      <th>min_time</th>
      <th>max_time</th>
      <th>year</th>
      <th>avg_rating</th>
      <th>geek_rating</th>
      <th>num_votes</th>
      <th>image_url</th>
      <th>age</th>
      <th>mechanic</th>
      <th>owned</th>
      <th>category</th>
      <th>designer</th>
      <th>weight</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3728</th>
      <td>3729</td>
      <td>https://boardgamegeek.com/boardgame/2471/zombies</td>
      <td>2471</td>
      <td>Zombies!!!</td>
      <td>2</td>
      <td>6</td>
      <td>60</td>
      <td>60</td>
      <td>60</td>
      <td>2001</td>
      <td>5.88023</td>
      <td>5.70313</td>
      <td>12304</td>
      <td>//cf.geekdo-images.com/images/pic2290485.jpg</td>
      <td>12</td>
      <td>Dice Rolling, Grid Movement, Hand Management, ...</td>
      <td>17363</td>
      <td>Exploration, Fighting, Horror, Miniatures, Mov...</td>
      <td>Todd Breitenstein, Kerry Breitenstein</td>
      <td>1.6209</td>
    </tr>
  </tbody>
</table>
</div>

Looking at the average number of maximum players, and how many records are in the data set in total. 


```python
df['avg_max_players'] = df['max_players'].mean()
df['avg_max_players'].value_counts()
```




    5.271054    4999
    Name: avg_max_players, dtype: int64


Calculating the average number of players, and how many games fall into each bucket. Truncated for brevity


```python
df['avg_players'] = (df['min_players'] + df['max_players']) / 2
df['avg_players'].value_counts()
```




    3.0      1273
    2.0       900
    3.5       771
    4.0       731
    2.5       319
    4.5       280
    5.0       181
    1.5       149
    1.0        92
    5.5        70
    6.0        61
    6.5        28
    7.0        26
    8.0        14
    50.5       12
    0.0        12
    Name: avg_players, dtype: int64



Graphing mean player counts. 

```python
df['avg_players'] = (df['min_players'] + df['max_players']) / 2
player_counts = df['avg_players'].value_counts()
player_counts = player_counts[player_counts > 10] # trim rare occurences
ax = sns.barplot(x=player_counts.index, y = player_counts)
sns.plt.title('Mean Player Counts')
ax.set(xlabel='player count')
plt.show()
```


![png](Board%20Game%20Analysis_files/Board%20Game%20Analysis_19_0.png)

