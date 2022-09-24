#INSTALL CUSTOM PACKAGES
#!pip install schedule
!pip install datetime
#!pip install ccxt
!pip install seaborn
!pip install yfinance

import yfinance as yf
#import schedule
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import numpy as np
from datetime import datetime
import time

def tr(data):
    data['previous_close'] = data['close'].shift(1)
    data['high-low'] = abs(data['high'] - data['low'])
    data['high-pc'] = abs(data['high'] - data['previous_close'])
    data['low-pc'] = abs(data['low'] - data['previous_close'])

    tr = data[['high-low', 'high-pc', 'low-pc']].max(axis=1)

    return tr

def atr(data, period):
    data['tr'] = tr(data)
    atr = data['tr'].rolling(period).mean()

    return atr

def supertrend(df, period=7, atr_multiplier=3):
    hl2 = (df['high'] + df['low']) / 2
    df['atr'] = atr(df, period)
    df['upperband'] = hl2 + (atr_multiplier * df['atr'])
    df['lowerband'] = hl2 - (atr_multiplier * df['atr'])
    df['in_uptrend'] = True

    for current in range(2, len(df.index)):
        previous = current - 1

        if df.loc[current, 'close'] > df.loc[previous, 'upperband']:
            df.loc[current, 'in_uptrend'] = True
        elif df.loc[current, 'close'] < df.loc[previous, 'lowerband']:
            df.loc[current, 'in_uptrend'] = False
        else:
            df.loc[current, 'in_uptrend'] = df.loc[previous, 'in_uptrend']

            if df.loc[current, 'in_uptrend'] and df.loc[current, 'lowerband'] < df.loc[previous, 'lowerband']:
              df.loc[current, 'lowerband'] = df.loc[previous, 'lowerband']

            if not df.loc[current, 'in_uptrend'] and df.loc[current, 'upperband'] > df.loc[previous, 'upperband']:
              df.loc[current, 'upperband'] = df.loc[previous, 'upperband']

    return df



def check_buy_sell_signals(df, ticker):
    global in_position

    print("checking for buy and sell signals")
    #print(df.tail())
    last_row_index = len(df.index) - 1
    previous_row_index = last_row_index - 1

    if not df.loc[previous_row_index, 'in_uptrend'] and df.loc[last_row_index, 'in_uptrend']:
        print("changed to uptrend, buy")
        if not in_position:
            #order = exchange.create_market_buy_order(ticker, 0.05)
            print('Buy')
            in_position = True
        else:
            print("already in position, nothing to do")
    if df.loc[previous_row_index, 'in_uptrend'] and not df.loc[last_row_index, 'in_uptrend']:
        if in_position:
            print("changed to downtrend, sell")
            #order = exchange.create_market_sell_order(ticker, 0.05)
            print('Sell')
            in_position = False
        else:
            print("You aren't in position, nothing to sell")

def run_bot(ticker, data_pts=300, freq='1d'):
    print(f"Fetching new bars for {datetime.now().isoformat()}")
    df = yf.download(ticker, period='1y')
    df.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'not_adj_close', 'Adj Close': 'close', 'Volume': 'volume'}, inplace = True)
    df.index.names = ['timestamp']
    df.reset_index(inplace=True)
    supertrend_data = supertrend(df)

    check_buy_sell_signals(supertrend_data, ticker)

    return df

#Create Buy Sell Image
def create_plot(df, ticker):
  print('Creating Plot')

  colors = {False:'#8B251C', True:'#277B4A'}
  mask = df['in_uptrend'].replace({False:'#8B251C', True:'#277B4A'})
  mask.name = 'color'
  df = pd.concat((df, mask), axis=1)
  #print(df.head())
  #df['color'] = df['in_uptrend'].map(colors)
  # Set figure size (width, height) in inches
  fig, ax = plt.subplots(figsize = (10,3), dpi=200)

  sns.lineplot(x='timestamp', y='previous_close', data=df, hue=df['in_uptrend'], marker='o', linestyle='', palette=colors)
  sns.despine()

  # Set label for x-axis
  ax.set_xlabel('Date' , size = 12 )
  date_form = DateFormatter("%m-%d-%y")
  ax.xaxis.set_major_formatter(date_form)

  # Set label for y-axis
  ax.set_ylabel('Price' , size = 12 )

  # Set title for plot
  ax.set_title(ticker , size = 18 )

  plt.legend(loc='best', frameon=False, labels=['Sell', 'Buy']) #best, upper left, upper right, lower left, lower right
  plt.xticks(rotation=90)

  # Display figure
  #plt.show()

  return fig

#Start Program
#pd.set_option('display.max_rows', None)
#pd.options.mode.chained_assignment = None  # default='warn'
in_position = False

ticker = 'QQQ'
df = run_bot(ticker, data_pts='1y').dropna()  # valid periods: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
fig = create_plot(df, ticker)
plt.show()

#Save Image
#filename = ticker + '.png'
#fig.savefig(filename, bbox_inches='tight')
#print('Saved PNG')

#schedule.every(10).seconds.do(run_bot)

print('Done')
