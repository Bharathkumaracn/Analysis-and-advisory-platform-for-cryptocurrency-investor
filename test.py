from turtle import title
from flask import Flask, render_template
from flask import request
import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import pandas_datareader as web
import datetime as dt
import json
import numpy as np
import requests


app = Flask(__name__)

@app.route('/')
def home():
    print()
    return render_template('home.html')

@app.route('/coin')
def coin():
    n=request.args.to_dict().keys()
    n=list(n)
    n=n[0].upper()
    
    # Getting historic data from yahoo finance
    
    crypto_currency= n
    against_currency='USD'
    start = dt.datetime(2021,1,1)
    end = dt.datetime.now()
    df = web.DataReader(f'{crypto_currency}-{against_currency}','yahoo', start,end)

    # Candle stick graph generation for historic data 

    figure = go.Figure( data = [
			go.Candlestick(
				x = df.index,
				low = df['Low'],
				high = df['High'],
				close = df['Close'],
				open = df['Open'],
				increasing_line_color = 'green',
				decreasing_line_color = 'red'
			)
		]
    )
    figure.update_layout(template="plotly_dark",title=n,yaxis_title="Price USD")
    graphJSON = json.dumps(figure, cls=plotly.utils.PlotlyJSONEncoder)
   
    return render_template('coin.html',graphJSON=graphJSON)

@app.route('/news')
def news():

    import nltk
    nltk.download('punkt')
    nltk.download('vader_lexicon')
    import requests
    import pandas as pd
    import re
    from nltk.tokenize import word_tokenize
    import numpy as np
    
    # Extracting news from cryptocompare using API

    URL = "https://min-api.cryptocompare.com/data/v2/news/?lang=EN&apiKey=e07c5948287ce37f024d2194a30a468be68f62dc6fcf2db77e65539ceb81baa3"

    r = requests.get(url = URL)
    
    # extracting data in json format
    data = r.json()
    li=[]
    for i in range(50):
        li.append(data['Data'][i]['title'])

    # Functions to prep data for sentiment analysis

    # removing emojis from text
    def remove_emoji(text):
        regrex_pattern = re.compile(pattern="["
                                    u"\U0001F600-\U0001F64F"
                                    u"\U0001F300-\U0001F5FF"  
                                    u"\U0001F680-\U0001F6FF"  
                                    u"\U0001F1E0-\U0001F1FF"
                                    "]+", flags=re.UNICODE)
        return regrex_pattern.sub(r'', text)

    # tokenizing the text 
    def filter_tweets(txt):
        
        txt = re.sub("@[A-Za-z0-9]+","",txt) 
        txt = re.sub(r"(?:\@|http?\://|https?\://|www)\S+", "", txt)
        txt = re.sub("&","",txt)
        stop_words = {'ourselves', 'hers', 'between', 'yourself', 'but', 'again', 'there', 'about', 'once', 'during', 'out', 'very', 'having', 'with', 'they', 'own', 'an', 'be', 'some', 'for', 'do', 'its', 'yours', 'such', 'into', 'of', 'most', 'itself', 'other', 'off', 'is', 's', 'am', 'or', 'who', 'as', 'from', 'him', 'each', 'the', 'themselves', 'until', 'below', 'are', 'we', 'these', 'your', 'his', 'through', 'don', 'nor', 'me', 'were', 'her', 'more', 'himself', 'this', 'down', 'should', 'our', 'their', 'while', 'above', 'both', 'up', 'to', 'ours', 'had', 'she', 'all', 'no', 'when', 'at', 'any', 'before', 'them', 'same', 'and', 'been', 'have', 'in', 'will', 'on', 'does', 'yourselves', 'then', 'that', 'because', 'what', 'over', 'why', 'so', 'can', 'did', 'not', 'now', 'under', 'he', 'you', 'herself', 'has', 'just', 'where', 'too', 'only', 'myself', 'which', 'those', 'i', 'after', 'few', 'whom', 't', 'being', 'if', 'theirs', 'my', 'against', 'a', 'by', 'doing', 'it', 'how', 'further', 'was', 'here', 'than'}
        word_tokens = word_tokenize(txt)
        filtered_sentence = [w for w in word_tokens if not w in stop_words]
        filtered_sentence = []
        for w in word_tokens:
            if w not in stop_words:
                filtered_sentence.append(w)
        filtered_sentence = ' '.join([str(elem) for elem in filtered_sentence])
        txt=filtered_sentence
        return(txt)

    # Using above functions on data 
    for i in range(len(li)):
        li[i]=filter_tweets(li[i])
        li[i]=remove_emoji(li[i])


    # Sentiment analysis of tweets

    from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA

    sia = SIA()
    results = []

    for line in li:
        pol_score = sia.polarity_scores(line)
        pol_score['headline'] = line
        results.append(pol_score)
        
    df = pd.DataFrame.from_records(results)
    print(df.head())

    compound_list = df['compound'].tolist()

    # counting total number of positive,negative and neutral tweets
    pos=0
    neg=0
    neu=0
    for i in compound_list:
        if(i>0):
            pos+=1
        elif(i<0):
            neg+=1
        else:
            neu+=1

    # Plotting Bar graph to display result of sentiment analysis 

    data = {'Positive':pos, 'Negative':neg, 'Neutral':neu}
    Sentiment = list(data.keys())
    No_of_headlines = list(data.values())
    
    import plotly.express as px

    y = ["Positive","Negative","Neutral"]
    x = [pos,neg,neu]
    fig = go.Figure([go.Bar(x=y, y=x)])
    fig.update_layout(title='Sentiment Analysis')
    fig.update_layout(template="plotly_dark")
   

    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    
    return render_template('news.html',graphJSON=graphJSON)

@app.route('/coin/prediction')
def prediction():

    # Extracting coin name from the url
    n=request.args.to_dict().keys()
    n=list(n)
    n=n[0].upper()
    
    import pandas as pd
    import yfinance as yf
    from datetime import date, timedelta
    from autots import AutoTS

    crypto_name=['bitcoin', 'ethereum', 'tether', 'usd-coin', 'bnb', 'xrp', 'cardano', 'binance-usd', 'solana', 'dogecoin', 'polkadot',
            'avalanche', 'wrapped-bitcoin', 'lido-staked-ether', 'tron', 'shiba-inu', 'dai', 'litecoin', 'cronos', 'leo-token', 
            'polygon', 'near-protocol', 'ftx', 'bitcoin-cash', 'stellar', 'chainlink', 'okb', 'flow', 'cosmos-hub', 'algorand', 
            'ethereum-classic', 'monero', 'apecoin', 'uniswap', 'vechain', 'hedera', 'internet-computer', 'elrond', 'terrausd', 
            'theta-fuel', 'terra', 'decentraland', 'magic-internet-money', 'filecoin', 'the-sandbox', 'ceth', 'axie-infinity', 'tezos', 
            'defichain', 'chain', 'frax', 'theta-network', 'maker', 'eos', 'pancakeswap', 'kucoin', 'cusdc', 'bittorrent', 'zcash',
            'trueusd', 'aave', 'huobi-btc', 'klaytn', 'the-graph', 'huobi', 'helium', 'bitcoin-sv', 'thorchain', 'iota', 
            'pax-dollar', 'quant', 'stepn', 'neutrino-usd', 'radix', 'ecash', 'cdai', 'fantom', 'gatetoken', 'bitdao', 'convex-finance',
            'zilliqa', 'nexo', 'neo', 'waves', 'gala', 'arweave', 'celo', 'enjin-coin', 'cusdt', 'bitcoin-cash-abc', 'amp', 'dash', 
            'kusama', 'chiliz', 'synthetix-network-token', 'basic-attention-token', 'frax-share', 'pax-gold', 'stacks', 'loopring', 
            'gnosis', 'harmony', 'osmosis', 'xdc-network', 'fei-usd', 'curve-dao-token', 'maiar-dex', 'mina-protocol', 'nem', 'lido-dao', 
            'compound', 'holo', 'nexus-mutual', 'decred', 'tether-gold', 'ecomi', 'tokenize-xchange', 'kadena', 'iost', 'qtum', 'kava', 
            'serum', '1inch', 'bancor-network-token', 'livepeer','usdd','synthetix-network','celsius-network','evmos','terraclassicusd','gate'
            ,'basic-attention','tenset']

    crypto_sym=['btc', 'eth', 'usdt', 'usdc', 'bnb', 'xrp', 'ada', 'busd', 'sol', 'doge', 'dot', 'avax', 'wbtc', 'steth', 'trx', 'shib', 
                'dai', 'ltc', 'cro', 'leo', 'matic', 'near', 'ftt', 'bch', 'xlm', 'link', 'okb', 'flow', 'atom', 'algo', 'etc', 'xmr', 'ape',
                'uni', 'vet', 'hbar', 'icp', 'egld', 'ust', 'tfuel', 'luna', 'mana', 'mim', 'fil', 'sand', 'ceth', 'axs', 'xtz', 'dfi', 
                'xcn', 'frax', 'theta', 'mkr', 'eos', 'cake', 'kcs', 'cusdc', 'btt', 'zec', 'tusd', 'aave', 'hbtc', 'klay', 'grt', 'ht', 
                'hnt', 'bsv', 'rune',  'miota', 'usdp', 'qnt', 'gmt', 'usdn', 'xrd', 'xec', 'cdai', 'ftm', 'gt', 'bit', 'cvx', 'zil', 'nexo',
                'neo', 'waves', 'gala', 'ar', 'celo', 'enj', 'cusdt', 'bcha', 'amp', 'dash', 'ksm', 'chz', 'snx', 'bat', 'fxs', 'paxg', 
                'stx', 'lrc', 'gno', 'one', 'osmo', 'xdc', 'fei', 'crv', 'mex', 'mina', 'xem', 'ldo', 'comp', 'hot', 'nxm', 'dcr', 'xaut', 
                'omi', 'tkx', 'kda', 'iost', 'qtum', 'kava', 'srm', '1inch', 'bnt', 'lpt','usdd','snx','cel','evm','ustc','gt','bat','10set']

    for i in range(len(crypto_sym)):
        if(crypto_sym[i]==n.lower()):
            coin_value=crypto_name[i]
            break
    
    # Getting historic data for Prediction from yahoo finance        
    
    today = date.today()
    d1 = today.strftime("%Y-%m-%d")
    end_date = d1
    d2 = date.today() - timedelta(days=365)
    d2 = d2.strftime("%Y-%m-%d")
    start_date = d2

    data = yf.download('{}-INR'.format(n), start=start_date, end=end_date, progress=False)
    data["Date"] = data.index
    data = data[["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]]
    data.reset_index(drop=True, inplace=True)

    # Prediction using AutoTS ML model

    model = AutoTS(forecast_length=10, frequency='infer',ensemble='simple', drop_data_older_than_periods=15,max_generations=30)
    model = model.fit(data, date_col='Date', value_col='Close', id_col=None)

    prediction = model.predict()
    forecast = prediction.forecast
    print(forecast)

    # Preparing data to plot graph   
    n=forecast.index.values
    date_p=[]
    close_p=list(forecast.Close)
    for i in n:
        temp=str(i)
        date_p.append(temp[:10])
        
    date_a=[]
    close_a=list(data['Close'].tail(30))
    m=list(data['Date'].tail(30))
    for i in m:
        temp=str(i)
        date_a.append(temp[:10])

    date_a.extend(date_p)
    close_a.extend(close_p)

    # Graph for prediction
    import plotly.express as px
    fig = px.line(x=date_a, y=close_a)
    fig.add_scatter(x=date_p, y=close_p, mode='lines')
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    # Web scraping 
    from selenium import webdriver   
    from selenium.webdriver.support.ui import WebDriverWait  
    from selenium.webdriver.chrome.options import Options  
    
    option = webdriver.ChromeOptions()
    option.add_argument('headless')
    driver = webdriver.Chrome(r"D:/Major Project/chromedriver.exe",options=option)
    driver.get('https://cryptopredictions.com/{}/'.format(coin_value))


    txt = driver.find_elements_by_css_selector("div.table-responsive")
    data_2022=txt[0].text
    data_2022=data_2022.split("\n")
    del(data_2022[0])
    a=[]
    for i in data_2022:
        temp=i.split()
        a.append(temp)
    
    month=[]
    year=[]
    min_price=[]
    max_price=[]
    avg_price=[]
    change=[]

    # adding scrapped data into lists to display in tabular format

    for i in range (len(a)):
        month.append(a[i][0])
        year.append(a[i][1])
        min_price.append(a[i][2])
        max_price.append(a[i][3])
        avg_price.append(a[i][4])
        change.append(a[i][5])

    dollar_price=78
    ln=len(month)
    return render_template('prediction.html',close=close_p,date=date_p,graphJSON=graphJSON,m=month,y=year,min=min_price,max=max_price,avg=avg_price,ch=change,dp=dollar_price,ln=ln)


@app.route('/advisory')
def advisory():
    
    import requests
    from operator import itemgetter

    # Getting top 100 coins using coingecko api 

    URL = "https://api.coingecko.com/api/v3/coins/markets?vs_currency=inr"
    r = requests.get(url = URL)
    data = r.json()

    # Creating Trending coin list 

    for i in range(len(data)):
        if(data[i]['price_change_percentage_24h']==None):
            data[i]['price_change_percentage_24h']=0


    li=sorted(data, key=itemgetter('price_change_percentage_24h'))
    li1=sorted(data, key=itemgetter('total_volume'))
    tr_list=[]
    

    for i in range(1,4):
        tr_list.append(li1[len(li1)-i])

    for i in range(1,4):
        tr_list.append(li[len(li)-i])
        
        
    for i in range(0,3):
        tr_list.append(li[i])

    # Creating lists of trending coins to display in table format
    symbol=[]
    price_change=[]  
    image=[]
    rank=[]
    cprice=[]
    cname=[]
    vol=[]
    symbol1=[]
    for i in range(len(tr_list)):
        cname.append(tr_list[i]['name'])
        price_change.append(tr_list[i]['price_change_percentage_24h'])
        symbol.append((tr_list[i]['symbol']).upper())
        image.append(tr_list[i]['image'])
        rank.append(tr_list[i]['market_cap_rank'])
        cprice.append(tr_list[i]['current_price'])
        vol.append(tr_list[i]['total_volume'])
        symbol1.append((tr_list[i]['symbol']))

    print(price_change)
    return render_template('advisory.html',r=rank,im=image,n=cname,s=symbol,p=cprice,v=vol,ch=price_change,s1=symbol1)


@app.route('/advisory/advisorycoin')
def advisorycoin():

    # Extracting coin name from the page url    
    n=request.args.to_dict().values()
    n=list(n)
    n=n[0]
    n=str(n)
    n=n.replace(' ','-')


    crypto_name=['bitcoin', 'ethereum', 'tether', 'usd-coin', 'bnb', 'xrp', 'cardano', 'binance-usd', 'solana', 'dogecoin', 'polkadot',
            'avalanche', 'wrapped-bitcoin', 'lido-staked-ether', 'tron', 'shiba-inu', 'dai', 'litecoin', 'cronos', 'leo-token', 
            'polygon', 'near-protocol', 'ftx', 'bitcoin-cash', 'stellar', 'chainlink', 'okb', 'flow', 'cosmos-hub', 'algorand', 
            'ethereum-classic', 'monero', 'apecoin', 'uniswap', 'vechain', 'hedera', 'internet-computer', 'elrond', 'terrausd', 
            'theta-fuel', 'terra', 'decentraland', 'magic-internet-money', 'filecoin', 'the-sandbox', 'ceth', 'axie-infinity', 'tezos', 
            'defichain', 'chain', 'frax', 'theta-network', 'maker', 'eos', 'pancakeswap', 'kucoin', 'cusdc', 'bittorrent', 'zcash',
            'trueusd', 'aave', 'huobi-btc', 'klaytn', 'the-graph', 'huobi', 'helium', 'bitcoin-sv', 'thorchain', 'iota', 
            'pax-dollar', 'quant', 'stepn', 'neutrino-usd', 'radix', 'ecash', 'cdai', 'fantom', 'gatetoken', 'bitdao', 'convex-finance',
            'zilliqa', 'nexo', 'neo', 'waves', 'gala', 'arweave', 'celo', 'enjin-coin', 'cusdt', 'bitcoin-cash-abc', 'amp', 'dash', 
            'kusama', 'chiliz', 'synthetix-network-token', 'basic-attention-token', 'frax-share', 'pax-gold', 'stacks', 'loopring', 
            'gnosis', 'harmony', 'osmosis', 'xdc-network', 'fei-usd', 'curve-dao-token', 'maiar-dex', 'mina-protocol', 'nem', 'lido-dao', 
            'compound', 'holo', 'nexus-mutual', 'decred', 'tether-gold', 'ecomi', 'tokenize-xchange', 'kadena', 'iost', 'qtum', 'kava', 
            'serum', '1inch', 'bancor-network-token', 'livepeer','usdd','synthetix-network','celsius-network','evmos']

    crypto_sym=['btc', 'eth', 'usdt', 'usdc', 'bnb', 'xrp', 'ada', 'busd', 'sol', 'doge', 'dot', 'avax', 'wbtc', 'steth', 'trx', 'shib', 
                'dai', 'ltc', 'cro', 'leo', 'matic', 'near', 'ftt', 'bch', 'xlm', 'link', 'okb', 'flow', 'atom', 'algo', 'etc', 'xmr', 'ape',
                'uni', 'vet', 'hbar', 'icp', 'egld', 'ust', 'tfuel', 'luna', 'mana', 'mim', 'fil', 'sand', 'ceth', 'axs', 'xtz', 'dfi', 
                'xcn', 'frax', 'theta', 'mkr', 'eos', 'cake', 'kcs', 'cusdc', 'btt', 'zec', 'tusd', 'aave', 'hbtc', 'klay', 'grt', 'ht', 
                'hnt', 'bsv', 'rune',  'miota', 'usdp', 'qnt', 'gmt', 'usdn', 'xrd', 'xec', 'cdai', 'ftm', 'gt', 'bit', 'cvx', 'zil', 'nexo',
                'neo', 'waves', 'gala', 'ar', 'celo', 'enj', 'cusdt', 'bcha', 'amp', 'dash', 'ksm', 'chz', 'snx', 'bat', 'fxs', 'paxg', 
                'stx', 'lrc', 'gno', 'one', 'osmo', 'xdc', 'fei', 'crv', 'mex', 'mina', 'xem', 'ldo', 'comp', 'hot', 'nxm', 'dcr', 'xaut', 
                'omi', 'tkx', 'kda', 'iost', 'qtum', 'kava', 'srm', '1inch', 'bnt', 'lpt','usdd','snx','cel','evm']

    for i in range(len(crypto_sym)):
        if(crypto_name[i]==n.lower()):
            coin_value=crypto_sym[i]
            break
    print(n)
    print(coin_value)

    import pandas
    import yfinance as yf
    from datetime import date, timedelta
    import plotly.express as px

    # Getting historic data from yahoo finance for calculating all the technical indicators

    today = date.today()
    d1 = today.strftime("%Y-%m-%d")
    end_date = d1
    d2 = date.today() - timedelta(days=730)  #data for 730 days
    d2 = d2.strftime("%Y-%m-%d")
    start_date = d2

    data = yf.download('{}-USD'.format(coin_value.upper()), start=start_date, end=end_date, progress=False)

    # RSI calculation
    def rsi(df, periods = 14):

        close_delta = df['Close'].diff()
        # Make two series: one for lower closes and one for higher closes
        up = close_delta.clip(lower=0)
        down = -1 * close_delta.clip(upper=0)

        ma_up = up.ewm(com = periods - 1, adjust=True, min_periods = periods).mean()
        ma_down = down.ewm(com = periods - 1, adjust=True, min_periods = periods).mean()

        rsi = ma_up / ma_down
        rsi = 100 - (100/(1 + rsi))
        return rsi


    RSI=rsi(data)

    # RSI result generation
    date=RSI.index 
    rsi_flag=0

    # 1,2,3,4  refers strong sell,sell,buy,strong buy respectively

    if((RSI[len(RSI)-1])>=85):
        rsi_result='Strong Sell'
        rsi_flag=2
    elif((RSI[len(RSI)-1])>=70):
        rsi_result='Sell'
        rsi_flag=3
    elif((RSI[len(RSI)-1])>=50):
        rsi_result='Neutral'
        rsi_flag=1
    elif((RSI[len(RSI)-1])>=20):
        rsi_result='Buy'
        rsi_flag=4
    else:
        rsi_result='Strong Buy'
        rsi_flag=5
    
    rsi_statement=[format(RSI[len(RSI)-1],'.4f'),rsi_result]
  
    # RSI graph generation
    fig = make_subplots(rows=2, cols=1,shared_xaxes=True,vertical_spacing=0)

    fig.append_trace(
        go.Scatter(
            x=data.index,
            y=data['Close'],
            line=dict(color='red', width=1),
            name='Closing Price',
            # showlegend=False,
            legendgroup='1',


        ), row=1, col=1
    )
    fig.append_trace(
        go.Scatter(
            x=data.index,
            y=RSI,
            line=dict(color='#ff9900', width=1),
            name='RSI',
            # showlegend=False,
            legendgroup='1',

        ), row=2, col=1
    )


        # Make it pretty
    layout = go.Layout(
        
        # Font Families
        font_family='Monospace',
        font_color='white',
        font_size=20,
        xaxis=dict(
            rangeslider=dict(
                visible=False
                
            )
            
        )
    )
    fig.add_hline(y=30, line_dash="dot", row=2, col=1, annotation_text="", 
              annotation_position="bottom right")

    fig.add_hline(y=70, line_dash="dot", row=2, col=1, annotation_text="", 
              annotation_position="bottom right")

    # Update options and show plot
    fig.update_layout(layout)
    fig.update_layout(template="plotly_dark")
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    # SMA calculation and graph generation

    SMA50 = data['Close'].rolling(window = 50).mean()
    SMA100 = data['Close'].rolling(window = 100).mean()
    
    fig_SMA = px.line(x=date, y=data['Close'])
    fig_SMA.add_scatter(x=data,y=data['Close'],mode='lines',name="Closing price")
    fig_SMA.update_traces(line_color='red')
    fig_SMA.add_scatter(x=date,y=SMA50, mode='lines',name="SMA50")
    fig_SMA.add_scatter(x=date,y=SMA100, mode='lines',name="SMA100")
    
    fig_SMA.update_layout(template="plotly_dark")

    graph_SMA = json.dumps(fig_SMA, cls=plotly.utils.PlotlyJSONEncoder)

    # EMA Calculation and graph generation

    sma50 = data['Close'].rolling(50).mean()
    sma100 = data['Close'].rolling(100).mean()

    modPrice50 = data['Close'].copy()
    modPrice50.iloc[0:50] = sma50[0:50]

    modPrice100 = data['Close'].copy()
    modPrice100.iloc[0:100] = sma100[0:100]

    ema50 = modPrice50.ewm(span=50, adjust=False).mean()
    ema100 = modPrice100.ewm(span=100, adjust=False).mean()

    # ema and crossovers for longterm ema50,100
    ema50a = ema50.reset_index()
    ema100a = ema100.reset_index()

    crossovers = pd.DataFrame()
    crossovers['Dates'] = ema50a['Date']
    crossovers['Price'] = [i for i in data['Close']]
    crossovers['EMA50'] = ema50a['Close']
    crossovers['EMA100'] = ema100a['Close']
    crossovers['position'] = crossovers['EMA50'] >= crossovers['EMA100']
    crossovers['pre-position'] = crossovers['position'].shift(1)
    crossovers['Crossover'] = np.where(crossovers['position'] == crossovers['pre-position'], False, True)
    crossovers['Crossover'][0] = False

    crossovers = crossovers.loc[crossovers['Crossover'] == True]
    crossovers = crossovers.reset_index()
    crossovers = crossovers.drop(['position', 'pre-position', 'Crossover', 'index'], axis=1)
    crossovers['Signal'] = np.nan
    crossovers['Binary_Signal'] = 0.0
    for i in range(len(crossovers['EMA50'])):
        if crossovers['EMA50'][i] > crossovers['EMA100'][i]:
            crossovers['Binary_Signal'][i] = 1.0
            crossovers['Signal'][i] = 'Buy'
        else:
            crossovers['Signal'][i] = 'Sell'

    sell_df=crossovers[crossovers['Signal'].str.contains('Sell')]
    buy_df=crossovers[crossovers['Signal'].str.contains('Buy')]

    fig_emal = px.line(x=date, y=data['Close'] )
    fig_emal.add_scatter(x=data,y=data['Close'],mode='lines',name="Closing price")
    fig_emal.update_traces(line_color='red')
    fig_emal.add_scatter(x=date,y=ema50, mode='lines',name="ema50")
    fig_emal.add_scatter(x=date,y=ema100, mode='lines',name="ema100")
    fig_emal.update_layout(template="plotly_dark")
    fig_emal.add_traces(go.Scatter(x=buy_df['Dates'], y=buy_df['EMA50'], mode="markers",
    marker=dict(color='lightgreen', symbol='triangle-up',size=15),name="Buy", hoverinfo="skip"))
    fig_emal.add_traces(go.Scatter(x=sell_df['Dates'], y=sell_df['EMA50'], mode="markers",
    marker=dict(color='red', symbol='triangle-down',size=15),name="Sell", hoverinfo="skip"))


    graph_emal = json.dumps(fig_emal, cls=plotly.utils.PlotlyJSONEncoder)

    # ema and crossovers for shortterm ema12,26

    sma12 = data['Close'].rolling(12).mean()
    sma26 = data['Close'].rolling(26).mean()

    modPrice12 = data['Close'].copy()
    modPrice12.iloc[0:12] = sma12[0:12]

    modPrice26 = data['Close'].copy()
    modPrice26.iloc[0:26] = sma26[0:26]

    ema12 = modPrice12.ewm(span=12, adjust=False).mean()
    ema26 = modPrice26.ewm(span=26, adjust=False).mean()

    ema12a = ema12.reset_index()
    ema26a = ema26.reset_index()

    crossovers1 = pd.DataFrame()
    crossovers1['Dates'] = ema12a['Date']
    crossovers1['Price'] = [i for i in data['Close']]
    crossovers1['EMA50'] = ema12a['Close']
    crossovers1['EMA100'] = ema26a['Close']
    crossovers1['position'] = crossovers1['EMA50'] >= crossovers1['EMA100']
    crossovers1['pre-position'] = crossovers1['position'].shift(1)
    crossovers1['Crossover'] = np.where(crossovers1['position'] == crossovers1['pre-position'], False, True)
    crossovers1['Crossover'][0] = False

    crossovers1 = crossovers1.loc[crossovers1['Crossover'] == True]
    crossovers1 = crossovers1.reset_index()
    crossovers1 = crossovers1.drop(['position', 'pre-position', 'Crossover', 'index'], axis=1)
    crossovers1['Signal'] = np.nan
    crossovers1['Binary_Signal'] = 0.0
    for i in range(len(crossovers1['EMA50'])):
        if crossovers1['EMA50'][i] > crossovers1['EMA100'][i]:
            crossovers1['Binary_Signal'][i] = 1.0
            crossovers1['Signal'][i] = 'Buy'
        else:
            crossovers1['Signal'][i] = 'Sell'

    sell1_df=crossovers1[crossovers1['Signal'].str.contains('Sell')]
    buy1_df=crossovers1[crossovers1['Signal'].str.contains('Buy')]

    fig_emal = px.line(x=date, y=data['Close'])
    fig_emal.add_scatter(x=data,y=data['Close'],mode='lines',name="Closing price")
    fig_emal.update_traces(line_color='red')
    fig_emal.add_scatter(x=date,y=ema12, mode='lines',name="ema12")
    fig_emal.add_scatter(x=date,y=ema26, mode='lines',name="ema26")
    
    fig_emal.update_layout(template="plotly_dark")
    fig_emal.add_traces(go.Scatter(x=buy1_df['Dates'], y=buy1_df['EMA50'], mode="markers",
    marker=dict(color='lightgreen', symbol='triangle-up',size=15),name="Buy", hoverinfo="skip"))
    fig_emal.add_traces(go.Scatter(x=sell1_df['Dates'], y=sell1_df['EMA50'], mode="markers",
    marker=dict(color='red', symbol='triangle-down',size=15),name="Sell", hoverinfo="skip"))
    graph_emas = json.dumps(fig_emal, cls=plotly.utils.PlotlyJSONEncoder)

    # MACD calculation and graph generation

    macd=ema12-ema26

    signal_line = macd.ewm(span=9, adjust=False).mean()

    fig_macd = make_subplots(rows=1, cols=1)

        # price Line
    fig_macd.append_trace(
        go.Scatter(
            x=data.index,
            y=macd,
            line=dict(color='red', width=2),
            name='macd',
            # showlegend=False,
            legendgroup='1',

        ), row=1, col=1
    )
    fig_macd.append_trace(
        go.Scatter(
            x=data.index,
            y=signal_line,
            line=dict(color='#ff9900', width=2),
            name='Signal Line',
            # showlegend=False,
            legendgroup='1',

        ), row=1, col=1
    )


    colors = np.where(macd-signal_line < 0, 'red', 'green')

    fig_macd.append_trace(
    go.Bar(
        x=data.index,
        y=macd-signal_line,
        showlegend=False,
        marker_color=colors,

    ), row=1, col=1
)

    layout = go.Layout(
    
        # Font Families
        font_family='Monospace',
        font_color='white',
        font_size=20,
        xaxis=dict(
            rangeslider=dict(
                visible=False
            )
        )
    )

    fig_macd.update_layout(layout)
    fig_macd.update_layout(template="plotly_dark")
    graph_macd = json.dumps(fig_macd, cls=plotly.utils.PlotlyJSONEncoder)

    # Advisory Logic

    # Calculating other ema pairs for main logic
    sma2 = data['Close'].rolling(window = 2).mean()
    sma5 = data['Close'].rolling(window = 5).mean()
    sma9 = data['Close'].rolling(window = 9).mean()
    sma13 = data['Close'].rolling(window = 13).mean()
    sma21 = data['Close'].rolling(window = 21).mean()

    modPrice2 = data['Close'].copy()
    modPrice2.iloc[0:2] = sma2[0:2]

    modPrice5 = data['Close'].copy()
    modPrice5.iloc[0:5] = sma5[0:5]

    modPrice9 = data['Close'].copy()
    modPrice9.iloc[0:9] = sma9[0:9]

    modPrice13 = data['Close'].copy()
    modPrice13.iloc[0:13] = sma13[0:13]

    modPrice21 = data['Close'].copy()
    modPrice21.iloc[0:21] = sma21[0:21]


    ema2 = modPrice2.ewm(span=2, adjust=False).mean()
    ema5 = modPrice5.ewm(span=5, adjust=False).mean()
    ema9 = modPrice9.ewm(span=9, adjust=False).mean()
    ema13 = modPrice13.ewm(span=13, adjust=False).mean()
    ema21 = modPrice21.ewm(span=21, adjust=False).mean()

    # crossover2 = ema5,13
    ema5a = ema5.reset_index()
    ema13a = ema13.reset_index()
    crossovers2 = pd.DataFrame()
    crossovers2['Dates'] = ema5a['Date']
    crossovers2['Price'] = [i for i in data['Close']]
    crossovers2['EMA50'] = ema5a['Close']
    crossovers2['EMA100'] = ema13a['Close']
    crossovers2['position'] = crossovers2['EMA50'] >= crossovers2['EMA100']
    crossovers2['pre-position'] = crossovers2['position'].shift(1)
    crossovers2['Crossover'] = np.where(crossovers2['position'] == crossovers2['pre-position'], False, True)
    crossovers2['Crossover'][0] = False

    crossovers2 = crossovers2.loc[crossovers2['Crossover'] == True]
    crossovers2 = crossovers2.reset_index()
    crossovers2 = crossovers2.drop(['position', 'pre-position', 'Crossover', 'index'], axis=1)
    crossovers2['Signal'] = np.nan
    crossovers2['Binary_Signal'] = 0.0
    for i in range(len(crossovers2['EMA50'])):
        if crossovers2['EMA50'][i] > crossovers2['EMA100'][i]:
            crossovers2['Binary_Signal'][i] = 1.0
            crossovers2['Signal'][i] = 'Buy'
        else:
            crossovers2['Signal'][i] = 'Sell'

    # crossoer3 = ema9,21  
    ema9a = ema9.reset_index()
    ema21a = ema21.reset_index()
    crossovers3 = pd.DataFrame()
    crossovers3['Dates'] = ema9a['Date']
    crossovers3['Price'] = [i for i in data['Close']]
    crossovers3['EMA50'] = ema9a['Close']
    crossovers3['EMA100'] = ema21a['Close']
    crossovers3['position'] = crossovers3['EMA50'] >= crossovers3['EMA100']
    crossovers3['pre-position'] = crossovers3['position'].shift(1)
    crossovers3['Crossover'] = np.where(crossovers3['position'] == crossovers3['pre-position'], False, True)
    crossovers3['Crossover'][0] = False

    crossovers3 = crossovers3.loc[crossovers3['Crossover'] == True]
    crossovers3 = crossovers3.reset_index()
    crossovers3 = crossovers3.drop(['position', 'pre-position', 'Crossover', 'index'], axis=1)
    crossovers3['Signal'] = np.nan
    crossovers3['Binary_Signal'] = 0.0
    for i in range(len(crossovers3['EMA50'])):
        if crossovers3['EMA50'][i] > crossovers3['EMA100'][i]:
            crossovers3['Binary_Signal'][i] = 1.0
            crossovers3['Signal'][i] = 'Buy'
        else:
            crossovers3['Signal'][i] = 'Sell'
            
    # crossoer4 = ema2,5  
    ema2a = ema2.reset_index()
    ema5a = ema5.reset_index()
    crossovers4 = pd.DataFrame()
    crossovers4['Dates'] = ema2a['Date']
    crossovers4['Price'] = [i for i in data['Close']]
    crossovers4['EMA50'] = ema2a['Close']
    crossovers4['EMA100'] = ema5a['Close']
    crossovers4['position'] = crossovers4['EMA50'] >= crossovers4['EMA100']
    crossovers4['pre-position'] = crossovers4['position'].shift(1)
    crossovers4['Crossover'] = np.where(crossovers4['position'] == crossovers4['pre-position'], False, True)
    crossovers4['Crossover'][0] = False

    crossovers4 = crossovers4.loc[crossovers4['Crossover'] == True]
    crossovers4 = crossovers4.reset_index()
    crossovers4 = crossovers4.drop(['position', 'pre-position', 'Crossover', 'index'], axis=1)
    crossovers4['Signal'] = np.nan
    crossovers4['Binary_Signal'] = 0.0
    for i in range(len(crossovers4['EMA50'])):
        if crossovers4['EMA50'][i] > crossovers4['EMA100'][i]:
            crossovers4['Binary_Signal'][i] = 1.0
            crossovers4['Signal'][i] = 'Buy'
        else:
            crossovers4['Signal'][i] = 'Sell'



    # Checking latest signal for all the ema pairs

    prev_signal_ema2_5 = crossovers4['Signal'][len(crossovers4)-1]
    prev_signal_ema5_13 = crossovers2['Signal'][len(crossovers2)-1]
    prev_signal_ema9_21 = crossovers3['Signal'][len(crossovers3)-1]
    prev_signal_ema12_26 = crossovers1['Signal'][len(crossovers1)-1]
    prev_signal_ema50_100 = crossovers['Signal'][len(crossovers)-1]

    prev_signal_date_ema2_5=crossovers4['Dates'][len(crossovers4)-1]
    prev_signal_date_ema5_13=crossovers2['Dates'][len(crossovers2)-1]
    prev_signal_date_ema9_21=crossovers3['Dates'][len(crossovers3)-1]
    prev_signal_date_ema12_26=crossovers1['Dates'][len(crossovers1)-1]
    prev_signal_date_ema50_100=crossovers['Dates'][len(crossovers)-1]

    def flag_ema(var):
        if(var=='Sell'):
            flag=0
        elif(var=='Buy'):
            flag=1
        return(flag)

    def flg(a,b):
        if(a>b):
            temp=1
        else:
            temp=0
        return(temp)

    # code for closing price strategy
    close_flag2=flg(data['Close'][len(data)-1],ema2[len(ema2)-1])
    close_flag5=flg(data['Close'][len(data)-1],ema5[len(ema5)-1])
    close_flag9=flg(data['Close'][len(data)-1],ema9[len(ema9)-1])
    close_flag12=flg(data['Close'][len(data)-1],ema12[len(ema12)-1])
    close_flag50=flg(data['Close'][len(data)-1],ema50[len(ema50)-1])

    # code for previous signal strategy
    flag2_5=flag_ema(prev_signal_ema2_5)
    flag5_13=flag_ema(prev_signal_ema5_13)
    flag9_21=flag_ema(prev_signal_ema9_21)
    flag12_26=flag_ema(prev_signal_ema12_26)
    flag50_100=flag_ema(prev_signal_ema50_100)

    def flg1(a,b):
        if(a==1 and b==1):
            temp=1
        else:
            temp=0
        return(temp)

    # combined strategy for ema
    flag2_5=flg1(flag2_5,close_flag2)
    flag5_13=flg1(flag5_13,close_flag5)
    flag9_21=flg1(flag9_21,close_flag9)
    flag12_26=flg1(flag12_26,close_flag12)
    flag50_100=flg1(flag50_100,close_flag50)

    # EMA result calculation using all ema pairs
    ema_flg= (0.20*flag2_5)+(0.30*flag5_13)+(0.25*flag9_21)+(0.15*flag12_26)+(0.10*flag50_100)
    ema_flag=0
    if(ema_flg<0.15):
        ema_res='Neutral'
        ema_flag=1
    elif(ema_flg<=0.35):
        ema_res='Weak Buy'
        ema_flag=2
    elif(ema_flg<0.65):
        ema_res='Buy'
        ema_flag=3
    elif(ema_flg>=0.65):
        ema_res='Strong Buy'
        ema_flag=4
    
    ema_statement= ema_res    
    
    # MACD result calculation
    macda = macd.reset_index()
    signal_linea = signal_line.reset_index()
    crossovers5 = pd.DataFrame()
    crossovers5['Dates'] = macda['Date']
    crossovers5['Price'] = [i for i in data['Close']]
    crossovers5['EMA50'] = macda['Close']
    crossovers5['EMA100'] = signal_linea['Close']
    crossovers5['position'] = crossovers5['EMA50'] >= crossovers5['EMA100']
    crossovers5['pre-position'] = crossovers5['position'].shift(1)
    crossovers5['Crossover'] = np.where(crossovers5['position'] == crossovers5['pre-position'], False, True)
    crossovers5['Crossover'][0] = False

    crossovers5 = crossovers5.loc[crossovers5['Crossover'] == True]
    crossovers5 = crossovers5.reset_index()
    crossovers5 = crossovers5.drop(['position', 'pre-position', 'Crossover', 'index'], axis=1)
    crossovers5['Signal'] = np.nan
    crossovers5['Binary_Signal'] = 0.0
    for i in range(len(crossovers5['EMA50'])):
        if crossovers5['EMA50'][i] > crossovers5['EMA100'][i]:
            crossovers5['Binary_Signal'][i] = 1.0
            crossovers5['Signal'][i] = 'Buy'
        else:
            crossovers5['Signal'][i] = 'Sell'

    prev_cross_macd=crossovers5['Signal'][len(crossovers5)-1]

    macd_statement= prev_cross_macd

    macd_flag=crossovers5['Binary_Signal'][len(crossovers5)-1]

    # flags for all indicators to calculate overall result
    if(rsi_flag==1):
        final_rsi_flag=0
    elif(rsi_flag==2):
        final_rsi_flag=-0.30
    elif(rsi_flag==3):
        final_rsi_flag=-0.15
    elif(rsi_flag==4):
        final_rsi_flag=0.15
    elif(rsi_flag==5):
        final_rsi_flag=0.30

    if(ema_flag==1):
        final_ema_flag=0
    elif(ema_flag==2):
        final_ema_flag=0.20
    elif(ema_flag==3):
        final_ema_flag=0.35
    elif(ema_flag==4):
        final_ema_flag=0.50

    if(macd_flag==0):
        final_macd_flag=-0.10
    elif(macd_flag==1):
        final_macd_flag=0.10

    king_of_flags= final_rsi_flag + final_macd_flag + final_ema_flag


    # overall result calculation
    if(king_of_flags>=0.50):
        final_result='Buy'
    else:
        final_result='Sell'
         
    print(final_result)
    print(king_of_flags)
    return render_template('advisorycoin.html',ress=final_result,graphJSON=graphJSON,r1=rsi_statement,graph_SMA=graph_SMA,graph_emal=graph_emal,r2=ema_statement,graph_emas=graph_emas,graph_macd=graph_macd,r3=macd_statement)


if __name__ == '__main__':
    app.run(debug=True)

