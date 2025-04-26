import re

import robin_stocks.robinhood as r
import pandas as pd
import numpy as np
from scipy import stats
import requests
import json
from textblob import TextBlob
import time

from sec_edgar_downloader import Downloader
from bs4 import BeautifulSoup
import os
import requests


r.authentication.login(username='', password='', expiresIn=86400, scope='internal',
                        store_session=True, pickle_name="", by_sms=False, mfa_code=)

headers = {'Accept': '*/*',
           'Accept-Language': 'en-US,en;q=0.5',
           'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/85.0.4183.83 Safari/537.36 Edg/85.0.564.41',
           'X-Requested-With': 'XMLHttpRequest'}


def get_financial_data_from_edgar(ticker, num_years=2):
    dl = Downloader(download_folder='./edgar_data', email_address='', company_name='')

    # Download the latest 10-K filings
    dl.get("10-K", ticker, limit=num_years)

    financial_data = []

    # Path where the files are saved
    base_path = f"./edgar_data/sec-edgar-filings/000{ticker}/10-K"

    # Get a list of all directories in the base path, sorted by date (newest first)
    dirs = sorted([d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))], reverse=True)

    for i, dir_name in enumerate(dirs[:num_years]):
        file_path = os.path.join(base_path, dir_name, "full-submission.txt")

        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()

        # Find the URL of the iXBRL file
        #ixbrl_url_match = re.search(r'<TYPE>10-K.*<FILENAME>(.*\.htm)', content, re.IGNORECASE)
        ixbrl_url_match = re.search(r'<FILENAME>(.*)', content, re.IGNORECASE)
        if ixbrl_url_match:
            ixbrl_filename = ixbrl_url_match.group(1)
            ixbrl_url = f"https://www.sec.gov/Archives/{ixbrl_filename}"

            # Download and parse the iXBRL file
            ixbrl_data = parse_ixbrl(ixbrl_url)
            financial_data.append(ixbrl_data)

    return financial_data


def parse_ixbrl(url):
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.content, 'html.parser')

    # Define the tags we're interested in
    tags_of_interest = {
        'Revenue': ['Revenues', 'SalesRevenueNet', 'SalesRevenueGoodsNet'],
        'NetIncome': ['NetIncomeLoss', 'ProfitLoss'],
        'Equity': ['StockholdersEquity', 'PartnersCapital']
    }

    data = {}

    for category, possible_tags in tags_of_interest.items():
        for tag in possible_tags:
            # Look for both ix:nonFraction and ix:nonNumeric tags
            #element = soup.find(['ix:nonfraction', 'ix:nonnumeric'], name=re.compile(tag, re.I))
            element = soup.find('ix:nonfraction', name=re.compile(tag, re.I))
            if element:
                # Extract the value and convert to float if possible
                value = element.get('value') or element.text.strip()
                try:
                    data[category] = float(value)
                except ValueError:
                    data[category] = value
                break  # Stop looking for alternative tags if we found one

    # Extract the context date
    context_ref = element.get('contextref') if element else None
    if context_ref:
        context = soup.find('xbrli:context', id=context_ref)
        if context:
            instant = context.find('xbrli:instant')
            if instant:
                data['Date'] = instant.text.strip()

    return data


def calculate_yoy_revenue_growth(financial_data):
    if len(financial_data) < 2:
        return None

    current_year = financial_data[0]
    previous_year = financial_data[1]

    if 'Revenue' not in current_year or 'Revenue' not in previous_year:
        return None

    return (current_year['Revenue'] - previous_year['Revenue']) / previous_year['Revenue']


def calculate_roe(financial_data):
    if not financial_data:
        return None

    current_year = financial_data[0]

    if 'NetIncome' not in current_year or 'Equity' not in current_year:
        return None

    return current_year['NetIncome'] / current_year['Equity']


def get_cik_from_ticker(ticker):
    url = f"https://www.sec.gov/include/ticker.txt"
    response = requests.get(url, headers=headers)
    for line in response.text.split('\n'):
        if line.startswith(ticker.lower()):
            return line.split('\t')[1]
    return None


def get_growth_stocks(sector, min_market_cap=1e9, min_revenue_growth=0.1, min_roe=0.15, max_pe=50):
    # Get list of S&P 500 stocks (you may need to update this URL or use a different data source)
    stocks = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
    sector_stocks = stocks[stocks['GICS Sector'] == sector]['Symbol'].tolist()

    growth_stocks = []
    for ticker in sector_stocks:
        # Get stock fundamentals from Robinhood
        fundamentals = r.stocks.get_fundamentals(ticker)[0]

        try:
            market_cap = float(fundamentals['market_cap'])
            pe_ratio = float(fundamentals['pe_ratio'])
        except (KeyError, ValueError, TypeError):
            continue

        # Get CIK for the ticker
        cik = get_cik_from_ticker(ticker)
        if not cik:
            continue

        # Get financial data from EDGAR
        financial_data = get_financial_data_from_edgar(cik)

        # Calculate revenue growth and ROE
        revenue_growth = calculate_yoy_revenue_growth(financial_data)
        roe = calculate_roe(financial_data)

        if revenue_growth is None or roe is None:
            continue

        # Apply growth criteria
        if (market_cap > min_market_cap and
                revenue_growth > min_revenue_growth and
                roe > min_roe and
                pe_ratio < max_pe):
            growth_stocks.append(ticker)

    return growth_stocks

def calculate_portfolio_risk(portfolio, lookback_days=252):
    historicals = {}
    for ticker in portfolio:
        historical_data = r.stocks.get_stock_historicals(ticker, interval='day', span='year')
        closes = [float(day['close_price']) for day in historical_data]
        historicals[ticker] = closes[-lookback_days:]  # Use last 252 trading days

    df = pd.DataFrame(historicals)

    # Calculate daily returns
    returns = df.pct_change()

    # Calculate covariance matrix
    cov_matrix = returns.cov()

    # Assume equal weighting for simplicity
    weights = np.array([1 / len(portfolio)] * len(portfolio))

    # Calculate portfolio volatility
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)

    return portfolio_volatility


def get_market_sentiment():
    # Using NewsAPI to get latest stock market news
    url = ('https://newsapi.org/v2/everything?'
           'q=stock+market&'
           'sortBy=publishedAt&'
           'apiKey=67de737394af42c58135609ba935e500')

    response = requests.get(url)
    articles = json.loads(response.text)['articles']

    sentiments = []
    for article in articles[:10]:  # Analyze sentiment of top 10 articles
        blob = TextBlob(article['title'] + " " + article['description'])
        sentiments.append(blob.sentiment.polarity)

    return np.mean(sentiments)


def get_macro_economic_risks():
    # This is a simplified example. In practice, you'd want to use a more comprehensive economic data API
    fed_url = "https://api.stlouisfed.org/fred/series/observations?series_id=FEDFUNDS&api_key=c441c6fec2eeaa578f17f5e714d7a285&file_type=json"
    vix_url = "https://www.alphavantage.co/query?function=MARKET_STATUS&apikey=8HRR9YAUSS00V613"

    fed_response = requests.get(fed_url)
    fed_rate = float(json.loads(fed_response.text)['observations'][-1]['value'])

    vix_response = requests.get(vix_url)
    vix = float(json.loads(vix_response.text)['VIX'])

    # Simplified risk assessment
    if fed_rate > 2.5 and vix > 20:
        return "High"
    elif fed_rate > 1.5 or vix > 15:
        return "Medium"
    else:
        return "Low"


def main():
    # Find growth stocks in technology sector
    tech_growth_stocks = get_growth_stocks('Information Technology')
    print("Growth Stocks:", tech_growth_stocks)

    # Calculate portfolio risk
    portfolio_risk = calculate_portfolio_risk(tech_growth_stocks)
    print(f"Portfolio Risk (Annualized Volatility): {portfolio_risk:.2%}")

    # Get market sentiment
    sentiment = get_market_sentiment()
    print(f"Market Sentiment: {sentiment:.2f} (-1 to 1 scale)")

    # Get macro-economic risks
    macro_risks = get_macro_economic_risks()
    print(f"Macro-economic Risk Level: {macro_risks}")

    # Implement stop-loss (example)
    stop_loss_percentage = 0.1  # 10% stop-loss
    print(f"Stop-loss set at {stop_loss_percentage:.0%} for each position")

    # Real-time monitoring (simplified example)
    while True:
        for ticker in tech_growth_stocks:
            current_price = float(r.stocks.get_latest_price(ticker)[0])
            quote = r.stocks.get_quotes(ticker)[0]
            avg_buy_price = float(quote['average_buy_price'])

            if current_price <= avg_buy_price * (1 - stop_loss_percentage):
                print(f"Stop-loss triggered for {ticker}. Selling position.")
                # r.orders.order_sell_market(ticker, 1)  # Uncomment to actually place sell order

        time.sleep(60)  # Check every minute


if __name__ == "__main__":
    main()
