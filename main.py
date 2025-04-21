import robin_stocks.robinhood as r
import pandas as pd
import numpy as np
from scipy import stats
import requests
import json
from textblob import TextBlob
import time

r.authentication.login(username='xyz@abc.com', password='', expiresIn=86400, scope='internal',
                        store_session=True, pickle_name="rr", by_sms=False, mfa_code=)


def get_growth_stocks(sector, min_market_cap=1e9, min_revenue_growth=0.1, min_roe=0.15, max_pe=50):
    # Get list of S&P 500 stocks (you may need to update this URL or use a different data source)
    stocks = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
    sector_stocks = stocks[stocks['GICS Sector'] == sector]['Symbol'].tolist()

    growth_stocks = []
    for ticker in sector_stocks:
        # Get stock fundamentals
        fundamentals = r.stocks.get_fundamentals(ticker)[0]

        # Get financial data
        try:
            market_cap = float(fundamentals['market_cap'])
            revenue_growth = float(fundamentals['year_over_year_revenue_growth'])
            roe = float(fundamentals['return_on_equity'])
            pe_ratio = float(fundamentals['pe_ratio'])
        except (KeyError, ValueError, TypeError):
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
