import os
from fredapi import Fred
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta


class EconomicIndicatorsFetcher:
    def __init__(self, api_key):
        """
        Initialize the Economic Indicators Fetcher with FRED API key

        :param api_key: Your FRED API key (get one at https://fred.stlouisfed.org/docs/api/fred/)
        """
        self.fred = Fred(api_key=api_key)

    def fetch_indicator(self, series_id, start_date=None, end_date=None):
        """
        Fetch a specific economic indicator from FRED

        :param series_id: FRED series ID for the economic indicator
        :param start_date: Start date for data retrieval (optional)
        :param end_date: End date for data retrieval (optional)
        :return: Pandas DataFrame with the indicator data
        """
        # If no dates provided, default to last 5 years
        if start_date is None:
            start_date = datetime.now() - timedelta(days=5 * 365)
        if end_date is None:
            end_date = datetime.now()

        try:
            # Fetch the data
            data = self.fred.get_series(series_id, start_date, end_date)
            return pd.DataFrame(data, columns=[series_id])
        except Exception as e:
            print(f"Error fetching {series_id}: {e}")
            return None

    def fetch_multiple_indicators(self, series_ids):
        """
        Fetch multiple economic indicators

        :param series_ids: List of FRED series IDs
        :return: Pandas DataFrame with multiple indicators
        """
        indicators = {}
        for series_id in series_ids:
            indicator = self.fetch_indicator(series_id)
            if indicator is not None:
                indicators[series_id] = indicator[series_id]

        return pd.DataFrame(indicators)

    def plot_indicators(self, indicators_df, title='Economic Indicators'):
        """
        Plot the fetched economic indicators

        :param indicators_df: DataFrame with economic indicators
        :param title: Title of the plot
        """
        plt.figure(figsize=(12, 6))
        for column in indicators_df.columns:
            plt.plot(indicators_df.index, indicators_df[column], label=column)

        plt.title(title)
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


def main():
    # Replace 'YOUR_API_KEY' with your actual FRED API key
    api_key = os.getenv('FRED_API_KEY', 'xxxxxxxxxxxx')

    # Create an instance of the fetcher
    fetcher = EconomicIndicatorsFetcher(api_key)

    # Example economic indicators (FRED series IDs)
    indicators = [
        'GDP',  # Gross Domestic Product
        'UNRATE',  # Unemployment Rate
        'CPIAUCSL',  # Consumer Price Index for All Urban Consumers
        'FEDFUNDS',  # Federal Funds Effective Rate
    ]

    # Fetch indicators
    economic_data = fetcher.fetch_multiple_indicators(indicators)

    # Print the data
    print(economic_data)

    # Plot the indicators
    fetcher.plot_indicators(economic_data, 'US Economic Indicators')


if __name__ == '__main__':
    main()

# Note: Before running this script:
# 1. Install required libraries:
#    pip install fredapi pandas matplotlib
# 2. Get a free API key from https://fred.stlouisfed.org/docs/api/fred/
# 3. Set your API key as an environment variable FRED_API_KEY
#    or replace 'YOUR_API_KEY' in the script
