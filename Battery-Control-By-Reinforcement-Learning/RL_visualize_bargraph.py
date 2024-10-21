import pandas as pd
import matplotlib.pyplot as plt

def main():
    # Load the CSV file
    df = pd.read_csv("Battery-Control-By-Reinforcement-Learning/result_dataframe.csv")

    # Convert date columns to datetime
    df['date'] = pd.to_datetime(df[['year', 'month', 'day']])

    # Sum total profits by day
    daily_profits = df.groupby('date').agg({
        'totalprofit_bid[Yen]': 'sum',
        'totalprofit_realtime[Yen]': 'sum',
        'totalprofit_actual_bid[Yen]': 'sum',
        'totalprofit_actual_realtime[Yen]': 'sum'
    }).dropna()

    # Plot the total daily profits
    daily_profits.plot(kind='bar', figsize=(15, 7))
    plt.xlabel('Date')
    plt.ylabel('Total Profit (Yen)')
    plt.title('Total Daily Profit')
    plt.legend(loc = "lower left", title='Profit Type')
     # Save the plot as a PDF file
    plt.savefig("Battery-Control-By-Reinforcement-Learning/total_daily_profit.png")
    # Save the plot as a PNG file
    plt.savefig("Battery-Control-By-Reinforcement-Learning/total_daily_profit.png")
    plt.show()

if __name__ == "__main__":
    main()
