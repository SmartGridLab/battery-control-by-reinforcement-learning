# このコードは、CSVファイル(weather_data_bid.csv)の日付が翌日であり、CSVファイル内の時間が30分ごとに増加していることを検証します。
# This is the unit test code for `weather_data_bid.py`
# - This test check if the datetime in the obtained csv file starts from 12AM tomorrow. 
# - Reason: `weather_data_bid.py` should generates the weather data for bidding for tomorrow.

# Read the CSV file
weather_data = pd.read_csv('./Battery-Control-By-Reinforcement-Learning/weather_data_bid.csv')

# Convert the date and hour columns to datetime format
weather_data['datetime'] = pd.to_datetime(weather_data[['year', 'month', 'day', 'hour']])

# Get today's date
today = pd.Timestamp.now()

# Calculate tomorrow's date
tomorrow = today + pd.DateOffset(days=1)

# Assert that all dates in the CSV are tomorrow
assert weather_data['datetime'].dt.date.nunique() == 1, f"Multiple dates found in CSV, expected only one date: {tomorrow.date()}"
assert weather_data['datetime'].dt.date.unique()[0] == tomorrow.date(), f"Date {weather_data['datetime'].dt.date.unique()[0]} in CSV is not tomorrow: {tomorrow.date()}"

# Create a date_range of expected times starting from 00:00 to 23:30 with a 30 min interval
expected_times = pd.date_range(start='00:00', end='23:59', freq='30min').time

# Now check if all these expected_times exist in the 'time' column of the DataFrame
missing_times = [time for time in expected_times if time not in weather_data['datetime'].dt.time.unique()]

# If any expected time is missing in the data, raise an AssertionError
assert not missing_times, f"The following times are missing in the data: {missing_times}"




