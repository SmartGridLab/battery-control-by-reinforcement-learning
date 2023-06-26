# このコードは、CSVファイル(weather_data_bid.csv)の日付が翌々日であり、CSVファイル内の時間が30分ごとに増加していることを検証します。
# This is the unit test code for `weather_data_bid.py`
# - This test check if the datetime in the obtained csv file starts from 12AM day after tomorrow. 
# - Reason: `weather_data_bid.py` should generates the weather data for bidding for the day after tomorrow.

import pandas as pd
import datetime

# Read the CSV file
weather_data = pd.read_csv('weather_data_bid.csv')

# Get today's date
today = datetime.date.today()

# Calculate the day after tomorrow's date
day_after_tomorrow = today + datetime.timedelta(days=2)

# Convert day_after_tomorrow to string in the format yyyy-mm-dd
day_after_tomorrow_str = day_after_tomorrow.strftime("%Y-%m-%d")

# Extract the date portion from the CSV's datetime
weather_data['date'] = weather_data[['year', 'month', 'day']].apply(lambda row: '-'.join(row.values.astype(str)), axis=1)

# Assert that all dates in the CSV are the day after tomorrow
assert weather_data['date'].nunique() == 1, f"Multiple dates found in CSV, expected only one date: {day_after_tomorrow_str}"
assert weather_data['date'].unique()[0] == day_after_tomorrow_str, f"Date {weather_data['date'].unique()[0]} in CSV is not the day after tomorrow: {day_after_tomorrow_str}"

# Convert the fractional hours to datetime.time format
def convert_to_time(fractional_hour):
    hours = int(fractional_hour)
    minutes = int((fractional_hour - hours) * 60)
    return datetime.time(hour=hours, minute=minutes)

weather_data['time'] = weather_data['hour'].apply(convert_to_time)

# Then create a list of expected times starting from 00:00 to 23:30 with a 30 min interval
expected_times = [datetime.time(hour, minute) for hour in range(24) for minute in [0, 30]]

# Now check if all these expected_times exist in the 'time' column of the DataFrame
missing_times = [time for time in expected_times if time not in weather_data['time'].tolist()]

# If any expected time is missing in the data, raise an AssertionError
assert not missing_times, f"The following times are missing in the data: {missing_times}"


