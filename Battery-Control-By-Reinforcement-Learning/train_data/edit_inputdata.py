import pandas as pd

df = pd.read_csv("Battery-Control-By-Reinforcement-Learning/input_data2022.csv")

# get the maximum and minimum values 
def get_maxmin_value(df, colmun_name):
    max_value = df[colmun_name].max()
    min_value = df[colmun_name].min()
    return max_value, min_value

def descr_value(df, column_name1, column_name2, column_name3):
    PVout_max, PVout_min = get_maxmin_value(df, column_name1)
    price_max, price_min = get_maxmin_value(df, column_name2)
    imbalance_max, imbalance_min = get_maxmin_value(df, column_name3)
    print("PVout_max: ", PVout_max)
    print("PVout_min: ", PVout_min)
    print("price_max: ", price_max)
    print("price_min: ", price_min)
    print("imbalance_max: ", imbalance_max)
    print("imbalance_min: ", imbalance_min)
    return PVout_max, PVout_min, price_max, price_min, imbalance_max, imbalance_min

# 無効な日付を修正する関数
def fix_valid_date(row):
    try:
        pd.to_datetime(f"{int(row['year'])}-{int(row['month'])}-{int(row['day'])}")
        return row
    except ValueError:
        print(f"Invalid date: {int(row['year'])}-{int(row['month'])}-{int(row['day'])}-{row['hour']}")
        # 12月31日の場合、次の年の1月1日にする
        if row["month"] == 12:
            row["year"] += 1
            row["month"] = 1
            row["day"] = 1
        else:
        # 12月以外の場合、次の月の1日にする
            row["month"] += 1
            
        row["day"] = 1
        return row

# 無効な日付が存在しないか確認する関数
def is_valid_date(row):
    try:
        pd.to_datetime(f"{int(row['year'])}-{int(row['month'])}-{int(row['day'])}")
        return True
    except ValueError:
        return False
# 1日のデータが48個に満たない日を探す関数
def find_incomplete_days(df):
    df["date"] = pd.to_datetime(df[["year", "month", "day"]])
    grouped = df.groupby('date')
    incomplete_days = [date for date, group in grouped if len(group) != 48]
    return incomplete_days


# show the maximum and minimum values of essential columns
PVout_max, PVout_min, price_max, price_min, imbalance_max, imbalance_min = descr_value(df, "PVout", "price", "imbalance")
# fix negative values (PVout is not negative)
negative_counts = (df["PVout"] < 0).sum()
print("修正前のPVoutが負になっている個数:", negative_counts)
# negative values are replaced with 0
df["PVout"] = df["PVout"].apply(lambda x: 0 if x < 0 else x)
negative_counts = (df["PVout"] < 0).sum()
print("修正後のPVoutが負になっている個数:", negative_counts)

# fix invalid dates
df = df.apply(fix_valid_date, axis = 1)
# Recheck invalid dates
invalid_dates = df[~df.apply(is_valid_date, axis = 1)]
if invalid_dates.empty:
    print("There are no invalid dates and the data is saved.")
else:
    print("Invalid dates: ")
    print(invalid_dates)

# find incomplete days
incomplete_days = find_incomplete_days(df)
while incomplete_days:
    print("Incomplete days (not equal to 48 data):")
    for day in incomplete_days:
        print("48個データがない日:", day)
    df = df[~df["date"].isin(incomplete_days)]
    incomplete_days = find_incomplete_days(df)
    if not incomplete_days:
        print("All days have 48 data.")
        break

# df["date"] = pd.to_datetime(df[['year', 'month', 'day']])
# ##-------------変更する期間----------------##
# start_date = "2022-06-01"
# end_date = "2022-06-02"
# ##----------------------------------------##
# mask = (df["date"] >= start_date) & (df["date"] <= end_date)
# ##--------------指定した期間の値を編集------##
# df.loc[mask, "PVout"] = 0.0
# df.loc[mask, "price"] = 0.01
# df.loc[mask, "imbalance"] = 200
# ##----------------------------------------##

print(len(df))
print(df["year"].iloc[0], df["month"].iloc[0], df["day"].iloc[0], df["hour"].iloc[0])
print(df["year"].iloc[17424], df["month"].iloc[17424], df["day"].iloc[17424], df["hour"].iloc[17424])

df.to_csv("Battery-Control-By-Reinforcement-Learning/train_data/input_data2022_edited.csv", index=False)