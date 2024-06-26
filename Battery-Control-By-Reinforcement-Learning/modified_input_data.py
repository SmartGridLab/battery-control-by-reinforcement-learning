import pandas as pd

# CSVファイルのパス
file_path = 'Battery-Control-By-Reinforcement-Learning/input_data2022.csv'

# CSVファイルの読み込み
df = pd.read_csv(file_path)

# hourが24の行を探し、hourを0に変更し、dayを1増やす
df.loc[df['hour'] == 24, 'hour'] = 0
df.loc[df['hour'] == 0, 'day'] += 1

# 月の日数を考慮してdayを調整する関数
def adjust_date(row):
    if row['day'] > 31:
        row['day'] = 1
        row['month'] += 1
    if row['month'] > 12:
        row['month'] = 1
        row['year'] += 1
    return row

# 全ての行に対して日付を調整
df = df.apply(adjust_date, axis=1)

# 新しいCSVファイルに保存
output_file_path = 'Battery-Control-By-Reinforcement-Learning/modified_data2022.csv'
df.to_csv(output_file_path, index=False)

print(f"Modified data has been saved to {output_file_path}")