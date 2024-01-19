import pandas as pd
import sqlite3

class BatteryControlDatabase:
    def __init__ (self, csv_file, dbname):
        # CSVファイルの読み込み
        df = pd.read_csv(csv_file)

        # 列名のスペースをアンダースコアに置換
        df.columns = df.columns.str.replace(' ', '_')

        # データベースに接続
        # dbname = 'battery_control.db'
        conn = sqlite3.connect(dbname)
        cursor = conn.cursor()

        # テーブルを作成
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS battery_control (
            PVout REAL,
            price REAL,
            imbalance REAL,
            yearSin REAL,
            yearCos REAL,
            monthSin REAL,
            monthCos REAL,
            hourSin REAL,
            hourCos REAL,
            PV_predict_bid REAL, 
            PV_predict_realtime REAL, 
            PV_actual REAL, 
            energyprice_predict_bid REAL, 
            energyprice_predict_realtime REAL, 
            energyprice_actual REAL, 
            imbalanceprice_predict_bid REAL, 
            imbalanceprice_predict_realtime REAL, 
            imbalanceprice_actual REAL, 
            charge_discharge_bid REAL, 
            charge_discharge_realtime REAL, 
            charge_discharge_actual_bid REAL, 
            charge_discharge_actual_realtime REAL, 
            SoC_bid REAL, 
            SoC_realtime REAL, 
            SoC_actual_bid REAL, 
            SoC_actual_realtime REAL, 
            energytransfer_bid REAL, 
            energytransfer_realtime REAL, 
            energytransfer_actual_bid REAL, 
            energytransfer_actual_realtime REAL, 
            energyprofit_bid REAL, 
            energyprofit_realtime REAL, 
            imbalancepenalty_bid REAL, 
            imbalancepenalty_realtime REAL, 
            imbalancepenalty_actual_bid REAL, 
            imbalancepenalty_actual_realtime REAL, 
            totalprofit_bid REAL, 
            totalprofit_realtime REAL, 
            totalprofit_actual_bid REAL, 
            totalprofit_actual_realtime REAL, 
            mode TEXT, 
            mode_realtime TEXT
        )
        ''')

        # データフレームの内容をSQLテーブルに挿入
        df.to_sql('battery_control', conn, if_exists='append', index=False)

        # データベース接続を閉じる
        conn.close()

    def update_data(self, data, dbname):
        # データベースに接続
        # dbname = 'battery_control.db'
        conn = sqlite3.connect(dbname)
        cursor = conn.cursor()
        # pandasのdataframe形式であるdataからyear, month, day, hourを取得
        year = int(data.loc[0, "year"])
        month = int(data.loc[0, "month"])
        day = int(data.loc[0, "day"])
        hour = int(data.loc[0, "hour"])

        # データをデータベースに更新するSQL文を作成
        update_sql = '''UPDATE battery_control
                        SET column1 = ?, column2 = ?, ...
                        WHERE year = ? AND month = ? AND day = ? AND hour = ?;'''

        # データを挿入
        cursor.execute(insert_sql, data)

        # 変更をコミットし、接続を閉じる
        conn.commit()
        conn.close()