# databaseを操作するためのクラス。2024/1/22時点では実装途中で動作しない。

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
        conn = sqlite3.connect(dbname)
        cursor = conn.cursor()

        # pandasのDataFrame形式であるdataからyear, month, day, hourを取得
        year = int(data.loc[0, "year"])
        month = int(data.loc[0, "month"])
        day = int(data.loc[0, "day"])
        hour = int(data.loc[0, "hour"])

        # 他の必要なカラムの値も取得
        # 例：column1 = data.loc[0, "column1"], column2 = data.loc[0, "column2"], ...

        # データが既に存在するか確認する
        cursor.execute("SELECT COUNT(*) FROM battery_control WHERE year = ? AND month = ? AND day = ? AND hour = ?", (year, month, day, hour))
        exists = cursor.fetchone()[0]

        if exists:
            # データを更新するSQL文
            update_sql = '''UPDATE battery_control
                            SET column1 = ?, column2 = ?, ...
                            WHERE year = ? AND month = ? AND day = ? AND hour = ?'''
            # 更新する値のタプルを用意
            update_values = (data.loc[0, "column1"], data.loc[0, "column2"], ..., year, month, day, hour)
            cursor.execute(update_sql, update_values)
        else:
            # データを挿入するSQL文
            insert_sql = '''INSERT INTO battery_control (year, month, day, hour, column1, column2, ...)
                            VALUES (?, ?, ?, ?, ?, ?, ...);'''
            # 挿入する値のタプルを用意
            insert_values = (year, month, day, hour, data.loc[0, "column1"], data.loc[0, "column2"], ...)
            cursor.execute(insert_sql, insert_values)

        # 変更をコミットし、接続を閉じる
        conn.commit()
        conn.close()
