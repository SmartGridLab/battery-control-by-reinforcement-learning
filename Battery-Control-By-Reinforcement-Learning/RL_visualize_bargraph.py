import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime

def main():
    # CSVファイルを読み込む
    df = pd.read_csv("Battery-Control-By-Reinforcement-Learning/result_dataframe.csv")

    # 日付列をdatetime型に変換する
    df['date'] = pd.to_datetime(df[['year', 'month', 'day']])

    # 各利益タイプに対応する色を定義
    colors = {
        'totalprofit_bid[Yen]': 'orange',
        'totalprofit_actual_bid[Yen]': 'red',
        'totalprofit_base[Yen]': 'blue'
    }

    # 'totalprofit_base[Yen]'が存在するか確認
    if 'totalprofit_base[Yen]' not in df.columns:
        print("Warning: 'totalprofit_base[Yen]'列がデータフレームに存在しません。追加してください。")

    # 元の棒グラフと折れ線グラフで使用する利益の列を定義
    profit_columns = [
        'totalprofit_bid[Yen]',
        'totalprofit_actual_bid[Yen]',
        'totalprofit_base[Yen]'
    ]
    # 存在する列のみ使用
    profit_columns = [col for col in profit_columns if col in df.columns]

    # 元のグラフ用の月ごとの合計を計算
    monthly_profits = df.groupby(df['date'].dt.to_period('M')).agg({
        'totalprofit_bid[Yen]': 'sum',
        'totalprofit_actual_bid[Yen]': 'sum',
        'totalprofit_base[Yen]': 'sum'
    }).dropna()

    # 元の棒グラフ
    fig1, ax1 = plt.subplots(figsize=(12, 8))
    monthly_profits[profit_columns].plot(kind='bar', color=[colors[col] for col in profit_columns], ax=ax1, width=0.5, legend=False)
    ax1.set_xlabel('monthly total', fontsize=24)
    ax1.set_ylabel('Total Profit (Yen)', fontsize=24)
    ax1.set_title('Total Monthly Profit Comparison', fontsize=24)
    ax1.tick_params(axis='x', labelsize=24, direction='in')
    ax1.tick_params(axis='y', labelsize=24, direction='in')
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    ax1.axhline(y=0, color='black', linewidth=1.5)
    plt.xticks(rotation=0)

    # 元の折れ線グラフ用の日ごとの合計を計算
    daily_profits = df.groupby(df['date'].dt.to_period('D')).agg({
        'totalprofit_bid[Yen]': 'sum',
        'totalprofit_actual_bid[Yen]': 'sum',
        'totalprofit_base[Yen]': 'sum'
    }).dropna()

    # 元の折れ線グラフ
    fig2, ax2 = plt.subplots(figsize=(12, 8))
    ax2.plot(daily_profits.index.to_timestamp(), daily_profits['totalprofit_bid[Yen]'], label='Total Profit Bid [Yen]', color=colors['totalprofit_bid[Yen]'], marker='o')
    ax2.plot(daily_profits.index.to_timestamp(), daily_profits['totalprofit_actual_bid[Yen]'], label='Total Profit Actual Bid [Yen]', color=colors['totalprofit_actual_bid[Yen]'], marker='o')
    ax2.plot(daily_profits.index.to_timestamp(), daily_profits['totalprofit_base[Yen]'], label='Total Profit Base [Yen]', color=colors['totalprofit_base[Yen]'], marker='o')
    ax2.set_xlabel('Date', fontsize=24)
    ax2.set_ylabel('Total Profit (Yen)', fontsize=24)
    ax2.set_title('Daily Total Profit Comparison', fontsize=24)
    ax2.tick_params(axis='x', labelsize=24, direction='in')
    ax2.tick_params(axis='y', labelsize=24, direction='in')
    ax2.grid(axis='y', which='major', linestyle='--', alpha=0.7)
    ax2.axhline(y=0, color='black', linewidth=1.5)

    # 特定の日付をX軸の目盛として設定
    specific_dates = ['09/01/2022', '09/15/2022', '09/30/2022']
    specific_dates = pd.to_datetime(specific_dates)
    ax2.set_xticks(specific_dates)
    ax2.set_xticklabels(specific_dates.strftime('%m/%d/%Y'), rotation=45, ha='center', fontsize=20)
    plt.xticks(rotation=0)

    # 新しい棒グラフ（'totalprofit_bid[Yen]'と'totalprofit_actual_bid[Yen]'のみ）
    fig3, ax3 = plt.subplots(figsize=(12, 8))
    monthly_profits[['totalprofit_bid[Yen]', 'totalprofit_actual_bid[Yen]']].plot(
        kind='bar', color=[colors['totalprofit_bid[Yen]'], colors['totalprofit_actual_bid[Yen]']],
        ax=ax3, width=0.5, legend=False
    )
    ax3.set_xlabel('monthly total', fontsize=24)
    ax3.set_ylabel('Total Profit (Yen)', fontsize=24)
    ax3.set_title('Total Monthly Profit Comparison (Bid and Actual)', fontsize=24)
    ax3.tick_params(axis='x', labelsize=24, direction='in')
    ax3.tick_params(axis='y', labelsize=24, direction='in')
    ax3.grid(axis='y', linestyle='--', alpha=0.7)
    ax3.axhline(y=0, color='black', linewidth=1.5)
    plt.xticks(rotation=0)

    # 新しい折れ線グラフ（'totalprofit_bid[Yen]'と'totalprofit_actual_bid[Yen]'のみ）
    fig4, ax4 = plt.subplots(figsize=(12, 8))
    ax4.plot(daily_profits.index.to_timestamp(), daily_profits['totalprofit_bid[Yen]'],
             label='Total Profit Bid [Yen]', color=colors['totalprofit_bid[Yen]'], marker='o')
    ax4.plot(daily_profits.index.to_timestamp(), daily_profits['totalprofit_actual_bid[Yen]'],
             label='Total Profit Actual Bid [Yen]', color=colors['totalprofit_actual_bid[Yen]'], marker='o')
    ax4.set_xlabel('Date', fontsize=24)
    ax4.set_ylabel('Total Profit (Yen)', fontsize=24)
    ax4.set_title('Daily Total Profit Comparison (Bid and Actual)', fontsize=24)
    ax4.tick_params(axis='x', labelsize=24, direction='in')
    ax4.tick_params(axis='y', labelsize=24, direction='in')
    ax4.grid(axis='y', which='major', linestyle='--', alpha=0.7)
    ax4.axhline(y=0, color='black', linewidth=1.5)

    # 特定の日付をX軸の目盛として設定
    specific_dates = pd.to_datetime(specific_dates)
    ax4.set_xticks(specific_dates)
    ax4.set_xticklabels(specific_dates.strftime('%m/%d/%Y'), rotation=45, ha='center', fontsize=20)
    plt.xticks(rotation=0)

    # 全グラフのレイアウト調整
    fig1.tight_layout()
    fig2.tight_layout()
    fig3.tight_layout()
    fig4.tight_layout()

    # PDFに保存
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    pdf_path = f"Battery-Control-By-Reinforcement-Learning/total_profit_comparison_{current_time}.pdf"
    with PdfPages(pdf_path) as pdf:
        pdf.savefig(fig1)
        pdf.savefig(fig2)
        pdf.savefig(fig3)
        pdf.savefig(fig4)

        # 各ペアのグラフに対する凡例
        for fig, ax in [(fig1, ax1), (fig2, ax2), (fig3, ax3), (fig4, ax4)]:
            fig_legend, ax_legend = plt.subplots(figsize=(12, 8))
            ax_legend.axis('off')
            handles, labels = ax.get_legend_handles_labels()
            ax_legend.legend(handles=handles, labels=labels, loc='center', fontsize=24, title="Profit Type", title_fontsize=24)
            pdf.savefig(fig_legend)
            plt.close(fig_legend)

    # グラフを閉じる
    plt.close(fig1)
    plt.close(fig2)
    plt.close(fig3)
    plt.close(fig4)

    print(f"グラフがPDFファイル '{pdf_path}' に保存されました。")

if __name__ == "__main__":
    main()
