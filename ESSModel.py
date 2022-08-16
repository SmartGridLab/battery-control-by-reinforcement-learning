class ESSmodel(gym.Env):
    def __init__(self, mode, pdf_day, train_days, test_day, PV_parameter):
        #パラメータ
        self.time = 0
        self.timeStep = 0
        self.battery = [[0]]
        self.planed_SoCkwh = [[0]]
        self.battery_true = [[0]]
        self.days = 1
        self.episode = 0
        self.battery_MAX = 4 # ４kWh
        self.Train_Days = train_days # 学習日
        self.rewards_timeInst = []
        self.rewards_episode = []
        self.all_PV_out_time = []
        self.all_PV_true_time = []
        self.all_soc = []
        self.all_soc_true = []
        self.all_battery = []
        self.all_price = []
        self.all_price_true = []

        # all_time: Time frames in a day from 0 to 48
        # all_timeStep: 学習回数. 30日分でall_timeStepが+1
        # all_action_fil: Forecasting phase. 
        # all_action_true: Observed phase. 
        #   - ESS charge/discharge amount which is modified to satisfy phisical constraints
        self.all_time = []  
        self.all_timeStep = [] 
        self.all_action = []
        self.all_action_fil = []    
        self.all_action_true = []

        self.MAX_reward = -10000
        self.mode = mode
        self.all_alpha = []
        self.all_beta = []
        self.sell_PVout = []
        self.sell_PVtrue = []
        self.PV_real = []
        self.K = 1.46   # parameters for inbalace rule
        self.L = 0.43   # parameters for inbalace rule
        
        #データのロード
        PV_data = pd.read_csv("train_and_test_data.csv", encoding="shift-jis") # train_and_test_data
        self.time_stamps = PV_data["hour"]
        price_all = PV_data["forecast_price"]   #電力価格の予測値
        true_all_price = PV_data["price[yen/kW30m]"]    # 電力価格の実測値
        alpha_all = PV_data["alpha"]    # インバランスの定数α
        beta_all = PV_data["beta"]  # インバランスの定数β
        self.PV = PV_parameter  # Forecast or Training
        PV_out_all = PV_data[self.PV]   # Training or test (specified by PV_parameter) data PV generation
        PV_true_all = PV_data["PVout_true"] # PV generation observed (PVout_true)
        self.test_days = test_day
        # 学習/テスト用データ作成
        # テスト期間は、学習期間の最終日の翌日＋test_day分を使う
        # ex) 4/1までが学習期間なら4/2+30日分がテスト期間（test_day=30日の場合）
        if self.mode == "learn":
            price_data = price_all[48*pdf_day:48*(self.Train_Days + pdf_day)]
            price_true_data = true_all_price[48*pdf_day:48*(self.Train_Days + pdf_day)]
            PV_out_data = PV_out_all[48*pdf_day:48*(self.Train_Days + pdf_day)]
            PV_true_data = PV_true_all[48*pdf_day:48*(self.Train_Days + pdf_day)]
            alpha_data = alpha_all[48*pdf_day:48*(self.Train_Days + pdf_day)]
            beta_data = beta_all[48*pdf_day:48*(self.Train_Days + pdf_day)]
        elif self.mode == "test":
            price_data = price_all[48*(self.Train_Days + pdf_day):48*(self.Train_Days + pdf_day + self.test_days)]
            price_true_data = true_all_price[48*(self.Train_Days + pdf_day):48*(self.Train_Days + pdf_day + self.test_days)]
            PV_out_data = PV_out_all[48*(self.Train_Days + pdf_day):48*(self.Train_Days + pdf_day + self.test_days)]
            PV_true_data = PV_true_all[48*(self.Train_Days + pdf_day):48*(self.Train_Days + pdf_day + self.test_days)]
            alpha_data = alpha_all[48*(self.Train_Days + pdf_day):48*(self.Train_Days + pdf_day + self.test_days)]
            beta_data = beta_all[48*(self.Train_Days + pdf_day):48*(self.Train_Days + pdf_day + self.test_days)]
        # Pandas data frame から numpy へ変換する
        #   ：tesnsor flowがnumpyしか読み込めないので
        PV_out_data = (PV_out_data.values)# 型変換
        self.PV_out = PV_out_data.reshape((len(PV_out_data), 1)) 
        PV_true_data = (PV_true_data.values)# 型変換
        self.PV_true = PV_true_data.reshape((len(PV_true_data), 1)) 
        price_data = (price_data.values)# 型変換
        self.price = price_data.reshape((len(price_data), 1)) 
        price_true_data = (price_true_data.values)# 型変換
        self.true_price = price_true_data.reshape((len(price_true_data), 1)) 
        alpha_data = (alpha_data.values)# 型変換
        self.alpha_data = alpha_data.reshape((len(alpha_data), 1)) 
        beta_data = (beta_data.values)# 型変換
        self.beta_data = beta_data.reshape((len(beta_data), 1))
        # Prepare the max value for normalization
        #  - max value changes day by day (The max values are overwritten in a loop)
        self.MAX_price = max(self.price[0:48])
        self.MAX_alpha = max(self.alpha_data[0:48])
        # Prepare for Reinforcement learning input (one time instance)        
        self.PV_out_time = self.PV_out[self.time]
        self.price_time = self.price[self.time]
        # ??        
        self.true_price_time = self.true_price[self.time]
        self.PV_true_time = self.PV_true[self.time]
        
        #アクション
        self.ACTION_NUM = 1 #アクションの数(台数増やすならここを増やす？RLの出力層のnuronがいくつあるかを設定)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape = (self.ACTION_NUM,))
        
        #状態の上限と下限の設定
        # Nurons for input layer（現状では４つ） 
        #   - 過去の実測値or予測値：self.PV_out_time
        #   - 電力価格：self.price_time
        #   - 現在の時間コマ：self.time
        #   - SoC: ??設定されてない...
        # LOW, HIGH: 
        #   - 入力のnuronの数と上下限値を設定
        #   - [0,1]を超えるとどうなるの・・・？よくわからん（調べたらいいかも？）
        LOW = np.array([0, 0, 0, 0])
        HIGH = np.array([1, 1, 1, 1])
        self.observation_space  = gym.spaces.Box(low=LOW, high=HIGH)

    def step(self, action): # rewardの決定
        #action > 0 →放電  action < 0 →充電
        # if ma.isnan(action[0]) == True:
        #  - なにかの原因？タイミングでNaNがactionに入ってくることがある。
        #    NaNだとエラーになるので、NaNをarrayの0.0にしている
        if ma.isnan(action[0]) == True:
            action = [0.0]
            action = np.array(action)
        # action[kwh]で正規化された-1~1を取る。
        # なので、30min分のkwhへ変換して、実験で使う電池の規格1.5kw定格へ変更する
        charge_discharge = 1.5*(action/2) 
        # rewards_timeInst: 各time instanceでの報酬
        #   - 初期化している
        rewards_timeInst = 0
        # self.timeとtimeをわけて考えたいので、入れ直している（後で使う）
        time = self.time    # ??
        timeStep = self.timeStep  # ??
        # battery_MAX: 容量[kWh]定格値。実験の電池は4kWh
        # self.battery: 計画段階でのSoC[kWh]の値（PVの出力は考慮していない理想的なSoC）
        # self.battery_true: PVの実測値で動作させた場合(PVの出力実績によって実現可能な実測値SoC)
        # planed_SoCkwh: 計画段階でのSoC[kWh]の値（PVの出力予測から実現可能な計画SoC）
        soc = (self.battery / self.battery_MAX) 
        soc_true = (self.battery_true / self.battery_MAX)
        battery = self.battery
        planed_SoCkwh = self.planed_SoCkwh
        battery_true = self.battery_true
        # done: ライブラリの中に含まれるフラグ。学習を止める際に使う
        #   - なにかの条件が満たされたら学習を止めたい場合につかうもの Trueで学習が強制的にとまる
        done = False
        # 変数を用意
        alpha = self.alpha_data[48*(self.days - 1) + self.time]
        beta = self.beta_data[48*(self.days - 1) + self.time]
        price = self.price[48*(self.days - 1) + self.time]  
        PV_out_time = self.PV_out[48*(self.days - 1) + self.time]
        PV_true_time = self.PV_true[48*(self.days - 1) + self.time]
        # グラフ描写用のデータ保管：all_*
        self.all_soc.append(soc*100)
        self.all_battery.append(battery)
        self.all_soc_true.append(soc_true*100)
        self.all_price.append(self.price[48*(self.days - 1) + self.time])
        self.all_price_true.append(self.true_price[48*(self.days - 1) + self.time])
        self.all_time.append(time/2)    # ??
        self.all_timeStep.append(timeStep/2)  # ??
        self.all_action.append(action*1.5)
        self.all_PV_out_time.append(PV_out_time[0])
        self.all_PV_true_time.append(PV_true_time[0])
        self.all_alpha.append(self.alpha_data[48*(self.days - 1) + self.time])
        self.all_beta.append(self.beta_data[48*(self.days - 1) + self.time])

        ## Graphの描写目的で、SoCkwhが現実と乖離しないように処理
        ## (PVの出力予測値と充放電量の整合性あわせ)
        # PV出力予測値がマイナスになることがあるので、０に補正
        if PV_out_time < 0: 
            PV_out_time = [0]
        # PVの発電量が充電予定量(-action*1.5)に足りないとき（action<0：充電）
        if PV_out_time < -action*1.5 and action < 0:
            action_real = -PV_out_time[0]
        # 電池の残SoCが、放電予定量に足りないとき（action>0: 放電）
        elif action > 0 and 0 < planed_SoCkwh < action*1.5:  # どういう状況??
            action_real = planed_SoCkwh
        # 電池の残SoCが100%なのに、充電したいとき
        elif planed_SoCkwh == self.battery_MAX and action < 0: # どういう状況??
            action_real = 0
        # 電池の残SoCが0%なのに、放電したいとき
        elif action > 0 and planed_SoCkwh == 0: # どういう状況??
            action_real = 0
        # 物理的な制約を受けない動作量のとき,そのまま動作させる
        else:
            action_real = action[0]*1.5
        self.all_action_fil.append(action_real)

        ## PVの出力実績値と充放電量の整合性あわせ
        if PV_true_time < -action*1.5 and action < 0:
            action_true = -PV_true_time[0]
        elif action > 0 and 0 < battery_true < action*1.5:
            action_true = battery_true
        elif battery_true == self.battery_MAX and action < 0:
            action_true = 0
        elif action > 0 and battery_true == 0:
            action_true = 0
        else:
            action_true = action[0]*1.5
        self.all_action_true.append(action_true)

        # time stepを進める際に、現在値を保管するために変数へ入れる
        prev_SoCkwh = planed_SoCkwh
        prev_SoCkwh_true = battery_true
        battery = battery - charge_discharge
        battery = battery[0]
        planed_SoCkwh = planed_SoCkwh - action_real/2
        planed_SoCkwh = planed_SoCkwh
        battery_true = battery_true - action_true/2
        battery_true = battery_true

        # Graphの描写目的で、SoCkwhが現実と乖離しないように処理
        #   - Error codeが出るように処理すべき部分kamo?
        #   - Graphの描写目的で設定しているので、学習には関係ないはず
        if planed_SoCkwh < 0:
            planed_SoCkwh = 0
        elif planed_SoCkwh > self.battery_MAX:
            planed_SoCkwh = np.array([self.battery_MAX])
            planed_SoCkwh = planed_SoCkwh[0]
        if battery_true < 0:
            battery_true = 0
        elif battery_true > self.battery_MAX:
            battery_true = np.array([self.battery_MAX])
            battery_true = battery_true[0]

        ## Rewardの設定
        # 充電する量がPV出力より高いならペナルティ（マイナス値）
        if -action*1.5 > PV_out_time:
            rewards_timeInst += (price/self.MAX_price)*(action*1.5 - PV_out_time[0])        
        # 放電量がSoCより大きいならペナルティ（マイナス値）
        if action*1.5 > battery:
            rewards_timeInst += (price/self.MAX_price)*(battery/4 - action)
        # 実際に売電で得られる金額をrewardへ追加（プラス値）
        elif action > 0 and action*1.5 <= battery:
            rewards_timeInst += (price/self.MAX_price)*action
        # SoCが最大容量より大きいならペナルティ
        if battery > self.battery_MAX:
            rewards_timeInst += (price)*(self.battery_MAX - battery)
            if self.mode == "test":
                battery = np.array([self.battery_MAX])
        # SoCがマイナスならペナルティ
        elif battery < 0:
            rewards_timeInst += (price)*battery
            if self.mode == "test":
                battery = 0
        # 充電に使った分を引く（予測値）
        if action_real < 0:
            PV_out_time = PV_out_time - (planed_SoCkwh - prev_SoCkwh) 
        # 充電に使った分を引く（実測値）
        if action_true < 0:
            PV_true_time = PV_true_time - (battery_true - prev_SoCkwh_true) # 充電に使った分を引く

        # total_forecast_time_real: インバランスの計算に使う
        # 予測値ベースで計画を立てたときの充放電量[kWh]の値を代入
        if action_real > 0: # 放電(30分) 1.5kw定格 -> kwhへ変換 
            total_forecast_time_real = action/(2*1.5)
        elif action_real <= 0:  # 充電
            total_forecast_time_real = 0
        # 実測値ベースで計画を立てたときの充放電量[kWh]の値を代入
        if action_true > 0:
            true_total_forecast_time = action_true/(1.5*2)
        elif action_true <= 0:
            true_total_forecast_time = 0

        ## インバランスによるreward（マイナス）の計算
        if true_total_forecast_time - total_forecast_time_real < 0:
        #不足インバランス
            imbalance = (alpha/self.MAX_alpha + beta + self.K)*(abs(true_total_forecast_time - total_forecast_time_real))
        elif true_total_forecast_time - total_forecast_time_real > 0:
        #余剰インバランス
            imbalance = (alpha/self.MAX_alpha + beta - self.L)*(abs(true_total_forecast_time - total_forecast_time_real))
        elif true_total_forecast_time - total_forecast_time_real == 0:
            imbalance = 0
        rewards_timeInst -= imbalance*100

        # HourとtimeStepを１つ進める
        # selfをつけないとClass外から参照できないのでself.へ入れる          
        self.time += 1
        time = self.time    # 下の方で使うので別の変数へ入れ直す
        self.timeStep += 1
        self.battery = battery
        self.planed_SoCkwh = planed_SoCkwh
        self.battery_true = battery_true
        soc = (self.battery / self.battery_MAX) # %
        soc_true = (self.battery_true / self.battery_MAX) # %
        ## 1日学習し終わったので処理
        if self.time == 48:
            self.days += 1  # 日付が１日進む
            self.time = 0   # hourが0に戻る
            # Calculate MAX_price for the next day
            if self.days - 1 != self.Train_Days and self.mode == "learn":   # Is the last day of learning?
                self.MAX_price = max(self.price[48*(self.days - 1):48*self.days])
                self.MAX_alpha = max(self.alpha_data[48*(self.days - 1):48*self.days])
            elif self.days != self.test_days and self.mode == "test":   # Is the last day of test?
                self.MAX_price = max(self.price[48*(self.days - 1):48*self.days])
                self.MAX_alpha = max(self.alpha_data[48*(self.days - 1):48*self.days])
        # Time instance ごとのrewardを追加する
        self.rewards_timeInst.append(rewards_timeInst)
        # Time instanceごとの充放電操作あとの売電量の計算
        self.sell_PVout.append(PV_out_time[0])  # PV_out_timeは、予測値-充電予定量の残り  
        self.sell_PVtrue.append(PV_true_time[0])    # PV_true_timeは、実績値-充電実績量の残り

        ## 学習の経過表示、リセット
        # Learningの最終日の最終コマか？
        if time == 48 and self.days - 1 == self.Train_Days and self.mode == "learn": 
            self.episode += 1   
            # Episodeごとのrewardを計算
            self.rewards_episode.append(np.sum(self.rewards_timeInst))
            # Epidode 1000回ごとに学習モデルとRewardをcsvへ保存        
            if self.episode % 1000 == 0 and 1 < self.episode :
                pdf_name = "result-" + self.mode + "_by_1000.pdf"
                pp = PdfPages(pdf_name) # PDF fileの作成
                self.model.save("ESS_learn_1000")   # Model
                reward_graph = self.graph(self.rewards_episode)
                pp.savefig(reward_graph)
                pp.close()    
            # 追加されたEpisodeのrewardと過去のEpoisodeでの最大のrewardと比較する
            if self.rewards_episode[-1] >= self.MAX_reward:
                pdf_name = "result-" + self.mode + "_max_reward.pdf"
                pp = PdfPages(pdf_name) # PDF fileの作成
                self.MAX_reward = self.rewards_episode[-1] # rewardの最高値を更新
                graph_1 = self.graph(self.rewards_episode)  # rewardの推移を描写
                graph_2 = self.schedule(self.all_action,self.all_PV_out_time,self.all_soc,"schedule_pre", mode = 0)
                graph_3 = self.schedule(self.all_action,self.all_PV_out_time,self.all_soc,"schedule_pre", mode = 1)
                graph_4 = self.schedule(self.all_action_true,self.all_PV_true_time,self.all_soc_true,"schedule_fil", mode = 0)
                graph_5 = self.schedule(self.all_action_true,self.all_PV_true_time,self.all_soc_true,"schedule_fil", mode = 1)
                pp.savefig(graph_1)
                pp.savefig(graph_2)
                pp.savefig(graph_3)
                pp.savefig(graph_4)
                pp.savefig(graph_5)
                pp.close()    
                # モデルの保存
                # - 一番報酬の高いモデルを保存しておく（が結局、Episodeが多いモデルのほうがtestデータに強そう）
                save_reward = pd.DataFrame(np.ravel(self.rewards_episode))
                label_name_reward = ["reward"]
                save_reward.columns = label_name_reward
                save_reward.to_csv("reward.csv")
                self.model.save("ESS_learn")
            

        ## Testの経過表示、リセット
        if self.mode == "test" and time == 48:
            self.episode += 1
            
        if self.mode == "test" and time == 48 and self.days == self.test_days:
            pdf_name = "result-" + self.mode + ".pdf"
            pp = PdfPages(pdf_name) # PDFの作成                
            self.MAX_reward = np.sum(self.rewards_timeInst) # rewardの最高値
            graph_4 = self.schedule(self.all_action,self.all_PV_out_time,self.all_soc,"schedule_pre", mode = 0)
            graph_5 = self.schedule(self.all_action,self.all_PV_out_time,self.all_soc,"schedule_pre", mode = 1)
            graph_6 = self.schedule(self.all_action_true,self.all_PV_true_time,self.all_soc_true,"schedule_fil", mode = 0)
            graph_7 = self.schedule(self.all_action_true,self.all_PV_true_time,self.all_soc_true,"schedule_fil", mode = 1)
            pp.savefig(graph_4)
            pp.savefig(graph_5)
            pp.savefig(graph_6)
            pp.savefig(graph_7)
            
            # インバランス料金の計算に必要な変数を用意
            self.all_action_true = pd.DataFrame(np.ravel(self.all_action_true))
            self.all_action_fil = pd.DataFrame(np.ravel(self.all_action_fil))
            self.sell_PVout = pd.DataFrame(np.ravel(self.sell_PVout))
            self.sell_PVtrue = pd.DataFrame(np.ravel(self.sell_PVtrue))
            self.all_alpha = pd.DataFrame(np.ravel(self.all_alpha))
            self.all_beta = pd.DataFrame(np.ravel(self.all_beta))

            # インバランス料金、利益等の算出(評価値の算出)
            imbalance = 0
            total_profit = 0
            profit = 0
            imbalance_PV = 0
            PV_profit_true = 0
            imb_all = []
            sell_all = []
            imb_PV = []
            sell_PV = []
            timeStepCpunt = 0   
            # インバランスを計算
            #   - 実績のcharge/dischargeを使って、Test期間のすべてのTime instanceについてインバランスを計算
            for i in range(0, len(self.all_action_true)):
                timeStepCpunt += 1
                # Pandasのdataframeから要素を抽出
                true_PV_forecast_time = self.sell_PVtrue[0][i]
                true_ESS_forecast_time = self.all_action_true[0][i]
                Forecast_ESS_time = self.all_action_fil[0][i]
                Forecast_PV_time = self.sell_PVout[0][i]
                alpha = self.all_alpha[0][i]
                beta = self.all_beta[0][i]
                price = self.true_price[i]
                PVtrue = self.all_PV_true_time[i]
                PVout = self.all_PV_out_time[i]

                ## 売電量の計算
                # 蓄電池の実績値が放電の場合
                if true_ESS_forecast_time > 0:
                    true_total_forecast_time = true_PV_forecast_time + true_ESS_forecast_time
                # 蓄電池の実績値が充電の場合 
                elif true_ESS_forecast_time <= 0:
                    true_total_forecast_time = true_PV_forecast_time    # true_PV_forecast_time：充電分を差し引いたPV発電量
                # 蓄電池の計画値が放電の場合
                if Forecast_ESS_time > 0:
                    total_forecast_time_real = Forecast_PV_time + Forecast_ESS_time
                # 蓄電池の計画値が充電の場合
                elif Forecast_ESS_time <= 0:
                    total_forecast_time_real = Forecast_PV_time # Forecast_PV_time：充電分を差し引いたPV発電量
                # 1 time instance分の売電売上の計算
                total_profit += true_total_forecast_time*price # PV＋ESSによる売電量の売上を計算
                # ESSがなく、PVを単に売電した場合の売上の計算
                PV_profit_true += PVtrue*price # PV実測のみによる売上

                ## ESSがある場合のインバランス量の計算
                #   - total_forecast_time_real: 予測において、どれだけPV+ESS出力から売電したか
                #   - true_total_forecast_time: 実績において、どれだけPV+ESS出力から売電したか
                # 実績が足りないとき（不足インバランス）    
                if true_total_forecast_time < total_forecast_time_real:             
                    imbalance -= (alpha + beta + self.K)*(abs(true_total_forecast_time - total_forecast_time_real))
                # 実績が多すぎのとき（余剰インバランス）
                elif true_total_forecast_time > total_forecast_time_real:
                    imbalance -= (alpha + beta - self.L)*(abs(true_total_forecast_time - total_forecast_time_real))
                # 計画値と実績値が一致するとき（インバランスなし）
                elif true_total_forecast_time == total_forecast_time_real:
                    imbalance -= 0

                ## ESSがない（PVのみの）場合のインバランス量の計算
                #   - PVout: PV予測値の売電
                #   - PVtrue: PV発電量実績の売電
                # 不足インバランス
                if PVout < PVtrue:                
                    imbalance_PV -= (alpha + beta + self.K)*(abs(PVtrue - PVout))
                # 余剰インバランス
                elif PVout > PVtrue:
                    imbalance_PV -= (alpha + beta - self.L)*(abs(PVtrue - PVout))
                # インバランスなし
                elif PVout == PVtrue:
                    imbalance_PV -= 0

                ## 1日ごとのインバランス料金の合計の計算（グラフ表示用）
                if timeStepCpunt == 48:
                    # -------配列の要素問題でエラーが出たらつかうかも・・・
                    #if i == 47:
                        #imb_all.append(imbalance)
                    #else:
                    # -------------------------------------------------------
                    imb_all.append(imbalance[0])
                    sell_all.append(total_profit[0])
                    sell_PV.append(PV_profit_true[0])
                    imb_PV.append(imbalance_PV)
                    timeStepCpunt = 0

            ## グラフ表示ようのデータ作成
            # Arrayへ変換
            sell_all = np.array(sell_all)
            imb_all = np.array(imb_all)
            sell_PV = np.array(sell_PV)
            imb_PV = np.array(imb_PV)
            # 利益の計算
            profit = sell_all+imb_all # ESS+PVの場合
            profit_PV = sell_PV+imb_PV # PVノミの場合
            imb_Graph = self.graph_imb(sell_all,imb_all,profit,"日", "金額[円]", "売上","インバランス料金","利益")
            imb_Graph_PV = self.graph_imb(sell_PV,imb_PV,profit_PV,"日", "金額[円]", "売上","インバランス料金","利益")
            pp.savefig(imb_Graph)
            pp.savefig(imb_Graph_PV)
            pp.close()    
            # 結果表示
            print("PV+ESS")
            print(total_profit)
            print((-1)*imbalance)
            print(total_profit + imbalance)
            print("PV")
            print(PV_profit_true)
            print((-1)*imbalance_PV)
            print(PV_profit_true + imbalance_PV)

        # Reset部分
        #   - 最終日の最後のコマの学習/テストが終わったら、初期化をする
        #   - SoCを戻す、csvの読み込みのカーソルを一番最初に戻す etc...
        if time == 48 and self.days - 1 == self.Train_Days and self.mode == "learn":
            state = self.reset()
        elif time == 48 and self.days == self.test_days and self.mode == "test":
            state = self.reset()
        else:
            state = [self.time/24, self.PV_out[48*(self.days - 1) + self.time]/2, soc, self.price[48*(self.days - 1) + self.time]/self.MAX_price]
        state = pd.DataFrame(state)
        state = (state.values).T

        return state, rewards_timeInst, done, {}
    
    def reset(self): # 状態を初期化
        self.time = 0
        self.timeStep = 0
        self.battery = 0    # Soc[kw]の初期値
        self.planed_SoCkwh = 0
        self.battery_true = 0 
        self.days = 1
        self.PV_out_time = self.PV_out[self.time]
        if self.PV_out_time < 0:
            self.PV_out_time = np.array([0])
        self.price_time = self.price[self.time]
        self.true_price_time = self.true_price[self.time]
        self.MAX_price = max(self.price[0:48])
        self.MAX_alpha = max(self.alpha_data[0:48])
        self.rewards_timeInst = []
        self.all_PV_out_time = []
        self.all_PV_true_time = []
        self.all_soc = []
        self.all_soc_true = []
        self.all_soc_real = []
        self.all_battery = []
        self.all_price = []
        self.all_price_true = []
        self.all_time = []
        self.all_timeStep = []
        self.all_action = []
        self.all_action_fil = []
        self.all_action_true = []
    
        # self.battery/4: 定格値の4kwでSoc[kw]を割って0~1の間のSoCへ変換
        state = [self.time/24, self.PV_out_time/2, self.battery/4, self.price_time/self.MAX_price]

        return state

    def render(self, mode='human', close=False):
        pass

    def close(self): 
        pass

    def seed(self): 
        pass
