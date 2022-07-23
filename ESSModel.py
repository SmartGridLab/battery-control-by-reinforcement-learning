class ESS_Model(gym.Env):
    def __init__(self, mode, pdf_day, train_days, test_day, PV_parameter):
        #パラメータ
        sns.set(font_scale = 3)
        self.time = 0
        self.count = 0
        self.battery = [[0]]
        self.battery_real = [[0]]
        self.battery_true = [[0]]
        self.days = 1
        self.episode = 0
        self.battery_MAX = 4 # ４kWh
        self.Train_Days = train_days # 学習日
        self.reward_action = []
        self.reward_soc = []
        self.reward_soc_real = []
        self.reward_PV = []
        self.optimize_rewards = []
        self.rewards_action = []
        self.rewards_soc = []
        self.rewards_soc_real = []
        self.rewards_PV = []
        self.total_rewards = []
        self.total_reward = []
        self.all_optimize_rewards = []
        self.all_PV_out_time = []
        self.all_soc = []
        self.all_soc_real = []
        self.all_battery = []
        self.all_price = []
        self.all_price_true = []
        self.all_time = []
        self.all_count = []
        self.all_action = []
        self.all_action_fil = []
        self.all_PV_real = []
        self.MAX_reward = -10000
        self.mode = mode
        self.soc_day = []
        self.PV_day = []
        self.total_day = []
        self.rewards_soc_day = []
        self.rewards_PV_day = []
        self.total_rewards_day = []
        self.all_alpha = []
        self.all_beta = []
        self.sell_PVout = []
        self.PV_real = []
        self.reward_PV_real = []
        self.rewards_PV_real = []
        self.total_reward_real = []
        self.total_rewards_real = []
        self.K = 1.46
        self.L = 0.43
        
        #データのロード
        PV_data = pd.read_csv("train_and_test_data.csv", encoding="shift-jis") # train_and_test_data
        self.time_stamps = PV_data["hour"]
        price_all = PV_data["forecast_price"]
        true_all_price = PV_data["price[yen/kW30m]"]
        alpha_all = PV_data["alpha"]
        beta_all = PV_data["beta"]
        lower_all = PV_data["lower"]
        upper_all = PV_data["upper"]
        self.PV = PV_parameter
        PV_out_all = PV_data[self.PV]
        PV_true_all = PV_data["PVout_true"]
        self.test_days = test_day
        # 確率密度関数の作成用データ
        PV_true_pdf = PV_true_all[0:48*pdf_day]
        for i in range(0, 30):
            PV_true_pdf_day = PV_true_pdf[48*i:48*(i+1)].reset_index(drop=True)
            if i == 0:
                pdf_data = PV_true_pdf_day
            else:
                pdf_data = pd.concat([pdf_data, PV_true_pdf_day],axis=1)
        pdf_data = np.array(pdf_data)
        pdf_data = np.sort(pdf_data)
        pdf_data = pd.DataFrame(pdf_data)
        pdf_data = pdf_data.T.values.tolist()
        pdf_data = pd.DataFrame(pdf_data) # 過去30日分の実測値
        self.pdf_data = pdf_data
        # 学習(テスト)用データ作成
        if self.mode == "learn":
            price_data = price_all[48*pdf_day:48*(self.Train_Days + pdf_day)]
            price_true_data = true_all_price[48*pdf_day:48*(self.Train_Days + pdf_day)]
            PV_out_data = PV_out_all[48*pdf_day:48*(self.Train_Days + pdf_day)]
            PV_true_data = PV_true_all[48*pdf_day:48*(self.Train_Days + pdf_day)]
            alpha_data = alpha_all[48*pdf_day:48*(self.Train_Days + pdf_day)]
            beta_data = beta_all[48*pdf_day:48*(self.Train_Days + pdf_day)]
            lower_data = lower_all[48*pdf_day:48*(self.Train_Days + pdf_day)]
            upper_data = upper_all[48*pdf_day:48*(self.Train_Days + pdf_day)]
        elif self.mode == "test":
            price_data = price_all[48*(self.Train_Days + pdf_day):48*(self.Train_Days + pdf_day + self.test_days)]
            price_true_data = true_all_price[48*(self.Train_Days + pdf_day):48*(self.Train_Days + pdf_day + self.test_days)]
            PV_out_data = PV_out_all[48*(self.Train_Days + pdf_day):48*(self.Train_Days + pdf_day + self.test_days)]
            PV_true_data = PV_true_all[48*(self.Train_Days + pdf_day):48*(self.Train_Days + pdf_day + self.test_days)]
            alpha_data = alpha_all[48*(self.Train_Days + pdf_day):48*(self.Train_Days + pdf_day + self.test_days)]
            beta_data = beta_all[48*(self.Train_Days + pdf_day):48*(self.Train_Days + pdf_day + self.test_days)]
            lower_data = lower_all[48*(self.Train_Days + pdf_day):48*(self.Train_Days + pdf_day + self.test_days)]
            upper_data = upper_all[48*(self.Train_Days + pdf_day):48*(self.Train_Days + pdf_day + self.test_days)]

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
        lower_data = (lower_data.values)# 型変換
        self.lower_data = lower_data.reshape((len(lower_data), 1)) 
        upper_data = (upper_data.values)# 型変換
        self.upper_data = upper_data.reshape((len(upper_data), 1)) 
        self.MAX_price = max(self.price[0:48])
        self.MAX_alpha = max(self.alpha_data[0:48])
        
        self.PV_out_time = self.PV_out[self.time]
        self.PV_true_time = self.PV_true[self.time]
        self.price_time = self.price[self.time]
        self.true_price_time = self.true_price[self.time]
        self.lower_time = self.lower_data[self.time]
        self.upper_time = self.upper_data[self.time]
        self.PV_pdf_time = self.pdf_data[self.time]
        self.alpha_data_time = self.alpha_data[self.time]/max(self.alpha_data[0:48])
        self.beta_data_time = self.beta_data[self.time]
        
        #アクション
        self.ACTION_NUM=1 #アクションの数
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape = (self.ACTION_NUM,))
        
        #状態の上限と下限の設定
        STATE_NUM = 6
        LOW = np.array([0, 0, 0, 0, 0, 0])
        HIGH = np.array([1, 1, 1, 1, 1, 1])
        self.observation_space  = gym.spaces.Box(low=LOW, high=HIGH)

    def step(self, action): # rewardの決定
        #action > 0 →放電  action < 0 →充電
        if ma.isnan(action[0]) == True:
            action = [0.0]
            action = np.array(action)
        charge_discharge = 1.5*(action/2) # kW・30分
        optimize_rewards = 0
        time = self.time
        count = self.count
        soc = (self.battery / self.battery_MAX) # %
        soc_real = (self.battery_real / self.battery_MAX)
        battery = self.battery
        battery_real = self.battery_real
        battery_true = self.battery_true
        done = False
        alpha = self.alpha_data[48*(self.days - 1) + self.time]
        beta = self.beta_data[48*(self.days - 1) + self.time]
        price = self.price[48*(self.days - 1) + self.time]

        # 確率密度関数の作成
        #if action < 0:
            #pdf_data = []
            #for i in range(0, 30):
                #if self.lower_time <= self.PV_pdf_time[i] <= self.upper_time:
                    #pdf_data.append(self.PV_pdf_time[i])
            #pdf_data.append(self.PV_out_time[0])
            #pdf_data.append(self.lower_time[0])
            #pdf_data.append(self.upper_time[0])
            #pdf_data = np.array(pdf_data)
            #pdf_data = np.sort(pdf_data)
            #n = len(pdf_data)
            # 確率密度関数描画用のx軸データ
            #if max(pdf_data)+max(pdf_data)/2 == 0:
                #x = np.linspace(0, 1, 1000)
            #else:
                #x = np.linspace(0, max(pdf_data)+max(pdf_data)/2, 1000)
            # 確率密度関数の値を取得
            #ys = []
            #h_scott=np.sqrt(np.var(pdf_data,ddof=1)*((n)**(-1/5))**2) # スコットのルール(バンド幅の決定)
            #for y in pdf_data:
                #ys.append(norm.pdf(x, loc=y, scale=h_scott) / n)
            #sumy = np.sum(ys, axis=0) # 確率密度関数
            #idx = np.argmin(np.abs(np.array(x) - (-action*1.5)))
            #pdf = sumy[idx]
            #pdf = pdf #/max(sumy)
            #if pdf < 0:
                #pdf = 0
            # 期待値を使用したreward
            #if min(x) <= -action*1.5 <= max(x) and ma.isnan(pdf) == False: # 充電量が区間内のとき
                #optimize_rewards += -action*pdf*(price/self.MAX_price)
                #optimize_rewards += -action*pdf*(price/self.MAX_price)

        self.all_soc.append(soc*100)
        self.all_battery.append(battery)
        self.all_soc_real.append(soc_real*100)
        self.all_price.append(self.price[48*(self.days - 1) + self.time])
        self.all_price_true.append(self.true_price[48*(self.days - 1) + self.time])
        self.all_time.append(time/2)
        self.all_count.append(count/2)
        self.all_action.append(action*1.5)
        self.all_PV_out_time.append(self.PV_out_time[0])
        self.all_alpha.append(self.alpha_data[48*(self.days - 1) + self.time])
        self.all_beta.append(self.beta_data[48*(self.days - 1) + self.time])
        
        PV_out_time = self.PV_out[48*(self.days - 1) + self.time]
        PV_true_time = self.PV_true[48*(self.days - 1) + self.time]

        if PV_out_time < 0:
            PV_out_time = [0]
                
        if PV_out_time < -action*1.5 and action < 0:
            action_real = -PV_out_time[0]
            PV_real = [0]
            self.all_action_fil.append(action_real)
        elif action > 0 and 0 < battery_real < action*1.5:
            action_real = battery_real
            self.all_action_fil.append(action_real)
        elif battery_real == self.battery_MAX and action < 0:
            action_real = 0
            self.all_action_fil.append(action_real)
        elif action > 0 and battery_real == 0:
            action_real = 0
            self.all_action_fil.append(action_real)
        else:
            if self.mode == "learn":
                action_real = action[0]*1.5
                self.all_action_fil.append(action_real)
            elif self.mode == "test":
                action_real = action[0]*1.5
                action_real = action_real[0]
                self.all_action_fil.append(action_real)

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

        pred_battery = battery
        battery = battery - charge_discharge
        battery = battery[0]
        battery_real = battery_real - action_real/2
        battery_real = battery_real
        battery_true = battery_true - action_true/2
        battery_true = battery_true

        if battery_real < 0:
            battery_real = 0
        elif battery_real > self.battery_MAX:
            battery_real = np.array([self.battery_MAX])
            battery_real = battery_real[0]
        
        if -action*1.5 > PV_out_time: # 充電する量がPV出力より高いならペナルティ
            optimize_rewards += (price/self.MAX_price)*(action - PV_out_time[0]/2)
        
        if action*1.5 > battery: # 放電量がSoCより大きいならペナルティ
            optimize_rewards += (price/self.MAX_price)*(battery/4 - action)
        elif action > 0 and action*1.5 <= battery:
            optimize_rewards += (price/self.MAX_price)*action
        
        if battery > self.battery_MAX: # SoCが最大容量より大きいならペナルティ
            optimize_rewards += (price)*(self.battery_MAX - battery)
            if self.mode == "test":
                battery = np.array([self.battery_MAX])
        elif battery < 0: # SoCがマイナスならペナルティ
            optimize_rewards += (price)*battery
            if self.mode == "test":
                battery = 0
            
        if action_real < 0:
            PV_out_time = PV_out_time - (battery - pred_battery) # 充電に使った分を引く
            PV_real = PV_out_time - (battery_real - pred_battery) # 充電に使った分を引く
        elif PV_out_time < -action*1.5 and action < 0:
            PV_real = [0]
        else:
            PV_real = PV_out_time

        if action_real > 0:
            total_forecast_time_real = action_real/(2*1.5)
        elif action_real <= 0:
            total_forecast_time_real = 0

        if action_true > 0:
            true_total_forecast_time = action_true/(2*1.5)
        elif action_true <= 0:
            true_total_forecast_time = 0

        if true_total_forecast_time - total_forecast_time_real <= 0:
        #不足インバランス
            imbalance = (alpha/self.MAX_alpha + beta + self.K)*(abs(true_total_forecast_time - total_forecast_time_real))
        elif true_total_forecast_time - total_forecast_time_real > 0:
        #余剰インバランス
            imbalance = (alpha/self.MAX_alpha + beta - self.L)*(abs(true_total_forecast_time - total_forecast_time_real))

        optimize_rewards -= imbalance
                          
        self.time += 1
        time = self.time
        self.count += 1
        self.battery = battery
        self.battery_real = battery_real
        self.battery_true = battery_true
        soc = (self.battery / self.battery_MAX) # %
        soc_real = (self.battery_real / self.battery_MAX) # %

        if self.time == 48:
            self.days += 1
            self.time = 0
            if self.days - 1 != self.Train_Days and self.mode == "learn":
                self.MAX_price = max(self.price[48*(self.days - 1):48*self.days])
                self.MAX_alpha = max(self.alpha_data[48*(self.days - 1):48*self.days])
            elif self.days != self.test_days and self.mode == "test":
                self.MAX_price = max(self.price[48*(self.days - 1):48*self.days])
                self.MAX_alpha = max(self.alpha_data[48*(self.days - 1):48*self.days])
        
        if self.mode == "learn":
            if time == 48 and self.days - 1 == self.Train_Days:
                state = [self.time/24, self.PV_out[48*(self.days-1) + self.time - 1]/2, soc, self.price[48*(self.days-1) + self.time - 1]/self.MAX_price,
                                self.alpha_data[48*(self.days-1) + self.time - 1]/self.MAX_alpha, self.beta_data[48*(self.days-1) + self.time - 1]]
                self.PV_out_time = self.PV_out[48*(self.days - 1) + self.time - 1]
                self.lower_time = self.lower_data[48*(self.days - 1) + self.time - 1]
                self.upper_time = self.upper_data[48*(self.days - 1) + self.time - 1]
            else:
                state = [self.time/24, self.PV_out[48*(self.days - 1) + self.time]/2, soc, self.price[48*(self.days - 1) + self.time]/self.MAX_price,
                                self.alpha_data[48*(self.days - 1) + self.time]/self.MAX_alpha, self.beta_data[48*(self.days - 1) + self.time]]
                self.PV_out_time = self.PV_out[48*(self.days - 1) + self.time]
                self.lower_time = self.lower_data[48*(self.days - 1) + self.time]
                self.upper_time = self.upper_data[48*(self.days - 1) + self.time]
        elif self.mode == "test":
            if time == 48 and self.days - 1 == self.test_days:
                state = [self.time/24, self.PV_out[48*(self.days-1) + self.time - 1]/2, soc, self.price[48*(self.days-1) + self.time - 1]/self.MAX_price,
                                self.alpha_data[48*(self.days-1) + self.time - 1]/self.MAX_alpha, self.beta_data[48*(self.days-1) + self.time - 1]]
                self.PV_out_time = self.PV_out[48*(self.days - 1) + self.time - 1]
                self.lower_time = self.lower_data[48*(self.days - 1) + self.time - 1]
                self.upper_time = self.upper_data[48*(self.days - 1) + self.time - 1]
            else:
                state = [self.time/24, self.PV_out[48*(self.days - 1) + self.time]/2, soc, self.price[48*(self.days - 1) + self.time]/self.MAX_price, 
                                self.alpha_data[48*(self.days - 1) + self.time]/self.MAX_alpha, self.beta_data[48*(self.days - 1) + self.time]]
                self.PV_out_time = self.PV_out[48*(self.days - 1) + self.time]
                self.lower_time = self.lower_data[48*(self.days - 1) + self.time]
                self.upper_time = self.upper_data[48*(self.days - 1) + self.time]
        
        state = pd.DataFrame(state)
        state = (state.values).T
        self.optimize_rewards.append(optimize_rewards)
        self.sell_PVout.append(PV_out_time[0])
        self.PV_real.append(PV_real[0])
        #self.PV_pdf_time = self.pdf_data[self.time]
        if self.PV_out_time < 0:
            self.PV_out_time = [0]
        
        if time == 48 and self.days - 1 == self.Train_Days and self.mode == "learn": #学習の経過表示、リセット
            self.episode += 1
            self.total_rewards.append(np.sum(self.total_reward))
            self.all_optimize_rewards.append(np.sum(self.optimize_rewards))
            
            if self.episode % 1000 == 0 and 1 < self.episode :
                self.graph(self.all_optimize_rewards, "reward", "episode", "reward", show = "yes")
                self.schedule(self.all_action,self.all_soc,"schedule",show = "yes", mode = 0)
                self.model.save("ESS_learn_1000")
                save_reward = pd.DataFrame(np.ravel(self.all_optimize_rewards))
                label_name_reward = ["reward"]
                save_reward.columns = label_name_reward
                save_reward.to_csv("reward_1000.csv")
                
            if np.sum(self.optimize_rewards) >= self.MAX_reward:
                pdf_name = "result-" + self.mode + "_prot.pdf"
                pp = PdfPages(pdf_name) # PDFの作成
                
                self.MAX_reward = np.sum(self.optimize_rewards) # rewardの最高値
                graph_1 = self.graph(self.all_optimize_rewards, "reward", "episode", "reward", show = "no")
                graph_2 = self.schedule(self.all_action,self.all_soc,"schedule_pre", show = "no", mode = 0)
                graph_3 = self.schedule(self.all_action,self.all_soc,"schedule_pre", show = "no", mode = 1)
                graph_4 = self.schedule(self.all_action_fil,self.all_soc_real,"schedule_fil", show = "no", mode = 0)
                graph_5 = self.schedule(self.all_action_fil,self.all_soc_real,"schedule_fil", show = "no", mode = 1)
                
                pp.savefig(graph_1)
                pp.savefig(graph_2)
                pp.savefig(graph_3)
                pp.savefig(graph_4)
                pp.savefig(graph_5)
                pp.close()
                
                #モデルの保存
                save_reward = pd.DataFrame(np.ravel(self.all_optimize_rewards))
                label_name_reward = ["reward"]
                save_reward.columns = label_name_reward
                save_reward.to_csv("reward.csv")
                self.model.save("ESS_learn")

            self.reset()

        if self.mode == "test" and time == 48:
            self.episode += 1
            
        if self.mode == "test" and time == 48 and self.days == self.test_days:
            pdf_name = "result-" + self.mode + ".pdf"
            pp = PdfPages(pdf_name) # PDFの作成
                
            self.MAX_reward = np.sum(self.optimize_rewards) # rewardの最高値

            graph_4 = self.schedule(self.all_action,self.all_soc,"schedule_pre", show = "no", mode = 0)
            graph_5 = self.schedule(self.all_action,self.all_soc,"schedule_pre", show = "no", mode = 1)
            graph_6 = self.schedule(self.all_action_fil,self.all_soc_real,"schedule_fil", show = "no", mode = 0)
            graph_7 = self.schedule(self.all_action_fil,self.all_soc_real,"schedule_fil", show = "no", mode = 1)
                
            pp.savefig(graph_4)
            pp.savefig(graph_5)
            pp.savefig(graph_6)
            pp.savefig(graph_7)
            pp.close()

            self.all_action = pd.DataFrame(np.ravel(self.all_action))
            self.all_action_fil = pd.DataFrame(np.ravel(self.all_action_fil))
            self.sell_PVout = pd.DataFrame(np.ravel(self.sell_PVout))
            self.PV_real = pd.DataFrame(np.ravel(self.PV_real))
            self.all_alpha = pd.DataFrame(np.ravel(self.all_alpha))
            self.all_beta = pd.DataFrame(np.ravel(self.all_beta))

            generation_data = pd.concat([self.all_action,self.sell_PVout,self.all_action_fil,self.PV_real,self.all_alpha, self.all_beta], axis=1)

            label_name = [self.PV + "_charge_discharge",self.PV + "_PV",self.PV + "_charge_discharge_real", self.PV + "_PV_real",self.PV + "_alpha",self.PV + "_beta"]
            generation_data.columns = label_name
            generation_data.to_csv(self.PV + "_generation.csv")

        return state, optimize_rewards, done, {}
    
    def reset(self): # 状態を初期化
        self.time = 0
        self.count = 0
        self.battery = 0
        self.battery_real = 0
        self.battery_true = 0 
        self.days = 1
        self.PV_out_time = self.PV_out[self.time]
        self.lower_time = self.lower_data[self.time]
        self.upper_time = self.upper_data[self.time]
        #self.PV_pdf_time = self.pdf_data[self.time]
        self.alpha_data_time = self.alpha_data[self.time]
        self.beta_data_time = self.beta_data[self.time]
        if self.PV_out_time < 0:
            self.PV_out_time = np.array([0])
        self.price_time = self.price[self.time]
        self.true_price_time = self.true_price[self.time]
        self.MAX_price = max(self.price[0:48])
        self.MAX_alpha = max(self.alpha_data[0:48])
        self.reward_action = []
        self.reward_soc = []
        self.reward_soc_real = []
        self.rewards_PV_real = []
        self.total_rewards_real = []
        self.reward_PV = []
        self.total_reward = []
        self.optimize_rewards = []
        self.all_PV_out_time = []
        self.all_soc = []
        self.all_soc_real = []
        self.all_battery = []
        self.all_price = []
        self.all_price_true = []
        self.all_time = []
        self.all_count = []
        self.all_action = []
        self.all_action_fil = []
    
        state = [self.time/24, self.PV_out_time/2, self.battery/4, self.price_time/self.MAX_price, self.alpha_data_time/self.MAX_alpha, self.beta_data_time]
        return state

    def render(self, mode='human', close=False):
        pass

    def close(self): 
        pass

    def seed(self): 
        pass

#メインルーチン    
    def main_root(self, mode, num_episodes, train_days, episode, model_name):
        if mode == "learn":
            self.model = PPO2("MlpPolicy", env, gamma = 0.9, verbose=0, learning_rate = 0.0001, n_steps = 48) # モデルの定義(A2C) 

            #モデルの学習
            self.model.learn(total_timesteps=num_episodes*train_days*episode)
        
        if mode == "test":
            #モデルのロード
            self.model = PPO2.load(model_name)
            #モデルのテスト
            obs = env.reset() # 最初のstate
            for i in range(0, num_episodes*(self.test_days - 1)):
                action, _ = self.model.predict(obs)
                obs, reward, done, _ = self.step(action)