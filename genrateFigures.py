Class getFigures():
    def schedule(self, action, soc, name, show, mode):
        fig = plt.figure(figsize=(16, 9), dpi=80)
        ax1 = fig.add_subplot(111)
        ax2 = ax1.twinx()
        ax2.set_ylim([-1,101])
        ax1.tick_params(axis='x', labelsize=35)
        ax1.tick_params(axis='y', labelsize=35)
        ax2.tick_params(axis='x', labelsize=35)
        ax2.tick_params(axis='y', labelsize=35)
        if self.mode == "learn":
            ax1.plot(self.all_time, action, "blue", drawstyle="steps-post",label="充放電")
            ax1.plot(self.all_time, self.all_PV_out_time, "Magenta",label="PV出力")
            ax2.plot(self.all_time, soc, "red",label="SoC")
        elif self.mode == "test":
            ax1.plot(self.all_count, action, "blue", drawstyle="steps-post",label="充放電")
            ax1.plot(self.all_count, self.all_PV_out_time, "Magenta",label="PV出力")
            ax2.plot(self.all_count, soc, "red",label="SoC")
        if mode == 0:
            if self.mode == "learn":
                ax1.plot(self.all_time, self.all_price, "green",drawstyle="steps-post",label="電力価格")
            elif self.mode == "test":
                ax1.plot(self.all_count, self.all_price_true, "green",drawstyle="steps-post",label="電力価格")
            ax1.set_ylabel("電力[kW] 電力価格[円]", fontname="MS Gothic",fontsize = 35)
        elif mode == 1:
            ax1.set_ylim([-2,2])
            ax1.set_ylabel("電力[kW]", fontname="MS Gothic",fontsize = 35)    
        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax1.legend(h1+h2, l1+l2, loc='upper left', prop={"family":"MS Gothic"},fontsize = 35).get_frame().set_alpha(0.0)
        if self.mode == "learn":
            ax1.set_xlim([0,23.5])
        elif self.mode == "test":
            ax1.set_xlim([0,23.5*(self.test_days - 1)])
        ax1.set_xlabel('時間[時]', fontname="MS Gothic",fontsize = 35)
        ax1.grid(True)
        ax2.set_ylabel("SoC[%]", fontname="MS Gothic",fontsize = 35)
        if show == "yes":
            plt.show()
        plt.close()

        return fig
    
    def graph(self, y, label_name, x_label, y_label, show):
        fig = plt.figure(figsize=(18, 9), dpi=80)
        plt.plot(np.arange(self.episode), y, label = "報酬")
        plt.legend(prop={"family":"MS Gothic"},fontsize = 35)
        plt.xlabel("学習回数", fontname="MS Gothic",fontsize = 30)
        plt.ylabel("報酬", fontname="MS Gothic",fontsize = 30)
        if show == "yes":
            plt.show()
        plt.close()
        
        return fig
