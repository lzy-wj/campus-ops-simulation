import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import graphviz
import matplotlib.font_manager as fm
import os
import heapq  # 用于排队仿真的优先队列

# ==========================================
# 1. 全局基础配置与工具
# ==========================================

st.set_page_config(
    page_title="CampusOps: 校园运营仿真系统",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

def setup_chinese_font():
    """配置中文字体，确保在不同环境下中文显示正常"""
    try:
        # 方案1：项目内字体
        font_path = os.path.join(os.path.dirname(__file__), 'fonts', 'SourceHanSansCN-Regular.otf')
        if os.path.exists(font_path):
            fm.fontManager.addfont(font_path)
            plt.rcParams['font.sans-serif'] = ['Source Han Sans CN', 'sans-serif']
            plt.rcParams['axes.unicode_minus'] = False
            return
        
        # 方案2：常见系统字体
        font_list = ['Noto Sans CJK SC', 'SimHei', 'Microsoft YaHei', 'PingFang SC', 'Heiti TC']
        available_fonts = [f.name for f in fm.fontManager.ttflist]
        for font in font_list:
            if font in available_fonts:
                plt.rcParams['font.sans-serif'] = [font]
                plt.rcParams['axes.unicode_minus'] = False
                return
        
        # 方案3：后备方案
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
    except Exception as e:
        st.warning(f"字体加载遇到问题: {e}")

plt.style.use('seaborn-v0_8-whitegrid')
setup_chinese_font()

# ==========================================
# 2. 核心类定义：库存与排队
# ==========================================

class InventorySimulation:
    """(s, S) 库存策略仿真核心类"""
    def __init__(self, params):
        self.params = params
        self.s = params['s']
        self.S = params['S']
        self.T = params['T']
        self.lam = params['lam']
        self.avg_demand = params.get('avg_demand', 1)
        self.L = params['L']
        self.r = params['r']
        self.K = params['K']
        self.c_unit = params['c']
        self.h = params['h']
        
        # 状态初始化
        self.t = 0.0
        self.x = self.S
        self.y = 0
        self.C = 0.0
        self.H = 0.0
        self.R = 0.0
        self.t_C = 0.0
        self.t_O = float('inf')
        self.history = []

    def _generate_next_arrival(self):
        # 避免除以0
        if self.lam <= 0: return float('inf')
        U = np.random.uniform(0, 1)
        return - (1.0 / self.lam) * np.log(U)

    def _generate_demand_size(self):
        return max(1, np.random.poisson(self.avg_demand))

    def _calculate_ordering_cost(self, quantity):
        if quantity <= 0: return 0
        return self.K + self.c_unit * quantity

    def run(self):
        np.random.seed(int(self.params['seed']))
        self.t = 0.0
        self.x = self.S
        self.y = 0
        self.C = 0.0
        self.H = 0.0
        self.R = 0.0
        self.t_C = self._generate_next_arrival()
        self.t_O = float('inf')
        
        self.history.append({
            '时间': 0.0, '现有库存': self.x, '在途订单': self.y, 
            '事件类型': '初始化', '累计利润': 0.0, '变动量': 0
        })
        
        while True:
            next_event_time = min(self.t_C, self.t_O)
            if next_event_time > self.T:
                break
                
            if self.t_C <= self.t_O: # 顾客到达事件
                event_time = self.t_C
                self.H += self.h * self.x * (event_time - self.t)
                self.t = event_time
                D = self._generate_demand_size()
                w = min(D, self.x)
                lost = D - w
                self.R += w * self.r
                self.x -= w
                triggered_order = False
                if self.x < self.s and self.y == 0:
                    self.y = self.S - self.x
                    self.t_O = self.t + self.L
                    triggered_order = True
                
                current_profit = self.R - self.C - self.H
                self.history.append({
                    '时间': self.t, '现有库存': self.x, '在途订单': self.y,
                    '事件类型': '缺货损失' if lost > 0 else ('顾客购买' if not triggered_order else '顾客购买并订货'),
                    '累计利润': current_profit, '变动量': -w
                })
                self.t_C = self.t + self._generate_next_arrival()
            else: # 订单送达事件
                event_time = self.t_O
                self.H += self.h * self.x * (event_time - self.t)
                self.t = event_time
                cost_order = self._calculate_ordering_cost(self.y)
                self.C += cost_order
                self.x += self.y
                arrived_qty = self.y
                self.y = 0
                self.t_O = float('inf')
                current_profit = self.R - self.C - self.H
                self.history.append({
                    '时间': self.t, '现有库存': self.x, '在途订单': self.y,
                    '事件类型': '订单送达', '累计利润': current_profit, '变动量': arrived_qty
                })

        self.H += self.h * self.x * (self.T - self.t)
        final_profit = self.R - self.C - self.H
        df_log = pd.DataFrame(self.history)
        summary = {
            'final_profit': final_profit,
            'total_revenue': self.R,
            'total_ordering_cost': self.C,
            'total_holding_cost': self.H
        }
        return df_log, summary

class CanteenSimulation:
    """食堂排队仿真核心类 (M/M/c + 非齐次泊松)"""
    def __init__(self, params):
        self.env_duration = params['duration']
        self.num_servers = params['servers']
        self.arrival_rate_base = params['arrival_rate']
        self.service_rate = params['service_rate']
        self.is_peak_hour = params.get('peak_mode', False)
        self.seed = params.get('seed', 42)
        
    def _get_arrival_rate(self, t):
        """非齐次泊松过程：模拟饭点流量激增"""
        if not self.is_peak_hour:
            return self.arrival_rate_base
        # 饭点逻辑：中间1/3的时间段流量翻倍
        if self.env_duration * 0.33 < t < self.env_duration * 0.66:
            return self.arrival_rate_base * 2.5
        return self.arrival_rate_base

    def run(self):
        np.random.seed(int(self.seed))
        t = 0.0
        queue_len = 0
        servers_busy = 0
        events = [] # 优先队列
        
        # 初始化第一个到达
        rate0 = self._get_arrival_rate(0)
        if rate0 > 0:
            inter_arrival = np.random.exponential(1.0 / rate0)
            heapq.heappush(events, (inter_arrival, 0, None)) # 0=Arrival, 1=Departure
        
        history = [{'time': 0, 'queue': 0, 'busy': 0, 'in_system': 0, 'arrivals': 0, 'departures': 0}]
        
        while events:
            curr_time, event_type, _ = heapq.heappop(events)
            
            if curr_time > self.env_duration:
                break
            
            # 记录上一刻状态
            last = history[-1]
            history.append({
                'time': curr_time, 
                'queue': last['queue'], 
                'busy': last['busy'], 
                'in_system': last['in_system'],
                'arrivals': last['arrivals'],
                'departures': last['departures']
            })
            
            t = curr_time
            
            if event_type == 0: # Arrive
                # Schedule next arrival
                rate_t = self._get_arrival_rate(t)
                if rate_t > 0:
                    next_dt = np.random.exponential(1.0 / rate_t)
                    if t + next_dt <= self.env_duration:
                        heapq.heappush(events, (t + next_dt, 0, None))
                
                # Handle current arrival
                last_arrivals = history[-1]['arrivals']
                # Update counters for this step
                curr_arrivals = last_arrivals + 1
                curr_departures = history[-1]['departures']
                
                if servers_busy < self.num_servers:
                    servers_busy += 1
                    # Avoid division by zero
                    if self.service_rate > 0:
                        srv_t = np.random.exponential(1.0 / self.service_rate)
                        heapq.heappush(events, (t + srv_t, 1, None))
                else:
                    queue_len += 1
            
            elif event_type == 1: # Depart
                # Update counters
                curr_arrivals = history[-1]['arrivals']
                curr_departures = history[-1]['departures'] + 1
                
                if queue_len > 0:
                    queue_len -= 1
                    if self.service_rate > 0:
                        srv_t = np.random.exponential(1.0 / self.service_rate)
                        heapq.heappush(events, (t + srv_t, 1, None))
                else:
                    servers_busy -= 1
            
            history.append({
                'time': t, 
                'queue': queue_len, 
                'busy': servers_busy, 
                'in_system': queue_len + servers_busy,
                'arrivals': curr_arrivals,
                'departures': curr_departures
            })
            
        return pd.DataFrame(history)

# ==========================================
# 3. 界面渲染逻辑：库存系统
# ==========================================

def render_inventory_ui(sim_params):
    st.title("📦 (s, S) 库存策略仿真与优化")
    st.markdown("**模块功能：** 模拟校园超市/小卖部的库存管理，寻找最优补货策略。")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📖 模型原理", 
        "💻 算法源码",
        "🕹️ 单次仿真", 
        "📈 敏感性分析", 
        "🎯 策略优化"
    ])

    # === Tab 1: 模型原理 ===
    with tab1:
        st.header("带丢失销售的库存模型原理")
        st.markdown("""
        **核心假设：**
        * **需求过程**：顾客到达服从泊松过程($\lambda$)，需求量服从泊松分布。
        * **补货策略**：$(s, S)$ 策略。当库存 $x < s$ 且无在途订单时，补货至 $S$。
        * **成本构成**：固定订货成本 $K$，单位变动成本 $c$，单位持有成本 $h$。
        * **丢失销售**：缺货时需求直接流失 (Lost Sales)。
        """)
        
        col_g1, col_g2 = st.columns(2)
        with col_g1:
            st.subheader("流程图")
            graph = graphviz.Digraph()
            graph.attr(rankdir='TB')
            graph.node('Start', '开始\nt=0, x=S', shape='oval')
            graph.node('Check', '下一事件?', shape='diamond')
            graph.node('Cust', '顾客到达\n需求 D', shape='box', color='blue')
            graph.node('Order', '订单送达\n入库 +Q', shape='box', color='green')
            graph.node('Decide', '需要订货?\nx < s', shape='diamond')
            graph.node('Place', '下订单\nQ = S-x', shape='box', style='filled', color='orange')
            
            graph.edge('Start', 'Check')
            graph.edge('Check', 'Cust', label='t_C min')
            graph.edge('Check', 'Order', label='t_O min')
            graph.edge('Cust', 'Decide')
            graph.edge('Decide', 'Place', label='Yes')
            graph.edge('Decide', 'Check', label='No')
            graph.edge('Place', 'Check')
            graph.edge('Order', 'Check')
            st.graphviz_chart(graph)
        
        with col_g2:
            st.subheader("目标函数")
            st.latex(r" \max \Pi = R - C_{order} - C_{hold} ")
            st.latex(r" C_{order} = \sum (K + c \cdot Q) ")
            st.latex(r" C_{hold} = \int_0^T h \cdot x(t) dt ")

    # === Tab 2: 算法源码 ===
    with tab2:
        st.header("核心仿真代码")
        st.code("""
# 核心事件循环逻辑
while True:
    next_event = min(t_arrival, t_order)
    if next_event > T: break
    
    if t_arrival <= t_order:
        # 处理顾客到达
        update_holding_cost()
        process_demand()
        if inventory < s and no_pending_order:
            place_order()
    else:
        # 处理订单到达
        update_holding_cost()
        receive_order()
        """, language='python')

    # === Tab 3: 单次仿真 ===
    with tab3:
        st.subheader(f"当前策略: (s={sim_params['s']}, S={sim_params['S']})")
        
        sim_engine = InventorySimulation(sim_params)
        df_result, summary = sim_engine.run()
        
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi1.metric("最终利润", f"{summary['final_profit']:,.2f}", delta_color="normal")
        kpi2.metric("总收入", f"{summary['total_revenue']:,.2f}")
        kpi3.metric("总订货成本", f"{summary['total_ordering_cost']:,.2f}", delta_color="inverse")
        kpi4.metric("总持有成本", f"{summary['total_holding_cost']:,.2f}", delta_color="inverse")

        st.markdown("### 📈 库存状态随时间变化图")
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        times = df_result['时间']
        inventory = df_result['现有库存']
        
        color_inv = 'tab:blue'
        ax1.set_xlabel('仿真时间')
        ax1.set_ylabel('现有库存量', color=color_inv)
        ax1.step(times, inventory, where='post', color=color_inv, label='现有库存', alpha=0.8, linewidth=2)
        
        ax1.axhline(y=sim_params['s'], color='orange', linestyle='--', label='再订货点 s')
        ax1.axhline(y=sim_params['S'], color='green', linestyle='--', label='最大库存 S')
        ax1.fill_between(times, 0, inventory, step='post', color=color_inv, alpha=0.1)

        # 标记特殊事件
        orders = df_result[df_result['事件类型'] == '顾客购买并订货']
        arrived = df_result[df_result['事件类型'] == '订单送达']
        stockouts = df_result[df_result['事件类型'] == '缺货损失']
        
        if not orders.empty:
            ax1.scatter(orders['时间'], orders['现有库存'], color='orange', marker='o', s=60, zorder=5, label='触发订货')
        if not arrived.empty:
            ax1.scatter(arrived['时间'], arrived['现有库存'], color='green', marker='^', s=80, zorder=5, label='订单送达')
        if not stockouts.empty:
            ax1.scatter(stockouts['时间'], stockouts['现有库存'], color='red', marker='x', s=80, zorder=5, label='发生缺货')

        ax1.legend(loc='upper right')
        st.pyplot(fig)

        col_pie, col_data = st.columns([1, 2])
        with col_pie:
            st.markdown("#### 成本结构")
            costs = [summary['total_holding_cost'], summary['total_ordering_cost']]
            if sum(costs) > 0:
                fig_pie, ax_pie = plt.subplots()
                ax_pie.pie(costs, labels=['持有成本', '订货成本'], autopct='%1.1f%%', colors=['#ff9999','#66b3ff'])
                st.pyplot(fig_pie)
        with col_data:
            st.markdown("#### 事件日志")
            format_dict = {
                '时间': '{:.2f}', 
                '现有库存': '{:.0f}', 
                '变动量': '{:.0f}', 
                '累计利润': '{:.2f}'
            }
            st.dataframe(df_result[['时间', '事件类型', '现有库存', '变动量', '累计利润']].style.format(format_dict))

    # === Tab 4: 敏感性分析 ===
    with tab4:
        st.header("📈 单参数敏感性分析")
        col_param, _ = st.columns([1, 2])
        with col_param:
            target = st.selectbox("选择分析变量", ["再订货点 s", "最大库存 S", "订货提前期 L"])
        
        results = []
        x_vals = []
        
        if target == "再订货点 s":
            x_range = range(0, int(sim_params['S']))
            for val in x_range:
                p = sim_params.copy(); p['s'] = val
                _, s = InventorySimulation(p).run()
                results.append(s['final_profit'])
                x_vals.append(val)
        elif target == "最大库存 S":
            x_range = range(int(sim_params['s'])+1, int(sim_params['s'])+51)
            for val in x_range:
                p = sim_params.copy(); p['S'] = val
                _, s = InventorySimulation(p).run()
                results.append(s['final_profit'])
                x_vals.append(val)
        elif target == "订货提前期 L":
            x_range = np.linspace(0.5, 10.0, 20)
            for val in x_range:
                p = sim_params.copy(); p['L'] = val
                _, s = InventorySimulation(p).run()
                results.append(s['final_profit'])
                x_vals.append(val)
        
        fig_sens, ax_sens = plt.subplots(figsize=(10, 4))
        ax_sens.plot(x_vals, results, marker='o', color='purple')
        ax_sens.set_xlabel(target)
        ax_sens.set_ylabel("总利润")
        ax_sens.set_title(f"参数 {target} 对利润的影响")
        st.pyplot(fig_sens)

    # === Tab 5: 策略优化 ===
    with tab5:
        st.header("🎯 (s, S) 全局策略优化")
        col_opt1, col_opt2 = st.columns(2)
        with col_opt1: s_max = st.slider("s 搜索上限", value=20, min_value=5, max_value=100, step=1)
        with col_opt2: S_max = st.slider("S 搜索上限", value=60, min_value=10, max_value=200, step=1)
        
        if st.button("🚀 开始优化计算"):
            progress = st.progress(0)
            heatmap_data = []
            
            step_s = max(2, s_max // 20)
            step_S = max(5, S_max // 20)
            
            s_vals = range(0, s_max+1, step_s)
            total = len(s_vals)
            
            for i, s_v in enumerate(s_vals):
                for S_v in range(s_v+5, S_max+1, step_S):
                    p = sim_params.copy(); p['s'] = s_v; p['S'] = S_v
                    _, res = InventorySimulation(p).run()
                    heatmap_data.append({'s': s_v, 'S': S_v, 'Profit': res['final_profit']})
                progress.progress((i+1)/total)
            
            if heatmap_data:
                df_hm = pd.DataFrame(heatmap_data).pivot(index='s', columns='S', values='Profit')
                fig_hm, ax_hm = plt.subplots(figsize=(10, 8))
                sns.heatmap(df_hm, cmap="viridis", ax=ax_hm, annot=False)
                ax_hm.invert_yaxis()
                ax_hm.set_title("利润热力图 (颜色越亮利润越高)")
                st.pyplot(fig_hm)
            else:
                st.warning("搜索范围无效，请调整参数。")

# ==========================================
# 4. 界面渲染逻辑：食堂排队系统
# ==========================================

def render_canteen_ui(params):
    st.title("🍔 校园食堂排队系统仿真")
    st.markdown("**模块功能：** 模拟饭点高峰期的人流拥堵情况，基于排队论优化窗口开设数量。")
    
    # 运行仿真
    sim = CanteenSimulation(params)
    df_res = sim.run()
    
    # 计算统计量
    if not df_res.empty:
        avg_q = df_res['queue'].mean()
        max_q = df_res['queue'].max()
        if params['servers'] > 0:
            utilization = df_res['busy'].mean() / params['servers']
        else:
            utilization = 0
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("平均排队人数", f"{avg_q:.1f} 人")
        col2.metric("排队峰值", f"{max_q} 人", delta_color="inverse")
        col3.metric("窗口利用率", f"{utilization:.1%}")
        state = "拥堵" if utilization > 0.85 else ("闲置" if utilization < 0.4 else "健康")
        col4.metric("系统状态", state)
        
        tab_v1, tab_v2, tab_v3 = st.tabs(["排队动态", "资源分析", "排队论原理"])
        
        with tab_v1:
            st.subheader("🚶‍♂️ 排队长度随时间变化")
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(df_res['time'], df_res['queue'], color='#ff6b6b', label='等待队列长度', linewidth=2)
            ax.plot(df_res['time'], df_res['in_system'], color='#4ecdc4', linestyle='--', alpha=0.6, label='系统总人数 (含就餐)')
            
            if params.get('peak_mode'):
                T = params['duration']
                ax.axvspan(T*0.33, T*0.66, color='yellow', alpha=0.15, label='饭点高峰时段')
            
            ax.set_xlabel("时间 (分钟)")
            ax.set_ylabel("人数")
            ax.legend()
            st.pyplot(fig)
            st.caption("红色实线表示正在排队且未获得服务的学生人数。当曲线激增时，表明服务能力不足。")
            
            st.divider()
            
            st.subheader("📊 累积流量图 (Cumulative Flow)")
            fig_cf, ax_cf = plt.subplots(figsize=(10, 5))
            ax_cf.plot(df_res['time'], df_res['arrivals'], color='blue', label='累积到达人数')
            ax_cf.plot(df_res['time'], df_res['departures'], color='green', label='累积离开人数')
            ax_cf.fill_between(df_res['time'], df_res['departures'], df_res['arrivals'], color='gray', alpha=0.2, label='滞留系统人数')
            ax_cf.set_xlabel("时间 (分钟)")
            ax_cf.set_ylabel("累积人数")
            ax_cf.legend()
            st.pyplot(fig_cf)
            st.caption("蓝色与绿色曲线之间的垂直距离代表系统内的总人数（排队+服务中）。")
        
        with tab_v2:
            col_res1, col_res2 = st.columns(2)
            with col_res1:
                st.subheader("窗口忙碌分布")
                fig2, ax2 = plt.subplots()
                sns.histplot(df_res['busy'], discrete=True, stat='probability', color='skyblue', ax=ax2)
                ax2.set_xticks(range(int(params['servers']) + 1))
                ax2.set_xlabel("同时忙碌的窗口数")
                st.pyplot(fig2)
            with col_res2:
                st.info("分析建议：\n\n如果直方图集中在最右侧，说明窗口几乎一直满负荷，需要增加窗口。\n\n如果集中在左侧，说明资源浪费。")
            
            st.divider()
            st.subheader("📉 窗口利用率随时间变化")
            fig_util, ax_util = plt.subplots(figsize=(10, 4))
            ax_util.fill_between(df_res['time'], 0, df_res['busy'], step='post', color='orange', alpha=0.5, label='忙碌窗口数')
            ax_util.axhline(y=params['servers'], color='red', linestyle='--', label='总窗口数')
            ax_util.set_xlabel("时间 (分钟)")
            ax_util.set_ylabel("窗口数")
            ax_util.set_ylim(0, params['servers'] + 1)
            ax_util.legend(loc='upper right')
            st.pyplot(fig_util)

        with tab_v3:
            st.markdown(r"""
            ### M/M/c 排队模型 (含非平稳扩展)
            
            本模块基于经典的排队论模型，但引入了**时间相关性**：
            """)
            
            st.latex(r"""
            \lambda(t) = \begin{cases} 
            \lambda_{base} & \text{非高峰期} \\
            2.5 \times \lambda_{base} & \text{饭点高峰期} 
            \end{cases} 
            """)
            
            st.markdown(r"""
            **参数定义：**
            * $c$: 服务台（窗口）数量
            * $\mu$: 单窗口服务率 ($1/\text{平均服务时间}$)
            * $\rho$: 系统利用率 $= \lambda / (c\mu)$
            """)
    else:
        st.error("仿真发生错误，未生成数据。")

# ==========================================
# 5. 主程序入口与侧边栏逻辑
# ==========================================

def main():
    st.sidebar.title("🏫 CampusOps")
    st.sidebar.info("随机过程大作业")
    
    app_mode = st.sidebar.radio("选择仿真场景:", 
        ["📦 库存管理", "🍔 食堂排队"])
    
    st.sidebar.markdown("---")
    
    if app_mode == "📦 库存管理":
        st.sidebar.subheader("⚙️ 库存模型参数")
        
        # --- UPDATE: 全部改为 Slider ---
        T = st.sidebar.slider("仿真周期 T (天)", value=100, min_value=10, max_value=365, step=10, help="仿真的总时间单位")
        lam = st.sidebar.slider("到达率 λ (人/天)", value=2.0, min_value=0.1, max_value=20.0, step=0.1, format="%.2f")
        L = st.sidebar.slider("提前期 L (天)", value=2.0, min_value=0.0, max_value=30.0, step=0.5, format="%.1f")
        
        with st.sidebar.expander("💰 成本与价格设置", expanded=False):
            r = st.slider("单位售价 r", value=50.0, min_value=1.0, max_value=200.0, step=1.0)
            c = st.slider("单位成本 c", value=20.0, min_value=1.0, max_value=150.0, step=1.0)
            h = st.slider("持有成本 h", value=1.0, min_value=0.1, max_value=50.0, step=0.1)
            K = st.slider("固定订货费 K", value=100.0, min_value=0.0, max_value=500.0, step=10.0)
        
        st.sidebar.subheader("📝 策略控制")
        s = st.sidebar.slider("再订货点 s", value=10, min_value=0, max_value=100, step=1)
        # 动态调整 S 的最小值，使其大于 s
        S = st.sidebar.slider("最大库存 S", value=max(s+1, 40), min_value=s+1, max_value=200, step=1)
        seed = st.sidebar.slider("随机种子", value=42, min_value=0, max_value=1000, step=1)
        
        inv_params = {
            'T': T, 'lam': lam, 'avg_demand': 1, 'L': L,
            'r': r, 'c': c, 'h': h, 'K': K,
            's': s, 'S': S, 'seed': seed
        }
        
        render_inventory_ui(inv_params)
        
    else: # 食堂排队模式
        st.sidebar.subheader("⚙️ 排队模型参数")
        
        # --- UPDATE: 全部改为 Slider ---
        duration = st.sidebar.slider("仿真时长 (分)", value=120, min_value=30, max_value=480, step=10)
        peak_mode = st.sidebar.checkbox("🔥 启用饭点高峰", value=True)
        
        st.sidebar.subheader("👥 人流设置")
        arrival_rate = st.sidebar.slider("基础到达率 (人/分)", value=2.0, min_value=0.5, max_value=10.0, step=0.1, format="%.1f")
        
        st.sidebar.subheader("🏪 窗口设置")
        servers = st.sidebar.slider("开放窗口数 c", value=3, min_value=1, max_value=20, step=1)
        service_time = st.sidebar.slider("平均打饭时间 (秒)", value=30, min_value=5, max_value=120, step=5)
        
        seed = st.sidebar.slider("随机种子", value=42, min_value=0, max_value=1000, step=1)
        
        q_params = {
            'duration': duration,
            'arrival_rate': arrival_rate,
            'peak_mode': peak_mode,
            'servers': servers,
            'service_rate': 60.0 / service_time if service_time > 0 else 60.0,
            'seed': seed
        }
        
        render_canteen_ui(q_params)

if __name__ == "__main__":
    main()