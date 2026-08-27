import os
import time
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import threading
import queue

# Import from the hybrid data generator
from data_generator import (
    generate_milp_feasible_instance, quick_feasibility_check,
    verify_milp_feasibility,
    induce_infeasibility,  
    save_instance_txt, assign_time_windows, load_instance_txt
)
# Import from the robust solver
from milp import solve_evrptw

def make_type_code(inst_type: str, tw_key: str) -> str:
    suffix = {'wide': '', 'medium': 'm', 'tight': 't'}[tw_key]
    return f"{inst_type}{suffix}"

def make_station_label(st_mode: str, k: int) -> str:
    return "Rnd" if st_mode == "Random" else f"{k}S"

# --- HELPER: Collapsible Frame ---
class CollapsibleFrame(ttk.Frame):
    def __init__(self, parent, title=""):
        super().__init__(parent)
        self.columnconfigure(0, weight=1)
        self.show = tk.IntVar(value=0)
        self.title_frame = ttk.Frame(self)
        self.title_frame.grid(row=0, column=0, sticky="ew")
        self.toggle_btn = ttk.Checkbutton(self.title_frame, text=f"▶ {title}", 
                                          variable=self.show, style='Toolbutton', 
                                          command=self.toggle)
        self.toggle_btn.pack(side="left", fill="x", expand=True)
        self.sub_frame = ttk.Frame(self, relief="sunken", borderwidth=1)

    def toggle(self):
        if self.show.get():
            self.sub_frame.grid(row=1, column=0, sticky="nsew", padx=2, pady=2)
            self.toggle_btn.configure(text=self.toggle_btn.cget("text").replace("▶", "▼"))
        else:
            self.sub_frame.grid_forget()
            self.toggle_btn.configure(text=self.toggle_btn.cget("text").replace("▼", "▶"))

class SynthChargeApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("SynthCharge Generator & Solver")
        self.geometry("1200x950")
        
        self.progress_queue = queue.Queue()
        self._preview_map = {} 
        self.after_id = None
        
        style = ttk.Style(self)
        style.theme_use('clam')
        
        self.create_widgets()
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.check_progress_queue()

    def on_closing(self):
        if self.after_id:
            self.after_cancel(self.after_id)
        self.destroy()

    def create_widgets(self):
        # HEADER
        header_frame = ttk.Frame(self)
        header_frame.pack(fill=tk.X, padx=10, pady=5)
        ttk.Label(header_frame, text="SynthCharge", font=("Segoe UI", 16, "bold")).pack(side=tk.LEFT)
        ttk.Button(header_frame, text="📘 How It Works (README)", command=self.show_readme).pack(side=tk.RIGHT)

        # MAIN TABS
        main_paned_window = ttk.PanedWindow(self, orient=tk.VERTICAL)
        main_paned_window.pack(fill=tk.BOTH, expand=True)

        self.tabs = ttk.Notebook(main_paned_window)
        main_paned_window.add(self.tabs, weight=1)

        self.gen_tab = ttk.Frame(self.tabs)
        self.exp_tab = ttk.Frame(self.tabs)
        self.solve_tab = ttk.Frame(self.tabs)
        self.view_tab = ttk.Frame(self.tabs)
        
        self.tabs.add(self.gen_tab,   text="Benchmark Generator")
        self.tabs.add(self.exp_tab,   text="Experimental Generator")
        self.tabs.add(self.solve_tab, text="Solve Instance")
        self.tabs.add(self.view_tab,  text="View Instance")

        self._create_benchmark_tab_widgets()
        self._create_experimental_tab_widgets()
        self._create_solver_tab_widgets()
        self._create_viewer_tab_widgets()

        status_frame = ttk.Frame(main_paned_window, height=25)
        main_paned_window.add(status_frame, weight=0)
        self.status_var = tk.StringVar(value="Status: Idle")
        status_label = ttk.Label(status_frame, textvariable=self.status_var, anchor=tk.W)
        status_label.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10, pady=2)

    def show_readme(self):
        win = tk.Toplevel(self)
        win.title("How It Works - SynthCharge Logic")
        win.geometry("700x600")
        txt = scrolledtext.ScrolledText(win, font=("Consolas", 10), padx=10, pady=10)
        txt.pack(fill=tk.BOTH, expand=True)
        manual = """SynthCharge Generation Pipeline (v2.0)
======================================
1. INITIALIZATION
   Input: N (Customers), S (Stations), Seed, Type (C/R/RC)
   Set Random State(Seed)

2. SPATIAL TOPOLOGY
   Step A: Place Depot D0 (Center/Random/Custom)
   Step B: Generate Customers (C_1...C_N) with Safety Radius (0.04)

3. PHYSICS ENGINE (Smart Config)
   IF Battery == 'Adaptive' (Default):
       BATTERY_CAPACITY (Q) = 0.4 * Max_Diagonal
   ELSE (Fixed):
       Use User_Defined Q
   RANGE = Q / Consumption_Rate (r)

4. INFRASTRUCTURE LAYER
   Step A: Adaptive Repair (Midpoints + Isolated Paths)
   Step B: Count Enforcement (Match User 'S')
   Step C: Depot Charging (Optional S1 at D0)

5. TEMPORAL LAYER
   Horizon H = User_Input (Default 10.0)
   Assign Service_Time ~ U(User_Min, User_Max)
   Assign Windows (Ready, Due) using Linear Staggering

6. FEASIBILITY FILTER
   Check 1 (Heuristic): Reachability & Capacity
   Check 2 (MILP - Exact): If N <= 10, solve optimizer.
"""
        txt.insert(tk.END, manual)
        txt.config(state='disabled')

    # --- BENCHMARK TAB ---
    def _create_benchmark_tab_widgets(self):
        main_frame = ttk.Frame(self.gen_tab, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        main_frame.columnconfigure(0, weight=1)

        # 1. CORE
        core_frame = ttk.LabelFrame(main_frame, text="Core Parameters", padding="10")
        core_frame.grid(row=0, column=0, sticky="ew", pady=(0, 5))
        core_frame.columnconfigure(1, weight=1)
        core_frame.columnconfigure(3, weight=1)
        
        ttk.Label(core_frame, text="Output Directory:").grid(row=0, column=0, sticky="e", padx=5)
        self.out_dir = tk.StringVar()
        ttk.Entry(core_frame, textvariable=self.out_dir).grid(row=0, column=1, columnspan=3, sticky="ew")
        ttk.Button(core_frame, text="Browse...", command=self.browse).grid(row=0, column=4, padx=5)
        
        ttk.Label(core_frame, text="# Customers:").grid(row=1, column=0, sticky="e", padx=5)
        self.n_cust = tk.IntVar(value=50)
        ttk.Entry(core_frame, textvariable=self.n_cust, width=8).grid(row=1, column=1, sticky="w")
        
        ttk.Label(core_frame, text="# Instances (per type):").grid(row=1, column=2, sticky="e", padx=5)
        self.n_inst = tk.IntVar(value=1)
        ttk.Entry(core_frame, textvariable=self.n_inst, width=8).grid(row=1, column=3, sticky="w")

        # 2. SPATIAL
        spatial_frame = ttk.LabelFrame(main_frame, text="Spatial Configuration", padding="10")
        spatial_frame.grid(row=1, column=0, sticky="ew", pady=(0, 5))
        spatial_frame.columnconfigure(1, weight=1)
        spatial_frame.columnconfigure(3, weight=1)

        ttk.Label(spatial_frame, text="Instance Type:").grid(row=0, column=0, sticky="e", padx=5)
        self.inst_type = tk.StringVar(value="RC")
        ttk.Combobox(spatial_frame, textvariable=self.inst_type, values=["C", "R", "RC", "All"], width=8, state="readonly").grid(row=0, column=1, sticky="w")

        ttk.Label(spatial_frame, text="# Ext. Stations:").grid(row=0, column=2, sticky="e", padx=5)
        self.n_stat = tk.IntVar(value=3)
        self.n_stat_entry = ttk.Entry(spatial_frame, textvariable=self.n_stat, width=8)
        self.n_stat_entry.grid(row=0, column=3, sticky="w")
        
        ttk.Label(spatial_frame, text="Station Mode:").grid(row=1, column=0, sticky="e", padx=5)
        self.station_mode = tk.StringVar(value="Fixed")
        ttk.Combobox(spatial_frame, textvariable=self.station_mode, values=["Fixed", "Random"], width=8, state="readonly").grid(row=1, column=1, sticky="w")
        
        self.cluster_label = ttk.Label(spatial_frame, text="# Clusters:")
        self.cluster_label.grid(row=1, column=2, sticky="e", padx=5)
        self.n_clusters = tk.IntVar(value=3)
        self.n_clusters_entry = ttk.Entry(spatial_frame, textvariable=self.n_clusters, width=8)
        self.n_clusters_entry.grid(row=1, column=3, sticky="w")

        self.cluster_mode_label = ttk.Label(spatial_frame, text="Cluster Mode:")
        self.cluster_mode_label.grid(row=2, column=0, sticky="e", padx=5)
        self.cluster_mode = tk.StringVar(value="Fixed")
        self.cluster_mode_combo = ttk.Combobox(spatial_frame, textvariable=self.cluster_mode, values=["Fixed", "Random"], width=8, state="readonly")
        self.cluster_mode_combo.grid(row=2, column=1, sticky="w")

        self.std_label = ttk.Label(spatial_frame, text="Spread (Std):")
        self.std_label.grid(row=2, column=2, sticky="e", padx=5)
        self.cluster_std = tk.DoubleVar(value=0.15)
        self.std_entry = ttk.Entry(spatial_frame, textvariable=self.cluster_std, width=8)
        self.std_entry.grid(row=2, column=3, sticky="w")

        self.ratio_label = ttk.Label(spatial_frame, text="Mix Ratio (%):")
        self.ratio_label.grid(row=3, column=0, sticky="e", padx=5)
        self.cluster_ratio = tk.DoubleVar(value=50.0)
        self.ratio_scale = ttk.Scale(spatial_frame, from_=0, to=100, variable=self.cluster_ratio, orient='horizontal')
        self.ratio_scale.grid(row=3, column=1, columnspan=2, sticky="ew", padx=5)
        self.ratio_entry = ttk.Entry(spatial_frame, textvariable=self.cluster_ratio, width=5)
        self.ratio_entry.grid(row=3, column=3, sticky="w")

        ttk.Label(spatial_frame, text="Depot Type:").grid(row=4, column=0, sticky="e", padx=5)
        self.depot_type = tk.StringVar(value="center")
        self.depot_type_combo = ttk.Combobox(spatial_frame, textvariable=self.depot_type, values=["center", "random", "custom"], width=8, state="readonly")
        self.depot_type_combo.grid(row=4, column=1, sticky="w")

        ttk.Label(spatial_frame, text="Depot X/Y:").grid(row=4, column=2, sticky="e", padx=5)
        self.depot_x = tk.DoubleVar(value=0.5)
        self.depot_y = tk.DoubleVar(value=0.5)
        fxy = ttk.Frame(spatial_frame)
        fxy.grid(row=4, column=3, sticky="w")
        self.depot_x_entry = ttk.Entry(fxy, textvariable=self.depot_x, width=4)
        self.depot_x_entry.pack(side=tk.LEFT)
        self.depot_y_entry = ttk.Entry(fxy, textvariable=self.depot_y, width=4)
        self.depot_y_entry.pack(side=tk.LEFT, padx=2)

        self.charger_at_depot = tk.BooleanVar(value=False)
        ttk.Checkbutton(spatial_frame, text="Charger at Depot", variable=self.charger_at_depot).grid(row=5, column=0, columnspan=2, sticky="w", padx=5, pady=2)

        # 3. TEMPORAL
        temp_frame = ttk.LabelFrame(main_frame, text="Temporal & Seed", padding="10")
        temp_frame.grid(row=2, column=0, sticky="ew", pady=(0, 5))
        temp_frame.columnconfigure(1, weight=1)
        temp_frame.columnconfigure(3, weight=1)
        
        ttk.Label(temp_frame, text="Time Window:").grid(row=0, column=0, sticky="e", padx=5)
        self.tw_type = tk.StringVar(value="wide")
        ttk.Combobox(temp_frame, textvariable=self.tw_type, values=["wide", "medium", "tight", "All"], width=8, state="readonly").grid(row=0, column=1, sticky="w")
        
        ttk.Label(temp_frame, text="Horizon (H):").grid(row=0, column=2, sticky="e", padx=5)
        self.time_horizon = tk.DoubleVar(value=10.0) 
        ttk.Entry(temp_frame, textvariable=self.time_horizon, width=8).grid(row=0, column=3, sticky="w")

        ttk.Label(temp_frame, text="Base Seed:").grid(row=1, column=0, sticky="e", padx=5)
        self.seed = tk.IntVar(value=42)
        self.seed_entry = ttk.Entry(temp_frame, textvariable=self.seed, width=8)
        self.seed_entry.grid(row=1, column=1, sticky="w")

        ttk.Label(temp_frame, text="Seed Mode:").grid(row=1, column=2, sticky="e", padx=5)
        self.seed_mode = tk.StringVar(value="Fixed")
        ttk.Combobox(temp_frame, textvariable=self.seed_mode, values=["Fixed", "Random"], width=8, state="readonly").grid(row=1, column=3, sticky="w")

        # 4. VEHICLE CONFIG (COLLAPSIBLE)
        self.veh_collapse = CollapsibleFrame(main_frame, "Vehicle & Service Configuration")
        self.veh_collapse.grid(row=3, column=0, sticky="ew", pady=5)
        vf = self.veh_collapse.sub_frame
        
        ttk.Label(vf, text="Fleet Size (K):").grid(row=0, column=0, sticky="e")
        self.max_vehicles = tk.IntVar(value=5)
        ttk.Spinbox(vf, from_=1, to=100, textvariable=self.max_vehicles, width=5).grid(row=0, column=1, sticky="w")
        
        ttk.Label(vf, text="Capacity (C):").grid(row=0, column=2, sticky="e")
        self.veh_capacity = tk.DoubleVar(value=1.5)
        ttk.Entry(vf, textvariable=self.veh_capacity, width=6).grid(row=0, column=3, sticky="w")
        
        ttk.Label(vf, text="Consumption (r):").grid(row=1, column=0, sticky="e")
        self.veh_consumption = tk.DoubleVar(value=0.25)
        vc_ent = ttk.Entry(vf, textvariable=self.veh_consumption, width=6)
        vc_ent.grid(row=1, column=1, sticky="w")
        vc_ent.bind("<KeyRelease>", self.update_range_label)
        
        ttk.Label(vf, text="Refuel Rate (g):").grid(row=1, column=2, sticky="e")
        self.veh_refuel = tk.DoubleVar(value=1.0)
        ttk.Entry(vf, textvariable=self.veh_refuel, width=6).grid(row=1, column=3, sticky="w")
        
        ttk.Label(vf, text="Velocity (v):").grid(row=1, column=4, sticky="e")
        self.veh_velocity = tk.DoubleVar(value=1.0)
        ttk.Entry(vf, textvariable=self.veh_velocity, width=6).grid(row=1, column=5, sticky="w")
        
        ttk.Label(vf, text="Battery Strategy:").grid(row=2, column=0, sticky="e")
        self.batt_strat = tk.StringVar(value="adaptive")
        frame_batt = ttk.Frame(vf)
        frame_batt.grid(row=2, column=1, columnspan=2, sticky="w")
        ttk.Radiobutton(frame_batt, text="Adaptive", variable=self.batt_strat, value="adaptive", command=self.update_range_label).pack(side="left")
        ttk.Radiobutton(frame_batt, text="Fixed", variable=self.batt_strat, value="fixed", command=self.update_range_label).pack(side="left")
        
        ttk.Label(vf, text="Fixed Capacity (Q):").grid(row=2, column=3, sticky="e")
        self.veh_battery = tk.DoubleVar(value=0.4)
        qb_ent = ttk.Entry(vf, textvariable=self.veh_battery, width=6)
        qb_ent.grid(row=2, column=4, sticky="w")
        qb_ent.bind("<KeyRelease>", self.update_range_label)
        
        # Service Time Range
        ttk.Label(vf, text="Service Time (Min/Max):").grid(row=3, column=0, sticky="e")
        self.serv_min = tk.DoubleVar(value=0.01)
        self.serv_max = tk.DoubleVar(value=0.03)
        frame_srv = ttk.Frame(vf)
        frame_srv.grid(row=3, column=1, columnspan=3, sticky="w")
        ttk.Entry(frame_srv, textvariable=self.serv_min, width=5).pack(side="left")
        ttk.Label(frame_srv, text="-").pack(side="left")
        ttk.Entry(frame_srv, textvariable=self.serv_max, width=5).pack(side="left")
        
        # LIVE RANGE LABEL
        self.range_lbl = ttk.Label(vf, text="Effective Range: (Adaptive)", foreground="blue")
        self.range_lbl.grid(row=3, column=4, columnspan=2, sticky="w")

        # 5. ACTIONS
        action_frame = ttk.LabelFrame(main_frame, text="Actions & Log", padding="10")
        action_frame.grid(row=4, column=0, sticky="ew")
        
        btn_subframe = ttk.Frame(action_frame)
        btn_subframe.pack(fill=tk.X, pady=5)
        
        self.btn_gen = ttk.Button(btn_subframe, text="Generate Instances", command=self.start_benchmark_generation)
        self.btn_gen.pack(side=tk.LEFT, padx=10)
        ttk.Button(btn_subframe, text="Preview Layout(s)", command=self.preview).pack(side=tk.LEFT, padx=10)
        
        self.bench_progress = ttk.Progressbar(action_frame, orient='horizontal', mode='determinate')
        self.bench_progress.pack(fill=tk.X, padx=10, pady=5)
        
        self.log_text = scrolledtext.ScrolledText(action_frame, height=6, state='disabled', font=("Consolas", 9))
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        preview_subframe = ttk.Frame(action_frame)
        preview_subframe.pack(pady=5, fill=tk.X, expand=True)
        ttk.Label(preview_subframe, text="Pick layout to open:").pack(side=tk.LEFT, padx=5)
        self.layout_choice = tk.StringVar()
        self.layout_menu = ttk.Combobox(preview_subframe, textvariable=self.layout_choice, width=25, state="readonly")
        self.layout_menu.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(preview_subframe, text="Open in New Window", command=self.open_layout).pack(side=tk.LEFT, padx=5)

        self.canvas_frame = ttk.Frame(main_frame)
        self.canvas_frame.grid(row=5, column=0, sticky="nsew", pady=10)
        main_frame.rowconfigure(5, weight=1)
        self._setup_dynamic_callbacks()
        self._update_widget_states()

    def update_range_label(self, event=None):
        try:
            strat = self.batt_strat.get()
            r = self.veh_consumption.get()
            if strat == "fixed":
                q = self.veh_battery.get()
                rng = q / r if r > 0 else 0
                self.range_lbl.config(text=f"Range: {rng:.2f} units")
            else:
                self.range_lbl.config(text="Range: Adaptive (Map dependent)")
        except:
            self.range_lbl.config(text="Range: Error")

    # --- EXPERIMENTAL TAB ---
    def _create_experimental_tab_widgets(self):
        main_frame = ttk.Frame(self.exp_tab, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        config_frame = ttk.LabelFrame(main_frame, text="Generation Parameters", padding="10")
        config_frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(config_frame, text="Num Customers (N):").grid(row=0, column=0, sticky="w", pady=2)
        self.exp_n_min = tk.IntVar(value=40)
        self.exp_n_max = tk.IntVar(value=100)
        ttk.Entry(config_frame, textvariable=self.exp_n_min, width=5).grid(row=0, column=1, padx=(0, 5))
        ttk.Label(config_frame, text="to").grid(row=0, column=2, padx=5)
        ttk.Entry(config_frame, textvariable=self.exp_n_max, width=5).grid(row=0, column=3)
        ttk.Label(config_frame, text="Num Stations (S):").grid(row=1, column=0, sticky="w", pady=2)
        self.exp_s_min = tk.IntVar(value=5)
        self.exp_s_max = tk.IntVar(value=12)
        ttk.Entry(config_frame, textvariable=self.exp_s_min, width=5).grid(row=1, column=1, padx=(0, 5))
        ttk.Label(config_frame, text="to").grid(row=1, column=2, padx=5)
        ttk.Entry(config_frame, textvariable=self.exp_s_max, width=5).grid(row=1, column=3)
        types_frame = ttk.Frame(config_frame)
        types_frame.grid(row=2, column=0, columnspan=6, sticky='w', pady=5)
        ttk.Label(types_frame, text="Randomize Types:").pack(side=tk.LEFT, anchor='w')
        self.exp_use_c = tk.BooleanVar(value=True)
        self.exp_use_r = tk.BooleanVar(value=True)
        self.exp_use_rc = tk.BooleanVar(value=True)
        ttk.Checkbutton(types_frame, text="Clustered", variable=self.exp_use_c).pack(side=tk.LEFT, padx=5)
        ttk.Checkbutton(types_frame, text="Random", variable=self.exp_use_r).pack(side=tk.LEFT, padx=5)
        ttk.Checkbutton(types_frame, text="Mixed", variable=self.exp_use_rc).pack(side=tk.LEFT, padx=5)
        tw_frame = ttk.Frame(config_frame)
        tw_frame.grid(row=3, column=0, columnspan=6, sticky='w', pady=2)
        self.exp_use_wide = tk.BooleanVar(value=True)
        self.exp_use_medium = tk.BooleanVar(value=True)
        self.exp_use_tight = tk.BooleanVar(value=True)
        ttk.Checkbutton(tw_frame, text="Wide TW", variable=self.exp_use_wide).pack(side=tk.LEFT, padx=(120, 5))
        ttk.Checkbutton(tw_frame, text="Medium TW", variable=self.exp_use_medium).pack(side=tk.LEFT, padx=5)
        ttk.Checkbutton(tw_frame, text="Tight TW", variable=self.exp_use_tight).pack(side=tk.LEFT, padx=5)
        self.exp_random_depot = tk.BooleanVar(value=True)
        ttk.Checkbutton(config_frame, text="Random Depot", variable=self.exp_random_depot).grid(row=4, column=1, columnspan=2, sticky='w')
        ttk.Label(config_frame, text="Feasibility Ratio (%):").grid(row=5, column=0, sticky="w", pady=5)
        self.exp_feasibility_ratio = tk.DoubleVar(value=100.0)
        def _update_ratio_label(*args):
            try:
                val = self.exp_feasibility_ratio.get()
                self.exp_ratio_label.config(text=f"{val:.1f}% Feasible")
            except: pass
        self.exp_feasibility_ratio.trace_add("write", _update_ratio_label)
        slider = ttk.Scale(config_frame, from_=0, to=100, orient=tk.HORIZONTAL, variable=self.exp_feasibility_ratio, length=200)
        slider.grid(row=5, column=1, columnspan=3, sticky="we")
        self.exp_ratio_label = ttk.Label(config_frame, text="100.0% Feasible")
        self.exp_ratio_label.grid(row=5, column=5, padx=5, sticky='w')
        ttk.Label(config_frame, text="Total Instances:").grid(row=6, column=0, sticky="w", pady=5)
        self.exp_total_instances = tk.IntVar(value=1000)
        ttk.Entry(config_frame, textvariable=self.exp_total_instances, width=10).grid(row=6, column=1, sticky='w')
        ttk.Label(config_frame, text="Output Directory:").grid(row=7, column=0, sticky="w", pady=2)
        self.exp_out_dir = tk.StringVar()
        ttk.Entry(config_frame, textvariable=self.exp_out_dir, width=40).grid(row=7, column=1, columnspan=4, sticky="we")
        ttk.Button(config_frame, text="Browse...", command=lambda: self.browse(self.exp_out_dir)).grid(row=7, column=5, padx=5)
        # --- NEW CONTROL ---
        ttk.Label(config_frame, text="Infeasibility Type:").grid(row=8, column=0, sticky="w")
        
        self.exp_infeas_type = tk.StringVar(value="Random")
        
        ttk.Combobox(
            config_frame,
            textvariable=self.exp_infeas_type,
            values=["Random", "energy", "load", "time", "stations"],
            width=12,
            state="readonly"
        ).grid(row=8, column=1, sticky="w")
        
        control_frame = ttk.LabelFrame(main_frame, text="Control & Progress", padding="10")
        control_frame.pack(fill=tk.X, pady=10)
        self.exp_generate_button = ttk.Button(control_frame, text="Generate Dataset", command=self.start_experimental_generation)
        self.exp_generate_button.pack(pady=5)
        self.exp_progress_label_f = ttk.Label(control_frame, text="Feasible Instances Collected: 0 / 0")
        self.exp_progress_label_f.pack()
        self.exp_progress_bar_f = ttk.Progressbar(control_frame, orient='horizontal', length=400, mode='determinate')
        self.exp_progress_bar_f.pack(pady=2, fill=tk.X, padx=20)
        self.exp_progress_label_i = ttk.Label(control_frame, text="Infeasible Instances Collected: 0 / 0")
        self.exp_progress_label_i.pack()
        self.exp_progress_bar_i = ttk.Progressbar(control_frame, orient='horizontal', length=400, mode='determinate')
        self.exp_progress_bar_i.pack(pady=2, fill=tk.X, padx=20)
        self.exp_status_label = ttk.Label(control_frame, text="Status: Idle", foreground="blue", font=("Segoe UI", 10, "bold"))
        self.exp_status_label.pack(pady=5)

    def _create_solver_tab_widgets(self):
        main_frame = ttk.Frame(self.solve_tab, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        sfrm = ttk.Frame(main_frame)
        sfrm.pack(fill=tk.X, pady=(0, 10))
        sfrm.columnconfigure(1, weight=1)
        ttk.Label(sfrm, text="Instance File:").grid(row=0, column=0, sticky="e", padx=5)
        self.solve_file = tk.StringVar()
        ttk.Entry(sfrm, textvariable=self.solve_file).grid(row=0, column=1, sticky="ew")
        ttk.Button(sfrm, text="Browse...", command=self.browse_solve_file).grid(row=0, column=2, padx=5)
        ttk.Label(sfrm, text="Phase:").grid(row=1, column=0, sticky="e", padx=5, pady=5)
        self.solve_phase = tk.StringVar(value="C")
        ttk.Combobox(sfrm, textvariable=self.solve_phase, values=["A", "B", "C", "All"], width=8, state="readonly").grid(row=1, column=1, sticky="w")
        ttk.Label(sfrm, text="Time Limit (s):").grid(row=2, column=0, sticky="e", padx=5)
        self.solve_tl = tk.IntVar(value=300)
        ttk.Entry(sfrm, textvariable=self.solve_tl, width=8).grid(row=2, column=1, sticky="w")
        ttk.Button(sfrm, text="Solve", command=self.solve_instance).grid(row=3, column=1, pady=10, sticky="w")
        output_frame = ttk.Frame(main_frame)
        output_frame.pack(fill=tk.BOTH, expand=True)
        self.route_text = tk.Text(output_frame, height=8, wrap='none', width=50)
        self.route_text.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        self.solve_canvas_frame = ttk.Frame(output_frame)
        self.solve_canvas_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    def _create_viewer_tab_widgets(self):
        main_frame = ttk.Frame(self.view_tab, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        vfrm = ttk.Frame(main_frame)
        vfrm.pack(fill=tk.X, pady=(0, 10))
        vfrm.columnconfigure(1, weight=1)
        ttk.Label(vfrm, text="Instance .txt File:").grid(row=0, column=0, sticky="e", padx=5)
        self.view_file = tk.StringVar()
        ttk.Entry(vfrm, textvariable=self.view_file).grid(row=0, column=1, sticky="ew")
        ttk.Button(vfrm, text="Browse...", command=self.browse_view_file).grid(row=0, column=2, padx=5)
        ttk.Button(vfrm, text="Load & Show", command=self.view_instance).grid(row=1, column=1, pady=5, sticky="w")
        self.view_canvas_frame = ttk.Frame(main_frame)
        self.view_canvas_frame.pack(fill=tk.BOTH, expand=True)

    def _setup_dynamic_callbacks(self):
        self.station_mode.trace_add("write", self._update_widget_states)
        self.cluster_mode.trace_add("write", self._update_widget_states)
        self.inst_type.trace_add("write", self._update_widget_states)
        self.depot_type.trace_add("write", self._update_widget_states)
        self.seed_mode.trace_add("write", self._update_widget_states)

    # --- SAFETY FIX: Checks for attribute existence ---
    def _update_widget_states(self, *args):
        try:
            is_fixed_stat = self.station_mode.get() == "Fixed"
            self.n_stat_entry.config(state=tk.NORMAL if is_fixed_stat else tk.DISABLED)
            
            is_fixed_seed = self.seed_mode.get() == "Fixed"
            self.seed_entry.config(state=tk.NORMAL if is_fixed_seed else tk.DISABLED)
            
            is_custom_depot = self.depot_type.get() == "custom"
            self.depot_x_entry.config(state=tk.NORMAL if is_custom_depot else tk.DISABLED)
            self.depot_y_entry.config(state=tk.NORMAL if is_custom_depot else tk.DISABLED)
            
            itype = self.inst_type.get()
            has_clusters = itype in ["C", "RC", "All"]
            
            for w in [self.cluster_label, self.n_clusters_entry, self.cluster_mode_label, 
                      self.cluster_mode_combo, self.std_label, self.std_entry]:
                w.config(state=tk.NORMAL if has_clusters else tk.DISABLED)
                
            if has_clusters and self.cluster_mode.get() == "Random":
                self.n_clusters_entry.config(state=tk.DISABLED)
            
            is_mixed = itype in ["RC", "All"]
            self.ratio_scale.state(['!disabled'] if is_mixed else ['disabled'])
            self.ratio_label.config(state=tk.NORMAL if is_mixed else tk.DISABLED)
            self.ratio_entry.config(state=tk.NORMAL if is_mixed else tk.DISABLED)
        except AttributeError:
            pass 

    def browse(self, target_var=None):
        if target_var is None: target_var = self.out_dir
        d = filedialog.askdirectory()
        if d: target_var.set(d)
    def browse_solve_file(self):
        f = filedialog.askopenfilename()
        if f: self.solve_file.set(f)
    def browse_view_file(self):
        f = filedialog.askopenfilename()
        if f: self.view_file.set(f)

    # --- BENCHMARK GENERATION (THREADED & LOGGING) ---
    def start_benchmark_generation(self):
        out_base = self.out_dir.get().strip()
        if not out_base:
            messagebox.showerror("Error", "Specify output directory.")
            return
        
        # Capture ALL Params (Including new Physics)
        params = {
            'out_base': out_base,
            'n_cust': self.n_cust.get(),
            'raw_type': self.inst_type.get(),
            'raw_tw': self.tw_type.get(),
            'n_inst': self.n_inst.get(),
            'seed_mode': self.seed_mode.get(),
            'seed_val': self.seed.get(),
            'depot_type': self.depot_type.get(),
            'depot_coords': [float(self.depot_x.get()), float(self.depot_y.get())] if self.depot_type.get()=='custom' else None,
            'n_clusters': self.n_clusters.get(),
            'cluster_mode': self.cluster_mode.get(),
            'station_mode': self.station_mode.get(),
            'n_stat': self.n_stat.get(),
            'cluster_std': self.cluster_std.get(),
            'cluster_ratio': self.cluster_ratio.get(),
            'max_vehicles': self.max_vehicles.get(),
            'charger_at_depot': self.charger_at_depot.get(),
            'time_horizon': self.time_horizon.get(),
            
            # PHYSICS
            'capacity': self.veh_capacity.get(),
            'consumption': self.veh_consumption.get(),
            'refuel': self.veh_refuel.get(),
            'velocity': self.veh_velocity.get(),
            'batt_strat': self.batt_strat.get(),
            'batt_fixed': self.veh_battery.get(),
            'serv_min': self.serv_min.get(),
            'serv_max': self.serv_max.get()
        }
        
        self.status_var.set("Initializing...")
        self.log_text.config(state='normal')
        self.log_text.delete('1.0', tk.END)
        self.log_text.insert(tk.END, "--- Starting Generation Sequence ---\n")
        self.log_text.config(state='disabled')
        
        self.btn_gen.config(state=tk.DISABLED)
        self.bench_progress['value'] = 0
        self.bench_progress['maximum'] = 100 
        
        threading.Thread(target=self.run_benchmark_generation, args=(params,), daemon=True).start()

    def run_benchmark_generation(self, p):
        try:
            os.makedirs(p['out_base'], exist_ok=True)
            log_path = os.path.join(p['out_base'], "generation_log.txt")
            def write_to_file(msg):
                with open(log_path, "a") as f:
                    f.write(msg + "\n")
            write_to_file(f"--- Started Generation at {time.strftime('%H:%M:%S')} ---")
            
            types_to_run = ["C", "R", "RC"] if p['raw_type'] == "All" else [p['raw_type']]
            tws_to_run = ["wide", "medium", "tight"] if p['raw_tw'] == "All" else [p['raw_tw']]
            total_batches = len(types_to_run) * len(tws_to_run)
            total_files_needed = total_batches * p['n_inst']
            self.progress_queue.put(("bench_init", total_files_needed))
            
            batch_count = 0
            total_generated = 0
            start_time = time.time()
            rng_seed = np.random.default_rng() if p['seed_mode'] == "Random" else None

            for it in types_to_run:
                for tw_name in tws_to_run:
                    batch_count += 1
                    tw_key = tw_name.lower()
                    frac = {"wide":0.8, "medium":0.4, "tight":0.2}[tw_key]
                    saved = 0
                    trials = 0
                    
                    log_header = f"Batch {batch_count}/{total_batches}: {it}-{tw_key}"
                    self.progress_queue.put(("log", log_header))
                    write_to_file(log_header)
                    
                    while saved < p['n_inst']:
                        trials += 1
                        iter_start = time.time()
                        if trials % 5 == 0:
                            msg = f"Batch {batch_count}/{total_batches} ({it}-{tw_key}) | Trials: {trials} | Rejected: {trials - saved}"
                            self.progress_queue.put(("bench_pulse", msg))
                        
                        if rng_seed: sd = int(rng_seed.integers(0, 2**32-1))
                        else: sd = p['seed_val'] + (total_generated + trials - 1)
                        
                        nc = 0
                        if it != "R":
                            if p['cluster_mode'] == "Random":
                                r_tmp = np.random.default_rng(sd)
                                max_c = max(2, int(np.sqrt(p['n_cust'])))
                                nc = int(r_tmp.integers(2, max_c + 1))
                            else: nc = p['n_clusters']
                        
                        use_adaptive = (p['station_mode'] == "Random")
                        s_user = p['n_stat']
                        if use_adaptive:
                            r_tmp2 = np.random.default_rng(sd+1)
                            s_user = int(r_tmp2.integers(1, max(1, p['n_cust']//5)+1))

                        # Call Generator with NEW Physics Params
                        inst = generate_milp_feasible_instance(
                            n_customers=p['n_cust'], n_stations=s_user, instance_type=it, 
                            random_seed=sd, n_clusters=nc, depot_mode=p['depot_type'], custom_depot=p['depot_coords'], 
                            cluster_std=p['cluster_std'], cluster_ratio=p['cluster_ratio']/100.0, 
                            max_vehicles=p['max_vehicles'], use_adaptive_stations=use_adaptive, charger_at_depot=p['charger_at_depot'], 
                            time_horizon=p['time_horizon'],
                            # New Physics
                            vehicle_capacity=p['capacity'], consumption_rate=p['consumption'], velocity=p['velocity'],
                            refuel_rate=p['refuel'], battery_strategy=p['batt_strat'], fixed_battery_val=p['batt_fixed'],
                            service_min=p['serv_min'], service_max=p['serv_max']
                        )
                        inst['time_windows'] = assign_time_windows(p['n_cust'], fraction=frac, time_horizon=p['time_horizon'])
                        
                        if quick_feasibility_check(inst):
                            is_valid = True
                            if p['n_cust'] <= 10:
                                self.progress_queue.put(("bench_pulse", f"Verifying MILP (Trial {trials})..."))
                                is_valid = verify_milp_feasibility(inst['depot'], inst['customers'], inst['stations'], inst['demands'], inst['service_times'], inst['time_windows'], inst['battery_capacity'], inst['load_capacity'], inst['consumption_rate'], inst['velocity'], time_horizon=p['time_horizon'])
                            if is_valid:
                                code = make_type_code(it, tw_key)
                                lbl = make_station_label(p['station_mode'], s_user)
                                save_instance_txt(inst, p['out_base'], code, p['n_cust'], lbl, saved)
                                saved += 1
                                total_generated += 1
                                t_elapsed = time.time() - iter_start
                                succ_rate = (saved / trials) * 100
                                log_msg = f"  -> Saved {code}_{saved:03d} | Seed {sd} | Rate {succ_rate:.1f}% | {t_elapsed:.2f}s"
                                self.progress_queue.put(("log", log_msg))
                                write_to_file(log_msg)
                                self.progress_queue.put(("bench_update", (total_generated, f"Saved {total_generated}")))
            
            elapsed = time.time() - start_time
            final_msg = f"Done! Generated: {total_generated} in {elapsed:.2f}s"
            write_to_file(final_msg)
            self.progress_queue.put(("benchmark_done", final_msg))
            self.progress_queue.put(("log", "--- Generation Complete ---"))
        except Exception as e:
            self.progress_queue.put(("error", str(e)))

    # --- EXPERIMENTAL GENERATION ---
    def start_experimental_generation(self):
        self.status_var.set("Status: Starting experimental generation...")
        self.exp_generate_button.config(state=tk.DISABLED)
        self.exp_status_label.config(text="Running...", foreground="orange")
        threading.Thread(target=self.generate_experimental_set, daemon=True).start()

    def generate_experimental_set(self):

        try:
    
            out_dir = self.exp_out_dir.get().strip()
            if not out_dir:
                self.progress_queue.put(("error", "Experimental output directory is empty."))
                return
    
            os.makedirs(out_dir, exist_ok=True)
            os.makedirs(os.path.join(out_dir, "feasible"), exist_ok=True)
            os.makedirs(os.path.join(out_dir, "infeasible"), exist_ok=True)
    
            csv_path = os.path.join(out_dir, "experimental_summary.csv")
    
            with open(csv_path, "w") as f:
                f.write("InstanceID,Status,Type,TimeWindow,N,Stations,Seed,Feasible,InfeasMode,Severity\n")
    
            total = int(self.exp_total_instances.get())
            ratio = float(self.exp_feasibility_ratio.get()) / 100.0
    
            target_feas = int(round(total * ratio))
            target_inf = total - target_feas
    
            self.progress_queue.put(("set_targets", (target_feas, target_inf)))
    
            n_min, n_max = int(self.exp_n_min.get()), int(self.exp_n_max.get())
            s_min, s_max = int(self.exp_s_min.get()), int(self.exp_s_max.get())
    
            types = [t for t,v in [("C",self.exp_use_c),("R",self.exp_use_r),("RC",self.exp_use_rc)] if v.get()]
            tws = [t for t,v in [("wide",self.exp_use_wide),("medium",self.exp_use_medium),("tight",self.exp_use_tight)] if v.get()]
    
            if not types:
                self.progress_queue.put(("error","No instance types selected."))
                return
    
            if not tws:
                self.progress_queue.put(("error","No time window types selected."))
                return
    
            rng = np.random.default_rng()
    
            horizon = float(self.time_horizon.get())
    
            count_feas = 0
            count_inf = 0
            instance_id = 0
    
            def log_row(iid, feas_flag, it, tw, n, s, seed, mode, sev):
    
                status = "Feasible" if feas_flag else "Infeasible"
    
                with open(csv_path,"a") as f:
                    f.write(f"{iid},{status},{it},{tw},{n},{s},{seed},{feas_flag},{mode},{sev}\n")
    
    
            while count_feas < target_feas or count_inf < target_inf:
    
                if count_feas >= target_feas:
                    want_feasible = False
                elif count_inf >= target_inf:
                    want_feasible = True
                else:
                    want_feasible = rng.random() < ratio
    
    
                max_tries = 40
    
                generated_inst = None
                generated_feas = None
                generated_meta = None
    
    
                for _ in range(max_tries):
    
                  
    
                    n = int(rng.integers(n_min, n_max+1))
                    s = int(rng.integers(s_min, s_max+1))
                    it = str(rng.choice(types))
                    tw = str(rng.choice(tws))
                    seed = int(rng.integers(0,2**32-1))
    
                    std = float(rng.uniform(0.02,0.15))
                    mix = float(rng.uniform(0.2,0.8))
    
                    depot_mode = "random" if self.exp_random_depot.get() else "center"
    
                    inst = generate_milp_feasible_instance(
                        n,
                        s,
                        it,
                        random_seed=seed,
                        cluster_std=std,
                        cluster_ratio=mix,
                        use_adaptive_stations=True,
                        depot_mode=depot_mode,
                        charger_at_depot=self.charger_at_depot.get(),
                        time_horizon=horizon,
                        vehicle_capacity=self.veh_capacity.get(),
                        consumption_rate=self.veh_consumption.get(),
                        velocity=self.veh_velocity.get(),
                        refuel_rate=self.veh_refuel.get(),
                        battery_strategy=self.batt_strat.get(),
                        fixed_battery_val=self.veh_battery.get(),
                        service_min=self.serv_min.get(),
                        service_max=self.serv_max.get()
                    )
    
                    frac = {"wide":0.8,"medium":0.4,"tight":0.2}[tw]
    
                    inst['time_windows'] = assign_time_windows(
                        n,
                        fraction=frac,
                        time_horizon=horizon
                    )
    
                    severity = 0.0
                    infeas_mode = ""
                    if not want_feasible:
    
                        if self.exp_infeas_type.get() != "Random":
                            infeas_mode = self.exp_infeas_type.get()
                        else:
    
                            INFEAS_TYPES = ["energy","energy","energy","load","time","stations"]
    
                            infeas_mode = str(rng.choice(INFEAS_TYPES))
    
                        severity = float(rng.uniform(0.08,0.35))
    
                        inst = induce_infeasibility(
                            inst,
                            rng=rng,
                            mode=infeas_mode,
                            severity=severity
                        )
    
                        inst["infeasibility_type"] = infeas_mode
                        inst["infeasibility_severity"] = severity
    
    
                    is_feasible = quick_feasibility_check(inst)
    
    
                    if is_feasible == want_feasible:
    
                        generated_inst = inst
                        generated_feas = is_feasible
                        generated_meta = (it,tw,n,s,seed,infeas_mode,severity)
                        break
    
    
                if generated_inst is None:
    
                    generated_inst = inst
                    generated_feas = is_feasible
                    generated_meta = (it,tw,n,s,seed,infeas_mode,severity)
    
    
                it,tw,n,s,seed,infeas_mode,severity = generated_meta
    
                instance_id += 1
    
                code = make_type_code(it,tw)
    
                severity = round(severity if not generated_feas else 0.0,3)
    
    
                if generated_feas and count_feas < target_feas:
    
                    save_instance_txt(
                        generated_inst,
                        os.path.join(out_dir,"feasible"),
                        code,
                        n,
                        f"{s}S",
                        count_feas
                    )
    
                    log_row(instance_id,True,it,tw,n,s,seed,"",0.0)
    
                    count_feas += 1

                elif not generated_feas and count_inf < target_inf:

                    # create tag for infeasibility type
                    inf_code = f"{code}_inf_{infeas_mode}"
                
                    save_instance_txt(
                        generated_inst,
                        os.path.join(out_dir,"infeasible"),
                        inf_code,
                        n,
                        f"{s}S",
                        count_inf
                    )
                
                    log_row(instance_id,False,it,tw,n,s,seed,infeas_mode,severity)
                
                    count_inf += 1
    
                # elif not generated_feas and count_inf < target_inf:
    
                #     save_instance_txt(
                #         generated_inst,
                #         os.path.join(out_dir,"infeasible"),
                #         code,
                #         n,
                #         f"{s}S",
                #         count_inf
                #     )
    
                #     log_row(instance_id,False,it,tw,n,s,seed,infeas_mode,severity)
    
                #     count_inf += 1
    
    
                if (count_feas + count_inf) % 5 == 0:
                    self.progress_queue.put(("progress",(count_feas,count_inf)))
    
    
            self.progress_queue.put(
                ("done",f"Complete. Feasible={count_feas}, Infeasible={count_inf}")
            )
    
    
        except Exception as e:
    
            self.progress_queue.put(("error",str(e)))
    # ===========================
    # def generate_experimental_set(self):
    #     """
    #     Guarantees the user-selected feasible/infeasible ratio by:
    #       1) deciding the target label first (want_feasible)
    #       2) generating until an instance matches that label
    #       3) if we want infeasible, forcibly 'break' feasibility (energy/load) before checking
    #     """
    #     try:
    #         out_dir = self.exp_out_dir.get().strip()
    #         if not out_dir:
    #             self.progress_queue.put(("error", "Experimental output directory is empty."))
    #             return
    
    #         os.makedirs(out_dir, exist_ok=True)
    #         os.makedirs(os.path.join(out_dir, "feasible"), exist_ok=True)
    #         os.makedirs(os.path.join(out_dir, "infeasible"), exist_ok=True)
    
    #         csv_path = os.path.join(out_dir, "experimental_summary.csv")
    #         with open(csv_path, "w") as f:
    #             f.write("InstanceID,Status,Type,TimeWindow,N,Stations,Seed,Feasible,InfeasMode,Severity\n")
    
    #         total = int(self.exp_total_instances.get())
    #         ratio = float(self.exp_feasibility_ratio.get()) / 100.0
    #         t_feas = int(round(total * ratio))
    #         t_inf = total - t_feas
    #         self.progress_queue.put(("set_targets", (t_feas, t_inf)))
    
    #         n_min, n_max = int(self.exp_n_min.get()), int(self.exp_n_max.get())
    #         s_min, s_max = int(self.exp_s_min.get()), int(self.exp_s_max.get())
    
    #         types = [t for t, v in [("C", self.exp_use_c), ("R", self.exp_use_r), ("RC", self.exp_use_rc)] if v.get()]
    #         tws = [t for t, v in [("wide", self.exp_use_wide), ("medium", self.exp_use_medium), ("tight", self.exp_use_tight)] if v.get()]
    #         if not types:
    #             self.progress_queue.put(("error", "No instance types selected in Experimental Generator."))
    #             return
    #         if not tws:
    #             self.progress_queue.put(("error", "No time-window types selected in Experimental Generator."))
    #             return
    
    #         rng = np.random.default_rng()
    #         horizon = float(self.time_horizon.get())
    
    #         c_feas, c_inf = 0, 0
    #         global_id = 0
    
    #         # local helper to log and push progress
    #         def _log_row(_id, feas_flag, it, tw, n, s, sd, mode, severity):
    #             status_str = "Feasible" if feas_flag else "Infeasible"
    #             with open(csv_path, "a") as f:
    #                 f.write(f"{_id},{status_str},{it},{tw},{n},{s},{sd},{feas_flag},{mode},{severity}\n")
    
    #         while c_feas < t_feas or c_inf < t_inf:
    #             # Decide what we WANT next (target-first logic)
    #             if c_feas >= t_feas:
    #                 want_feas = False
    #             elif c_inf >= t_inf:
    #                 want_feas = True
    #             else:
    #                 want_feas = (rng.random() < ratio)
    
    #             # Generate until we match target label (bounded)
    #             max_tries = 40
    #             made_inst = None
    #             made_is_feas = None
    #             made_meta = None  # (it, tw, n, s, sd, infeas_mode)

    #             severity = 0.0
    #             infeas_mode = ""
                
    #             for _ in range(max_tries):
    #                 global_id += 1
    
    #                 n = int(rng.integers(n_min, n_max + 1))
    #                 s = int(rng.integers(s_min, s_max + 1))
    #                 it = str(rng.choice(types))
    #                 tw = str(rng.choice(tws))
    #                 sd = int(rng.integers(0, 2**32 - 1))
    #                 std = float(rng.uniform(0.02, 0.15))
    #                 r_mix = float(rng.uniform(0.2, 0.8))
    #                 dm = "random" if self.exp_random_depot.get() else "center"
    
    #                 inst = generate_milp_feasible_instance(
    #                     n, s, it,
    #                     random_seed=sd,
    #                     cluster_std=std,
    #                     cluster_ratio=r_mix,
    #                     use_adaptive_stations=True,
    #                     depot_mode=dm,
    #                     charger_at_depot=self.charger_at_depot.get(),
    #                     time_horizon=horizon,
    #                     vehicle_capacity=self.veh_capacity.get(),
    #                     consumption_rate=self.veh_consumption.get(),
    #                     velocity=self.veh_velocity.get(),
    #                     refuel_rate=self.veh_refuel.get(),
    #                     battery_strategy=self.batt_strat.get(),
    #                     fixed_battery_val=self.veh_battery.get(),
    #                     service_min=self.serv_min.get(),
    #                     service_max=self.serv_max.get()
    #                 )
    
    #                 frac = {"wide": 0.8, "medium": 0.4, "tight": 0.2}[tw]
    #                 inst['time_windows'] = assign_time_windows(n, fraction=frac, time_horizon=horizon)

                  
                   
    #                 if not want_feas:

    #                     # infeas_mode = str(rng.choice(["energy","load","time","stations"]))
    #                     INFEAS_TYPES = ["energy","load","time","stations"]

    #                     if hasattr(self, "exp_infeas_type") and self.exp_infeas_type.get() != "Random":
    #                         infeas_mode = self.exp_infeas_type.get()
    #                     else:
    #                         infeas_mode = str(rng.choice(INFEAS_TYPES))
                    
    #                     severity = float(rng.uniform(0.08,0.35))
                    
    #                     inst = induce_infeasibility(
    #                         inst,
    #                         rng=rng,
    #                         mode=infeas_mode,
    #                         severity=severity
    #                     )
                    
    #                     # store metadata (useful for analysis)
    #                     inst["infeasibility_type"] = infeas_mode
    #                     inst["infeasibility_severity"] = severity
    #                 # if not want_feas:
    #                 #     # Force infeasibility FAST (compatible with quick_feasibility_check)
    #                 #     infeas_mode = str(rng.choice(["energy", "energy", "energy", "load", "load"]))
    #                 #     inst = induce_infeasibility(inst, rng=rng, mode=infeas_mode)
    
    #                 is_feas = quick_feasibility_check(inst)
    
    #                 if is_feas == want_feas:
    #                     made_inst = inst
    #                     made_is_feas = is_feas
    #                     made_meta = (it, tw, n, s, sd, infeas_mode)
    #                     break
    
    #             # If we couldn't match the label after max_tries, fallback to last inst
    #             if made_inst is None:
    #                 made_inst = inst
    #                 made_is_feas = is_feas
    #                 made_meta = (it, tw, n, s, sd, infeas_mode if 'infeas_mode' in locals() else "")
    
    #             # it, tw, n, s, sd, infeas_mode = made_meta
    #             # code = make_type_code(it, tw)
    
    #             # # Save respecting remaining quotas
    #             # if made_is_feas and c_feas < t_feas:
    #             #     save_instance_txt(made_inst, os.path.join(out_dir, "feasible"), code, n, f"{s}S", c_feas)
    #             #     _log_row(global_id, True, it, tw, n, s, sd, infeas_mode, severity)
    #             #     c_feas += 1
    
    #             # elif (not made_is_feas) and c_inf < t_inf:
    #             #     save_instance_txt(made_inst, os.path.join(out_dir, "infeasible"), code, n, f"{s}S", c_inf)
    #             #     _log_row(global_id, False, it, tw, n, s, sd, infeas_mode, severity)
    #             #     c_inf += 1

    #             # unpack metadata (now including severity)
    #             it, tw, n, s, sd, infeas_mode = made_meta
                
    #             # round severity for clean CSV logging
    #             severity = round(severity if not made_is_feas else 0.0, 3)
    #             # severity = round(severity, 3)
                
    #             code = make_type_code(it, tw)
                
    #             # ---------- SAVE FEASIBLE ----------
    #             if made_is_feas and c_feas < t_feas:
                
    #                 save_instance_txt(
    #                     made_inst,
    #                     os.path.join(out_dir, "feasible"),
    #                     code,
    #                     n,
    #                     f"{s}S",
    #                     c_feas
    #                 )
                
    #                 _log_row(global_id, True, it, tw, n, s, sd, "", 0.0)
                
    #                 c_feas += 1
                
                
    #             # ---------- SAVE INFEASIBLE ----------
    #             elif (not made_is_feas) and c_inf < t_inf:
                
    #                 # encode infeasibility type + severity in filename
    #                 # tag = f"{infeas_mode}{int(severity*100)}"
                
    #                 # save_instance_txt(
    #                 #     made_inst,
    #                 #     os.path.join(out_dir, "infeasible"),
    #                 #     f"{code}_{tag}",
    #                 #     n,
    #                 #     f"{s}S",
    #                 #     c_inf
    #                 # )
    #                 save_instance_txt(
    #                     made_inst,
    #                     os.path.join(out_dir, "infeasible"),
    #                     code,
    #                     n,
    #                     f"{s}S",
    #                     c_inf
    #                 )
                
    #                 _log_row(global_id, False, it, tw, n, s, sd, infeas_mode, severity)
                
    #                 c_inf += 1
    
    #             # progress updates
    #             if (c_feas + c_inf) % 5 == 0:
    #                 self.progress_queue.put(("progress", (c_feas, c_inf)))
    
    #         self.progress_queue.put(("done", f"Complete. Feasible={c_feas}, Infeasible={c_inf}"))
    #     except Exception as e:
    #         self.progress_queue.put(("error", str(e)))
    # def generate_experimental_set(self):
    #     try:
    #         out_dir = self.exp_out_dir.get().strip()
    #         if not out_dir: return
    #         csv_path = os.path.join(out_dir, "experimental_summary.csv")
    #         with open(csv_path, "w") as f: f.write("InstanceID,Status,Type,TimeWindow,N,Stations,Seed,Feasible\n")
    #         total = self.exp_total_instances.get()
    #         ratio = self.exp_feasibility_ratio.get() / 100.0
    #         t_feas = int(total * ratio)
    #         t_inf = total - t_feas
    #         self.progress_queue.put(("set_targets", (t_feas, t_inf)))
    #         n_min, n_max = self.exp_n_min.get(), self.exp_n_max.get()
    #         s_min, s_max = self.exp_s_min.get(), self.exp_s_max.get()
    #         types = [t for t,v in [("C",self.exp_use_c), ("R",self.exp_use_r), ("RC",self.exp_use_rc)] if v.get()]
    #         tws = [t for t,v in [("wide",self.exp_use_wide), ("medium",self.exp_use_medium), ("tight",self.exp_use_tight)] if v.get()]
    #         if not types or not tws: return
    #         os.makedirs(os.path.join(out_dir, "feasible"), exist_ok=True)
    #         os.makedirs(os.path.join(out_dir, "infeasible"), exist_ok=True)
    #         c_feas, c_inf = 0, 0
    #         rng = np.random.default_rng()
    #         horizon = self.time_horizon.get()
    #         global_id = 0
    #         while c_feas < t_feas or c_inf < t_inf:
    #             global_id += 1
    #             n = rng.integers(n_min, n_max + 1)
    #             s = rng.integers(s_min, s_max + 1)
    #             it = rng.choice(types)
    #             tw = rng.choice(tws)
    #             sd = rng.integers(0, 2**32-1)
    #             std = rng.uniform(0.02, 0.15)
    #             r_mix = rng.uniform(0.2, 0.8)
    #             dm = "random" if self.exp_random_depot.get() else "center"
                
    #             # Retrieve Physics from GUI for Experimental run too
    #             inst = generate_milp_feasible_instance(
    #                 n, s, it, random_seed=sd, cluster_std=std, cluster_ratio=r_mix, 
    #                 use_adaptive_stations=True, depot_mode=dm, charger_at_depot=self.charger_at_depot.get(), 
    #                 time_horizon=horizon,
    #                 vehicle_capacity=self.veh_capacity.get(), consumption_rate=self.veh_consumption.get(), 
    #                 velocity=self.veh_velocity.get(), refuel_rate=self.veh_refuel.get(),
    #                 battery_strategy=self.batt_strat.get(), fixed_battery_val=self.veh_battery.get(),
    #                 service_min=self.serv_min.get(), service_max=self.serv_max.get()
    #             )
    #             frac = {"wide":0.8, "medium":0.4, "tight":0.2}[tw]
    #             inst['time_windows'] = assign_time_windows(n, fraction=frac, time_horizon=horizon)
                
    #             is_feas = quick_feasibility_check(inst)
    #             if is_feas and n <= 10:
    #                 is_feas = verify_milp_feasibility(inst['depot'], inst['customers'], inst['stations'], inst['demands'], inst['service_times'], inst['time_windows'], inst['battery_capacity'], inst['load_capacity'], inst['consumption_rate'], inst['velocity'], time_horizon=horizon)

    #             status_str = "Feasible" if is_feas else "Infeasible"
    #             with open(csv_path, "a") as f: f.write(f"{global_id},{status_str},{it},{tw},{n},{s},{sd},{is_feas}\n")
    #             code = make_type_code(it, tw)
    #             if is_feas and c_feas < t_feas:
    #                 save_instance_txt(inst, os.path.join(out_dir, "feasible"), code, n, f"{s}S", c_feas)
    #                 c_feas += 1
    #             elif not is_feas and c_inf < t_inf:
    #                 save_instance_txt(inst, os.path.join(out_dir, "infeasible"), code, n, f"{s}S", c_inf)
    #                 c_inf += 1
    #             if (c_feas+c_inf) % 5 == 0: self.progress_queue.put(("progress", (c_feas, c_inf)))
    #         self.progress_queue.put(("done", "Complete."))
    #     except Exception as e:
    #         self.progress_queue.put(("error", str(e)))

    # --- PREVIEW ---
    def preview(self):
        self.status_var.set("Previewing...")
        for w in self.canvas_frame.winfo_children(): w.destroy()
        n = self.n_cust.get()
        n_inst_per_combo = self.n_inst.get()
        raw_type = self.inst_type.get()
        types_to_run = ["C", "R", "RC"] if raw_type == "All" else [raw_type]
        raw_tw = self.tw_type.get()
        tws_to_run = ["wide", "medium", "tight"] if raw_tw == "All" else [raw_tw]
        tasks = []
        for it in types_to_run:
            for tw in tws_to_run:
                for i in range(n_inst_per_combo):
                    tasks.append((it, tw, i))
        total_plots = len(tasks)
        if total_plots > 9:
            messagebox.showwarning("Preview Limit", f"Requesting {total_plots} layouts. Showing first 9.")
            tasks = tasks[:9]
            total_plots = 9
        if total_plots == 0: return
        cols = min(3, total_plots)
        rows = (total_plots + cols - 1) // cols
        fig, axs = plt.subplots(rows, cols, figsize=(cols*5, rows*5), constrained_layout=True)
        axs = axs.flatten() if hasattr(axs, "flatten") else [axs]
        seed_rng = np.random.default_rng()
        horizon = self.time_horizon.get()
        self._preview_map = {}
        dropdown_values = []
        
        for idx, (it, tw, k) in enumerate(tasks):
            ax = axs[idx]
            if self.seed_mode.get() == "Random": sd = int(seed_rng.integers(0, 2**32-1))
            else: sd = self.seed.get() + idx
            dm = self.depot_type.get()
            cd = None
            if dm == 'custom':
                try: cd = [float(self.depot_x.get()), float(self.depot_y.get())]
                except: pass
            nc = 0
            if it != "R":
                if self.cluster_mode.get() == "Random":
                    rng2 = np.random.default_rng(sd)
                    max_c = max(2, int(np.sqrt(n)))
                    nc = int(rng2.integers(2, max_c + 1))
                else: nc = self.n_clusters.get()
            use_adaptive = (self.station_mode.get() == "Random")
            s_user = self.n_stat.get()
            if use_adaptive:
                rng3 = np.random.default_rng(sd)
                s_user = int(rng3.integers(1, max(1, n//5)+1))
            
            # Retrieve Physics for Preview Call
            inst = generate_milp_feasible_instance(
                n_customers=n, n_stations=s_user, instance_type=it, random_seed=sd, n_clusters=nc, depot_mode=dm, custom_depot=cd, 
                cluster_std=self.cluster_std.get(), cluster_ratio=self.cluster_ratio.get()/100.0, max_vehicles=self.max_vehicles.get(), 
                use_adaptive_stations=use_adaptive, charger_at_depot=self.charger_at_depot.get(), time_horizon=horizon,
                vehicle_capacity=self.veh_capacity.get(), consumption_rate=self.veh_consumption.get(), 
                velocity=self.veh_velocity.get(), refuel_rate=self.veh_refuel.get(),
                battery_strategy=self.batt_strat.get(), fixed_battery_val=self.veh_battery.get(),
                service_min=self.serv_min.get(), service_max=self.serv_max.get()
            )
            
            label = f"{it} | {tw} | Seed {sd}"
            dropdown_values.append(label)
            # Store params so open_layout can replicate
            self._preview_map[label] = {
                'n': n, 's': s_user, 'it': it, 'sd': sd, 'nc': nc, 'dm': dm, 'cd': cd, 'std': self.cluster_std.get(), 
                'ratio': self.cluster_ratio.get()/100.0, 'k': self.max_vehicles.get(), 'adap': use_adaptive, 
                'chg': self.charger_at_depot.get(), 'hz': horizon,
                # Store physics too
                'cap': self.veh_capacity.get(), 'cons': self.veh_consumption.get(), 'vel': self.veh_velocity.get(),
                'ref': self.veh_refuel.get(), 'b_strat': self.batt_strat.get(), 'b_fix': self.veh_battery.get(),
                's_min': self.serv_min.get(), 's_max': self.serv_max.get()
            }
            
            C, S, D = inst['customers'], inst['stations'], inst['depot']
            lbls = inst['cluster_labels']
            cmap = plt.get_cmap('tab10')
            ax.scatter(D[0], D[1], c='red', marker='s', s=120, label='Depot', zorder=10)
            ax.text(D[0], D[1]+0.02, "D0", fontsize=9, fontweight='bold', ha='center')
            if len(S) > 1:
                field_stats = S[1:]
                ax.scatter(field_stats[:,0], field_stats[:,1], c='black', marker='^', s=90, label='Station', zorder=9)
                for si, s_pt in enumerate(field_stats, 1):
                    ax.text(s_pt[0], s_pt[1]+0.02, f"S{si}", fontsize=8, ha='center')
            unique_lbls = np.unique(lbls)
            for lbl in unique_lbls:
                pts = C[lbls == lbl]
                if len(pts) > 0:
                    color = cmap(lbl % cmap.N)
                    ax.scatter(pts[:,0], pts[:,1], c=[color], s=50, label=f"Clust {lbl}", zorder=5)
                    indices = np.where(lbls == lbl)[0] + 1
                    for c_idx, pt in zip(indices, pts):
                        ax.text(pt[0], pt[1]+0.015, f"C{c_idx:02d}", fontsize=7, ha='center')
            ax.set_title(label)
            ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.05, 1.05)
            ax.set_aspect('equal')
            ax.grid(True, linestyle='--', alpha=0.3)
            ax.legend(loc='upper right', fontsize='xx-small')

        for j in range(total_plots, len(axs)): axs[j].axis('off')
        canvas = FigureCanvasTkAgg(fig, master=self.canvas_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.layout_menu.configure(values=dropdown_values)
        if dropdown_values: self.layout_choice.set(dropdown_values[0])
        self.status_var.set("Idle.")

    def open_layout(self):
        sel = self.layout_choice.get()
        if not sel or sel not in self._preview_map: return messagebox.showwarning("Selection Error", "Please generate a preview and select a layout.")
        d = self._preview_map[sel]
        
        # Regenerate using stored physics from preview map
        inst = generate_milp_feasible_instance(
            n_customers=d['n'], n_stations=d['s'], instance_type=d['it'], random_seed=d['sd'], 
            n_clusters=d['nc'], depot_mode=d['dm'], custom_depot=d['cd'], cluster_std=d['std'], 
            cluster_ratio=d['ratio'], max_vehicles=d['k'], use_adaptive_stations=d['adap'], 
            charger_at_depot=d['chg'], time_horizon=d['hz'],
            vehicle_capacity=d['cap'], consumption_rate=d['cons'], velocity=d['vel'], refuel_rate=d['ref'],
            battery_strategy=d['b_strat'], fixed_battery_val=d['b_fix'], service_min=d['s_min'], service_max=d['s_max']
        )
        
        win = tk.Toplevel(self)
        win.title(f"Detailed View: {sel}")
        win.geometry("900x900")
        fig, ax = plt.subplots(figsize=(8,8))
        C, S, D = inst['customers'], inst['stations'], inst['depot']
        lbls = inst['cluster_labels']
        cmap = plt.get_cmap('tab10')
        ax.scatter(D[0], D[1], c='red', marker='s', s=150, label='Depot', zorder=10)
        ax.text(D[0], D[1]+0.02, "D0", fontsize=10, fontweight='bold', ha='center')
        if len(S) > 1:
            st = S[1:]
            ax.scatter(st[:,0], st[:,1], c='black', marker='^', s=100, label='Station', zorder=5)
            for i, s_pt in enumerate(st, 1):
                ax.text(s_pt[0], s_pt[1]+0.02, f"S{i}", fontsize=9, ha='center')
        for lbl in np.unique(lbls):
            pts = C[lbls == lbl]
            if len(pts) > 0:
                color = cmap(lbl % cmap.N)
                ax.scatter(pts[:,0], pts[:,1], c=[color], marker='o', s=60, label=f"Cluster {lbl}", zorder=4)
                indices = np.where(lbls == lbl)[0] + 1
                for idx, pt in zip(indices, pts):
                    ax.text(pt[0], pt[1]+0.015, f"C{idx}", fontsize=8, ha='center')
        ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.05, 1.05)
        ax.set_aspect('equal')
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        canvas = FigureCanvasTkAgg(fig, master=win)
        canvas.draw()
        toolbar = NavigationToolbar2Tk(canvas, win)
        toolbar.update()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def solve_instance(self):
        path = self.solve_file.get().strip()
        if not path: return messagebox.showerror("Error", "Select file.")
        self.status_var.set("Status: Solving...")
        self.route_text.delete('1.0', tk.END)
        for w in self.solve_canvas_frame.winfo_children(): w.destroy()
        self.update_idletasks()
        try:
            tl = self.solve_tl.get()
            phase = self.solve_phase.get()
            if phase == "All":
                fig, axs = plt.subplots(1, 3, figsize=(15, 5))
                for i, ph in enumerate(["A","B","C"]):
                    self.route_text.insert(tk.END, f"--- Phase {ph} ---\n")
                    res = solve_evrptw(path, phase_key=ph, time_limit=tl, return_data=True)
                    if res:
                        nodes, arcs, obj, gap = res
                        self.route_text.insert(tk.END, f"Obj: {obj:.2f} | Gap: {gap:.4f}\n")
                        self.route_text.insert(tk.END, f"Routes ({len(arcs)} arcs):\n")
                        for u, v in arcs: self.route_text.insert(tk.END, f"  {u} -> {v}\n")
                        self.route_text.insert(tk.END, "\n")
                        self._draw_solution(axs[i], nodes, arcs, title=f"Phase {ph}")
                    else: self.route_text.insert(tk.END, "Infeasible or Timed Out\n\n")
                canvas = FigureCanvasTkAgg(fig, master=self.solve_canvas_frame)
            else:
                res = solve_evrptw(path, phase_key=phase, time_limit=tl, return_data=True)
                if res:
                    nodes, arcs, obj, gap = res
                    fig, ax = plt.subplots(figsize=(6,6))
                    self._draw_solution(ax, nodes, arcs, title=f"Phase {phase}")
                    canvas = FigureCanvasTkAgg(fig, master=self.solve_canvas_frame)
                    self.route_text.insert(tk.END, f"Solved Phase {phase}\nObj: {obj:.2f} | Gap: {gap:.4f}\n")
                    for a,b in arcs: self.route_text.insert(tk.END, f"{a} -> {b}\n")
            if 'canvas' in locals():
                canvas.draw()
                canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            self.status_var.set("Status: Solved.")
        except Exception as e:
            self.status_var.set("Status: Error.")
            messagebox.showerror("Solver Error", str(e))

    def _draw_solution(self, ax, nodes, arcs, title=""):
        for _, (lab, x, y, t) in nodes.items():
            if t == 'd': 
                ax.scatter(x, y, c='red', marker='s', s=100, zorder=5, label='Depot')
                ax.text(x, y+0.02, lab, ha='center', fontweight='bold')
            elif t == 'f':
                ax.scatter(x, y, c='black', marker='^', s=80, zorder=5, label='Station')
                ax.text(x, y+0.02, lab, ha='center')
            else:
                ax.scatter(x, y, c='blue', marker='o', s=40, zorder=4, label='Customer')
                ax.text(x, y+0.02, lab, ha='center', fontsize=8)
        for a_id, b_id in arcs:
            xa, ya = next((n[1], n[2]) for n in nodes.values() if n[0]==a_id)
            xb, yb = next((n[1], n[2]) for n in nodes.values() if n[0]==b_id)
            ax.annotate("", xy=(xb, yb), xytext=(xa, ya), arrowprops=dict(arrowstyle="->", lw=1.0))
        ax.set_title(title)
        ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.05, 1.05)
        ax.set_aspect('equal')
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize='small')

    def check_progress_queue(self):
        try:
            while True:
                msg, data = self.progress_queue.get_nowait()
                if msg == "log":
                    self.log_text.config(state='normal')
                    self.log_text.insert(tk.END, data + "\n")
                    self.log_text.see(tk.END)
                    self.log_text.config(state='disabled')
                elif msg == "status":
                    self.status_var.set(data)
                elif msg == "bench_init":
                    self.bench_progress['maximum'] = data
                    self.bench_progress['value'] = 0
                elif msg == "bench_update":
                    count, text = data
                    self.bench_progress['value'] = count
                    self.status_var.set(text)
                elif msg == "bench_pulse":
                    self.status_var.set(data)
                elif msg == "benchmark_done":
                    self.btn_gen.config(state=tk.NORMAL)
                    self.status_var.set("Status: Idle.")
                    messagebox.showinfo("Success", data)
                elif msg == "set_targets":
                    self.exp_progress_bar_f['maximum'] = data[0]
                    self.exp_progress_bar_i['maximum'] = data[1]
                elif msg == "progress":
                    self.exp_progress_bar_f['value'] = data[0]
                    self.exp_progress_bar_i['value'] = data[1]
                    self.exp_progress_label_f.config(text=f"Feasible: {data[0]}")
                    self.exp_progress_label_i.config(text=f"Infeasible: {data[1]}")
                elif msg == "done":
                    self.exp_generate_button.config(state=tk.NORMAL)
                    self.exp_status_label.config(text="Status: Complete!", foreground="green")
                    messagebox.showinfo("Success", data)
                elif msg == "error":
                    self.btn_gen.config(state=tk.NORMAL)
                    self.exp_generate_button.config(state=tk.NORMAL)
                    self.status_var.set("Status: Error")
                    self.exp_status_label.config(text="Status: Error!", foreground="red")
                    messagebox.showerror("Error", data)
        except queue.Empty: pass
        finally: self.after_id = self.after(100, self.check_progress_queue)


    def view_instance(self):
            path = self.view_file.get().strip()
            if not path: return
            try:
                inst = load_instance_txt(path)
                for w in self.view_canvas_frame.winfo_children(): w.destroy()
                fig, ax = plt.subplots(figsize=(8,8))
                D, S, C = inst['depot'], inst['stations'], inst['customers']
                ax.scatter(D[0], D[1], c='red', marker='s', s=150, label='Depot', zorder=5)
                ax.text(D[0], D[1]+0.02, "D0", fontsize=10, fontweight='bold', ha='center')
                if len(S) > 0:
                    sx = [s[0] for s in S]
                    sy = [s[1] for s in S]
                    ax.scatter(sx, sy, c='black', marker='^', s=100, label='Stations', zorder=5)
                    for i, s in enumerate(S, 1):
                        ax.text(s[0], s[1]+0.02, f"S{i}", fontsize=9, ha='center')
                if len(C) > 0:
                    cx = [c[0] for c in C]
                    cy = [c[1] for c in C]
                    ax.scatter(cx, cy, c='green', marker='o', s=60, label='Customers', zorder=4)
                    for i, c in enumerate(C, 1):
                        ax.text(c[0], c[1]+0.015, f"C{i}", fontsize=7, ha='center')
                ax.set_title(os.path.basename(path))
                ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.05, 1.05)
                ax.set_aspect('equal')
                ax.grid(True, linestyle='--', alpha=0.3)
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                canvas = FigureCanvasTkAgg(fig, master=self.view_canvas_frame)
                canvas.draw()
                canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
                self.status_var.set("Status: Idle.")
            except Exception as e:
                self.status_var.set("Status: Error! Could not load file.")
                messagebox.showerror("Error", str(e))

if __name__ == "__main__":
    app = SynthChargeApp()
    app.mainloop()