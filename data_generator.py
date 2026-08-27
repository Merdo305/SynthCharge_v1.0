import os
import numpy as np
import pulp
import copy

MAX_SEED = 2**32 - 1

# ==============================================================================
# SECTION 2: SPATIAL DISTRIBUTION
# ==============================================================================
def generate_customers(n_customers, instance_type, random_seed=None, n_clusters=None, 
                       cluster_std=0.05, cluster_ratio=0.5, 
                       avoid_points=None, min_dist=0.04):
    rng = np.random.default_rng(random_seed % MAX_SEED if random_seed else None)
    coords = np.zeros((n_customers, 2))
    labels = np.full(n_customers, -1, dtype=int)
    existing = avoid_points if avoid_points is not None else np.empty((0, 2))
    t = instance_type.upper()
    
    def is_too_close(p, others, d):
        if len(others) == 0: return False
        return np.min(np.linalg.norm(others-p, axis=1)) < d

    if t == 'R':
        for i in range(n_customers):
            for _ in range(10):
                cand = rng.random(2)
                if not is_too_close(cand, existing, min_dist): break
            coords[i] = cand
            labels[i] = 0
            existing = np.vstack([existing, cand.reshape(1,2)])
    elif t == 'C':
        k = n_clusters or max(2, int(np.sqrt(n_customers)))
        centers = rng.random((k, 2))
        sizes = rng.multinomial(n_customers, [1/k]*k)
        idx = 0
        for c, sz in enumerate(sizes):
            pts = np.clip(rng.normal(centers[c], cluster_std, (sz, 2)), 0, 1)
            coords[idx:idx+sz] = pts
            labels[idx:idx+sz] = c
            idx += sz
    elif t == 'RC':
        n_c = int(n_customers * cluster_ratio)
        n_r = n_customers - n_c
        k = n_clusters or max(2, int(np.sqrt(n_c))) if n_c>0 else 1
        c_pts = np.zeros((n_c, 2))
        if n_c > 0:
            centers = rng.random((k, 2))
            sizes = rng.multinomial(n_c, [1/k]*k)
            idx = 0
            for c, sz in enumerate(sizes):
                pts = np.clip(rng.normal(centers[c], cluster_std, (sz, 2)), 0, 1)
                c_pts[idx:idx+sz] = pts
                labels[idx:idx+sz] = c
                idx += sz
            existing = np.vstack([existing, c_pts])
        r_pts = np.zeros((n_r, 2))
        for i in range(n_r):
            for _ in range(10):
                cand = rng.random(2)
                if not is_too_close(cand, existing, min_dist): break
            r_pts[i] = cand
            existing = np.vstack([existing, cand.reshape(1,2)])
        if n_c > 0 and n_r > 0: coords = np.vstack([c_pts, r_pts])
        elif n_c > 0: coords = c_pts
        else: coords = r_pts
        if n_r > 0: labels[n_c:] = k
    return coords, labels

# ==============================================================================
# SECTION 3: ADAPTIVE CHARGING STATIONS (FIXED & UNIQUE)
# ==============================================================================
def place_necessary_charging_stations(depot, customers, battery_capacity, consumption_rate=0.25, rng=None):
    if rng is None: rng = np.random.default_rng()
    
    stations = [depot.copy()]
    max_dist = battery_capacity / consumption_rate
    
    # 1. Midpoint Heuristic
    for i, c1 in enumerate(customers):
        for c2 in customers[i+1:]:
            d = np.linalg.norm(c1-c2)
            if d*consumption_rate > battery_capacity*0.8:
                mid = 0.5*(c1+c2)
                # FIX: Use the instance-specific RNG for noise, not global np.random
                noise = (rng.random(2)-0.5) * 0.1 
                stations.append(np.clip(mid+noise, 0, 1))
                
    # 2. Depot Isolation Heuristic
    for c in customers:
        d0 = np.linalg.norm(c-depot)
        if d0*consumption_rate > battery_capacity*0.7:
            mind = min(np.linalg.norm(c-s) for s in stations)
            if mind*consumption_rate > battery_capacity*0.5:
                direction = (c-depot)/np.linalg.norm(c-depot)
                stations.append(np.clip(depot + direction*(max_dist*0.6), 0, 1))
    
    # 3. Filtering & Overlap Prevention
    filtered = [stations[0]]
    sep = max_dist * 0.3
    min_cust_dist = 0.04
    
    for s in stations[1:]:
        # Check against stations
        if any(np.linalg.norm(s - e) < sep for e in filtered): continue
        # Check against customers
        if np.any(np.linalg.norm(customers - s, axis=1) < min_cust_dist): continue
            
        filtered.append(s)
        
    return np.array(filtered)

# ==============================================================================
# SECTION 4: VEHICLE PARAMETERS 
# ==============================================================================
def determine_battery_capacity(customers, depot, consumption_rate=0.25, strategy="adaptive", fixed_val=0.4):
    """
    Calculates Battery Capacity based on strategy.
    """
    if strategy == "fixed":
        return fixed_val
    
    # Adaptive Logic
    dists = []
    for i, c1 in enumerate(customers):
        for c2 in customers[i+1:]: dists.append(np.linalg.norm(c1-c2))
    max_d = max(dists) if dists else 0.5
    
    # Set B to ~40% of max diagonal (converted to energy)
    B = min(0.4, max_d*0.8/consumption_rate)
    B = max(B, 0.15)
    return B

def generate_demands_and_services(n_customers, load_capacity=1.5, service_min=0.01, service_max=0.03, rng=None):
    if rng is None: rng = np.random.default_rng()
    # Demands
    d = rng.uniform(0.02*load_capacity, 0.3*load_capacity, n_customers)
    if d.sum() > 3*load_capacity: d *= (3*load_capacity)/d.sum()
    # Services (User Defined Range)
    s = rng.uniform(service_min, service_max, n_customers)
    return d, s

# ==============================================================================
# SECTION 5: TIME WINDOWS
# ==============================================================================
def assign_time_windows(n_customers, fraction=0.8, time_horizon=10.0, rng=None):
    tw = []
    width = time_horizon * fraction
    for i in range(n_customers):
        st = (i/n_customers)*(time_horizon*0.3)
        en = st + width
        if en > time_horizon: en=time_horizon; st=en-width
        tw.append((st, en))
    return tw

# ==============================================================================
# SECTION 6: FEASIBILITY & MASTER
# ==============================================================================
def verify_milp_feasibility(depot, customers, stations, demands, services, 
                            time_windows, B, load, consumption_rate, v, 
                            time_horizon=10.0, time_limit=60):
    n = len(customers); m = len(stations); K = min(3, n)
    all_pts = np.vstack([depot.reshape(1,2), customers, stations])
    P = len(all_pts)
    dist = np.linalg.norm(all_pts[:,None,:]-all_pts[None,:,:], axis=2)
    prob = pulp.LpProblem("EVRPTW_Check", pulp.LpMinimize)
    x = pulp.LpVariable.dicts("x", (range(P), range(P), range(K)), 0, 1, cat='Binary')
    t = pulp.LpVariable.dicts("t", (range(P), range(K)), 0)
    
    M_val = max(1000, time_horizon * 2)

    prob += pulp.lpSum(dist[i][j]*x[i][j][k] for i in range(P) for j in range(P) for k in range(K) if i!=j)
    for i in range(1, n+1): prob += pulp.lpSum(x[j][i][k] for j in range(P) for k in range(K) if j!=i) == 1
    for i in range(P):
        for k in range(K): prob += pulp.lpSum(x[i][j][k] for j in range(P) if j!=i) == pulp.lpSum(x[j][i][k] for j in range(P) if j!=i)
    for k in range(K):
        prob += pulp.lpSum(x[0][j][k] for j in range(1,P)) <= 1
        prob += pulp.lpSum(x[j][0][k] for j in range(1,P)) <= 1
    for idx in range(1, n+1):
        a, bw = time_windows[idx-1]
        for k in range(K):
            prob += t[idx][k] >= a * pulp.lpSum(x[j][idx][k] for j in range(P) if j!=idx)
            inc = pulp.lpSum(x[j][idx][k] for j in range(P) if j!=idx)
            prob += t[idx][k] <= bw * inc + M_val * (1 - inc)
            
    solver = pulp.PULP_CBC_CMD(timeLimit=time_limit, msg=False)
    prob.solve(solver)
    return pulp.LpStatus[prob.status] == 'Optimal'

def check_charging_necessity(depot, customers, stations, B, consumption_rate):
    for i, c1 in enumerate(customers):
        for c2 in customers[i+1:]:
            if np.linalg.norm(c1-c2)*consumption_rate > 0.8*B: return True
    for c in customers:
        if np.linalg.norm(depot-c)*2*consumption_rate > 0.8*B: return True
    return False

# def quick_feasibility_check(inst):
#     if not inst['charging_required']: return False
#     pts = np.vstack([inst['depot'].reshape(1,2), inst['stations']])
#     maxd = inst['battery_capacity'] / inst['consumption_rate']
#     for c in inst['customers']:
#         if np.min(np.linalg.norm(c - pts, axis=1)) > maxd: return False
#     if inst['demands'].sum() > 3*inst['load_capacity']: return False
#     return True

# def quick_feasibility_check(inst):
#     pts = np.vstack([inst['depot'].reshape(1,2), inst['stations']])
#     maxd = inst['battery_capacity'] / inst['consumption_rate']

#     for c in inst['customers']:
#         if np.min(np.linalg.norm(c - pts, axis=1)) > maxd:
#             return False

#     if inst['demands'].sum() > 3 * inst['load_capacity']:
#         return False

#     return True

def quick_feasibility_check(inst):

    depot = inst['depot']
    stations = inst['stations']
    customers = inst['customers']

    battery = inst['battery_capacity']
    rate = inst['consumption_rate']

    max_range = battery / rate

    # ------------------------------------------------
    # ENERGY REACHABILITY
    # ------------------------------------------------

    pts = np.vstack([depot.reshape(1,2), stations])

    for c in customers:
        if np.min(np.linalg.norm(c - pts, axis=1)) > max_range:
            return False


    # ------------------------------------------------
    # LOAD CAPACITY
    # ------------------------------------------------

    if inst['demands'].sum() > 3 * inst['load_capacity']:
        return False


    # ------------------------------------------------
    # TIME WINDOW SANITY
    # ------------------------------------------------

    for ready, due in inst['time_windows']:
        if due <= ready:
            return False


    # ------------------------------------------------
    # STATION GRAPH CONNECTIVITY
    # ------------------------------------------------

    nodes = np.vstack([depot.reshape(1,2), stations])

    for i in range(len(nodes)):
        for j in range(i+1, len(nodes)):

            d = np.linalg.norm(nodes[i] - nodes[j])

            if d > max_range * 2:
                return False

    return True

# def induce_infeasibility(inst, rng=None, mode="energy"):
#     """
#     Make the instance fail quick_feasibility_check() deterministically.

#     quick_feasibility_check fails iff:
#       (A) ∃ customer with min distance to depot/stations > maxd (=B/r), OR
#       (B) sum(demands) > 3*load_capacity
#     """
#     if rng is None:
#         rng = np.random.default_rng()

#     inst = copy.deepcopy(inst)

#     if mode == "energy":
#         # Guarantee failure of reachability: make maxd tiny
#         # maxd = B/r ; make B so small that maxd < 1e-6
#         inst['battery_capacity'] = 1e-6 * inst['consumption_rate']

#     elif mode == "load":
#         # Guarantee capacity failure
#         inst['demands'] = inst['demands'].copy()
#         inst['demands'][0] = 4.0 * inst['load_capacity']  # ensures sum > 3*load_capacity

#     else:
#         raise ValueError("Unsupported infeasibility mode")

#     return inst

import copy
import numpy as np

def induce_infeasibility(inst, rng=None, mode="energy", severity=0.15):
    """
    Inject controlled infeasibility while keeping the instance realistic.

    Modes:
        energy   -> make a customer unreachable by reducing battery
        load     -> exceed fleet capacity (assumes 3 vehicles)
        time     -> create invalid time windows
        stations -> remove charging stations to break reachability
    """

    if rng is None:
        rng = np.random.default_rng()

    inst = copy.deepcopy(inst)

    customers = inst["customers"]
    depot = inst["depot"]

    # ------------------------------------------------
    # ENERGY INFEASIBILITY
    # ------------------------------------------------
    if mode == "energy":

        inst["battery_capacity"] *= (1 - severity)

        max_range = inst["battery_capacity"] / inst["consumption_rate"]

        dists = np.linalg.norm(customers - depot, axis=1)

        farthest = np.argmax(dists)

        if dists[farthest] <= max_range:
            inst["battery_capacity"] *= 0.4


    # ------------------------------------------------
    # LOAD INFEASIBILITY (guaranteed)
    # ------------------------------------------------
    elif mode == "load":

        fleet_capacity = 3 * inst["load_capacity"]

        total_demand = np.sum(inst["demands"])

        if total_demand <= fleet_capacity:

            excess = fleet_capacity * (1.2 + severity)

            scale = excess / total_demand

            inst["demands"] = inst["demands"] * scale


    # ------------------------------------------------
    # TIME WINDOW INFEASIBILITY
    # ------------------------------------------------
    elif mode == "time":

        tw = list(inst["time_windows"])
        n = len(tw)

        if n > 0:

            idx = rng.integers(0, n)

            ready, due = tw[idx]

            # force invalid window
            tw[idx] = (ready, ready - 0.01)

            inst["time_windows"] = tw


    # ------------------------------------------------
    # STATION INFEASIBILITY
    # ------------------------------------------------
    elif mode == "stations":

        if len(inst["stations"]) > 1:

            # keep only depot charger or one station
            inst["stations"] = inst["stations"][:1]


    else:
        raise ValueError("Unsupported infeasibility mode")

    return inst
# def induce_infeasibility(inst, rng=None, mode="energy", severity=0.1):
#     """
#     Inject controlled infeasibility while keeping the instance realistic.

#     Modes:
#         energy   -> reduce battery so some customers become unreachable
#         load     -> exceed vehicle capacity
#         time     -> shrink time windows
#         stations -> remove charging stations
#     """

#     if rng is None:
#         rng = np.random.default_rng()

#     inst = copy.deepcopy(inst)

#     if mode == "energy":

#         # Reduce battery enough to break reachability
#         inst['battery_capacity'] *= (1 - severity)

#         # Ensure at least one customer becomes unreachable
#         maxd = inst['battery_capacity'] / inst['consumption_rate']
#         depot = inst['depot']
#         customers = inst['customers']

#         farthest = np.argmax(np.linalg.norm(customers - depot, axis=1))
#         c = customers[farthest]

#         if np.linalg.norm(c - depot) <= maxd:
#             inst['battery_capacity'] *= 0.5

#     elif mode == "load":

#         # Increase demand so capacity constraint breaks
#         idx = rng.integers(0, len(inst['demands']))
#         inst['demands'][idx] += severity * inst['load_capacity'] * 4


#     elif mode == "time":

#         horizon = inst["time_horizon"]
    
#         tw = inst["time_windows"]
    
#         n = len(tw)
    
#         # choose 20–40% customers
#         k = max(1, int(n * rng.uniform(0.2,0.4)))
    
#         idx = rng.choice(n, size=k, replace=False)
    
#         for i in idx:
    
#             ready, due = tw[i]
    
#             width = due - ready
    
#             # shrink window heavily
#             new_due = ready + width * (0.1 + severity)
    
#             tw[i] = (ready, new_due)
    
#         inst["time_windows"] = tw

#     # elif mode == "time":

#     #     # Shrink time windows heavily
#     #     new_tw = []
#     #     for (a, b) in inst['time_windows']:

#     #         width = b - a
#     #         shrink = width * severity

#     #         new_a = a + shrink
#     #         new_b = b - shrink

#     #         if new_b <= new_a:
#     #             new_b = new_a + 0.01

#     #         new_tw.append((new_a, new_b))

#     #     inst['time_windows'] = new_tw

#     elif mode == "stations":

#         # Remove stations to break reachability
#         if len(inst['stations']) > 2:
#             inst['stations'] = inst['stations'][:-2]
#         elif len(inst['stations']) > 1:
#             inst['stations'] = inst['stations'][:-1]

#     else:
#         raise ValueError("Unsupported infeasibility mode")

#     return inst



# --- MASTER GENERATOR (FIXED) ---
def generate_milp_feasible_instance(n_customers, n_stations, instance_type,
                                    random_seed=None, n_clusters=None,
                                    depot_mode='center', custom_depot=None,
                                    cluster_std=0.05, cluster_ratio=0.5,
                                    max_vehicles=3,
                                    use_adaptive_stations=True,
                                    charger_at_depot=False,
                                    time_horizon=10.0,
                                    # Physics Params
                                    vehicle_capacity=1.5,
                                    consumption_rate=0.25,
                                    velocity=1.0,
                                    refuel_rate=1.0,
                                    battery_strategy='adaptive',
                                    fixed_battery_val=0.4,
                                    service_min=0.01,
                                    service_max=0.03):
    
    # 1. Initialize ONE Random Generator for the whole instance
    # This ensures "Seed 42" always produces the exact same result, and "Seed 43" is totally different.
    rng = np.random.default_rng(random_seed % (2**32 - 1) if random_seed is not None else None)
    
    # 2. Depot
    if depot_mode == 'random': depot = rng.random(2)
    elif depot_mode == 'custom' and custom_depot is not None: depot = np.array(custom_depot)
    else: depot = np.array([0.5, 0.5])

    # 3. Customers (Pass the RNG)
    # Note: We must modify generate_customers to accept 'rng' object instead of 'random_seed' integer
    # Or we just pass the seed. To keep it simple with your previous code, 
    # we will rely on the integer seed logic inside generate_customers, which is fine.
    customers, labels = generate_customers(n_customers, instance_type, random_seed, n_clusters, cluster_std, cluster_ratio, avoid_points=depot.reshape(1, 2))
    
    # 4. Physics 
    B = determine_battery_capacity(customers, depot, consumption_rate, battery_strategy, fixed_battery_val)
    
    # 5. Stations (CRITICAL FIX: Pass 'rng')
    candidates = place_necessary_charging_stations(depot, customers, B, consumption_rate, rng=rng)
    
    # Select/Generate Stations to match N_Stations count
    field_candidates = candidates[1:]
    target_field_count = n_stations
    
    if len(field_candidates) > target_field_count:
        # Use RNG to shuffle, so even the selection is deterministic per seed
        rng.shuffle(field_candidates) 
        field_stations = field_candidates[:target_field_count]
    elif len(field_candidates) < target_field_count:
        needed = target_field_count - len(field_candidates)
        extras = []
        all_pts = np.vstack([depot.reshape(1,2), customers, field_candidates]) if len(field_candidates)>0 else np.vstack([depot.reshape(1,2), customers])
        for _ in range(needed):
            for _ in range(20):
                cand = rng.random(2) # Use Instance RNG
                if np.min(np.linalg.norm(all_pts-cand, axis=1)) > 0.04: break
            extras.append(cand)
            all_pts = np.vstack([all_pts, cand.reshape(1,2)])
        field_stations = np.vstack([field_candidates, np.array(extras)]) if len(field_candidates)>0 else np.array(extras)
    else:
        field_stations = field_candidates

    final_list = [depot] 
    if charger_at_depot: final_list.append(depot.copy())
    if len(field_stations) > 0:
        for s in field_stations: final_list.append(s)
            
    stations = np.array(final_list)

    # 6. Demands & Windows (Pass RNG)
    d, s = generate_demands_and_services(n_customers, vehicle_capacity, service_min, service_max, rng=rng)
    tw = assign_time_windows(n_customers, fraction=0.8, time_horizon=time_horizon, rng=rng) 
    req = check_charging_necessity(depot, customers, stations, B, consumption_rate)

    return {
        'depot': depot, 'customers': customers, 'stations': stations, 
        'cluster_labels': labels, 'battery_capacity': B, 'load_capacity': vehicle_capacity, 
        'consumption_rate': consumption_rate, 'velocity': velocity, 'refuel_rate': refuel_rate,
        'demands': d, 'service_times': s, 'time_windows': tw, 'charging_required': req,
        'time_horizon': time_horizon
    }

def save_instance_txt(inst, out_dir, type_code, n_customers, st_label, idx):
    folder = os.path.join(out_dir, f"{type_code}", f"N{n_customers}")
    os.makedirs(folder, exist_ok=True)
    fname = f"{type_code}_N{n_customers}_{st_label}_{idx:03d}.txt"
    path = os.path.join(folder, fname)
    
    th = inst.get('time_horizon', 10.0)
    
    with open(path, 'w') as f:
        f.write("StringID   Type       x          y          demand     ReadyTime  DueDate    ServiceTime\n")
        d = inst['depot']
        f.write(f"D0         d     {d[0]:.4f} {d[1]:.4f} 0.0000    0.0000    {th:.4f}    0.0000\n")
        for i, s in enumerate(inst['stations'][1:], 1):
            f.write(f"S{i}         f     {s[0]:.4f} {s[1]:.4f} 0.0000    0.0000    {th:.4f}    0.0000\n")
        for i, c in enumerate(inst['customers'], 1):
            t0, t1 = inst['time_windows'][i-1]
            f.write(f"C{i:02d}       c     {c[0]:.4f} {c[1]:.4f} {inst['demands'][i-1]:.4f}    {t0:.4f}    {t1:.4f}    {inst['service_times'][i-1]:.4f}\n")
        f.write(f"\nQ Vehicle fuel tank capacity /{inst['battery_capacity']:.4f}/\n")
        f.write(f"C Vehicle load capacity /{inst['load_capacity']:.4f}/\n")
        f.write(f"r fuel consumption rate /{inst['consumption_rate']:.4f}/\n")
        f.write(f"g inverse refueling rate /{inst['refuel_rate']:.4f}/\n")
        f.write(f"v average Velocity /{inst['velocity']:.4f}/\n")

def load_instance_txt(path):
    if not os.path.exists(path): raise FileNotFoundError(path)
    depot, stations, customers = None, [], []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("StringID"): continue
            if "/" in line: break
            parts = line.split()
            if len(parts) < 4: continue
            typ, x, y = parts[1], float(parts[2]), float(parts[3])
            if typ == 'd': depot = np.array([x, y])
            elif typ == 'f': stations.append([x, y])
            elif typ == 'c': customers.append([x, y])
    return {'depot': depot, 'stations': np.array(stations), 'customers': np.array(customers)}