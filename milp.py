import os, sys, string, numpy as np
from gurobipy import Model, GRB, quicksum
import matplotlib.pyplot as plt

def read_instance(path: str):
    if not os.path.isfile(path): raise FileNotFoundError(path)
    xc, yc, q, e, l, s, vid, typ = [], [], [], [], [], [], [], []
    Q, C, h, g, V = 100.0, 200.0, 1.0, 1.0, 1.0 # Defaults

    with open(path) as fh: lines = fh.readlines()
    reading_nodes = False
    for ln in lines:
        ln = ln.strip()
        if ln == "": continue
        if ln.startswith("StringID"): reading_nodes = True; continue
        if "/" in ln and ("Vehicle" in ln or "rate" in ln or "Velocity" in ln or "capacity" in ln):
            reading_nodes = False
            try:
                parts = ln.split('/')
                if len(parts) >= 2:
                    val = float(parts[1])
                    if ln.startswith("Q"): Q = val
                    elif ln.startswith("C"): C = val
                    elif ln.startswith("r"): h = val
                    elif ln.startswith("g"): g = val
                    elif ln.startswith("v"): V = val
            except: pass
            continue
        if reading_nodes:
            parts = ln.split()
            if len(parts) < 8: continue
            name, t_type, x, y, d, rt, dt, st = parts[:8]
            dup = 2 if t_type == "f" else 1
            for k in range(dup):
                suf = string.ascii_lowercase[k] if t_type == "f" else ""
                xc.append(float(x)); yc.append(float(y)); q.append(float(d))
                e.append(float(rt)); l.append(float(dt)); s.append(float(st))
                vid.append(f"{name}{suf}"); typ.append(t_type)
    if len(xc) > 0:
        xc.append(xc[0]); yc.append(yc[0]); q.append(q[0]); e.append(e[0]); l.append(l[0]); s.append(s[0]); vid.append(vid[0] + "_dup"); typ.append("d")
    return dict(xc=xc, yc=yc, q=q, e=e, l=l, s=s, vid=vid, typ=typ, Q=Q, C=C, h=h, g=g, V=V)

def solve_evrptw(path, *, phase_key="A", time_limit=300, return_data=False):
    phase = phase_key.upper()
    use_bat = phase in ("B", "C")
    use_time = phase == "C"
    try: D = read_instance(path)
    except: return None
    if phase == "A":
        keep = [t != "f" for t in D["typ"]]
        for k in ("xc","yc","q","e","l","s","vid","typ"): D[k] = [D[k][i] for i, kpt in enumerate(keep) if kpt]
    n = len(D["xc"])
    if n == 0: return None
    dep0, depN = 0, n-1
    cust = [i for i,t in enumerate(D["typ"]) if t == "c"]
    sta = [i for i,t in enumerate(D["typ"]) if t == "f"]
    SC0, SCN = [dep0] + sta + cust, sta + cust + [depN]
    A = [(i,j) for i in SC0 for j in SCN if not (i in sta and j in sta)]
    dist = {(i,j): np.hypot(D["xc"][i]-D["xc"][j], D["yc"][i]-D["yc"][j]) for i,j in A}
    velocity = D["V"] if D["V"] is not None and D["V"] > 0 else 1.0
    tm = {k: d / velocity for k,d in dist.items()}
    
    m = Model(f"EVRPTW_{phase}")
    m.Params.OutputFlag = 0
    m.Params.TimeLimit = time_limit
    x = m.addVars(A, vtype=GRB.BINARY, name="x")
    u = m.addVars(n, name="u")
    t = m.addVars(n, name="t")
    y = m.addVars(n, name="y") if use_bat else None
    
    f_out = lambda i: (x[i,j] for j in SCN if (i,j) in x)
    f_in = lambda j: (x[i,j] for i in SC0 if (i,j) in x)
    m.addConstrs(quicksum(f_out(i)) == 1 for i in cust)
    if sta: m.addConstrs(quicksum(f_out(i)) <= 1 for i in sta)
    m.addConstrs(quicksum(f_in(i)) == quicksum(f_out(i)) for i in sta+cust)
    
    if phase == "A":
        order = m.addVars(cust, lb=1, ub=len(cust), name="ord")
        for i in cust:
            for j in cust:
                if i!=j and (i,j) in x: m.addConstr(order[j] >= order[i] + 1 - len(cust)*(1-x[i,j]))
    
    m.addConstrs(u[j] <= u[i] - D["q"][i]*x[i,j] + D["C"]*(1-x[i,j]) for i in SC0 for j in SCN if (i,j) in x)
    m.addConstr(u[dep0] <= D["C"])
    
    if use_bat:
        m.addConstrs(y[j] <= y[i] - D["h"]*dist[i,j]*x[i,j] + D["Q"]*(1-x[i,j]) for i in cust for j in SCN if (i,j) in x)
        for i in [dep0] + sta: m.addConstrs(y[j] <= D["Q"] - D["h"]*dist[i,j]*x[i,j] for j in SCN if (i,j) in x)
    
    if use_time:
        m.addConstrs(t[i] + (tm[i,j] + D["s"][i])*x[i,j] - D["l"][dep0]*(1-x[i,j]) <= t[j] for i in cust+[dep0] for j in SCN if (i,j) in x)
        if sta: m.addConstrs(t[i] + tm[i,j]*x[i,j] + D["g"]*(D["Q"]-y[i]) - (D["l"][dep0]+D["Q"]*D["g"])*(1-x[i,j]) <= t[j] for i in sta for j in SCN if (i,j) in x)
        m.addConstrs(t[k] <= D["l"][k] for k in range(n))
        m.addConstrs(t[k] >= D["e"][k] for k in range(n))
        
    obj = quicksum(dist[i,j]*x[i,j] for i,j in A) + 500*quicksum(x[dep0,j] for j in SCN if (dep0,j) in x)
    m.setObjective(obj, GRB.MINIMIZE)
    m.optimize()
    
    if m.Status not in (GRB.OPTIMAL, GRB.TIME_LIMIT) or m.SolCount == 0: return None
    sol = m.getAttr("X", x)
    arcs = [(D["vid"][i].replace("_dup",""), D["vid"][j].replace("_dup","")) for i,j in A if sol[i,j] > .5]
    nodes = {k: (D["vid"][k].replace("_dup",""), D["xc"][k], D["yc"][k], D["typ"][k]) for k in range(n)}
    return nodes, arcs, m.ObjVal, m.MIPGap