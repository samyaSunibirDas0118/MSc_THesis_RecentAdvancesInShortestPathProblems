"""
SSSP Algorithm Benchmark + Visualizations
==========================================
Algorithms : Dijkstra, DMMSY (Duan et al. 2025), Δ-stepping, Bellman-Ford, BFS
Graph types: Random sparse→dense, Erdős–Rényi, Road network, Scaling

Requirements: Python 3.8+,  pip install matplotlib
Outputs (PDF + PNG):
  fig1_random_graphs   fig2_erdos_renyi   fig3_road_network
  fig4_scaling         fig5_duan_speedup  fig6_summary_grid
  benchmark_results.json
"""

import time, random, heapq, math, json
from collections import defaultdict, deque

import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── import the verified DMMSY implementation ─────────────────────────────────
from dmmsy import dmmsy

random.seed(42)

# ── plot style ────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':'serif','font.size':10,
    'axes.titlesize':11,'axes.labelsize':10,
    'xtick.labelsize':9,'ytick.labelsize':9,'legend.fontsize':8.5,
    'axes.spines.top':False,'axes.spines.right':False,
    'axes.grid':True,'grid.linestyle':':','grid.alpha':0.45,
    'figure.dpi':150,'savefig.dpi':300,'savefig.bbox':'tight',
})
COLORS  = {'dijkstra':'#378ADD','dmmsy':'#1D9E75',
           'delta':'#BA7517','bf':'#E24B4A','bfs':'#888780'}
LABELS  = {'dijkstra':'Dijkstra','dmmsy':'DMMSY (Duan 2025)',
           'delta':'Δ-stepping','bf':'Bellman-Ford','bfs':'BFS (lower bound)'}
MARKERS = {'dijkstra':'o','dmmsy':'s','delta':'^','bf':'D','bfs':'x'}
DASHES  = {'dijkstra':(),'dmmsy':(5,2),'delta':(3,2),'bf':(7,2,2,2),'bfs':(2,2)}
ALGOS   = list(COLORS)

# ══════════════════════════════════════════════════════════════════════════════
# Graph generators
# ══════════════════════════════════════════════════════════════════════════════

def gen_random(n, m):
    adj = defaultdict(list); added = set(); c = 0
    while c < m:
        u,v = random.randint(0,n-1), random.randint(0,n-1)
        if u!=v and (u,v) not in added:
            w = random.randint(1,20)
            adj[u].append((v,w)); adj[v].append((u,w))
            added.add((u,v)); added.add((v,u)); c += 1
    return adj

def gen_er(n, p):
    adj = defaultdict(list)
    for u in range(n):
        for v in range(u+1,n):
            if random.random() < p:
                w = random.randint(1,20)
                adj[u].append((v,w)); adj[v].append((u,w))
    return adj

def gen_road(n):
    adj = defaultdict(list); side = int(math.sqrt(n))
    for i in range(side):
        for j in range(side):
            nd = i*side+j
            if nd >= n: continue
            r,d = i*side+j+1,(i+1)*side+j
            if j+1<side and r<n:
                w=random.randint(1,10); adj[nd].append((r,w)); adj[r].append((nd,w))
            if i+1<side and d<n:
                w=random.randint(1,10); adj[nd].append((d,w)); adj[d].append((nd,w))
    for _ in range(int(n*0.02)):
        u,v=random.randint(0,n-1),random.randint(0,n-1)
        if u!=v:
            w=random.randint(5,50); adj[u].append((v,w)); adj[v].append((u,w))
    return adj

def edges(adj,n): return sum(len(adj[u]) for u in range(n))//2

# ══════════════════════════════════════════════════════════════════════════════
# Other algorithms
# ══════════════════════════════════════════════════════════════════════════════

def dijkstra(adj, n, src):
    d=[float('inf')]*n; d[src]=0; pq=[(0,src)]
    while pq:
        dd,u=heapq.heappop(pq)
        if dd>d[u]: continue
        for v,w in adj.get(u,()):
            nd=dd+w
            if nd<d[v]: d[v]=nd; heapq.heappush(pq,(nd,v))
    return d

def bellman_ford(adj, n, src):
    d=[float('inf')]*n; d[src]=0
    es=[(u,v,w) for u in range(n) for v,w in adj.get(u,())]
    for _ in range(n-1):
        upd=False
        for u,v,w in es:
            if d[u]!=float('inf') and d[u]+w<d[v]: d[v]=d[u]+w; upd=True
        if not upd: break
    return d

def bfs(adj, n, src):
    d=[-1]*n; d[src]=0; q=deque([src])
    while q:
        u=q.popleft()
        for v,_ in adj.get(u,()):
            if d[v]==-1: d[v]=d[u]+1; q.append(v)
    return d

def delta_stepping(adj, n, src, delta=None):
    INF=float('inf'); tent=[INF]*n; tent[src]=0; dist=[INF]*n
    ws=[w for u in range(n) for _,w in adj.get(u,())]
    if not ws: return dist
    if delta is None: delta=max(1,max(ws)//4)
    NB=1024; bkts=[set() for _ in range(NB)]
    def relax(v,d):
        if d<tent[v]:
            ob=int(tent[v]/delta)%NB if tent[v]<INF else None
            if ob is not None: bkts[ob].discard(v)
            tent[v]=d; bkts[int(d/delta)%NB].add(v)
    relax(src,0); i=itr=0
    while itr<NB*2:
        while not bkts[i%NB] and itr<NB*2: i+=1; itr+=1
        if itr>=NB*2: break
        bi=i%NB; S=set()
        while bkts[bi]:
            R=list(bkts[bi]); bkts[bi]=set()
            for u in R:
                for v,w in adj.get(u,()):
                    if w<=delta: relax(v,tent[u]+w)
            S.update(R)
        for u in S:
            if tent[u]<INF: dist[u]=tent[u]
            for v,w in adj.get(u,()):
                if w>delta: relax(v,tent[u]+w)
        i+=1; itr+=1
    for v in range(n): dist[v]=tent[v]
    return dist

def tm(fn,*args):
    t=time.perf_counter(); fn(*args); return round((time.perf_counter()-t)*1000,4)

# ══════════════════════════════════════════════════════════════════════════════
# Benchmark runner
# ══════════════════════════════════════════════════════════════════════════════

def run_benchmarks():
    R={}; src=0

    print("=== Random Graphs (n=2000) ===")
    rows=[]
    for lbl,mt in [("sparse",4000),("medium",10000),("dense",30000),("very_dense",60000)]:
        adj=gen_random(2000,mt); m=edges(adj,2000)
        row=dict(label=lbl,n=2000,m=m,
            dijkstra =tm(dijkstra,      adj,2000,src),
            dmmsy    =tm(dmmsy,         adj,2000,src),
            delta    =tm(delta_stepping,adj,2000,src),
            bf       =tm(bellman_ford,  adj,2000,src) if m<8000 else None,
            bfs      =tm(bfs,           adj,2000,src))
        rows.append(row); print(f"  {lbl}: m={m}")
    R["random_graphs"]=rows

    print("\n=== Erdős–Rényi (n=2000) ===")
    rows=[]
    for lbl,p in [("p=0.002",0.002),("p=0.005",0.005),("p=0.01",0.01),
                  ("p=0.03",0.03),("p=0.07",0.07)]:
        adj=gen_er(2000,p); m=edges(adj,2000)
        row=dict(label=lbl,n=2000,m=m,p=p,
            dijkstra =tm(dijkstra,      adj,2000,src),
            dmmsy    =tm(dmmsy,         adj,2000,src),
            delta    =tm(delta_stepping,adj,2000,src),
            bf       =tm(bellman_ford,  adj,2000,src) if m<10000 else None,
            bfs      =tm(bfs,           adj,2000,src))
        rows.append(row); print(f"  {lbl}: m={m}")
    R["erdos_renyi"]=rows

    print("\n=== Road Network ===")
    rows=[]
    for n in [1000,2000,4000,8000]:
        adj=gen_road(n); m=edges(adj,n)
        row=dict(label=f"n={n}",n=n,m=m,
            dijkstra =tm(dijkstra,      adj,n,src),
            dmmsy    =tm(dmmsy,         adj,n,src),
            delta    =tm(delta_stepping,adj,n,src),
            bf       =tm(bellman_ford,  adj,n,src) if m<12000 else None,
            bfs      =tm(bfs,           adj,n,src))
        rows.append(row); print(f"  n={n}: m={m}")
    R["road_network"]=rows

    print("\n=== Scaling (avg degree ~8) ===")
    rows=[]
    for n in [500,1000,2000,4000,6000]:
        adj=gen_random(n,n*4); m=edges(adj,n)
        row=dict(label=f"n={n}",n=n,m=m,
            dijkstra =tm(dijkstra,      adj,n,src),
            dmmsy    =tm(dmmsy,         adj,n,src),
            delta    =tm(delta_stepping,adj,n,src),
            bf       =tm(bellman_ford,  adj,n,src) if m<10000 else None,
            bfs      =tm(bfs,           adj,n,src))
        rows.append(row); print(f"  n={n}: m={m}")
    R["scaling"]=rows

    with open("benchmark_results.json","w") as f: json.dump(R,f,indent=2)
    print("\nSaved benchmark_results.json")
    return R

# ══════════════════════════════════════════════════════════════════════════════
# Plotting
# ══════════════════════════════════════════════════════════════════════════════

def legend_handles(algos):
    return [mpatches.Patch(color=COLORS[a],label=LABELS[a]) for a in ALGOS if a in algos]

def plot_lines(ax, rows, xkey, title, xlabel, xticks=None, xlabels=None):
    xs=[r[xkey] for r in rows]
    for a in ALGOS:
        ys=[r.get(a) for r in rows]
        vx=[x for x,y in zip(xs,ys) if y is not None]
        vy=[y for y in ys if y is not None]
        if not vx: continue
        ax.plot(vx,vy,color=COLORS[a],marker=MARKERS[a],
                dashes=DASHES[a] if DASHES[a] else (),
                linewidth=1.8,markersize=5)
    ax.set_title(title,pad=8); ax.set_xlabel(xlabel); ax.set_ylabel("Runtime (ms)")
    if xticks: ax.set_xticks(xticks)
    if xlabels: ax.set_xticklabels(xlabels)
    ax.legend(handles=legend_handles(ALGOS),frameon=True,framealpha=0.9,
              edgecolor='#cccccc',fontsize=8,loc='upper left')

def save(fig,name):
    fig.savefig(f"{name}.pdf"); fig.savefig(f"{name}.png")
    plt.close(fig); print(f"  saved {name}.pdf / .png")

def make_figs(R):
    # Fig 1: Random graphs
    fig,ax=plt.subplots(figsize=(6.5,4.2))
    plot_lines(ax,R["random_graphs"],'m',
               "Fig 1 — Random graphs (n=2 000): runtime vs density",
               "Edge count $m$",[4000,10000,30000,60000],["4k","10k","30k","60k"])
    fig.tight_layout(); save(fig,"fig1_random_graphs")

    # Fig 2: Erdős–Rényi
    fig,ax=plt.subplots(figsize=(6.5,4.2))
    rows=R["erdos_renyi"]
    plot_lines(ax,rows,'m',
               "Fig 2 — Erdős–Rényi G(n,p), n=2 000: runtime vs density",
               "Edge count $m$")
    # annotate crossover if present
    dijk_vals=[r['dijkstra'] for r in rows]
    dmmsy_vals=[r['dmmsy'] for r in rows]
    ms=[r['m'] for r in rows]
    for i in range(1,len(rows)):
        if dmmsy_vals[i] < dijk_vals[i] and dmmsy_vals[i-1] >= dijk_vals[i-1]:
            ax.axvline(ms[i],color='#cccccc',linestyle=':',lw=1.2)
            ax.annotate(f"DMMSY wins\nm≈{ms[i]//1000}k",
                        xy=(ms[i],dmmsy_vals[i]),xytext=(ms[i]*1.15,dmmsy_vals[i]*1.5),
                        fontsize=8,color=COLORS['dmmsy'],
                        arrowprops=dict(arrowstyle='->',color=COLORS['dmmsy'],lw=0.9))
            break
    fig.tight_layout(); save(fig,"fig2_erdos_renyi")

    # Fig 3: Road network
    fig,ax=plt.subplots(figsize=(6.5,4.2))
    plot_lines(ax,R["road_network"],'n',
               "Fig 3 — Road network (grid + highways): runtime vs $n$",
               "Nodes $n$",[1000,2000,4000,8000])
    fig.tight_layout(); save(fig,"fig3_road_network")

    # Fig 4: Scaling
    fig,ax=plt.subplots(figsize=(6.5,4.2))
    plot_lines(ax,R["scaling"],'n',
               "Fig 4 — Scaling at fixed avg degree ≈ 8: runtime vs $n$",
               "Nodes $n$",[500,1000,2000,4000,6000])
    fig.tight_layout(); save(fig,"fig4_scaling")

    # Fig 5: DMMSY speedup
    labels,speedups,bar_colors=[],[],[]
    for key,prefix in [("random_graphs","RG"),("erdos_renyi","ER"),
                       ("road_network","Road"),("scaling","Scale")]:
        for r in R[key]:
            if r['dijkstra'] and r['dmmsy']:
                sp=round(r['dijkstra']/r['dmmsy'],3)
                labels.append(f"{prefix} {r['label']}")
                speedups.append(sp)
                bar_colors.append(COLORS['dmmsy'] if sp>=1 else '#D3D1C7')
    fig,ax=plt.subplots(figsize=(7,len(labels)*0.38+1.2))
    y=np.arange(len(labels))
    bars=ax.barh(y,speedups,color=bar_colors,height=0.6)
    ax.axvline(1.0,color='#E24B4A',linestyle='--',lw=1.4,label='Break-even (1×)')
    ax.set_yticks(y); ax.set_yticklabels(labels,fontsize=8)
    ax.set_xlabel("Speedup (Dijkstra time / DMMSY time)  — values >1 mean DMMSY wins")
    ax.set_title("Fig 5 — DMMSY speedup over Dijkstra across all scenarios\n"
                 "Green: DMMSY wins  |  Grey: Dijkstra wins",pad=8)
    for bar,sp in zip(bars,speedups):
        ax.text(bar.get_width()+0.02,bar.get_y()+bar.get_height()/2,
                f"{sp:.2f}×",va='center',fontsize=7.5)
    ax.set_xlim(0,max(speedups)*1.2)
    ax.legend(frameon=True,framealpha=0.9,fontsize=8)
    fig.tight_layout(); save(fig,"fig5_duan_speedup")

    # Fig 6: 2×2 summary
    fig,axes=plt.subplots(2,2,figsize=(12,8.5))
    fig.suptitle("SSSP Algorithm Benchmark — Summary",fontsize=13,y=1.01)
    panels=[
        (axes[0,0],"random_graphs",'m',"Edge count $m$","Random graphs (n=2 000)",
         [4000,10000,30000,60000],["4k","10k","30k","60k"]),
        (axes[0,1],"erdos_renyi",'m',"Edge count $m$","Erdős–Rényi (n=2 000)",None,None),
        (axes[1,0],"road_network",'n',"Nodes $n$","Road network",None,None),
        (axes[1,1],"scaling",'n',"Nodes $n$","Scaling (avg degree ≈ 8)",None,None),
    ]
    for ax,key,xk,xl,title,xticks,xlabels in panels:
        rows=R[key]; xs=[r[xk] for r in rows]
        for a in ALGOS:
            ys=[r.get(a) for r in rows]
            vx=[x for x,y in zip(xs,ys) if y is not None]
            vy=[y for y in ys if y is not None]
            if not vx: continue
            ax.plot(vx,vy,color=COLORS[a],marker=MARKERS[a],
                    dashes=DASHES[a] if DASHES[a] else (),linewidth=1.6,markersize=4)
        ax.set_title(title,fontsize=10); ax.set_xlabel(xl); ax.set_ylabel("ms")
        if xticks: ax.set_xticks(xticks)
        if xlabels: ax.set_xticklabels(xlabels)
    handles=[mpatches.Patch(color=COLORS[a],label=LABELS[a]) for a in ALGOS]
    fig.legend(handles=handles,loc='lower center',ncol=5,
               frameon=True,framealpha=0.9,fontsize=8.5,bbox_to_anchor=(0.5,-0.04))
    fig.tight_layout()
    fig.savefig("fig6_summary_grid.pdf",bbox_inches='tight')
    fig.savefig("fig6_summary_grid.png",bbox_inches='tight')
    plt.close(fig); print("  saved fig6_summary_grid.pdf / .png")

# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Running benchmarks...")
    R = run_benchmarks()
    print("\nGenerating figures...")
    make_figs(R)
    print("\nDone.")
