"""kg_report.py — explainable recommendation example + figures (KG visualisation, results charts)."""
import json
from collections import defaultdict
from pathlib import Path
import numpy as np, pandas as pd, networkx as nx, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent; KG = ROOT / "kg"; RES = ROOT / "results"; FIG = ROOT / "figures"; FIG.mkdir(exist_ok=True)
On = pd.read_csv(KG / "nodes_occupation.csv"); Sn = pd.read_csv(KG / "nodes_skill.csv"); Cn = pd.read_csv(KG / "nodes_course.csv")
REQ = pd.read_csv(KG / "edges_requires.csv"); TEA = pd.read_csv(KG / "edges_teaches.csv"); REL = pd.read_csv(KG / "edges_related_to.csv")
oid = {k: i for i, k in enumerate(On.id)}; sid = {k: i for i, k in enumerate(Sn.id)}
emb = np.load(RES / "embeddings_seed0.npz"); H = emb["hgt_req"]                     # HGT occupation×skill logits (seed 0)

# ------------------------------------------------------------------ Phase 9: recommendation + explanation
def recommend(profile, k=3, n_missing=5):
    sk = [sid[s] for s in profile if s in sid]
    score = (1 / (1 + np.exp(-H[:, sk]))).mean(1)
    out = []
    for o in np.argsort(-score)[:k]:
        code = On.id[o]; d = REQ[REQ.occ == code].copy(); d["priority"] = d.weight            # weight = demand × confidence
        matched = d[d.skill.isin(profile)].sort_values("priority", ascending=False)
        missing = d[~d.skill.isin(profile)].sort_values("priority", ascending=False).head(n_missing)
        courses = []
        for s in missing.skill:
            for c in TEA[TEA.skill == s].course.head(2): courses.append(dict(course=c, teaches=s))
        paths = [f"{c['course']} —TEACHES→ {c['teaches']} ←REQUIRES— {On.name[o]}" for c in courses[:2]]
        cov = matched.weight.sum() / d.weight.sum()
        out.append(dict(occupation=On.name[o], domain=On.domain[o], score=round(float(score[o]), 3), weighted_coverage=round(float(cov), 3),
                        matched=[(s, round(w, 2)) for s, w in zip(matched.skill.head(6), matched.weight.head(6))],
                        missing=[(s, round(w, 2), round(c, 2)) for s, w, c in zip(missing.skill, missing.weight, missing.confidence)],
                        courses=courses[:4], paths=paths))
    return out

PROFILE = ["Python", "SQL", "Excel", "Tableau", "Power BI", "Statistics", "Data Visualization", "Data Analysis"]
rec = recommend(PROFILE)
json.dump(dict(profile=PROFILE, recommendations=rec), open(RES / "explanation_example.json", "w"), indent=1)
for r in rec:
    print(f"\n{r['occupation']}  score {r['score']}  coverage {r['weighted_coverage']}")
    print("  matched:", r["matched"]); print("  missing:", r["missing"]); print("  courses:", [c["course"] for c in r["courses"]]); print("  path:", r["paths"][:1])

# ------------------------------------------------------------------ KG visualisation (real graph)
# occupations fixed on a ring grouped by domain; the 45 most important skills placed by force layout inside
DOM_COL = {"AI & Machine Learning": "#2E9E4F", "Data Analytics & BI": "#1F77B4", "Cloud & DevOps": "#8C564B",
           "Software & Web": "#9467BD", "Cybersecurity": "#D62728", "UI/UX & Design": "#E377C2"}
occ = On.sort_values(["domain", "name"]).reset_index(drop=True); dom = dict(zip(On.id, On.domain))
top_sk = list(REQ.groupby("skill").weight.sum().sort_values(ascending=False).head(45).index)
G = nx.Graph()
pos = {}
for i, r in occ.iterrows():
    a_ = 2 * np.pi * i / len(occ); G.add_node(("O", r.id)); pos[("O", r.id)] = np.array([np.cos(a_), np.sin(a_)])
for a_, b_, w in zip(REQ.occ, REQ.skill, REQ.weight):
    if b_ in top_sk and w >= 0.3: G.add_edge(("O", a_), ("S", b_), weight=w)
# (skills without a strong edge are left out so nothing floats free in the layout)
init = {n: (pos[n] if n in pos else np.random.default_rng(1).normal(0, .2, 2)) for n in G}
pos = nx.spring_layout(G, pos=init, fixed=[n for n in G if n[0] == "O"], k=0.34, iterations=400, seed=3, weight="weight")
fig, ax = plt.subplots(figsize=(12, 9)); ax.axis("off"); ax.set_aspect("equal")
for (u, v, d) in G.edges(data=True):
    o = u if u[0] == "O" else v
    ax.plot(*zip(pos[u], pos[v]), color=DOM_COL[dom[o[1]]], alpha=.10 + .35 * d["weight"], lw=.5 + 1.2 * d["weight"], zorder=1)
sk = [n for n in G if n[0] == "S"]; deg = dict(G.degree())
ax.scatter([pos[n][0] for n in sk], [pos[n][1] for n in sk], s=[30 + 22 * deg[n] for n in sk], c="#F28E2B", edgecolors="white", lw=.6, zorder=3)
top_lbl = set(sorted(sk, key=lambda n: -deg[n])[:22])
for n in sk:
    if n in top_lbl: ax.text(pos[n][0], pos[n][1], n[1], fontsize=7.4, ha="center", va="center", zorder=4,
                             bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=.75))
for i, r in occ.iterrows():
    x_, y_ = pos[("O", r.id)]; c_ = DOM_COL[r.domain]
    ax.scatter([x_], [y_], s=150, c=c_, edgecolors="white", lw=1, zorder=5)
    ang = np.degrees(np.arctan2(y_, x_)); flip = 90 < ang % 360 < 270
    ax.text(x_ * 1.07, y_ * 1.07, r["name"], fontsize=8.2, rotation=ang + (180 if flip else 0), rotation_mode="anchor",
            ha="right" if flip else "left", va="center", color=c_, fontweight="bold", zorder=6)
for d_, c_ in DOM_COL.items(): ax.scatter([], [], c=c_, s=60, label=d_)
ax.scatter([], [], c="#F28E2B", s=60, label="Skill (size = number of occupations)")
ax.legend(loc="upper left", bbox_to_anchor=(-.08, 1.08), frameon=False, fontsize=9)
ax.set_xlim(-1.75, 1.75); ax.set_ylim(-1.55, 1.55)
fig.savefig(FIG / "kg_real_graph.png", dpi=170, bbox_inches="tight", pad_inches=0.1); plt.close()

# ------------------------------------------------------------------ results charts
M = json.load(open(RES / "metrics.json"))["summary"]
def g(k, m): return M[k][m]["mean"], M[k][m]["sd"]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
fig, axs = plt.subplots(1, 2, figsize=(11, 3.8))
models = ["Popularity", "Node2Vec", "HGT"]; colors = ["#BFBFBF", "#E2B64B", "#A58BCB"]
for ax, rel in zip(axs, ["REQUIRES", "TEACHES"]):
    x = np.arange(3); w = .26
    for i, (mdl, c) in enumerate(zip(models, colors)):
        vals = [g(f"LP/{rel}/{mdl}", m) for m in ("AUC", "Hits@10", "MRR")]
        ax.bar(x + (i - 1) * w, [v[0] for v in vals], w, yerr=[v[1] for v in vals], color=c, label=mdl, capsize=2)
    ax.set_xticks(x); ax.set_xticklabels(["AUC", "Hits@10", "MRR"]); ax.set_ylim(0, 1.05)
    ax.set_title(f"Link prediction — {rel} (held-out edges)")
axs[0].legend(frameon=False, fontsize=9)
fig.tight_layout(); fig.savefig(FIG / "results_link_prediction.png", dpi=200); plt.close()

fig, axs = plt.subplots(1, 2, figsize=(11, 3.8))
models = ["Random", "Skill overlap", "Node2Vec", "HGT"]; colors = ["#E0E0E0", "#9DB3D6", "#E2B64B", "#A58BCB"]
for ax, pn, title in zip(axs, ["simulated", "heldout_postings"], ["Simulated profiles (PPT method)", "Real held-out job postings"]):
    x = np.arange(3); w = .2
    for i, (mdl, c) in enumerate(zip(models, colors)):
        vals = [g(f"REC/{pn}/{mdl}", m) for m in ("MRR", "Recall@10", "NDCG@10")]
        ax.bar(x + (i - 1.5) * w, [v[0] for v in vals], w, yerr=[v[1] for v in vals], color=c, label=mdl, capsize=2)
    ax.set_xticks(x); ax.set_xticklabels(["MRR", "Recall@10", "NDCG@10"]); ax.set_ylim(0, 1.05); ax.set_title(f"Occupation recommendation — {title}")
axs[0].legend(frameon=False, fontsize=9, ncol=2)
fig.tight_layout(); fig.savefig(FIG / "results_recommendation.png", dpi=200); plt.close()
print("\nfigures written")
