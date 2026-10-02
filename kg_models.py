"""
kg_models.py — Phases 5–9
  5 Node2Vec embedding        biased random walks (p, q) on the TRAIN graph + Skip-Gram (gensim)
  6/7 Temporal & uncertainty  edge weights from kg_build.py (recency-weighted demand × confidence) bias the walks
  8 HGT                       2-layer Heterogeneous Graph Transformer (PyG), Node2Vec vectors as input features,
                              residual + LayerNorm + LeakyReLU, relation-specific DistMult decoder, link-prediction loss
  9 Recommendation & explain  rank occupations / courses for a learner; matched + missing skills + graph paths
Evaluation (no leakage: val/test edges are removed before ANY embedding is trained)
  · link prediction on held-out REQUIRES and TEACHES edges — AUC, filtered Hits@10, filtered MRR
  · occupation recommendation on (a) simulated profiles (PPT method) and (b) real held-out job postings
  · course retrieval for a missing skill on held-out TEACHES edges
  · 3 random seeds, mean ± sd, against popularity and skill-overlap baselines
"""
import json, math, random, sys, time
from collections import defaultdict
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from gensim.models import Word2Vec
from sklearn.metrics import roc_auc_score
from torch_geometric.data import HeteroData
from torch_geometric.nn import HGTConv

ROOT = Path(__file__).resolve().parent.parent; KG = ROOT / "kg"; RES = ROOT / "results"; RES.mkdir(exist_ok=True)
SEEDS = [int(s) for s in (sys.argv[1].split(",") if len(sys.argv) > 1 else ["0", "1", "2"])]
DIM, HID, HEADS, LAYERS = 64, 64, 4, 2
torch.set_num_threads(1)

# ------------------------------------------------------------------ load graph
On = pd.read_csv(KG / "nodes_occupation.csv"); Sn = pd.read_csv(KG / "nodes_skill.csv"); Cn = pd.read_csv(KG / "nodes_course.csv")
REQ = pd.read_csv(KG / "edges_requires.csv"); TEA = pd.read_csv(KG / "edges_teaches.csv"); REL = pd.read_csv(KG / "edges_related_to.csv")
oid = {k: i for i, k in enumerate(On.id)}; sid = {k: i for i, k in enumerate(Sn.id)}; cid = {k: i for i, k in enumerate(Cn.id)}
nO, nS, nC = len(oid), len(sid), len(cid)
req_all = [(oid[a], sid[b], w) for a, b, w in zip(REQ.occ, REQ.skill, REQ.weight)]
tea_all = [(cid[a], sid[b], 1.0) for a, b in zip(TEA.course, TEA.skill)]
rel_all = [(sid[a], sid[b], v) for a, b, v in zip(REL.src, REL.dst, REL.npmi)]
req_pos = defaultdict(set); tea_pos = defaultdict(set)
for o, s, _ in req_all: req_pos[o].add(s)
for c, s, _ in tea_all: tea_pos[c].add(s)

def split_edges(seed):
    rng = random.Random(seed)
    tr, va, te = [], [], []
    by = defaultdict(list)
    for e in req_all: by[e[0]].append(e)
    for o, es in by.items():                         # 80/10/10 per occupation
        rng.shuffle(es); n = len(es); a, b = int(.8 * n), int(.9 * n)
        tr += es[:a]; va += es[a:b]; te += es[b:]
    ttr, tva, tte = [], [], []
    byc = defaultdict(list)
    for e in tea_all: byc[e[0]].append(e)
    for c, es in byc.items():                        # hold out one edge per course that teaches ≥ 2 skills
        rng.shuffle(es)
        if len(es) >= 2: (tva if rng.random() < .5 else tte).append(es[0]); ttr += es[1:]
        else: ttr += es
    return tr, va, te, ttr, tva, tte

def prepares_from(req_tr, tea_tr):                   # PREPARES_FOR recomputed from TRAIN edges only (no leakage)
    occ_w = defaultdict(dict); c_sk = defaultdict(set)
    for o, s, w in req_tr: occ_w[o][s] = w
    for c, s, _ in tea_tr: c_sk[c].add(s)
    out = []
    for o, wm in occ_w.items():
        top = set(sorted(wm, key=wm.get, reverse=True)[:20]); tot = sum(wm.values())
        cand = [(c, sum(wm.get(s, 0) for s in ss) / tot) for c, ss in c_sk.items() if ss & top]
        out += [(c, o, cov) for c, cov in sorted(cand, key=lambda t: -t[1])[:10]]
    return out

# ------------------------------------------------------------------ Phase 5: Node2Vec
def node2vec(req_tr, tea_tr, rel, prep, seed, p=1.0, q=0.5, walks=20, length=30):
    rng = np.random.default_rng(seed)
    off_S, off_C = nO, nO + nS; N = nO + nS + nC
    adj = defaultdict(dict)
    def add(u, v, w): adj[u][v] = max(adj[u].get(v, 0), w); adj[v][u] = max(adj[v].get(u, 0), w)
    for o, s, w in req_tr: add(o, off_S + s, w)
    for c, s, w in tea_tr: add(off_C + c, off_S + s, w)
    for a, b, w in rel: add(off_S + a, off_S + b, w)
    for c, o, w in prep: add(off_C + c, o, max(w * 5, .2))
    nb = {u: (np.array(list(d.keys())), np.array(list(d.values()), dtype=float)) for u, d in adj.items()}
    sents = []
    nodes = [u for u in range(N) if u in nb]
    for _ in range(walks):
        rng.shuffle(nodes)
        for start in nodes:
            walk = [start]
            while len(walk) < length:
                cur = walk[-1]; ns, ws = nb[cur]
                if len(walk) == 1: nxt = rng.choice(ns, p=ws / ws.sum())
                else:
                    prev = walk[-2]; prev_nb = adj[prev]
                    bias = np.where(ns == prev, 1 / p, np.where(np.isin(ns, list(prev_nb)), 1.0, 1 / q))
                    pr = ws * bias; nxt = rng.choice(ns, p=pr / pr.sum())
                walk.append(int(nxt))
            sents.append([str(x) for x in walk])
    m = Word2Vec(sents, vector_size=DIM, window=5, min_count=0, sg=1, negative=5, workers=1, epochs=5, seed=seed)
    E = np.zeros((N, DIM), dtype=np.float32)
    for u in range(N):
        if str(u) in m.wv: E[u] = m.wv[str(u)]
    return E[:nO], E[nO:nO + nS], E[nO + nS:]

# ------------------------------------------------------------------ Phase 8: HGT
class HGT(nn.Module):
    def __init__(self, metadata):
        super().__init__()
        self.inp = nn.ModuleDict({t: nn.Linear(DIM, HID) for t in metadata[0]})
        self.convs = nn.ModuleList([HGTConv(HID, HID, metadata, heads=HEADS) for _ in range(LAYERS)])
        self.norms = nn.ModuleList([nn.ModuleDict({t: nn.LayerNorm(HID) for t in metadata[0]}) for _ in range(LAYERS)])
        self.rel = nn.ParameterDict({"requires": nn.Parameter(torch.ones(HID)), "teaches": nn.Parameter(torch.ones(HID))})
        self.drop = nn.Dropout(0.2)
    def forward(self, data):
        h = {t: F.leaky_relu(self.inp[t](x)) for t, x in data.x_dict.items()}
        for conv, ln in zip(self.convs, self.norms):
            out = conv(h, data.edge_index_dict)
            h = {t: ln[t](h[t] + self.drop(F.leaky_relu(out[t]))) for t in h}   # residual + LayerNorm + LeakyReLU
        return h
    def score(self, h, rel, src_t, src, dst):          # DistMult decoder
        return (h[src_t][src] * self.rel[rel] * h["skill"][dst]).sum(-1)

def hetero(req_tr, tea_tr, rel, prep, Eo, Es, Ec):
    d = HeteroData()
    d["occupation"].x = torch.tensor(Eo); d["skill"].x = torch.tensor(Es); d["course"].x = torch.tensor(Ec)
    T = lambda L: torch.tensor(L, dtype=torch.long).t().contiguous() if L else torch.empty((2, 0), dtype=torch.long)
    d["occupation", "requires", "skill"].edge_index = T([(o, s) for o, s, _ in req_tr])
    d["skill", "rev_requires", "occupation"].edge_index = T([(s, o) for o, s, _ in req_tr])
    d["course", "teaches", "skill"].edge_index = T([(c, s) for c, s, _ in tea_tr])
    d["skill", "rev_teaches", "course"].edge_index = T([(s, c) for c, s, _ in tea_tr])
    d["course", "prepares_for", "occupation"].edge_index = T([(c, o) for c, o, _ in prep])
    d["occupation", "rev_prepares_for", "course"].edge_index = T([(o, c) for c, o, _ in prep])
    d["skill", "related_to", "skill"].edge_index = T([(a, b) for a, b, _ in rel] + [(b, a) for a, b, _ in rel])
    return d

def train_hgt(data, req_tr, tea_tr, req_va, seed, epochs=300):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    m = HGT(data.metadata()); opt = torch.optim.Adam(m.parameters(), lr=5e-3, weight_decay=1e-4)
    ro = torch.tensor([o for o, _, _ in req_tr]); rs = torch.tensor([s for _, s, _ in req_tr])
    rw = torch.tensor([w for _, _, w in req_tr], dtype=torch.float)
    tc = torch.tensor([c for c, _, _ in tea_tr]); ts = torch.tensor([s for _, s, _ in tea_tr])
    best, best_state, K = -1, None, 5
    for ep in range(1, epochs + 1):
        m.train(); opt.zero_grad(); h = m(data)
        pos_r = m.score(h, "requires", "occupation", ro, rs)
        neg_r = m.score(h, "requires", "occupation", ro.repeat(K), torch.tensor(rng.integers(0, nS, len(ro) * K)))
        pos_t = m.score(h, "teaches", "course", tc, ts)
        neg_t = m.score(h, "teaches", "course", tc.repeat(K), torch.tensor(rng.integers(0, nS, len(tc) * K)))
        # edge weight (demand × confidence) scales the positive REQUIRES loss → temporal/uncertainty info used in training
        loss = (F.binary_cross_entropy_with_logits(pos_r, torch.ones_like(pos_r), weight=0.5 + rw)
                + F.binary_cross_entropy_with_logits(neg_r, torch.zeros_like(neg_r))
                + F.binary_cross_entropy_with_logits(pos_t, torch.ones_like(pos_t))
                + F.binary_cross_entropy_with_logits(neg_t, torch.zeros_like(neg_t)))
        loss.backward(); opt.step()
        if ep % 10 == 0:
            m.eval()
            with torch.no_grad(): h = m(data)
            S = lambda o: m.score(h, "requires", "occupation", torch.full((nS,), o), torch.arange(nS)).detach().numpy()
            mrr = rank_metrics(req_va, S, req_known(req_va))["MRR"]
            if mrr > best: best, best_state = mrr, {k: v.clone() for k, v in m.state_dict().items()}
    m.load_state_dict(best_state); m.eval()
    with torch.no_grad(): h = m(data)
    return m, h

# ------------------------------------------------------------------ metrics
def req_known(_): return req_pos
def rank_metrics(test, score_fn, known, k=10):
    rr, hit = [], []
    cache = {}
    for u, v, _ in test:
        if u not in cache: cache[u] = score_fn(u)
        sc = cache[u].copy(); others = [x for x in known[u] if x != v]
        sc[others] = -np.inf                                           # filtered setting
        rank = 1 + int((sc > sc[v]).sum())
        rr.append(1 / rank); hit.append(rank <= k)
    return {"MRR": float(np.mean(rr)), "Hits@10": float(np.mean(hit))}

def auc(test, pair_fn, known, n_items, seed):
    rng = np.random.default_rng(seed); y, s = [], []
    for u, v, _ in test:
        neg = int(rng.integers(0, n_items))
        while neg in known[u]: neg = int(rng.integers(0, n_items))
        y += [1, 0]; s += [pair_fn(u, v), pair_fn(u, neg)]
    return float(roc_auc_score(y, s))

def rec_metrics(ranks_lists, k=10):          # ranks_lists: list of (list of ranks of relevant items, n_relevant)
    mrr, rec, ndcg, h1 = [], [], [], []
    for ranks, nrel in ranks_lists:
        ranks = sorted(ranks)
        mrr.append(1 / ranks[0]); h1.append(ranks[0] == 1)
        rec.append(sum(r <= k for r in ranks) / nrel)
        dcg = sum(1 / math.log2(r + 1) for r in ranks if r <= k); idcg = sum(1 / math.log2(i + 2) for i in range(min(nrel, k)))
        ndcg.append(dcg / idcg)
    return {"Hits@1": float(np.mean(h1)), "MRR": float(np.mean(mrr)), "Recall@10": float(np.mean(rec)), "NDCG@10": float(np.mean(ndcg))}

def cos(A, B):
    A = A / (np.linalg.norm(A, axis=-1, keepdims=True) + 1e-9); B = B / (np.linalg.norm(B, axis=-1, keepdims=True) + 1e-9)
    return A @ B.T

# ------------------------------------------------------------------ profiles
def simulated_profiles(req_tr, seed, per_occ=30):
    """PPT method: a simulated learner holds a weighted random subset (3–8) of one occupation's
    required skills (train edges only) plus 2 random 'noise' skills; the source occupation is the target."""
    rng = np.random.default_rng(seed); by = defaultdict(list)
    for o, s, w in req_tr: by[o].append((s, w))
    prof = []
    for o, L in by.items():
        ss = np.array([s for s, _ in L]); ww = np.array([w for _, w in L]); ww = ww / ww.sum()
        for _ in range(per_occ):
            k = int(rng.integers(3, 9)); k = min(k, len(ss))
            pick = set(rng.choice(ss, size=k, replace=False, p=ww).tolist()) | set(rng.integers(0, nS, 2).tolist())
            prof.append((sorted(pick), o))
    return prof

HELD = json.load(open(KG / "heldout_profiles.json"))
def heldout_profiles(seed, cap=60):
    rng = random.Random(seed); by = defaultdict(list)
    for p in HELD:
        if p["occ"] in oid: by[p["occ"]].append(p)
    out = []
    for o, L in by.items():
        rng.shuffle(L); out += [([sid[s] for s in p["skills"] if s in sid], oid[o]) for p in L[:cap]]
    return [x for x in out if len(x[0]) >= 3]

def occ_ranks(profiles, score_fn):
    out = []
    for skills, target in profiles:
        sc = score_fn(skills); rank = 1 + int((sc > sc[target]).sum()); out.append(([rank], 1))
    return out

# ------------------------------------------------------------------ run one seed
def run(seed):
    t0 = time.time()
    req_tr, req_va, req_te, tea_tr, tea_va, tea_te = split_edges(seed)
    prep = prepares_from(req_tr, tea_tr)
    Eo, Es, Ec = node2vec(req_tr, tea_tr, rel_all, prep, seed)
    data = hetero(req_tr, tea_tr, rel_all, prep, Eo, Es, Ec)
    model, h = train_hgt(data, req_tr, tea_tr, req_va, seed)
    Ho, Hs, Hc = h["occupation"], h["skill"], h["course"]
    with torch.no_grad():
        HGT_req = torch.stack([model.score(h, "requires", "occupation", torch.full((nS,), o), torch.arange(nS)) for o in range(nO)]).numpy()
        HGT_tea = torch.stack([model.score(h, "teaches", "course", torch.full((nS,), c), torch.arange(nS)) for c in range(nC)]).numpy()
    N2V_req = cos(Eo, Es); N2V_tea = cos(Ec, Es)
    deg_s = np.zeros(nS)
    for _, s, _ in req_tr: deg_s[s] += 1
    for _, s, _ in tea_tr: deg_s[s] += 1
    POP_req = np.tile(deg_s, (nO, 1)); POP_tea = np.tile(deg_s, (nC, 1))

    R = {"seed": seed}
    # --- link prediction
    for rel, test, mats, known, nItems in [("REQUIRES", req_te, {"Popularity": POP_req, "Node2Vec": N2V_req, "HGT": HGT_req}, req_pos, nS),
                                           ("TEACHES", tea_te, {"Popularity": POP_tea, "Node2Vec": N2V_tea, "HGT": HGT_tea}, tea_pos, nS)]:
        for name, M in mats.items():
            r_ = rank_metrics(test, lambda u, M=M: M[u], known)
            r_["AUC"] = auc(test, lambda u, v, M=M: M[u, v] + 1e-9 * np.random.rand(), known, nItems, seed)
            R[f"LP/{rel}/{name}"] = r_
    # --- occupation recommendation
    W = np.zeros((nO, nS))
    for o, s, w in req_tr: W[o, s] = w
    def overlap(sk): return W[:, sk].sum(1) / W.sum(1)
    def n2v(sk): return N2V_req[:, sk].mean(1)
    def hgt(sk): return (1 / (1 + np.exp(-HGT_req[:, sk]))).mean(1)
    rng = np.random.default_rng(seed)
    def rnd(sk): return rng.random(nO)
    for pname, profs in [("simulated", simulated_profiles(req_tr, seed)), ("heldout_postings", heldout_profiles(seed))]:
        R[f"n_profiles/{pname}"] = len(profs)
        for name, fn in [("Random", rnd), ("Skill overlap", overlap), ("Node2Vec", n2v), ("HGT", hgt)]:
            R[f"REC/{pname}/{name}"] = rec_metrics(occ_ranks(profs, fn))
    # --- course retrieval for a missing skill (held-out TEACHES edges)
    by_skill = defaultdict(set)
    for c, s, _ in tea_te: by_skill[s].add(c)
    known_c = defaultdict(set)
    for c, s, _ in tea_tr + tea_va: known_c[s].add(c)
    deg_c = np.bincount([c for c, _, _ in tea_tr], minlength=nC).astype(float)
    for name, M in [("Popularity", np.tile(deg_c, (nS, 1)).T), ("Node2Vec", N2V_tea), ("HGT", HGT_tea)]:
        lists = []
        for s, rel_c in by_skill.items():
            sc = M[:, s].astype(float).copy(); sc[list(known_c[s] - rel_c)] = -np.inf
            ranks = [1 + int((sc > sc[c]).sum()) for c in rel_c]; lists.append((ranks, len(rel_c)))
        R[f"COURSE/{name}"] = rec_metrics(lists)
    R["n_test"] = {"REQUIRES": len(req_te), "TEACHES": len(tea_te), "course_queries": len(by_skill)}
    R["seconds"] = round(time.time() - t0, 1)
    if seed == SEEDS[0]:
        np.savez(RES / "embeddings_seed0.npz", hgt_occ=Ho.numpy(), hgt_skill=Hs.numpy(), hgt_course=Hc.numpy(), n2v_occ=Eo, n2v_skill=Es, n2v_course=Ec,
                 hgt_req=HGT_req, hgt_tea=HGT_tea)
    print(f"seed {seed} done in {R['seconds']} s", flush=True)
    return R

if __name__ == "__main__":
    runs = [run(s) for s in SEEDS]
    keys = [k for k in runs[0] if "/" in k and isinstance(runs[0][k], dict)]
    summary = {}
    for k in keys:
        summary[k] = {m: {"mean": float(np.mean([r[k][m] for r in runs])), "sd": float(np.std([r[k][m] for r in runs]))} for m in runs[0][k]}
    out = {"seeds": SEEDS, "summary": summary, "runs": runs,
           "setup": dict(dim=DIM, hidden=HID, heads=HEADS, layers=LAYERS, node2vec=dict(p=1.0, q=0.5, walks=20, length=30, window=5),
                         split="REQUIRES 80/10/10 per occupation; TEACHES: one edge per multi-skill course held out (val/test)",
                         ranking="filtered, against all skills (link prediction) / all occupations / all courses")}
    json.dump(out, open(RES / "metrics.json", "w"), indent=1)
    for k, v in summary.items():
        print(f"{k:42s} " + "  ".join(f"{m} {d['mean']:.4f}±{d['sd']:.4f}" for m, d in v.items()))
