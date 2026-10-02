"""export_web.py — packages the knowledge graph, model scores and evaluation results for the website (data/kg_web.json).
Run after kg_build.py, kg_models.py and kg_report.py:   python pipeline/src/export_web.py raw
"""
import json, re, sys
from pathlib import Path
import numpy as np, pandas as pd

P = Path(__file__).resolve().parent.parent; KG = P / "kg"; RES = P / "results"; WEB = P.parent / "data"; WEB.mkdir(exist_ok=True)
RAW = Path(sys.argv[1]) if len(sys.argv) > 1 else P.parent / "raw"
sys.path.insert(0, str(Path(__file__).parent))
from kg_build import CONCEPTS, AMBIG

On = pd.read_csv(KG / "nodes_occupation.csv"); Sn = pd.read_csv(KG / "nodes_skill.csv"); Cn = pd.read_csv(KG / "nodes_course.csv")
REQ = pd.read_csv(KG / "edges_requires.csv"); TEA = pd.read_csv(KG / "edges_teaches.csv")
PRE = pd.read_csv(KG / "edges_prepares_for.csv"); REL = pd.read_csv(KG / "edges_related_to.csv")
E = np.load(RES / "embeddings_seed0.npz")
def unit(A): return A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-9)
N2V = unit(E["n2v_occ"]) @ unit(E["n2v_skill"]).T                 # cosine, occupation × skill
HGT = E["hgt_req"]                                                  # DistMult logits, occupation × skill
HGT_T = E["hgt_tea"]                                                # course × skill

desc, tasks = {}, {}
try:
    od = pd.read_excel(RAW / "db_31_0_excel" / "Occupation Data.xlsx").set_index("O*NET-SOC Code")
    ts = pd.read_excel(RAW / "db_31_0_excel" / "Task Statements.xlsx")
    for c in On.id:
        desc[c] = od.loc[c, "Description"] if c in od.index else ""
        t = ts[ts["O*NET-SOC Code"] == c]
        tasks[c] = t.sort_values("Incumbents Responding", ascending=False)["Task"].head(4).tolist()
except Exception as e:
    print("  (O*NET text not found — descriptions left empty)", e)

IMG = {"15-1299.04": "1550751827-4bd374c3f58b", "15-1299.06": "1526374965328-7f61d4dc18c5", "15-1299.05": "1563986768609-322da13575f3",
       "15-1212.00": "1555949963-ff9fe0c870eb", "15-1299.07": "1639762681485-074b7f938ba0", "15-1255.01": "1511512578047-dfb367046420",
       "15-1255.00": "1561070791-2526d30994b5", "27-1024.00": "1626785774573-4b799315345d", "15-2051.00": "1551288049-bebda4e38f71",
       "15-1221.00": "1485827404703-89b55fcc595e", "15-2041.00": "1504868584819-f8e8b4b6d7e3", "15-2051.01": "1460925895917-afdab827c52f",
       "15-2031.00": "1507925921958-8a62f3d1a50d", "15-1243.00": "1544197150-b99a580bb7a8", "15-1242.00": "1558494949-ef010cbdcc31",
       "15-1241.00": "1451187580459-43490279c0fa", "15-1299.08": "1518432031352-d6fc5c10da5a", "15-1244.00": "1544197150-b99a580bb7a8",
       "11-3021.00": "1519389950473-47ba0277781c", "15-1254.00": "1461749280684-dccba630e2f6", "15-1253.00": "1517694712202-14dd9538aa97",
       "15-1251.00": "1542831371-29b0f74f9713", "15-1252.00": "1498050108023-c5249f4df085", "15-1211.00": "1531482615713-2afd69097998",
       "13-1111.00": "1542744173-8e7e53415bb0", "13-1082.00": "1552664730-d307ca884978", "13-1161.01": "1432888498266-38ffec3eaf0a"}
U = "https://images.unsplash.com/photo-{}?auto=format&fit=crop&w=900&q=70"

sk_index = {s: i for i, s in enumerate(Sn.id)}
occ = []
for i, r in On.iterrows():
    d = REQ[REQ.occ == r.id].sort_values("weight", ascending=False)
    occ.append(dict(id=r.id, name=r["name"], domain=r.domain, onet_title=r.onet_title, description=desc.get(r.id, ""), tasks=tasks.get(r.id, []),
                    image=U.format(IMG[r.id]) if r.id in IMG else "", postings=int(d.n_postings.max()) if len(d) else 0,
                    requires=[[sk_index[s], round(float(w), 3), round(float(dm), 3), round(float(c), 3), src]
                              for s, w, dm, c, src in zip(d.skill, d.weight, d.demand, d.confidence, d.source)]))
c_index = {c: i for i, c in enumerate(Cn.id)}
courses = [dict(name=r["name"], org=r.org if isinstance(r.org, str) else "", level=r.level if isinstance(r.level, str) else "",
                rating=None if pd.isna(r.rating) else float(r.rating),
                url="https://www.coursera.org/search?query=" + re.sub(r"\s+", "+", str(r["name"])),
                teaches=sorted(sk_index[s] for s in TEA[TEA.course == r.id].skill)) for _, r in Cn.iterrows()]
prepares = {o: [c_index[c] for c in PRE[PRE.occ == o].sort_values("coverage", ascending=False).course] for o in On.id}
related = [[sk_index[a], sk_index[b], round(float(v), 3)] for a, b, v in zip(REL.src, REL.dst, REL.npmi)]
hgt_courses = {int(s): [int(c) for c in np.argsort(-HGT_T[:, s])[:3]] for s in range(len(Sn))}   # model-suggested courses per skill

M = json.load(open(RES / "metrics.json")); ST = json.load(open(KG / "kg_stats.json"))
out = dict(
    meta=dict(title="Knowledge Graph Embedding on Skill and Career Recommendation", seeds=M["seeds"], setup=M["setup"]),
    stats=ST, summary=M["summary"], n_profiles={k: v for k, v in M["runs"][0].items() if k.startswith("n_")},
    occupations=occ, skills=[dict(name=r["name"], kind=r.kind) for _, r in Sn.iterrows()], courses=courses,
    prepares=prepares, related=related, hgt_courses=hgt_courses,
    scores=dict(hgt=np.round(HGT, 3).tolist(), node2vec=np.round(N2V, 3).tolist()),
    lexicon=dict(concepts=CONCEPTS, ambiguous=AMBIG))
json.dump(out, open(WEB / "kg_web.json", "w"), separators=(",", ":"))
print(f"wrote {WEB/'kg_web.json'}  ({(WEB/'kg_web.json').stat().st_size/1e6:.2f} MB) · {len(occ)} occupations · {len(Sn)} skills · {len(courses)} courses")
