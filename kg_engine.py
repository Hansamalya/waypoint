"""kg_engine.py — the website's ranking and explanation logic in Python (mirrors js/app.js and pipeline/src/kg_models.py)."""
import json, math, re
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data" / "kg_web.json"
norm = lambda t: " " + re.sub(r"[^a-z0-9+#./]+", " ", str(t).lower()) + " "

class Engine:
    def __init__(self, path=DATA):
        self.db = db = json.loads(Path(path).read_text(encoding="utf-8"))
        self.skill_index = {s["name"].lower(): i for i, s in enumerate(db["skills"])}
        self.req = [{e[0]: dict(w=e[1], d=e[2], c=e[3], src=e[4]) for e in o["requires"]} for o in db["occupations"]]
        self.teachers = [[] for _ in db["skills"]]
        for ci, c in enumerate(db["courses"]):
            for s in c["teaches"]: self.teachers[s].append(ci)
        self.rel = [[] for _ in db["skills"]]
        for a, b, v in db["related"]: self.rel[a].append((b, v)); self.rel[b].append((a, v))
        self.conc = {c: re.compile(p, re.I) for c, p in db["lexicon"]["concepts"].items()}
        self.amb = {c: re.compile(p) for c, p in db["lexicon"]["ambiguous"].items()}

    def skills(self, names):
        return sorted({self.skill_index[n.strip().lower()] for n in names if n.strip().lower() in self.skill_index})

    def extract(self, text):
        nt = norm(text); out = set()
        for i, s in enumerate(self.db["skills"]):
            n = s["name"]
            if s["kind"] == "concept": ok = bool(self.conc[n].search(nt)) if n in self.conc else False
            elif n in self.amb: ok = bool(self.amb[n].search(text))
            else: ok = norm(n) in nt
            if ok: out.add(i)
        return [self.db["skills"][i]["name"] for i in sorted(out)]

    def rank(self, names, model="hgt", top=None):
        U = self.skills(names); res = []
        for oi, o in enumerate(self.db["occupations"]):
            tot = sum(e["w"] for e in self.req[oi].values()); have = sum(self.req[oi][s]["w"] for s in U if s in self.req[oi])
            hgt = sum(1 / (1 + math.exp(-self.db["scores"]["hgt"][oi][s])) for s in U) / len(U) if U else 0.0
            n2v = sum(self.db["scores"]["node2vec"][oi][s] for s in U) / len(U) if U else 0.0
            cov = have / tot if tot else 0.0
            res.append(dict(id=o["id"], name=o["name"], domain=o["domain"], hgt=round(hgt, 4), node2vec=round(n2v, 4), coverage=round(cov, 4),
                            score=round({"hgt": hgt, "node2vec": n2v}.get(model, cov), 4)))
        res.sort(key=lambda r: -r["score"])
        return res[:top] if top else res

    def explain(self, names, occ_id, n=6):
        U = set(self.skills(names)); oi = next(i for i, o in enumerate(self.db["occupations"]) if o["id"] == occ_id)
        o = self.db["occupations"][oi]; S = self.db["skills"]; C = self.db["courses"]
        req = sorted(self.req[oi].items(), key=lambda kv: -kv[1]["w"])
        matched = [dict(skill=S[s]["name"], weight=e["w"]) for s, e in req if s in U]
        missing = [dict(skill=S[s]["name"], weight=e["w"], demand=e["d"], confidence=e["c"], source=e["src"]) for s, e in req if s not in U][:n]
        courses, paths = [], []
        for s, e in req:
            if s in U: continue
            for ci in self.teachers[s][:1]:
                courses.append(dict(course=C[ci]["name"], teaches=S[s]["name"], url=C[ci]["url"]))
                paths.append(f'{C[ci]["name"]} —TEACHES→ {S[s]["name"]} ←REQUIRES— {o["name"]}')
            bridge = [(x, v) for x, v in self.rel[s] if x in U]
            if bridge:
                x = max(bridge, key=lambda t: t[1])[0]; paths.append(f'{S[x]["name"]} —RELATED_TO→ {S[s]["name"]} ←REQUIRES— {o["name"]}')
            if len(courses) >= 4: break
        return dict(occupation=o["name"], matched=matched, missing=missing, courses=courses, paths=paths[:4])
