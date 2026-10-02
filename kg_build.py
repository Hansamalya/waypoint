"""
kg_build.py — Phases 1–4 of the framework
  1 Data ingestion        O*NET 31.0, LinkedIn postings, Coursera catalogue
  2 Entity standardisation tool names → canonical skills; concept lexicon; course titles → skills
  3 KG construction       Occupation / Skill / Course nodes; REQUIRES, TEACHES, PREPARES_FOR, RELATED_TO edges
                          with recency-weighted demand (temporal modelling) and confidence (uncertainty modelling)
  4 KG evaluation         schema conformance, connectivity, consistency
Postings are split 80/20 per occupation BEFORE any weight is computed; the 20 % is kept for evaluation only.
"""
import json, math, re, sys, random
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")

RAW = Path(sys.argv[1] if len(sys.argv) > 1 else "raw")
OUT = Path(__file__).resolve().parent.parent / "kg"; OUT.mkdir(exist_ok=True)
rng = random.Random(42)
HALF_LIFE_DAYS = 14          # temporal modelling: a posting loses half its weight every 14 days
MAX_POSTINGS = 3000          # per occupation, to keep the build light

# ---------------------------------------------------------------- occupations (6 domains, as in the PPT)
OCC = [  # O*NET code, display name, domain, posting-title pattern (most specific first)
 ("15-1299.04", "Penetration Tester", "Cybersecurity", r"penetration test|pen tester|ethical hacker|red team"),
 ("15-1299.06", "Digital Forensics Analyst", "Cybersecurity", r"forensic"),
 ("15-1299.05", "Information Security Engineer", "Cybersecurity", r"security engineer|security architect|cloud security"),
 ("15-1212.00", "Information Security Analyst", "Cybersecurity", r"security analyst|cyber ?security|soc analyst|information security"),
 ("15-1299.07", "Blockchain Engineer", "Software & Web", r"blockchain|web3|solidity|smart contract"),
 ("15-1255.01", "Video Game Designer", "UI/UX & Design", r"game (designer|developer|programmer)|level designer|unity developer"),
 ("15-1255.00", "UX / UI Designer", "UI/UX & Design", r"\bux\b|\bui\b.*designer|user experience|product designer|interaction designer"),
 ("27-1024.00", "Graphic Designer", "UI/UX & Design", r"graphic designer|visual designer|brand designer"),
 ("15-2051.00", "Data Scientist", "AI & Machine Learning", r"data scien|machine learning|\bml engineer|ai engineer|deep learning"),
 ("15-1221.00", "AI Research Scientist", "AI & Machine Learning", r"research scientist|applied scientist|research engineer"),
 ("15-2041.00", "Statistician", "Data Analytics & BI", r"statistician|biostatistic"),
 ("15-2051.01", "BI / Data Analyst", "Data Analytics & BI", r"business intelligence|\bbi (analyst|developer)|data analyst|reporting analyst"),
 ("15-2031.00", "Operations Research Analyst", "Data Analytics & BI", r"operations research|supply chain analyst|operations analyst"),
 ("15-1243.00", "Database Architect", "Data Analytics & BI", r"data architect|database architect"),
 ("15-1242.00", "Database Administrator / Data Engineer", "Data Analytics & BI", r"database (admin|engineer|developer)|\bdba\b|data engineer|etl developer"),
 ("15-1241.00", "Cloud / Network Architect", "Cloud & DevOps", r"cloud (architect|engineer)|network (architect|engineer)|solutions architect"),
 ("15-1299.08", "Systems / DevOps Engineer", "Cloud & DevOps", r"devops|site reliability|\bsre\b|platform engineer|systems engineer"),
 ("15-1244.00", "Network & Systems Administrator", "Cloud & DevOps", r"system administrator|systems administrator|network administrator|sysadmin"),
 ("11-3021.00", "IT Manager", "Cloud & DevOps", r"\bit manager|it director|director of (it|information technology)|information technology manager"),
 ("15-1254.00", "Web Developer", "Software & Web", r"web developer|front[- ]?end (engineer|developer)|react developer|wordpress developer"),
 ("15-1253.00", "QA / Test Engineer", "Software & Web", r"\bqa\b|quality assurance (analyst|engineer|tester)|test (engineer|automation)|\bsdet\b"),
 ("15-1251.00", "Computer Programmer", "Software & Web", r"\bprogrammer\b|programmer analyst"),
 ("15-1252.00", "Software Developer", "Software & Web", r"software (engineer|developer)|back[- ]?end (engineer|developer)|full[- ]?stack|mobile developer|ios developer|android developer"),
 ("15-1211.00", "Computer Systems Analyst", "Software & Web", r"systems analyst|business systems analyst"),
 ("13-1111.00", "Business Analyst / Consultant", "Data Analytics & BI", r"business analyst|management (consultant|analyst)|strategy (consultant|analyst)"),
 ("13-1082.00", "Project Manager", "Software & Web", r"project manager|program manager|scrum master"),
 ("13-1161.01", "SEO / Digital Marketing Strategist", "UI/UX & Design", r"\bseo\b|\bsem\b|search engine|digital marketing|\bppc\b|paid (search|media)"),
]

# ---------------------------------------------------------------- concept-skill lexicon (Phase 2)
CONCEPTS = {
 "Machine Learning": r"machine learning|\bml\b", "Deep Learning": r"deep learning|neural network", "NLP": r"natural language processing|\bnlp\b",
 "Computer Vision": r"computer vision|image recognition", "Generative AI": r"generative ai|\bllms?\b|large language model|\bgenai\b",
 "Statistics": r"statistic", "Data Analysis": r"data analy", "Data Visualization": r"data visuali|dashboards?\b",
 "Data Engineering": r"data engineering|data pipeline", "ETL": r"\betl\b|extract, transform", "Data Warehousing": r"data warehous",
 "Big Data": r"big data", "Data Modeling": r"data model", "Database Design": r"database design|relational database",
 "Cloud Computing": r"cloud computing|cloud platform|cloud infrastructure|cloud services", "DevOps": r"devops", "CI/CD": r"ci/cd|continuous integration|continuous deployment|continuous delivery",
 "Infrastructure as Code": r"infrastructure as code|\biac\b", "Microservices": r"microservice", "Networking": r"networking|tcp/ip|\blan\b|\bwan\b",
 "Network Security": r"network security|firewall", "Penetration Testing": r"penetration test|pen test", "Cryptography": r"cryptograph|encryption",
 "Incident Response": r"incident response", "Risk Assessment": r"risk assessment|risk management", "Vulnerability Management": r"vulnerabilit",
 "Web Development": r"web development|web applications?", "Front-end Development": r"front[- ]?end", "Back-end Development": r"back[- ]?end",
 "REST APIs": r"rest(ful)? api|\bapis?\b", "Mobile Development": r"mobile (app|development)|\bios\b|android",
 "Software Testing": r"software testing|test automation|automated test|unit test", "Object-Oriented Programming": r"object[- ]oriented",
 "Algorithms": r"algorithm", "Version Control": r"version control|\bgit\b",
 "UX Research": r"ux research|user research|usability", "Wireframing": r"wirefram", "Prototyping": r"prototyp",
 "Interaction Design": r"interaction design", "Visual Design": r"visual design|typography", "Game Development": r"game development|game design|game engine",
 "Blockchain": r"blockchain|smart contract", "Agile": r"\bagile\b|\bscrum\b|kanban", "Project Management": r"project management",
 "Digital Marketing": r"digital marketing|social media marketing", "SEO": r"\bseo\b|search engine optimi", "Requirements Analysis": r"requirements (gathering|analysis)",
 "Communication": r"communication skills|written and verbal|verbal and written", "Problem Solving": r"problem[- ]solving", "Leadership": r"leadership",
 "Optimization": r"optimi[sz]ation|operations research", "Business Intelligence": r"business intelligence",
}

# ---------------------------------------------------------------- tool-name standardisation (from O*NET technology names)
NAME_FIX = {
 "Amazon Web Services AWS software": "AWS", "Microsoft Azure software": "Azure", "Google Cloud software": "Google Cloud", "Microsoft Power BI": "Power BI",
 "Microsoft Excel": "Excel", "Microsoft SQL Server": "SQL Server", "Oracle Java": "Java", "The MathWorks MATLAB": "MATLAB", "Adobe Photoshop": "Photoshop",
 "Adobe Illustrator": "Illustrator", "Adobe InDesign": "InDesign", "Adobe After Effects": "After Effects", "Atlassian JIRA": "Jira", "Atlassian Confluence": "Confluence",
 "Apache Spark": "Spark", "Apache Kafka": "Kafka", "Apache Hadoop": "Hadoop", "Apache Airflow": "Airflow", "Salesforce software": "Salesforce", "SAP software": "SAP",
 "IBM SPSS Statistics": "SPSS", "Microsoft PowerPoint": "PowerPoint", "Microsoft Visual Basic for Applications VBA": "VBA", "Google Angular": "Angular",
 "Google Android": "Android SDK", "Apple iOS": "iOS SDK", "Unity Technologies Unity": "Unity", "Epic Games Unreal Engine": "Unreal Engine", "Amazon Redshift": "Redshift",
 "Microsoft .NET Framework": ".NET", "Oracle Database": "Oracle DB", "Splunk Enterprise": "Splunk", "Red Hat Enterprise Linux": "Linux", "Linux": "Linux",
 "Microsoft PowerShell": "PowerShell", "Microsoft SharePoint": "SharePoint", "Google Analytics": "Google Analytics", "Ansible software": "Ansible",
 "Docker": "Docker", "Kubernetes": "Kubernetes", "HashiCorp Terraform": "Terraform", "Figma": "Figma", "Snowflake": "Snowflake", "Alteryx software": "Alteryx",
}
GENERIC = {"Cloud","COM","Reporting","Reports","Database","Statistical","BASIC","Google","Analytics","Email","Tax","Testing","Pricing","LinkedIn","Optimization",
           "Simulation","Debugging","Deployment","Mathematical","Windows","Exchange","Instagram","Facebook","BI","Business intelligence","Spreadsheet","Word processing",
           "Scheduling","Data entry","Presentation","Graphics","Internet","Web browser","Calendar","Accounting","Inventory","Payroll","Budgeting","Compliance",
           "Document management","Project management","Works","Assessment","Job posting","Billing","YouTube","Microsoft Office","Office","Word","Access","Outlook"}
AMBIG = {"R": r"(?<![\w-])R(?=[,/)]| programming| language| studio)", "C": r"(?<![\w#+-])C(?=[,/)]| programming| language)",
         "Go": r"\bGolang\b|(?<![\w-])Go(?=[,/)]| programming| language)", "Swift": r"\bSwift(?:UI)?\b"}

CONCEPT_CANON = {c.lower(): c for c in CONCEPTS}

def display_name(raw):
    if raw in NAME_FIX: return NAME_FIX[raw]
    s = re.sub(r"\s+software$", "", raw).strip(); parts = s.split()
    if len(parts) > 1 and re.fullmatch(r"[A-Z]{2,6}", parts[-1]):
        acr = parts[-1]; words = parts[:-1][-len(acr):]
        if len(words) == len(acr) and "".join(w[0] for w in words).upper() == acr: return acr
    return s   # keep full product names ("Microsoft Teams"), so common words like "Teams" never match ordinary text

def norm(text):  # lower-case, keep + # . for C++, C#, Node.js
    return " " + re.sub(r"[^a-z0-9+#./]+", " ", text.lower()) + " "

def matcher(name):
    """Fast containment test on normalised text; regex only for ambiguous short names."""
    if name in AMBIG: rx = re.compile(AMBIG[name]); return lambda raw, nt: bool(rx.search(raw))
    key = norm(name)
    return lambda raw, nt: key in nt

def wilson_lb(k, n, z=1.96):
    if n == 0: return 0.0
    p = k / n; d = 1 + z*z/n
    return max(0.0, (p + z*z/(2*n) - z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))) / d)

def main():
    X = lambda f: pd.read_excel(RAW / "db_31_0_excel" / f"{f}.xlsx")
    print("Phase 1 · ingestion")
    occ_db = X("Occupation Data").set_index("O*NET-SOC Code")
    OCC_ = [o for o in OCC if o[0] in occ_db.index]
    missing = [o[0] for o in OCC if o[0] not in occ_db.index]
    if missing: print("  ! not in O*NET 31:", missing)
    codes = [o[0] for o in OCC_]
    tech = X("Software Skills"); tech = tech[tech["O*NET-SOC Code"].isin(codes)]
    pf = RAW / "postings.csv" if (RAW / "postings.csv").exists() else RAW / "postings_sample.csv"
    P = pd.read_csv(pf, usecols=["job_id", "title", "description", "skills_desc", "listed_time"])
    P["raw"] = P["title"].fillna("") + " \n " + P["description"].fillna("") + " \n " + P["skills_desc"].fillna("")
    P = P.drop(columns=["description", "skills_desc"])
    C = pd.read_csv(RAW / "coursea_data.csv").drop(columns=["Unnamed: 0"], errors="ignore")
    print(f"  O*NET occupations used: {len(OCC_)} · postings: {len(P):,} · courses: {len(C)}")

    # ---- assign each posting to at most one occupation (first matching pattern)
    lab = pd.Series([None] * len(P), index=P.index, dtype=object)
    for code, name, dom, pat in OCC_:
        m = lab.isna() & P["title"].str.contains(pat, case=False, regex=True, na=False)
        lab[m] = code
    P["occ"] = lab; P = P[P["occ"].notna()].copy()
    P = P.sample(frac=1, random_state=1).groupby("occ").head(MAX_POSTINGS).copy()
    # 80/20 split per occupation — held-out postings are never used to build the graph
    P["split"] = "build"
    for code, d in P.groupby("occ"):
        idx = d.sample(frac=0.2, random_state=7).index; P.loc[idx, "split"] = "heldout"
    t_max = P["listed_time"].max()
    P["age_days"] = (t_max - P["listed_time"]) / 86_400_000
    P["tau"] = np.power(0.5, P["age_days"] / HALF_LIFE_DAYS)             # recency weight
    P["nt"] = P["raw"].map(norm)
    print(f"  postings matched to occupations: {len(P):,} (build {int((P.split=='build').sum()):,}, held-out {int((P.split=='heldout').sum()):,})")

    print("Phase 2 · entity standardisation")
    onet_tools = defaultdict(dict)       # occ -> {tool: flag strength}
    for r_ in tech.itertuples():
        name = display_name(r_._3)
        name = CONCEPT_CANON.get(name.lower(), name)          # "Data analysis" → concept "Data Analysis"
        if name in GENERIC or len(name) < 2 and name not in AMBIG: continue
        flag = 0.9 if (r_._6 == "Y" or r_._7 == "Y") else 0.7
        onet_tools[r_._1][name] = max(flag, onet_tools[r_._1].get(name, 0))
    tool_vocab = sorted({t for d in onet_tools.values() for t in d if t not in CONCEPTS})
    match_tool = {t: matcher(t) for t in tool_vocab}
    match_con = {c: re.compile(p, re.I) for c, p in CONCEPTS.items()}

    def skills_in(raw, nt):
        s = {t for t, f in match_tool.items() if f(raw, nt)}
        s |= {c for c, rx in match_con.items() if rx.search(nt)}
        return s
    P["skills"] = [skills_in(a, b) for a, b in zip(P["raw"], P["nt"])]
    print(f"  tool vocabulary: {len(tool_vocab)} · concept lexicon: {len(CONCEPTS)}")

    print("Phase 3 · KG construction (recency-weighted demand + confidence)")
    B = P[P.split == "build"]
    req = []                              # (occ, skill, demand, confidence, weight, source)
    for code, d in B.groupby("occ"):
        n = len(d); tau = d["tau"].values; tsum = tau.sum()
        cnt, wcnt = Counter(), defaultdict(float)
        for sk, t in zip(d["skills"], tau):
            for s in sk: cnt[s] += 1; wcnt[s] += t
        cand = set(onet_tools[code]) | {s for s, k in cnt.items() if k / n >= 0.05 and k >= 5}
        rows = []
        for s in cand:
            k = cnt.get(s, 0)
            demand = wcnt.get(s, 0.0) / tsum                                   # recency-weighted share of postings
            e_onet = onet_tools[code].get(s, 0.0)                              # O*NET evidence
            e_post = 1 - math.exp(-k / 20)                                     # posting evidence saturates with count
            conf = 1 - (1 - e_onet) * (1 - e_post)                             # noisy-OR of the two sources
            if e_onet == 0 and wilson_lb(k, n) < 0.03: continue                # postings-only edge must be statistically supported
            if e_onet > 0 and k == 0 and e_onet < 0.9: continue                # O*NET example never seen in postings and not hot → drop
            src = "both" if (e_onet and k) else ("onet" if e_onet else "postings")
            rows.append([code, s, demand, conf, src, k, n])
        dmax = max((r_[2] for r_ in rows), default=1) or 1
        for r_ in rows:
            imp = r_[2] / dmax
            req.append(dict(occ=r_[0], skill=r_[1], demand=round(r_[2], 5), confidence=round(r_[3], 4),
                            weight=round(r_[3] * (0.3 + 0.7 * imp), 5), source=r_[4], mentions=r_[5], n_postings=r_[6]))
    REQ = pd.DataFrame(req)
    skills_used = set(REQ.skill)

    # TEACHES: course titles containing a skill name (exact standardised match)
    teach = []
    for c in C.itertuples():
        title = c.course_title.strip(); nt = norm(title)
        for s in skills_used:
            ok = match_tool[s](title, nt) if s in match_tool else bool(match_con[s].search(nt))
            if ok: teach.append(dict(course=title, skill=s, confidence=1.0, weight=1.0))
    TEA = pd.DataFrame(teach).drop_duplicates(["course", "skill"])
    crs = C.drop_duplicates("course_title").set_index(C.drop_duplicates("course_title").course_title.str.strip())

    # RELATED_TO: skill co-occurrence in build postings, normalised PMI
    sk_lists = [list(s & skills_used) for s in B["skills"]]
    single, pair = Counter(), Counter()
    for L in sk_lists:
        single.update(L)
        for i in range(len(L)):
            for j in range(i + 1, len(L)): pair[tuple(sorted((L[i], L[j])))] += 1
    N = len(sk_lists); rel = []
    for (a, b_), k in pair.items():
        if k < 20: continue
        pab, pa, pb = k / N, single[a] / N, single[b_] / N
        npmi = math.log(pab / (pa * pb)) / -math.log(pab)
        if npmi >= 0.25: rel.append(dict(src=a, dst=b_, npmi=round(npmi, 4), co=k))
    REL = pd.DataFrame(rel)
    if len(REL):  # keep top-5 neighbours per skill
        both = pd.concat([REL, REL.rename(columns={"src": "dst", "dst": "src"})])
        keep = both.sort_values("npmi", ascending=False).groupby("src").head(5)
        REL = keep[keep.src < keep.dst].drop_duplicates(["src", "dst"])

    # PREPARES_FOR: course teaches ≥1 of the occupation's 20 most important skills; keep the 10 best-covering courses
    occ_w = {o: dict(zip(d.skill, d.weight)) for o, d in REQ.groupby("occ")}
    c_sk = TEA.groupby("course")["skill"].apply(set).to_dict()
    prep = []
    for code, wmap in occ_w.items():
        top = set(sorted(wmap, key=wmap.get, reverse=True)[:20]); tot = sum(wmap.values())
        cand = [(c, sum(wmap.get(s, 0) for s in ss) / tot) for c, ss in c_sk.items() if ss & top]
        for c, cov in sorted(cand, key=lambda t: -t[1])[:10]:
            prep.append(dict(course=c, occ=code, coverage=round(cov, 4)))
    PRE = pd.DataFrame(prep)

    # ---------------------------------------------------------------- nodes
    occ_nodes = pd.DataFrame([dict(id=o[0], name=o[1], domain=o[2], onet_title=occ_db.loc[o[0], "Title"]) for o in OCC_ if o[0] in set(REQ.occ)])
    skill_nodes = pd.DataFrame([dict(id=s, name=s, kind="concept" if s in CONCEPTS else "tool") for s in sorted(skills_used)])
    course_ids = sorted(set(TEA.course))
    course_nodes = pd.DataFrame([dict(id=c, name=c, org=crs.loc[c, "course_organization"] if c in crs.index else "",
                                      level=crs.loc[c, "course_difficulty"] if c in crs.index else "",
                                      rating=float(crs.loc[c, "course_rating"]) if c in crs.index else None) for c in course_ids])
    occ_nodes.to_csv(OUT / "nodes_occupation.csv", index=False); skill_nodes.to_csv(OUT / "nodes_skill.csv", index=False)
    course_nodes.to_csv(OUT / "nodes_course.csv", index=False)
    REQ.to_csv(OUT / "edges_requires.csv", index=False); TEA.to_csv(OUT / "edges_teaches.csv", index=False)
    PRE.to_csv(OUT / "edges_prepares_for.csv", index=False); REL.to_csv(OUT / "edges_related_to.csv", index=False)

    # held-out posting profiles (skills found in postings the graph never saw)
    H = P[(P.split == "heldout")]
    prof = [dict(job_id=int(j), occ=o, skills=sorted(s & skills_used)) for j, o, s in zip(H.job_id, H.occ, H.skills) if len(s & skills_used) >= 3]
    json.dump(prof, open(OUT / "heldout_profiles.json", "w"))

    print("Phase 4 · KG evaluation")
    import networkx as nx
    G = nx.Graph()
    for n_ in occ_nodes.id: G.add_node(("O", n_))
    for n_ in skill_nodes.id: G.add_node(("S", n_))
    for n_ in course_nodes.id: G.add_node(("C", n_))
    E = {"REQUIRES": [(("O", a), ("S", b)) for a, b in zip(REQ.occ, REQ.skill)],
         "TEACHES": [(("C", a), ("S", b)) for a, b in zip(TEA.course, TEA.skill)],
         "PREPARES_FOR": [(("C", a), ("O", b)) for a, b in zip(PRE.course, PRE.occ)] if len(PRE) else [],
         "RELATED_TO": [(("S", a), ("S", b)) for a, b in zip(REL.src, REL.dst)] if len(REL) else []}
    schema = {"REQUIRES": ("O", "S"), "TEACHES": ("C", "S"), "PREPARES_FOR": ("C", "O"), "RELATED_TO": ("S", "S")}
    bad = sum(1 for r_, es in E.items() for u, v in es if (u[0], v[0]) != schema[r_])
    for es in E.values(): G.add_edges_from(es)
    n_nodes = G.number_of_nodes(); n_edges = sum(len(v) for v in E.values())
    lcc = max(nx.connected_components(G), key=len)
    dup = sum(len(v) - len(set(v)) for v in E.values())
    dangling = sum(1 for es in E.values() for u, v in es if u not in G or v not in G)
    stats = dict(
        nodes=dict(total=n_nodes, occupation=len(occ_nodes), skill=len(skill_nodes), course=len(course_nodes)),
        edges=dict(total=n_edges, **{k: len(v) for k, v in E.items()}),
        avg_degree=round(2 * n_edges / n_nodes, 2), lcc_pct=round(100 * len(lcc) / n_nodes, 2),
        components=nx.number_connected_components(G), isolated=sum(1 for n_ in G if G.degree(n_) == 0),
        schema_violations=bad, duplicate_edges=dup, dangling_edges=dangling,
        requires_sources=REQ.source.value_counts().to_dict(),
        postings=dict(matched=int(len(P)), build=int((P.split == "build").sum()), heldout=int(len(H)), heldout_profiles=len(prof)),
        half_life_days=HALF_LIFE_DAYS, date_range=[str(pd.to_datetime(P.listed_time.min(), unit="ms").date()), str(pd.to_datetime(t_max, unit="ms").date())],
    )
    json.dump(stats, open(OUT / "kg_stats.json", "w"), indent=1)
    print(json.dumps(stats, indent=1))

if __name__ == "__main__":
    main()
