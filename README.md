# SkillGraph — Knowledge Graph Embedding on Skill and Career Recommendation
## Pages
| Page | What it shows |
|---|---|
| Dashboard | Skills (autocomplete over 475 graph skills), résumé upload, model choice, top matches, graph analytics |
| Careers | 27 occupations ranked by **HGT**, **Node2Vec** or **skill overlap**; skill gap with weight, demand and confidence; matched / missing skills; RELATED_TO and TEACHES graph paths; courses |
| Roadmap | Quick wins (missing skills related to ones you have) → core gaps → courses → portfolio projects from O*NET tasks |
| Knowledge graph | Interactive D3 graph: occupations on a ring by domain, skills between them, optional courses |
| Results | All evaluation metrics (mean ± sd over 3 seeds) with honest notes |

## How the scores work (same as pipeline/src/kg_models.py)
- **HGT:** mean over your skills s of sigmoid(HGT DistMult score(occupation, s))
- **Node2Vec:** mean cosine similarity between the occupation and your skills
- **Skill overlap:** Σ weight of your skills ÷ Σ weight of all the occupation's skills
- **Edge weight** w = c · (0.3 + 0.7 · d / d_max), where d = recency-weighted demand (14-day half-life) and
  c = 1 − (1 − e_O*NET)(1 − e_postings) is the confidence
- **Skill gap:** matched = REQUIRES(o) ∩ your skills; missing = REQUIRES(o) − your skills, ranked by w; courses = TEACHES edges

## Rebuilding the data (optional)
The website ships with `data/kg_web.json`. To regenerate it from the raw datasets:
```bash
pip install -r pipeline/requirements.txt          # PyTorch, PyTorch Geometric, gensim … (large)
python pipeline/src/kg_build.py raw               # phases 1–4  → pipeline/kg/
python pipeline/src/kg_models.py 0,1,2            # phases 5–9  → pipeline/results/metrics.json
python pipeline/src/kg_report.py                  # explanation example + figures
python pipeline/src/export_web.py raw             # → data/kg_web.json
```
`raw/` must contain `db_31_0_excel/`, `postings.csv` and `coursea_data.csv`. VS Code has a launch configuration for each step.

## API (backend/app.py)
`GET /api/health` · `GET /api/occupations` · `POST /api/recommend {"skills": [...], "model": "hgt"|"node2vec"|"overlap"}` ·
`POST /api/explain {"skills": [...], "occupation": "15-2051.00"}` · `GET /api/metrics` · `POST /api/resume` (multipart `file`)

## Limitations
Postings cover 15 days in 2024, mostly US; course–skill links come from course titles only; accounts are stored in the browser.
Simulated profiles are drawn from the graph itself, so the held-out-posting results on the Results page are the realistic estimate.
