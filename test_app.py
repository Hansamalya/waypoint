"""test_app.py — checks data, engine and API.   python backend/test_app.py"""
import io, sys, unittest
from pathlib import Path
HERE = Path(__file__).resolve().parent; ROOT = HERE.parent; sys.path.insert(0, str(HERE))
from kg_engine import Engine
from app import app

class Tests(unittest.TestCase):
    eng = Engine(); c = app.test_client()
    def test_data_shape(self):
        d = self.eng.db
        self.assertEqual(len(d["occupations"]), 27); self.assertEqual(len(d["scores"]["hgt"]), 27); self.assertEqual(len(d["scores"]["hgt"][0]), len(d["skills"]))
        self.assertEqual(d["stats"]["schema_violations"], 0)
        self.assertAlmostEqual(d["stats"]["avg_degree"], round(2 * d["stats"]["edges"]["total"] / d["stats"]["nodes"]["total"], 2))
    def test_site_files(self):
        for f in ("index.html", "login.html", "dashboard.html", "recommendation.html", "roadmap.html", "results.html", "graphs/knowledge_graph.html",
                  "js/app.js", "css/style.css", "js/vendor/d3.min.js", "js/vendor/chart.umd.min.js", "js/vendor/pdf.min.js", "js/vendor/mammoth.browser.min.js"):
            self.assertTrue((ROOT / f).exists(), f)
    def test_rank_models(self):
        sk = ["Python", "Machine Learning", "Deep Learning", "Statistics", "SQL"]
        for m in ("hgt", "node2vec", "overlap"):
            r = self.eng.rank(sk, m, 5); self.assertEqual(len(r), 5)
            self.assertIn("AI & Machine Learning", {x["domain"] for x in r}, m)
    def test_security_profile(self):
        top = [x["domain"] for x in self.eng.rank(["Network Security", "Penetration Testing", "Incident Response", "Vulnerability Management", "Splunk"], "hgt", 3)]
        self.assertIn("Cybersecurity", top)
    def test_explain(self):
        e = self.eng.explain(["Python", "SQL", "Statistics"], "15-2051.00")
        self.assertTrue(e["matched"]); self.assertTrue(e["missing"]); self.assertTrue(e["paths"])
    def test_extract(self):
        s = self.eng.extract((ROOT / "samples" / "resume_web_developer.txt").read_text(encoding="utf-8"))
        for k in ("JavaScript", "React", "Git"): self.assertIn(k, s)
    def test_api(self):
        self.assertTrue(self.c.get("/api/health").get_json()["ok"])
        self.assertEqual(len(self.c.post("/api/recommend?top=3", json={"skills": ["Figma", "Prototyping", "Wireframing"], "model": "hgt"}).get_json()), 3)
        self.assertEqual(self.c.post("/api/recommend", json={"skills": [], "model": "bad"}).status_code, 400)
        self.assertEqual(self.c.post("/api/explain", json={"skills": [], "occupation": "nope"}).status_code, 404)
        self.assertIn("summary", self.c.get("/api/metrics").get_json())
        for p in ("/", "/results.html", "/data/kg_web.json"): self.assertEqual(self.c.get(p).status_code, 200, p)
        self.assertEqual(self.c.get("/pipeline/src/kg_build.py").status_code, 404)
    def test_resume_upload(self):
        for ext in ("pdf", "docx", "txt"):
            f = ROOT / "samples" / f"resume_data_analyst.{ext}"
            r = self.c.post("/api/resume", data={"file": (io.BytesIO(f.read_bytes()), f.name)}, content_type="multipart/form-data")
            self.assertEqual(r.status_code, 200, ext); self.assertIn("SQL", r.get_json()["skills"], ext)

if __name__ == "__main__": unittest.main(verbosity=2)
