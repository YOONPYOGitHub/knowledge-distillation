"""Offline aggregate/checksum verification only; never load models/corpus."""
import hashlib,json,math,pathlib,re
ROOT=pathlib.Path(__file__).resolve().parent

def load(name):
 return json.loads((ROOT/name).read_text(encoding="utf-8"))
def close(a,b):
 assert math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-12),(a,b)
def main():
 manifest=load("MANIFEST.json")
 for name,m in manifest["files"].items():
  p=ROOT/name
  assert p.is_relative_to(ROOT) and not p.is_symlink()
  raw=p.read_bytes()
  assert len(raw)==m["bytes"] and hashlib.sha256(raw).hexdigest()==m["sha256"],name
  if name.endswith(".py"):compile(raw,str(p),"exec")
  if name.endswith(".json"):json.loads(raw)
  text=raw.decode()
  for pattern in [r"/Users/",r"/subscriptions/",r"AccountKey=",r"-----BEGIN [A-Z ]*PRIVATE KEY",r"[?&]sig=",r"https?://[^\s]+blob\.core\.windows\.net",r"\bvm-a100-00[12]\b",r"stkda100"]:
   # This verifier itself contains the deny-pattern literals, not secrets.
   if name != "verify_projection.py": assert not re.search(pattern,text,re.I),(name,pattern)
 trials=[load(str(p.relative_to(ROOT))) for p in sorted((ROOT/"trials").glob("*.json"))]
 gens=[t for t in trials if "status_receipt" in t]
 assert len(trials)==8 and len(gens)==6
 assert sum(t["status_receipt"]["fields"]["/generation_calls"] for t in gens)==1124
 checked_gates=0
 for t in gens:
  q=t["quality_result"]
  if q is None:
   assert t["status_receipt"]["fields"]["/actual_metrics"] is None
   assert t["status_receipt"]["fields"]["/teacher_advantage_gate"] is None
   continue
  fields=q["fields"];s=fields["/summary"];teacher=s["teacher"];student=s["student"]
  c=t["configurations"]["phase2a/contract.json"]["fields"]["/gate"]
  lower=fields["/cluster_uncertainty"]["answerable_f1_delta_ci95"][0]
  gate=(teacher["answerable_f1"]-student["answerable_f1"]>=c["answerable_f1_delta_min"] and lower>c["answerable_f1_cluster_ci_lower_exclusive"] and teacher["exact"]-student["exact"]>=c["overall_exact_delta_min"] and teacher["answerability"]-student["answerability"]>=c["answerability_delta_min"] and teacher["impossible_false_answer_rate"]-student["impossible_false_answer_rate"]<=c["impossible_false_answer_delta_max"])
  assert gate==fields["/teacher_advantage_gate"]==False
  for role,row in t["aggregate_counts"].items():
   assert row["n"]==124 and row["answerable_n"]==93 and row["impossible_n"]==31
   close(row["exact_count"]/row["n"],s[role]["exact"])
   close(row["false_answer_count"]/31,s[role]["impossible_false_answer_rate"])
  checked_gates+=1
 assert checked_gates==3
 a=load("training/student-A.json");h=a["history"]
 assert [r["step"] for r in h]==list(range(0,1001,100))
 best=min(h,key=lambda r:r["response_ce"]);assert best["step"]==900
 teacher=load("training/teacher-FT.json")
 assert [r["step"] for r in teacher["history"]]==list(range(100,601,100))
 assert min(teacher["history"],key=lambda r:r["response_ce"])["step"]==500
 assert a["final_completion"]==teacher["final_completion"]==False
 assert load("training/unexecuted-B-C.json")["executed"]=={"B":0,"C":0}
 return {"status":"PASS","manifest_files_verified":len(manifest["files"]),"trial_documents":len(trials),"generation_trials":len(gens),"generation_calls":1124,"completed_gate_recalculations":checked_gates,"student_history_points":len(h),"teacher_history_points":len(teacher["history"]),"student_CE_best_relative_reduction_pct":100*(1-best["response_ce"]/h[0]["response_ce"]),"student_last_vs_best_worsening_pct":100*(h[-1]["response_ce"]/best["response_ce"]-1),"no_model_or_corpus_access":True,"scope":"checksums, syntax, aggregate arithmetic only; no QA rescoring or bootstrap"}
if __name__=="__main__":print(json.dumps(main(),ensure_ascii=False,indent=2))
