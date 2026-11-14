"""Frozen dev124 evaluator candidate; no model loading or execution launcher."""
import pathlib,json,hashlib,os
BASE=pathlib.Path(__file__).resolve().parent
ROOT=BASE.parents[1]
SEAM_SHA='f70f3170c1e8a2ab1c14525520fd40a67d8eadc536951656ba4e413678f71688'
CONTRACT_SHA='a52eaa48317703d8f633ef3a3cae710d474bb8502140c0d047c8a270bddeb0f7'
def contract():
 raw=(BASE/'FROZEN_DEV_CONTRACT.json').read_bytes()
 if hashlib.sha256(raw).hexdigest()!=CONTRACT_SHA: raise ValueError('contract SHA mismatch')
 return json.loads(raw)
def validate_members(rows):
 c=contract(); fields=('guid','cluster_id','document_id_proxy','normalized_context_sha256','original_row_index')
 if len(rows)!=124 or [{k:r[k] for k in fields} for r in rows]!=c['members'] or len({r['guid'] for r in rows})!=124 or len({r['cluster_id'] for r in rows})!=c['clusters']: raise ValueError('complete ordered dev membership required')
 for r in rows:
  if r.get('split')!='dev' or any(r.get('provenance',{}).get(k)!=v for k,v in {'repo':'klue/klue','config':'mrc','upstream_split':'train','revision':c['revision']}.items()): raise ValueError('dev provenance mismatch')
def load_dev(path=None):
 c=contract(); allowed=pathlib.Path(c['dev']['path']); path=allowed if path is None else pathlib.Path(path)
 if path!=allowed or path.resolve()!=allowed: raise ValueError('exact dev source path required')
 pub=pathlib.Path(c['publication_path']).read_bytes()
 if hashlib.sha256(pub).hexdigest()!=c['publication_sha256'] or json.loads(pub)['files']['dev']!=c['dev']: raise ValueError('publication mismatch')
 fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW)
 with os.fdopen(fd,'rb') as f: data=f.read()
 if len(data)!=c['dev']['bytes'] or hashlib.sha256(data).hexdigest()!=c['dev']['sha256']: raise ValueError('dev source SHA/size mismatch')
 rows=[json.loads(line) for line in data.splitlines()]; validate_members(rows)
 return rows

def valid_ce(value):
 import math
 if type(value) not in (int,float) or not math.isfinite(value) or value<0: raise ValueError('finite nonnegative CE required')
 return value

def response_sum(logits,targets):
 import torch
 import torch.nn.functional as F
 count=int((targets!=-100).sum().item())
 if count<=0 or logits.shape[:-1]!=targets.shape: raise ValueError('positive aligned response tokens required')
 loss=F.cross_entropy(logits.float().reshape(-1,logits.shape[-1]),targets.reshape(-1),ignore_index=-100,reduction='sum')
 return valid_ce(loss.item()),count

class Evaluator:
 def __init__(self,tokenizer,max_length=8192):
  import importlib.util
  if type(max_length) is not int or not 1<=max_length<=8192: raise ValueError('bounded length required')
  seam_path=ROOT/'phase2e/authorized_loader_20261103T0124/seam.py'
  if hashlib.sha256(seam_path.read_bytes()).hexdigest()!=SEAM_SHA: raise ValueError('accepted seam SHA mismatch')
  spec=importlib.util.spec_from_file_location('dev_accepted_seam',seam_path); seam=importlib.util.module_from_spec(spec); spec.loader.exec_module(seam)
  runner,generator=seam.dependencies()
  from transformers import AutoTokenizer
  cache=ROOT/'phase2e/integrated_adaptation_20261103T0100/tokenizer_cache/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a'
  official=AutoTokenizer.from_pretrained(cache,local_files_only=True,trust_remote_code=False)
  backend=tokenizer.backend_tokenizer.to_str()
  if backend!=official.backend_tokenizer.to_str() or tokenizer.chat_template!=official.chat_template or tokenizer.eos_token_id!=official.eos_token_id: raise ValueError('accepted tokenizer required')
  rows=load_dev(); self._items=[]; self._guids=tuple(r['guid'] for r in rows)
  for row in rows:
   # Reuse accepted gold schema validation without opening or admitting train payload.
   gold_row=dict(row,split='train'); _,gold=runner.train_messages_gold(gold_row)
   messages=[{'role':'user','content':generator.q.prompt(row)}]
   item=runner.encode_response(tokenizer,messages,gold,max_length)
   _,request=seam.eval_input(tokenizer,row)
   if item['input_ids'][:item['boundary']]!=request['input_ids'].tolist()[0]: raise ValueError('accepted train/eval prefix mismatch')
   self._items.append(item)
  c=contract()
  import torch,transformers,peft
  self.identity={'evaluator_source_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),'dev_contract_sha256':CONTRACT_SHA,'publication_sha256':c['publication_sha256'],'dev_path':c['dev']['path'],'dev_sha256':c['dev']['sha256'],'membership_sha256':hashlib.sha256(json.dumps(c['members'],sort_keys=True).encode()).hexdigest(),'accepted_seam_sha256':SEAM_SHA,'tokenizer_backend_sha256':hashlib.sha256(backend.encode()).hexdigest(),'max_length':max_length,'policy':'complete ordered dev124; no truncation; response+EOS; shifted FP32 CE sum / actual tokens; inference_mode; exact module mode restoration; response-position logits_to_keep','torch':torch.__version__,'transformers':transformers.__version__,'peft':peft.__version__}
 def __call__(self,model,step):
  import torch,math,copy
  modes={n:m.training for n,m in model.named_modules()}; per=[]
  try:
   model.eval()
   with torch.inference_mode():
    device=next(model.parameters()).device
    for guid,item in zip(self._guids,self._items,strict=True):
     ids=torch.tensor([item['input_ids']],device=device)
     boundary=item['boundary']; positions=torch.arange(boundary-1,len(item['input_ids'])-1,device=device)
     logits=model(input_ids=ids,logits_to_keep=positions,use_cache=False).logits
     targets=torch.tensor([item['labels'][boundary:]],device=device)
     loss,tokens=response_sum(logits,targets)
     per.append({'guid':guid,'tokens':tokens,'loss_sum':loss})
  finally:
   for name,module in model.named_modules(): module.training=modes[name]
  if len(per)!=124 or [r['guid'] for r in per]!=list(self._guids): raise ValueError('incomplete evaluation')
  tokens=sum(r['tokens'] for r in per); total=math.fsum(r['loss_sum'] for r in per)
  return {'split':'dev','step':step,'response_ce':valid_ce(total/tokens),'tokens':tokens,'rows':len(per),'guids':list(self._guids),'per_row':per,'identity':copy.deepcopy(self.identity)}
