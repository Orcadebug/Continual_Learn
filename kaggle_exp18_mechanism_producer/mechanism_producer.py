"""Exp18-Reconstructed engineering smoke producer; not a learning study."""
import hashlib, json, os, random, subprocess, sys, time
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "laya==0.4.6"], check=True)
from laya.train import collate_items, encode_item, items_from_rows, load_checkpoint

OUT, DEVICE, SEED = Path("/kaggle/working"), torch.device("cuda" if torch.cuda.is_available() else "cpu"), 1820
torch.manual_seed(SEED); random.seed(SEED)
if DEVICE.type == "cuda": torch.cuda.manual_seed_all(SEED)
torch.use_deterministic_algorithms(True, warn_only=False)

def find(name):
    hits=list(Path("/kaggle/input").rglob(name))
    if len(hits)!=1: raise RuntimeError(f"expected exactly one {name}, got {hits}")
    return hits[0]
def sha_file(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for x in iter(lambda:f.read(1<<20),b""): h.update(x)
    return h.hexdigest()
def digest(model, head):
    h=hashlib.sha256()
    for prefix,module in (("m",model),("h",head)):
        for n,t in module.state_dict().items(): h.update(prefix.encode()+n.encode()+t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
def items(tok, rows):
    q={"act":{"type":"choice","instructions":"Choose.","criteria":{"north":"north","south":"south"}}}
    source=[{"state":text,"questions":q,"expected":{"act":"north"}} for text in rows]
    parsed, skipped=items_from_rows(tok,source,128,64)
    assert not skipped
    return [encode_item(tok,x,128,64) for x in parsed]
def forward(model,head,tok,examples):
    b=collate_items([[x] for x in examples],tok.pad_token_id)
    ids,att,mp,mm,qt=(b[k].to(DEVICE) for k in ("input_ids","attention_mask","marker_pos","marker_mask","qtype"))
    hidden=model.encoder(input_ids=ids,attention_mask=att).last_hidden_state+model.type_emb(qt)[:,None,:]
    if model.head is not None:
        for layer in model.head.layers: hidden=layer(hidden,src_key_padding_mask=~att.bool())
    chosen=torch.gather(hidden,1,mp.clamp(min=0)[:,:,None].expand(-1,-1,hidden.size(-1)))
    return model.scorer(chosen).squeeze(-1).float().masked_fill(~mm,-1e4),head(hidden[:,0].float()).squeeze(-1)
def loss(model,head,tok,examples,targets):
    logits,value=forward(model,head,tok,examples)
    return F.cross_entropy(logits,torch.tensor(targets,device=DEVICE))+0.01*value.square().mean()
def make_model(state):
    model,tok,_=load_checkpoint(state["training_config"]["model_repo"]); model=model.to(DEVICE)
    head=nn.Sequential(nn.LayerNorm(model.encoder.config.hidden_size),nn.Linear(model.encoder.config.hidden_size,1)).to(DEVICE)
    model.load_state_dict(state["model"]); head.load_state_dict(state["value_head"])
    return model,head,tok
def optimizer(model,head,mult):
    return torch.optim.AdamW([{"params":model.parameters(),"lr":1e-5*mult},{"params":head.parameters(),"lr":1e-3*mult}],weight_decay=0.0)
def update(model,head,opt,tok,new,new_y,ref,ref_y,valve):
    params=list(model.parameters())+list(head.parameters()); opt.zero_grad(set_to_none=True)
    ref_before=loss(model,head,tok,ref,ref_y); raw=torch.autograd.grad(ref_before,params,allow_unused=True)
    rg=[None if x is None else x.detach().float().cpu() for x in raw]; del raw
    proposed=loss(model,head,tok,new,new_y); proposed.backward()
    dot=sum(float((p.grad.detach().float()*g.to(DEVICE)).sum()) for p,g in zip(params,rg) if p.grad is not None and g is not None)
    ns=sum(float((p.grad.detach().float()**2).sum()) for p in params if p.grad is not None); rs=sum(float((g**2).sum()) for g in rg if g is not None)
    active=bool(valve and dot<0 and rs>1e-12); change=0.0
    if active:
        scale=dot/(rs+1e-12)
        for p,g in zip(params,rg):
            if p.grad is not None and g is not None:
                before=p.grad.detach().float().clone(); p.grad.sub_(g.to(DEVICE,dtype=p.grad.dtype),alpha=scale); change+=float(((p.grad.detach().float()-before)**2).sum())
    torch.nn.utils.clip_grad_norm_(params,1.0)
    before=[p.detach().cpu().clone() for p in params]; opt.step()
    delta_sq=ref_dot_delta=0.0
    for p,old,g in zip(params,before,rg):
        d=p.detach().cpu()-old; delta_sq+=float((d*d).sum())
        if g is not None: ref_dot_delta+=float((g*d).sum())
    ref_after=float(loss(model,head,tok,ref,ref_y).detach())
    return {"loss":float(proposed.detach()),"reference_loss_before":float(ref_before.detach()),"reference_loss_after":ref_after,"gradient_cosine":dot/max((ns*rs)**.5,1e-12),"conflict_detected":dot<0,"projection_active":active,"projection_magnitude":change**.5,"parameter_delta_norm":delta_sq**.5,"reference_dot_parameter_delta":ref_dot_delta}

source=find("gate3_checkpoint.pt"); manifest=find("manifest.json"); initial=torch.load(source,map_location="cpu",weights_only=False)
if sha_file(source)!=json.loads(manifest.read_text())["files"]["gate3_checkpoint.pt"]: raise RuntimeError("recovered checkpoint hash mismatch")
new_ids=["stream-001","stream-002"]; ref_ids=["retained-001","retained-002"]
assert not set(new_ids)&set(ref_ids)
rows_new=["An agent is north of its goal.","An agent is south of its goal."]; rows_ref=["A retained route begins north.","A retained route begins south."]
arms={}; valve_checkpoint=None; start=time.time()
for arm,mult,valve in (("normal",1.0,False),("fixed_lr",0.30,False),("valve",1.0,True)):
    model,head,tok=make_model(initial); new,ref=items(tok,rows_new),items(tok,rows_ref); opt=optimizer(model,head,mult)
    result=update(model,head,opt,tok,new,[0,1],ref,[1,0],valve); result["learning_rate_multiplier"]=mult; result["update_ids"]=new_ids; result["reference_ids"]=ref_ids; result["exact_id_overlap"]=0
    arms[arm]=result
    if arm=="valve":
        # Full post-update state plus an uninterrupted predetermined next update reference.
        saved={"schema":1,"study":"Exp18-Reconstructed mechanism-only","arm":arm,"position":{"optimizer_step":1},"model":model.state_dict(),"value_head":head.state_dict(),"optimizer":opt.state_dict(),"memory":[{"experience_id":x} for x in new_ids+ref_ids],"controller":{"kind":"agem_projection"},"training_config":initial["training_config"],"rng":{"python":random.getstate(),"torch":torch.get_rng_state(),"cuda":torch.cuda.get_rng_state_all() if DEVICE.type=="cuda" else None},"batch_ids":{"new":new_ids,"ref":ref_ids}}
        tmp=OUT/"exp18_mechanism_checkpoint.tmp"; final=OUT/"exp18_mechanism_checkpoint.pt"; torch.save(saved,tmp); os.replace(tmp,final)
        second=update(model,head,opt,tok,new,[1,0],ref,[0,1],True)
        reference={"checkpoint_sha256":sha_file(final),"next_loss":second["loss"],"next_parameter_sha256":digest(model,head)}
        (OUT/"exp18_mechanism_reference.json").write_text(json.dumps(reference,indent=2))
    del model,head,opt
# Frozen evaluation is a pure forward pass and must mutate neither state nor optimizer.
model,head,tok=make_model(initial); opt=optimizer(model,head,1.0); pre=digest(model,head); _=forward(model,head,tok,items(tok,["Frozen evaluation."])); frozen_ok=pre==digest(model,head) and len(opt.state)==0
result={"passed":True,"wall_seconds":time.time()-start,"device":str(DEVICE),"source_checkpoint_sha256":sha_file(source),"arms":arms,"frozen_evaluation_unchanged":frozen_ok,"checkpoint":"exp18_mechanism_checkpoint.pt"}
(OUT/"exp18_mechanism_result.json").write_text(json.dumps(result,indent=2)); (OUT/"exp18_mechanism_manifest.json").write_text(json.dumps({n:sha_file(OUT/n) for n in ["exp18_mechanism_checkpoint.pt","exp18_mechanism_reference.json","exp18_mechanism_result.json"]},indent=2))
print(json.dumps(result,indent=2),flush=True)
