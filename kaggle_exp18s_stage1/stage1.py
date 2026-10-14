"""Exp18-S Stage 1: bounded-buffer pressure and real-Laya update-rate probes."""
import json, subprocess, sys, time
from collections import deque
from pathlib import Path
import torch
import torch.nn.functional as F
subprocess.run([sys.executable,"-m","pip","install","-q","laya==0.4.6"],check=True)
from laya.train import collate_items, encode_item, items_from_rows, load_checkpoint

OUT=Path("/kaggle/working"); device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
model,tok,_=load_checkpoint("convaiinnovations/laya"); model=model.to(device); model.train()
q={"act":{"type":"choice","instructions":"Choose.","criteria":{"north":"north","south":"south"}}}
rows=[{"state":f"Recorded pressure stream observation {i}.","questions":q,"expected":{"act":"north"}} for i in range(64)]
parsed,skipped=items_from_rows(tok,rows,128,64); assert not skipped
enc=[encode_item(tok,x,128,64) for x in parsed]
opt=torch.optim.AdamW(model.parameters(),lr=1e-5,weight_decay=0.0)
def step(batch):
    b=collate_items([[x] for x in batch],tok.pad_token_id); ids,att,mp,mm,qt=(b[k].to(device) for k in ("input_ids","attention_mask","marker_pos","marker_mask","qtype"))
    opt.zero_grad(set_to_none=True); h=model.encoder(input_ids=ids,attention_mask=att).last_hidden_state+model.type_emb(qt)[:,None,:]
    if model.head:
        for layer in model.head.layers: h=layer(h,src_key_padding_mask=~att.bool())
    logits=model.scorer(torch.gather(h,1,mp.clamp(min=0)[:,:,None].expand(-1,-1,h.size(-1)))).squeeze(-1).float().masked_fill(~mm,-1e4)
    loss=F.cross_entropy(logits,torch.tensor([i%2 for i in range(len(batch))],device=device)); loss.backward(); opt.step()
    if device.type=="cuda": torch.cuda.synchronize()
    return float(loss.detach())
# Actual learner consumption rate; stream records remain fixed and carry stable IDs.
t=time.perf_counter(); losses=[step(enc[:8]) for _ in range(2)]; consume_eps=16/(time.perf_counter()-t)
def pressure(rate,capacity):
    queue=deque(); produced=processed=evicted=max_backlog=0; first_update={}; ages=[]; now=0
    # Virtual seconds: fixed stream, deterministic arrival timestamps, bounded queue.
    for i in range(512):
        now=i/rate; queue.append((i,now)); produced+=1
        while len(queue)>capacity: queue.popleft(); evicted+=1
        budget=int((i+1)*consume_eps/rate)-processed
        for _ in range(max(0,budget)):
            if not queue: break
            ident,born=queue.popleft(); processed+=1; first_update.setdefault(ident,now); ages.append(now-born)
        max_backlog=max(max_backlog,len(queue))
    return {"rate_multiplier":rate/consume_eps,"capacity":capacity,"produced":produced,"processed":processed,"evicted":evicted,"remaining":len(queue),"max_backlog":max_backlog,"mean_first_update_delay_s":sum(ages)/len(ages) if ages else None}
pressure_results=[pressure(consume_eps*m,cap) for m in (1,4,16) for cap in (64,4096)]
# Fixed-data reuse only: ratios are update counts per 8 unique experiences, no held-out learning claim.
reuse=[]
for ratio in (1,4,16):
    start=time.perf_counter(); values=[step(enc[:8]) for _ in range(ratio)]; reuse.append({"updates_per_8_unique_experiences":ratio,"updates":ratio,"loss_first":values[0],"loss_last":values[-1],"wall_seconds":time.perf_counter()-start})
result={"study":"Exp18-S Stage1","device":str(device),"learner_consumption_experiences_per_second":consume_eps,"warmup_losses":losses,"pressure":pressure_results,"fixed_dataset_reuse":reuse,"interpretation":"engineering throughput only; no behavioral learning or valve claim"}
(OUT/"exp18s_stage1_result.json").write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2),flush=True)
