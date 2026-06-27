import os,sys,json,re,torch
os.environ.setdefault("HF_HOME","/data1/jiahui/hf_cache")
from transformers import AutoTokenizer, AutoModelForCausalLM
SHARD=int(sys.argv[1]); NSH=int(sys.argv[2]); BS=int(sys.argv[3]) if len(sys.argv)>3 else 8
M="FreedomIntelligence/HuatuoGPT-o1-8B"
OUT=f"/data1/jiahui/MedTVT-R1_repo/ceiling/huatuo_pred_shard{SHARD}.jsonl"
FMT="\n\nBased on this admission presentation, what are the patient's diagnoses? End your final response with one line: DIAGNOSES: <name1>; <name2>; ..."
tok=AutoTokenizer.from_pretrained(M); tok.padding_side="left"
if tok.pad_token is None: tok.pad_token=tok.eos_token
model=AutoModelForCausalLM.from_pretrained(M,dtype=torch.bfloat16,device_map="cuda").eval()
rows=[json.loads(l) for l in open("/data1/jiahui/MedTVT-R1_repo/ceiling/sft_test.jsonl")]
rows=[(i,r) for i,r in enumerate(rows) if i%NSH==SHARD]
done=set()
if os.path.exists(OUT):
    for l in open(OUT):
        try:done.add(json.loads(l)["idx"])
        except:pass
fout=open(OUT,"a")
def gold_of(m):return [x.strip() for x in m[-1]["content"].split("<answer>")[-1].replace("</answer>","").split(";") if x.strip()]
def parse(g):
    m=re.search(r'DIAGNOSES:\s*(.+)',g,re.S)
    seg=m.group(1) if m else (g.split("## Final Response")[-1] if "## Final Response" in g else "")
    seg=seg.split("\n\n")[0]
    out=[]
    for x in re.split(r'[;\n]',seg):
        x=re.sub(r'^\s*\d+[\.\)]\s*','',x).strip(' .*-')
        if x and len(x)<=80 and not x.lower().startswith(('the ','this ','based','please','note','possible diagnos')):out.append(x)
    return out[:8]
def flush(batch):
    prompts=[]
    for _,r in batch:
        inp=r["messages"][0]["content"].split("\n\nBased on")[0]
        prompts.append(tok.apply_chat_template([{"role":"user","content":inp+FMT}],tokenize=False,add_generation_prompt=True))
    enc=tok(prompts,return_tensors="pt",padding=True,truncation=True,max_length=3500,add_special_tokens=False).to("cuda")
    with torch.no_grad():
        o=model.generate(**enc,max_new_tokens=2048,do_sample=False)
    gen=o[:,enc.input_ids.shape[1]:]
    for (idx,r),row in zip(batch,gen):
        g=tok.decode(row,skip_special_tokens=True)
        pred=parse(g)
        th=g.split("## Final Response")[0].replace("## Thinking","").strip()[:1500]
        fout.write(json.dumps({"idx":idx,"input":r["messages"][0]["content"],"gold":gold_of(r["messages"]),
            "pred":pred,"think":th,"gen":g,"fmt":bool(pred)},ensure_ascii=False)+"\n")
    fout.flush()
batch=[];n=0
for idx,r in rows:
    if idx in done:continue
    batch.append((idx,r))
    if len(batch)>=BS:flush(batch);n+=len(batch);batch=[];print(f"ht-shard{SHARD}: {n}",flush=True)
if batch:flush(batch);n+=len(batch)
fout.close();print(f"ht-shard{SHARD} DONE {n}",flush=True)
