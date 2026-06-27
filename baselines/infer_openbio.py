import os,sys,json,re,torch
os.environ.setdefault("HF_HOME","/data1/jiahui/hf_cache")
from transformers import AutoTokenizer, AutoModelForCausalLM
SHARD=int(sys.argv[1]); NSH=int(sys.argv[2]); BS=int(sys.argv[3]) if len(sys.argv)>3 else 16
M="aaditya/Llama3-OpenBioLLM-8B"
OUT=f"/data1/jiahui/MedTVT-R1_repo/ceiling/openbio_pred_shard{SHARD}.jsonl"
SYS="You are an expert emergency physician."
FMT="\n\nList this patient's diagnoses as a concise semicolon-separated list of diagnosis names only. Do not write sentences or explanations."
PRE="<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"+SYS+"<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
POST="<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nDiagnoses: "
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
    g=re.sub(r'^\s*diagnoses[:;]?\s*','',g.strip(),flags=re.I)
    out=[]
    for x in re.split(r'[;\n]',g):
        x=re.sub(r'^\s*\d+[\.\)]\s*','',x).strip(' .*-')
        if x and len(x)<=80 and not x.lower().startswith(('the ','this ','no ','please','note')):out.append(x)
    return out[:8]
def flush(batch):
    prompts=[PRE+r["messages"][0]["content"].split("\n\nBased on")[0]+FMT+POST for _,r in batch]
    enc=tok(prompts,return_tensors="pt",padding=True,truncation=True,max_length=3500,add_special_tokens=False).to("cuda")
    with torch.no_grad():
        o=model.generate(**enc,max_new_tokens=256,do_sample=False)
    gen=o[:,enc.input_ids.shape[1]:]
    for (idx,r),row in zip(batch,gen):
        g=tok.decode(row,skip_special_tokens=True)
        pred=parse(g)
        fout.write(json.dumps({"idx":idx,"input":r["messages"][0]["content"],"gold":gold_of(r["messages"]),
            "pred":pred,"think":"","gen":g,"fmt":bool(pred)},ensure_ascii=False)+"\n")
    fout.flush()
batch=[];n=0
for idx,r in rows:
    if idx in done:continue
    batch.append((idx,r))
    if len(batch)>=BS:flush(batch);n+=len(batch);batch=[];print(f"ob-shard{SHARD}: {n}",flush=True)
if batch:flush(batch);n+=len(batch)
fout.close();print(f"ob-shard{SHARD} DONE {n}",flush=True)
