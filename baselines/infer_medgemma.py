import os,sys,json,re,torch
os.environ["HF_TOKEN"]="REPLACE_WITH_YOUR_HF_TOKEN"
from transformers import AutoProcessor, AutoModelForImageTextToText
SHARD=int(sys.argv[1]); NSH=int(sys.argv[2]); BS=int(sys.argv[3]) if len(sys.argv)>3 else 6
M="google/medgemma-4b-it"
OUT=f"/data1/jiahui/MedTVT-R1_repo/ceiling/medgemma_pred_shard{SHARD}.jsonl"
FMT="\n\nFirst give a BRIEF reasoning inside <think>...</think> (a few sentences), then the final diagnoses inside <answer>diagnosis1; diagnosis2</answer>. Always close every tag."
proc=AutoProcessor.from_pretrained(M); proc.tokenizer.padding_side="left"
model=AutoModelForImageTextToText.from_pretrained(M,dtype=torch.bfloat16,device_map="cuda").eval()
rows=[json.loads(l) for l in open("/data1/jiahui/MedTVT-R1_repo/ceiling/sft_test.jsonl")]
rows=[(i,r) for i,r in enumerate(rows) if i%NSH==SHARD]
done=set()
if os.path.exists(OUT):
    for l in open(OUT):
        try:done.add(json.loads(l)["idx"])
        except:pass
fout=open(OUT,"a")
def gold_of(m):return [x.strip() for x in m[-1]["content"].split("<answer>")[-1].replace("</answer>","").split(";") if x.strip()]
def flush(batch):
    msgs=[]
    for _,r in batch:
        inp=r["messages"][0]["content"].split("\n\nBased on")[0]
        msgs.append([{"role":"user","content":[{"type":"text","text":inp+FMT}]}])
    enc=proc.apply_chat_template(msgs,add_generation_prompt=True,tokenize=True,padding=True,return_tensors="pt",return_dict=True).to("cuda")
    with torch.no_grad():
        o=model.generate(**enc,max_new_tokens=2500,do_sample=False)
    gen=o[:,enc["input_ids"].shape[1]:]
    for (idx,r),row in zip(batch,gen):
        g=proc.decode(row,skip_special_tokens=True)
        a=re.search(r"<answer>(.*?)</answer>",g,re.S)
        pred=[x.strip() for x in (a.group(1).split(";") if a else []) if x.strip()]
        th=re.search(r"<think>(.*?)</think>",g,re.S)
        fout.write(json.dumps({"idx":idx,"input":r["messages"][0]["content"],"gold":gold_of(r["messages"]),
            "pred":pred,"think":(th.group(1).strip() if th else ""),"gen":g,"fmt":bool(a)},ensure_ascii=False)+"\n")
    fout.flush()
batch=[];n=0
for idx,r in rows:
    if idx in done:continue
    batch.append((idx,r))
    if len(batch)>=BS:flush(batch);n+=len(batch);batch=[];print(f"mg-shard{SHARD}: {n}",flush=True)
if batch:flush(batch);n+=len(batch)
fout.close();print(f"mg-shard{SHARD} DONE {n}",flush=True)
