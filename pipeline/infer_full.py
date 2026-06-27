import os,sys,json,torch,re
os.environ.setdefault("HF_HOME","/data1/jiahui/hf_cache")
from transformers import AutoTokenizer,AutoModelForImageTextToText
SHARD=int(sys.argv[1]); NSH=int(sys.argv[2])
CKPT="/data1/jiahui/MedTVT-R1_repo/ceiling/qwen35_2b_cotsft"
OUT=f"/data1/jiahui/MedTVT-R1_repo/ceiling/qwen_pred_shard{SHARD}.jsonl"
BS=16
tok=AutoTokenizer.from_pretrained("Qwen/Qwen3.5-2B")
tok.padding_side="left"
if tok.pad_token is None: tok.pad_token=tok.eos_token
model=AutoModelForImageTextToText.from_pretrained(CKPT,dtype=torch.bfloat16,device_map="cuda").eval()
rows=[json.loads(l) for l in open("/data1/jiahui/MedTVT-R1_repo/ceiling/sft_test.jsonl")]
rows=[r for i,r in enumerate(rows) if i%NSH==SHARD]
done=set()
if os.path.exists(OUT):
    for l in open(OUT):
        try:done.add(json.loads(l)["idx"])
        except:pass
fout=open(OUT,"a")
def gold_of(m):return [x.strip() for x in m[-1]["content"].split("<answer>")[-1].replace("</answer>","").split(";") if x.strip()]
batch=[]
def flush(batch):
    prompts=[tok.apply_chat_template(b[1]["messages"][:-1],tokenize=False,add_generation_prompt=False)+"<|im_start|>assistant\n" for b in batch]
    enc=tok(prompts,return_tensors="pt",add_special_tokens=False,padding=True,truncation=True,max_length=3000).to("cuda")
    with torch.no_grad():
        o=model.generate(**enc,max_new_tokens=2048,do_sample=False)
    gen=o[:,enc.input_ids.shape[1]:]
    for (idx,r),row in zip(batch,gen):
        g=tok.decode(row,skip_special_tokens=True)
        a=re.search(r"<answer>(.*?)</answer>",g,re.S)
        pred=[x.strip() for x in (a.group(1).split(";") if a else []) if x.strip()]
        th=re.search(r"<think>(.*?)</think>",g,re.S)
        fout.write(json.dumps({"idx":idx,"input":r["messages"][0]["content"],"gold":gold_of(r["messages"]),
            "pred":pred,"think":(th.group(1).strip() if th else ""),"gen":g,"fmt":bool(a and "<think>" in g)},ensure_ascii=False)+"\n")
    fout.flush()
n=0
for i,r in enumerate(rows):
    if i in done:continue
    batch.append((i,r))
    if len(batch)>=BS:flush(batch);n+=len(batch);batch=[];print(f"shard{SHARD}: {n}",flush=True)
if batch:flush(batch);n+=len(batch)
fout.close();print(f"shard{SHARD} DONE {n}",flush=True)
