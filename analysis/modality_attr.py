import json,re,asyncio,httpx,os
HERE="/data1/jiahui/MedTVT-R1_repo/ceiling"
keys=json.load(open(HERE+"/or_keys.json"))
KC=[0];lock=asyncio.Lock()
CATS=["Laboratory","Imaging (CT/X-ray/US)","ECG","Echocardiography",
      "Prior history/records","Medications","Vital signs","Symptoms/chief complaint","Physical exam"]
rows=[json.loads(l) for l in open("cohort_cot_final.jsonl")]
# 均匀抽样2000
step=max(1,len(rows)//2000)
sample=rows[::step][:2000]
P='''Below is a clinician's reasoning that leads to a diagnosis. Identify the PRIMARY type of evidence the reasoning relies on to reach the diagnosis. Choose EXACTLY ONE from:
Laboratory; Imaging (CT/X-ray/US); ECG; Echocardiography; Prior history/records; Medications; Vital signs; Symptoms/chief complaint; Physical exam.
REASONING: {T}
Return ONLY JSON: {"primary": "<one category exactly as listed>"}'''
out=open("modality_attr.jsonl","a")
done=0
if os.path.exists("modality_attr.jsonl"):
    done=sum(1 for _ in open("modality_attr.jsonl"))
async def one(cl,sem,i,r):
    a=r["messages"][-1]["content"]
    th=re.search(r"<think>(.*?)</think>",a,re.S)
    if not th:return
    t=th.group(1).strip()[:1500]
    p=P.replace("{T}",json.dumps(t))
    async with sem:
        for _ in range(5):
            k=keys[KC[0]%len(keys)];KC[0]+=1
            try:
                resp=await cl.post("https://openrouter.ai/api/v1/chat/completions",headers={"Authorization":f"Bearer {k}"},
                    json={"model":"openrouter/owl-alpha","messages":[{"role":"user","content":p}],"max_tokens":40,"temperature":0},timeout=60)
                if resp.status_code!=200:await asyncio.sleep(2);continue
                c=resp.json()["choices"][0]["message"].get("content") or ""
                m=re.search(r'"primary"\s*:\s*"([^"]+)"',c)
                if m:
                    async with lock:out.write(json.dumps({"i":i,"primary":m.group(1)})+"\n");out.flush()
                    return
            except:await asyncio.sleep(2)
async def main():
    sem=asyncio.Semaphore(100)
    todo=list(enumerate(sample))[done:]
    print(f"sample={len(sample)}, done={done}, todo={len(todo)}",flush=True)
    async with httpx.AsyncClient() as cl:
        await asyncio.gather(*[one(cl,sem,i,r) for i,r in todo])
    print("ATTR_DONE",flush=True)
asyncio.run(main())
