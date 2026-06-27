import json,re,asyncio,httpx
HERE="/data1/jiahui/MedTVT-R1_repo/ceiling"
keys=json.load(open(HERE+"/or_keys.json"));KC=[0];lock=asyncio.Lock()
rows=[json.loads(l) for l in open("cohort_cot_final.jsonl")]
# 单标签, 分层 sup/partial 各50
pool={'supported':[],'partial':[]}
for r in rows:
    kv=r['kept_verdicts'];kv=eval(kv) if isinstance(kv,str) else kv
    if len(kv)!=1:continue
    vd=kv[0].get('verdict')
    if vd in pool and len(pool[vd])<2000:
        a=r['messages'][-1]['content']
        th=re.search(r"<think>(.*?)</think>",a,re.S)
        if th:pool[vd].append((th.group(1).strip(),kv[0]['dx']))
sample=[]
for vd in ['supported','partial']:
    step=max(1,len(pool[vd])//50)
    for th,dx in pool[vd][::step][:50]:sample.append((th,dx,vd))
print("sampled",len(sample),flush=True)
P='''A clinician wrote the reasoning below to support a diagnosis. Based ONLY on the reasoning, how strongly does it ground the diagnosis in admission-time evidence?
- "supported": the reasoning cites a specific direct finding (lab/imaging/ECG) confirming it.
- "partial": the reasoning relies on indirect cues (prior history, medications, context) only.
- "unsupported": the reasoning provides no real evidence.
REASONING: {T}
DIAGNOSIS: {D}
Return ONLY JSON: {"verdict": "supported"|"partial"|"unsupported"}'''
out=open("cot_agree.jsonl","w")
async def one(cl,sem,th,dx,gold):
    p=P.replace("{T}",th[:1500]).replace("{D}",dx)
    async with sem:
        for _ in range(5):
            k=keys[KC[0]%len(keys)];KC[0]+=1
            try:
                r=await cl.post("https://openrouter.ai/api/v1/chat/completions",headers={"Authorization":f"Bearer {k}"},
                    json={"model":"openrouter/owl-alpha","messages":[{"role":"user","content":p}],"max_tokens":30,"temperature":0},timeout=60)
                if r.status_code!=200:await asyncio.sleep(2);continue
                m=re.search(r'"verdict"\s*:\s*"(\w+)"',r.json()["choices"][0]["message"].get("content") or "")
                if m:
                    async with lock:out.write(json.dumps({"gold":gold,"cot_rate":m.group(1).lower(),"dx":dx})+"\n");out.flush()
                    return
            except:await asyncio.sleep(2)
async def main():
    sem=asyncio.Semaphore(50)
    async with httpx.AsyncClient() as cl:
        await asyncio.gather(*[one(cl,sem,th,dx,vd) for th,dx,vd in sample])
    print("COTAGREE_DONE",flush=True)
asyncio.run(main())
