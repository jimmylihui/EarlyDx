import json,re,asyncio,httpx,glob
HERE="/data1/jiahui/MedTVT-R1_repo/ceiling"
keys=json.load(open(HERE+"/or_keys.json"));KC=[0];lock=asyncio.Lock()
# 从带verdict的cohort抽 (input, dx, verdict)
pool={'supported':[],'partial':[],'unsupported':[]}
for f in glob.glob("cohort_*0000.json"):
    for r in json.load(open(f)):
        u=r['messages'][0]['content'] if isinstance(r['messages'],list) else None
        if not u:continue
        for v in r.get('label_verdicts',[]):
            vd=v.get('verdict','').lower()
            if vd in pool and len(pool[vd])<400:
                pool[vd].append((u,v['dx']))
# 分层各34/33/33
import itertools
sample=[]
for vd,n in [('supported',34),('partial',33),('unsupported',33)]:
    step=max(1,len(pool[vd])//n)
    for u,dx in pool[vd][::step][:n]:sample.append((u,dx,vd))
print("sampled",len(sample),{k:len(pool[k]) for k in pool},flush=True)
P='''You are a clinical evidence auditor. Given admission-time information and a diagnosis label, decide how well the available evidence supports the label:
- "supported": a specific finding in the input directly substantiates the diagnosis.
- "partial": only indirect/circumstantial evidence (e.g., prior history, a home medication).
- "unsupported": the diagnosis cannot be derived from the input.
INPUT:
{U}
DIAGNOSIS: {D}
Return ONLY JSON: {"verdict": "supported"|"partial"|"unsupported"}'''
out=open("verifier_agree.jsonl","w")
async def one(cl,sem,u,dx,gold):
    p=P.replace("{U}",u[:3500]).replace("{D}",dx)
    async with sem:
        for _ in range(5):
            k=keys[KC[0]%len(keys)];KC[0]+=1
            try:
                r=await cl.post("https://openrouter.ai/api/v1/chat/completions",headers={"Authorization":f"Bearer {k}"},
                    json={"model":"openrouter/owl-alpha","messages":[{"role":"user","content":p}],"max_tokens":30,"temperature":0},timeout=60)
                if r.status_code!=200:await asyncio.sleep(2);continue
                m=re.search(r'"verdict"\s*:\s*"(\w+)"',r.json()["choices"][0]["message"].get("content") or "")
                if m:
                    async with lock:out.write(json.dumps({"gold":gold,"rerate":m.group(1).lower(),"dx":dx})+"\n");out.flush()
                    return
            except:await asyncio.sleep(2)
async def main():
    sem=asyncio.Semaphore(50)
    async with httpx.AsyncClient() as cl:
        await asyncio.gather(*[one(cl,sem,u,dx,vd) for u,dx,vd in sample])
    print("AGREE_DONE",flush=True)
asyncio.run(main())
