"""Parallel, resumable label<->evidence consistency judge (async httpx + key rotation).
Usage: python judge_labels.py <samples.json> [concurrency]
 - Loads 212 OpenRouter keys from or_keys.json; rotates keys per request + on retry.
 - Judges only samples lacking 'label_verdicts'; writes back to the file (resumable).
 - Each request has a hard timeout; failures retry across different keys.
"""
import os,sys,json,re,asyncio,httpx
HERE=os.path.dirname(os.path.abspath(__file__))
PATH=sys.argv[1]
CONC=int(sys.argv[2]) if len(sys.argv)>2 else 30
MODEL=os.environ.get("JUDGE_MODEL","nex-agi/nex-n2-pro:free")
KEYS=json.load(open(os.environ.get("OR_KEYS_FILE",os.path.join(HERE,"or_keys.json"))))
if os.environ.get("ORK"):KEYS=[os.environ["ORK"]]+KEYS
PROMPT="""You are a clinical evidence auditor for an admission-time diagnosis dataset.
Given the admission-time INPUT and the RECORDED DIAGNOSES, judge for EACH diagnosis whether the INPUT contains evidence to support it AT ADMISSION.
Verdict options:
- "supported": clear/direct evidence in input (labs, imaging, vitals, exam, or clearly consistent presentation)
- "partial": only indirect/suggestive, or supported only by past history / home meds (chronic comorbidity), not acute presentation
- "unsupported": no evidence, OR input clearly points to a different problem (mismatch with chief complaint/findings)
INPUT:
{INPUT}
RECORDED DIAGNOSES:
{DXS}
Output ONLY a JSON object: {"verdicts":[{"dx":"<dx>","verdict":"supported|partial|unsupported","reason":"<=15 words"}]}"""
def parse(txt):
    for c in reversed(re.findall(r'\{.*?\}',txt,re.S)+re.findall(r'\{.*\}',txt,re.S)):
        try:
            j=json.loads(c)
            if "verdicts" in j:return j["verdicts"]
        except:continue
    return None
def mkprompt(s):
    inp=s['messages'][0]['content'].split("\n\nBased on")[0]
    dxs=[x.strip() for x in s['messages'][1]['content'].replace("<answer>","").replace("</answer>","").split(";") if x.strip()]
    return PROMPT.replace("{INPUT}",inp).replace("{DXS}","\n".join(f"{i+1}. {x}" for i,x in enumerate(dxs)))
KC=[0]  # 全局轮转计数器: 每次请求用下一个key, 均匀摊到212个key
async def one(client,sem,s,idx,ndone):
    prompt=mkprompt(s)
    async with sem:
        for att in range(8):
            key=KEYS[KC[0]%len(KEYS)];KC[0]+=1
            try:
                r=await client.post("https://openrouter.ai/api/v1/chat/completions",
                    headers={"Authorization":f"Bearer {key}","Content-Type":"application/json"},
                    json={"model":MODEL,"messages":[{"role":"user","content":prompt}],"max_tokens":4000,"temperature":0},
                    timeout=70)
                if r.status_code==429:
                    await asyncio.sleep(3);continue
                if r.status_code!=200:
                    await asyncio.sleep(1);continue
                c=r.json()["choices"][0]["message"].get("content") or ""
                vd=parse(c) if c.strip() else None
                if vd is not None:
                    s["label_verdicts"]=vd;ndone[0]+=1
                    if ndone[0]%200==0:print(f"  [{ndone[0]}] judged",flush=True)
                    return True
            except Exception:
                await asyncio.sleep(1)
        return False
async def _checkpointer(d):
    while True:
        await asyncio.sleep(60)
        tmp=PATH+".tmp";json.dump(d,open(tmp,"w"),indent=2);os.replace(tmp,PATH)
        print(f"    [checkpoint] {sum(1 for s in d if 'label_verdicts' in s)} judged saved",flush=True)
async def main():
    d=json.load(open(PATH))
    todo=[(i,s) for i,s in enumerate(d) if "label_verdicts" not in s]
    print(f"{len(d)} samples, {len(todo)} to judge, conc={CONC}, keys={len(KEYS)}",flush=True)
    sem=asyncio.Semaphore(CONC);ndone=[0]
    cp=asyncio.ensure_future(_checkpointer(d))
    async with httpx.AsyncClient() as client:
        await asyncio.gather(*[one(client,sem,s,i,ndone) for i,s in todo])
    cp.cancel()
    tmp=PATH+".tmp";json.dump(d,open(tmp,"w"),indent=2);os.replace(tmp,PATH)
    from collections import Counter
    cnt=Counter(v.get('verdict') for s in d if s.get('label_verdicts') for v in s['label_verdicts'])
    miss=sum(1 for s in d if "label_verdicts" not in s)
    print(f"\n新判 {ndone[0]}/{len(todo)}, 仍缺 {miss}. verdict累计: {dict(cnt)}",flush=True)
    print("DONE",flush=True)
asyncio.run(main())
