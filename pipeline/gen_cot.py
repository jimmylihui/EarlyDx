"""Full CoT generation over cohort_clean.jsonl (owl-alpha, parallel, key-rotation, resumable).
Usage: python gen_cot.py [concurrency]
Appends each completed record (with 'cot' field) to cohort_cot.jsonl. Re-run to fill failures.
"""
import os,sys,json,re,asyncio,httpx
HERE=os.path.dirname(os.path.abspath(__file__))
IN=os.path.join(HERE,"cohort_clean.jsonl");OUT=os.path.join(HERE,"cohort_cot.jsonl")
CONC=int(sys.argv[1]) if len(sys.argv)>1 else 100
MODEL=os.environ.get("JUDGE_MODEL","openrouter/owl-alpha")
KEYS=json.load(open(os.environ.get("OR_KEYS_FILE",os.path.join(HERE,"or_keys.json"))))
PROMPT="""You are an expert emergency physician reasoning through a patient at admission. Think as a clinician does: FIRST work through the relevant findings (vitals, labs with values, imaging, history, medications), interpret them, then arrive at each diagnosis as a CONCLUSION at the end of the reasoning thread.
Rules:
- Do NOT start a sentence with the diagnosis name; let it emerge as the conclusion.
- Use ONLY findings present in the input; do not invent.
- If a diagnosis is a known chronic condition rather than an acute finding, say so honestly.
- If imaging/labs are negative or only equivocal for a diagnosis, explicitly state the evidence is limited and the diagnosis is presumptive/clinical — do not overstate.
- Be succinct.
INPUT:
{INPUT}
CONFIRMED DIAGNOSES (reasoning must converge to these): {DX}
Output:
<think>
[evidence-first reasoning converging to each diagnosis]
</think>
<answer>{DX}</answer>"""
KC=[0]
def mkprompt(s):
    inp=s['messages'][0]['content'].split("\n\nBased on")[0]
    dx=s['messages'][1]['content'].replace("<answer>","").replace("</answer>","")
    return PROMPT.replace("{INPUT}",inp).replace("{DX}",dx)
lock=asyncio.Lock()
async def one(client,sem,s,fout,ndone):
    prompt=mkprompt(s)
    async with sem:
        for att in range(8):
            key=KEYS[KC[0]%len(KEYS)];KC[0]+=1
            try:
                r=await client.post("https://openrouter.ai/api/v1/chat/completions",
                    headers={"Authorization":f"Bearer {key}","Content-Type":"application/json"},
                    json={"model":MODEL,"messages":[{"role":"user","content":prompt}],"max_tokens":1200,"temperature":0.3},
                    timeout=90)
                if r.status_code==429:await asyncio.sleep(3);continue
                if r.status_code!=200:await asyncio.sleep(1);continue
                c=r.json()["choices"][0]["message"].get("content") or ""
                if "<answer>" in c and "<think>" in c:
                    rec={"subject_id":s["subject_id"],"hadm_id":s["hadm_id"],
                         "messages":[s["messages"][0],{"role":"assistant","content":c.strip()}],
                         "kept_verdicts":s.get("kept_verdicts")}
                    async with lock:
                        fout.write(json.dumps(rec,ensure_ascii=False)+"\n");fout.flush();ndone[0]+=1
                        if ndone[0]%200==0:print(f"  [{ndone[0]}] done",flush=True)
                    return True
            except Exception:await asyncio.sleep(1)
        return False
async def main():
    rows=[json.loads(l) for l in open(IN)]
    done=set()
    if os.path.exists(OUT):
        for l in open(OUT):
            try:done.add(json.loads(l)["hadm_id"])
            except:pass
    todo=[s for s in rows if s["hadm_id"] not in done]
    print(f"{len(rows)} total, {len(done)} done, {len(todo)} to generate, conc={CONC}, keys={len(KEYS)}",flush=True)
    sem=asyncio.Semaphore(CONC);ndone=[0]
    fout=open(OUT,"a")
    async with httpx.AsyncClient() as client:
        await asyncio.gather(*[one(client,sem,s,fout,ndone) for s in todo])
    fout.close()
    print(f"\n本轮生成 {ndone[0]}/{len(todo)}. 累计 {len(done)+ndone[0]}/{len(rows)}",flush=True)
    print("DONE",flush=True)
asyncio.run(main())
