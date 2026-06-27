import json,re,asyncio,httpx
HERE="/data1/jiahui/MedTVT-R1_repo/ceiling"
keys=json.load(open(HERE+"/or_keys.json"));KC=[0];lock=asyncio.Lock()
rows=[json.loads(l) for l in open("cohort_cot_final.jsonl")]
step=max(1,len(rows)//100);sample=rows[::step][:100]
P='''You are an expert physician reviewing an AI's diagnostic reasoning. Given the admission INPUT, the REASONING, and the final DIAGNOSIS, judge whether the reasoning is clinically CORRECT: it must use evidence actually present in the input, contain no hallucinated findings, and the cited evidence must plausibly support the diagnosis.
Rate:
- "correct": clinically sound, evidence-grounded, no errors.
- "partial": mostly sound but with a minor issue (overstatement, weak link, slight misread).
- "incorrect": flawed reasoning, hallucinated/contradicted evidence, or unsupported conclusion.
INPUT:
{U}
REASONING: {T}
DIAGNOSIS: {D}
Return ONLY JSON: {"rating":"correct|partial|incorrect","reason":"<8 words>"}'''
out=open("cot_review.jsonl","w")
async def one(cl,sem,i,r):
    u=r['messages'][0]['content'];a=r['messages'][-1]['content']
    th=re.search(r'<think>(.*?)</think>',a,re.S);an=re.search(r'<answer>(.*?)</answer>',a,re.S)
    if not th:return
    p=P.replace("{U}",u[:3000]).replace("{T}",th.group(1).strip()[:1200]).replace("{D}",an.group(1) if an else "")
    async with sem:
        for _ in range(5):
            k=keys[KC[0]%len(keys)];KC[0]+=1
            try:
                rr=await cl.post("https://openrouter.ai/api/v1/chat/completions",headers={"Authorization":f"Bearer {k}"},
                    json={"model":"openrouter/owl-alpha","messages":[{"role":"user","content":p}],"max_tokens":60,"temperature":0},timeout=60)
                if rr.status_code!=200:await asyncio.sleep(2);continue
                c=rr.json()["choices"][0]["message"].get("content") or ""
                m=re.search(r'"rating"\s*:\s*"(\w+)"',c);rs=re.search(r'"reason"\s*:\s*"([^"]*)"',c)
                if m:
                    async with lock:out.write(json.dumps({"i":i,"rating":m.group(1).lower(),"reason":rs.group(1) if rs else "","dx":an.group(1) if an else ""})+"\n");out.flush()
                    return
            except:await asyncio.sleep(2)
async def main():
    sem=asyncio.Semaphore(50)
    async with httpx.AsyncClient() as cl:
        await asyncio.gather(*[one(cl,sem,i,r) for i,r in enumerate(sample)])
    print("REVIEW_DONE",flush=True)
asyncio.run(main())
