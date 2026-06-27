import os,json,re,asyncio,httpx
HERE="/data1/jiahui/MedTVT-R1_repo/ceiling"
keys=json.load(open(HERE+"/or_keys.json"))
OUT=HERE+"/nemotron_pred.jsonl"
MODEL="nvidia/nemotron-3-ultra-550b-a55b:free"
rows=[json.loads(l) for l in open(HERE+"/sft_test.jsonl")]
PROMPT="""You are an expert emergency physician. Given ONLY the admission-time presentation, reason from the evidence and determine THIS encounter's diagnoses. You are NOT given the answer.
INPUT:
{INPUT}
End your response with the final diagnoses on one line:
<answer>diagnosis1; diagnosis2</answer>"""
KC=[0];lock=asyncio.Lock()
def gold_of(m):return [x.strip() for x in m[-1]["content"].split("<answer>")[-1].replace("</answer>","").split(";") if x.strip()]
async def one(cl,sem,idx,r,f,n):
    inp=r["messages"][0]["content"].split("\n\nBased on")[0]
    p=PROMPT.replace("{INPUT}",inp)
    async with sem:
        for _ in range(6):
            k=keys[KC[0]%len(keys)];KC[0]+=1
            try:
                resp=await cl.post("https://openrouter.ai/api/v1/chat/completions",headers={"Authorization":f"Bearer {k}","Content-Type":"application/json"},
                    json={"model":MODEL,"messages":[{"role":"user","content":p}],"max_tokens":4000,"temperature":0.2},timeout=150)
                if resp.status_code==429:await asyncio.sleep(3);continue
                if resp.status_code!=200:await asyncio.sleep(2);continue
                c=resp.json()["choices"][0]["message"].get("content") or ""
                a=re.search(r"<answer>(.*?)</answer>",c,re.S)
                if a:
                    pred=[x.strip() for x in a.group(1).split(";") if x.strip()]
                    th=re.search(r"<think>(.*?)</think>",c,re.S)
                    async with lock:
                        f.write(json.dumps({"idx":idx,"input":inp,"gold":gold_of(r["messages"]),"pred":pred,
                            "think":(th.group(1).strip() if th else ""),"gen":c},ensure_ascii=False)+"\n");f.flush();n[0]+=1
                        if n[0]%100==0:print(f"nemo {n[0]}",flush=True)
                    return
            except:await asyncio.sleep(2)
async def main():
    done=set()
    if os.path.exists(OUT):
        for l in open(OUT):
            try:done.add(json.loads(l)["idx"])
            except:pass
    todo=[(i,r) for i,r in enumerate(rows) if i not in done]
    print(f"{len(rows)} total, {len(todo)} todo",flush=True)
    sem=asyncio.Semaphore(40);n=[0];f=open(OUT,"a")
    async with httpx.AsyncClient() as cl:
        await asyncio.gather(*[one(cl,sem,i,r,f,n) for i,r in todo])
    f.close();print(f"nemo DONE {n[0]}",flush=True)
asyncio.run(main())
