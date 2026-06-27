import json,re,asyncio,httpx,os,statistics as st
KEY="REPLACE_WITH_YOUR_OPENROUTER_KEY"
MODEL="anthropic/claude-opus-4.8"
OUT="opus_pred.jsonl"
rows=[json.loads(l) for l in open("sft_test.jsonl")][:100]
PROMPT="""You are an expert emergency physician. Given ONLY the admission-time presentation, reason from the evidence and determine THIS encounter's diagnoses. You are NOT given the answer.
INPUT:
{INPUT}
Output:
<think>[brief reasoning]</think>
<answer>diagnosis1; diagnosis2</answer>"""
def gold_of(m):return [x.strip() for x in m[-1]["content"].split("<answer>")[-1].replace("</answer>","").split(";") if x.strip()]
done=set()
if os.path.exists(OUT):
    for l in open(OUT):
        try:done.add(json.loads(l)["idx"])
        except:pass
lock=asyncio.Lock();f=open(OUT,"a")
async def judge(cl,G,P):
    if not P or not G:return 0
    J=f'''You are a clinical coding judge. Two diagnoses MATCH if same condition (synonyms/abbrev/specificity count).
Compute 1-to-1 matching. GOLD: {json.dumps(G)} PRED: {json.dumps(P)}
Return ONLY JSON: {{"matched_pairs": <n>}}'''
    for _ in range(5):
        try:
            r=await cl.post("https://openrouter.ai/api/v1/chat/completions",headers={"Authorization":f"Bearer {KEY}"},
                json={"model":MODEL,"messages":[{"role":"user","content":J}],"max_tokens":80,"temperature":0},timeout=120)
            if r.status_code!=200:await asyncio.sleep(3);continue
            return min(int(re.search(r'(\d+)',r.json()["choices"][0]["message"]["content"]).group(1)),len(G),len(P))
        except:await asyncio.sleep(3)
    return 0
async def one(cl,sem,idx,r,n):
    if idx in done:return
    inp=r["messages"][0]["content"].split("\n\nBased on")[0]
    p=PROMPT.replace("{INPUT}",inp);gold=gold_of(r["messages"])
    async with sem:
        for _ in range(5):
            try:
                resp=await cl.post("https://openrouter.ai/api/v1/chat/completions",headers={"Authorization":f"Bearer {KEY}"},
                    json={"model":MODEL,"messages":[{"role":"user","content":p}],"max_tokens":2000,"temperature":0.2},timeout=120)
                if resp.status_code!=200:await asyncio.sleep(3);continue
                c=resp.json()["choices"][0]["message"]["content"] or ""
                a=re.search(r"<answer>(.*?)</answer>",c,re.S)
                pred=[x.strip() for x in a.group(1).split(";") if x.strip()] if a else []
                th=re.search(r"<think>(.*?)</think>",c,re.S)
                m=await judge(cl,gold,pred)
                async with lock:
                    f.write(json.dumps({"idx":idx,"input":inp,"gold":gold,"pred":pred,
                        "think":(th.group(1).strip() if th else ""),"gen":c,"fmt":bool(a),"matched_pairs":m},ensure_ascii=False)+"\n");f.flush()
                    n[0]+=1
                    if n[0]%10==0:print(f"{n[0]} done",flush=True)
                return
            except:await asyncio.sleep(3)
async def main():
    sem=asyncio.Semaphore(8);n=[0]
    async with httpx.AsyncClient() as cl:
        await asyncio.gather(*[one(cl,sem,i,r,n) for i,r in enumerate(rows)])
    f.close()
    # 汇总
    rr=[json.loads(l) for l in open(OUT)]
    TP=sum(x["matched_pairs"] for x in rr);G=sum(len(x["gold"]) for x in rr);P=sum(len(x["pred"]) for x in rr)
    miP=TP/P;miR=TP/G;miF=2*miP*miR/(miP+miR)
    exP=st.mean(min(x["matched_pairs"],len(x["pred"]))/len(x["pred"]) if x["pred"] else 0 for x in rr)
    exR=st.mean(x["matched_pairs"]/len(x["gold"]) if x["gold"] else 0 for x in rr)
    exF=st.mean((lambda pr,rc:2*pr*rc/(pr+rc) if pr+rc else 0)(min(x["matched_pairs"],len(x["pred"]))/len(x["pred"]) if x["pred"] else 0, x["matched_pairs"]/len(x["gold"]) if x["gold"] else 0) for x in rr)
    jac=st.mean(x["matched_pairs"]/(len(x["gold"])+len(x["pred"])-x["matched_pairs"]) if (len(x["gold"])+len(x["pred"])-x["matched_pairs"])>0 else 0 for x in rr)
    res={"model":MODEL,"n":len(rr),"fmt_ok_pct":round(100*sum(x["fmt"] for x in rr)/len(rr),1),
         "avg_pred":round(st.mean(len(x["pred"]) for x in rr),2),"gold_avg":round(st.mean(len(x["gold"]) for x in rr),2),
         "micro_P":round(miP,3),"micro_R":round(miR,3),"micro_F1":round(miF,3),
         "ex_P":round(exP,3),"ex_R":round(exR,3),"ex_F1":round(exF,3),"jaccard":round(jac,3)}
    json.dump(res,open("eval_opus100.json","w"),indent=2)
    print("RESULT",json.dumps(res),flush=True)
asyncio.run(main())
