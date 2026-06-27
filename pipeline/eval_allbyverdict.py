import json,re,asyncio,httpx,os,hashlib,glob,statistics as st
HERE="/data1/jiahui/MedTVT-R1_repo/ceiling"
keys=json.load(open(HERE+"/or_keys.json"))
CACHEL=HERE+"/judge_cache.jsonl";cache={}
for l in open(CACHEL):
    try:o=json.loads(l);cache[o["k"]]=o["m"]
    except:pass
cf=open(CACHEL,"a");KC=[0];lock=asyncio.Lock()
def ck(G,P):return hashlib.md5(json.dumps([sorted(G),sorted(P)],ensure_ascii=False).encode()).hexdigest()
J='You are a clinical coding judge. Two diagnoses MATCH if same clinical condition (synonyms/abbrev/specificity count). Compute 1-to-1 matching.\nGOLD: {G}\nPRED: {P}\nReturn ONLY JSON: {"matched_pairs": <number>}'
async def judge(cl,G,P):
    if not G or not P:return 0
    k=ck(G,P)
    if k in cache:return cache[k]
    p=J.replace("{G}",json.dumps(G)).replace("{P}",json.dumps(P))
    for _ in range(6):
        kk=keys[KC[0]%len(keys)];KC[0]+=1
        try:
            r=await cl.post("https://openrouter.ai/api/v1/chat/completions",headers={"Authorization":f"Bearer {kk}"},
                json={"model":"openrouter/owl-alpha","messages":[{"role":"user","content":p}],"max_tokens":120,"temperature":0},timeout=90)
            if r.status_code!=200:await asyncio.sleep(2);continue
            mm=re.search(r'"matched_pairs"\s*:\s*(\d+)',r.json()["choices"][0]["message"].get("content") or "")
            if mm:
                m=min(int(mm.group(1)),len(G),len(P))
                async with lock:
                    if k not in cache:cache[k]=m;cf.write(json.dumps({"k":k,"m":m})+"\n");cf.flush()
                return m
        except:await asyncio.sleep(2)
    return 0
def trunc(s):return s.split("\n\nBased on")[0]
vmap={}
for l in open("cohort_clean.jsonl"):
    r=json.loads(l);m=r["messages"]
    if isinstance(m,str):m=eval(m)
    kv=r.get("kept_verdicts")
    if isinstance(kv,str):kv=eval(kv)
    vmap[trunc(m[0]["content"])]={v["dx"].strip().lower():v.get("verdict","?") for v in kv}
MODELS={
 "GPT-5.5":["gpt55_pred.jsonl"],"Claude-Opus-4.8":["opus_pred.jsonl"],
 "GLM-5.2":["owl_pred.jsonl"],"Nemotron-550B":["nemotron_pred.jsonl"],
 "MedGemma-4B":[f"medgemma_pred_shard{s}.jsonl" for s in(0,1,2)],
 "OpenBioLLM-8B":[f"openbio_pred_shard{s}.jsonl" for s in(0,1,2)],
 "HuatuoGPT-o1-8B":[f"huatuo_pred_shard{s}.jsonl" for s in(0,1,2)],
 "Qwen3.5-2B-CoT":[f"qwen_pred_shard{s}.jsonl" for s in(0,1,2)],
 "Qwen3.5-2B-direct":[f"qwen2b_nothink_pred_shard{s}.jsonl" for s in(0,1,2)],
 "Qwen3.5-4B-CoT":[f"qwen4b_sft_pred_shard{s}.jsonl" for s in(0,1,2)],
}
def load(fs):
    out=[]
    for f in fs:
        if os.path.exists(f):
            for l in open(f):
                try:out.append(json.loads(l))
                except:pass
    return out
def agg(triples):
    TP=G=P=0;Ps=[];Rs=[];Fs=[];Js=[]
    for gset,m,pred in triples:
        if not gset:continue
        g=len(gset);p=len(pred);TP+=m;G+=g;P+=p
        pr=m/p if p else 0;rc=m/g if g else 0
        Ps.append(pr);Rs.append(rc);Fs.append(2*pr*rc/(pr+rc) if pr+rc else 0)
        Js.append(m/(g+p-m) if (g+p-m)>0 else 0)
    miP=TP/P if P else 0;miR=TP/G if G else 0
    return dict(n=len(Ps),P=round(miP,3),R=round(miR,3),F1=round(2*miP*miR/(miP+miR),3) if miP+miR else 0,
                J=round(st.mean(Js),3) if Js else 0)
async def main():
    sem=asyncio.Semaphore(120);OUT={}
    async with httpx.AsyncClient() as cl:
        for name,fs in MODELS.items():
            rows=load(fs)
            if not rows:continue
            async def proc(r):
                vm=vmap.get(trunc(r["input"]),{})
                gold=r["gold"];pred=r["pred"]
                gs=[g for g in gold if vm.get(g.strip().lower())=="supported"]
                gp=[g for g in gold if vm.get(g.strip().lower())=="partial"]
                async with sem:
                    ma=await judge(cl,gold,pred)
                    ms=await judge(cl,gs,pred) if gs else 0
                    mp=await judge(cl,gp,pred) if gp else 0
                return (gold,gs,gp,pred,ma,ms,mp)
            res=await asyncio.gather(*[proc(r) for r in rows])
            OUT[name]={"N":len(rows),
                "all":agg([(g,ma,p) for g,gs,gp,p,ma,ms,mp in res]),
                "supported":agg([(gs,ms,p) for g,gs,gp,p,ma,ms,mp in res]),
                "partial":agg([(gp,mp,p) for g,gs,gp,p,ma,ms,mp in res])}
            print(name,OUT[name]["N"],"done",flush=True)
    json.dump(OUT,open("eval_all_byverdict.json","w"),indent=2)
    print("ALLSAVED",flush=True)
asyncio.run(main())
