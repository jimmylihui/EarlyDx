import json,os,re,asyncio,httpx,statistics,hashlib,sys
HERE="/data1/jiahui/MedTVT-R1_repo/ceiling"
keys=json.load(open(HERE+"/or_keys.json"))
CACHE=HERE+"/judge_cache.json"
CACHEL=HERE+"/judge_cache.jsonl"
cache={}
if os.path.exists(CACHE):cache.update(json.load(open(CACHE)))
if os.path.exists(CACHEL):
    for l in open(CACHEL):
        try:o=json.loads(l);cache[o["k"]]=o["m"]
        except:pass
print(f"已有缓存: {len(cache)}",flush=True)
_cf=open(CACHEL,"a")
clock=asyncio.Lock()
def load(files):
    out=[]
    for fn in files:
        if os.path.exists(fn):
            for l in open(fn):
                try:out.append(json.loads(l))
                except:pass
    return out
MODELS={
 "qwen":load([f"qwen_pred_shard{s}.jsonl" for s in (0,1,2)]),
 "owl":load(["owl_pred.jsonl"]),
 "nemotron":load(["nemotron_pred.jsonl"]),
 "qwen4b_sft":load([f"qwen4b_sft_pred_shard{s}.jsonl" for s in (0,1,2)]),
 "qwen2b_nothink":load([f"qwen2b_nothink_pred_shard{s}.jsonl" for s in (0,1,2)]),
 "medgemma":load([f"medgemma_pred_shard{s}.jsonl" for s in (0,1,2)]),
 "openbio":load([f"openbio_pred_shard{s}.jsonl" for s in (0,1,2)]),
 "huatuo":load([f"huatuo_pred_shard{s}.jsonl" for s in (0,1,2)]),
}
WANT=sys.argv[1:] if len(sys.argv)>1 else ["qwen","owl"]
for m in WANT:print(f"{m}: {len(MODELS[m])} 条",flush=True)
def ckey(G,P):
    return hashlib.md5(json.dumps([sorted(G),sorted(P)],ensure_ascii=False).encode()).hexdigest()
KC=[0]
J="""You are a clinical coding judge. GOLD diagnoses (truth) and PRED diagnoses (model) for one admission.
Two diagnoses MATCH if they refer to the same clinical condition (synonyms/abbreviations/specificity differences count as match; e.g. "CAD"="coronary artery disease", "AKI"="acute kidney injury").
Compute a 1-to-1 matching between GOLD and PRED (each item used at most once).
GOLD: {G}
PRED: {P}
Return ONLY JSON: {{"matched_pairs": <number of matched GOLD-PRED pairs>}}"""
async def judge(cl,G,P):
    if not P or not G:return (0,)
    k=ckey(G,P)
    if k in cache:return (cache[k],)
    p=J.replace("{G}",json.dumps(G)).replace("{P}",json.dumps(P))
    for _ in range(6):
        kk=keys[KC[0]%len(keys)];KC[0]+=1
        try:
            r=await cl.post("https://openrouter.ai/api/v1/chat/completions",headers={"Authorization":f"Bearer {kk}"},
                json={"model":"openrouter/owl-alpha","messages":[{"role":"user","content":p}],"max_tokens":120,"temperature":0},timeout=90)
            if r.status_code!=200:await asyncio.sleep(2);continue
            c=r.json()["choices"][0]["message"].get("content") or ""
            mm=re.search(r'"matched_pairs"\s*:\s*(\d+)',c)
            if mm:
                mp=min(int(mm.group(1)),len(G),len(P))
                async with clock:
                    if k not in cache:
                        cache[k]=mp
                        _cf.write(json.dumps({"k":k,"m":mp})+"\n");_cf.flush()
                return (mp,)
        except:await asyncio.sleep(2)
    return None
async def main():
    sem=asyncio.Semaphore(120);out={}
    async with httpx.AsyncClient() as cl:
        for name in WANT:
            recs=MODELS[name]
            async def one(r):
                async with sem:return (r,await judge(cl,r["gold"],r["pred"]))
            df=open(HERE+f"/eval_detail_{name}.jsonl","w")
            TP=G=P=0;ps=[];rs=[];fs=[];js=[];nfail=0;fmtok=0;has_fmt=0;nj=0
            tasks=[asyncio.ensure_future(one(r)) for r in recs]
            for t in asyncio.as_completed(tasks):
                r,v=await t
                if v is None:continue
                m=v[0];g=len(r["gold"]);pp=len(r["pred"])
                G+=g;P+=pp;TP+=m;nj+=1
                pr=m/pp if pp else 0;rc=m/g if g else 0
                f=2*pr*rc/(pr+rc) if(pr+rc)else 0
                jac=m/(g+pp-m) if(g+pp-m)>0 else 0
                ps.append(pr);rs.append(rc);fs.append(f);js.append(jac)
                if not pp:nfail+=1
                if "fmt" in r:has_fmt+=1;fmtok+=1 if r["fmt"] else 0
                df.write(json.dumps({"gold":r["gold"],"pred":r["pred"],"matched_pairs":m,
                    "P":round(pr,3),"R":round(rc,3),"F1":round(f,3),"Jaccard":round(jac,3)},ensure_ascii=False)+"\n")
                df.flush()
            df.close()
            miP=TP/P if P else 0;miR=TP/G if G else 0
            miF=2*miP*miR/(miP+miR) if(miP+miR)else 0
            out[name]={"n":len(recs),"judged":nj,
              "avg_dx":round(statistics.mean(len(r["pred"]) for r in recs),2),
              "gold_avg_dx":round(statistics.mean(len(r["gold"]) for r in recs),2),
              "empty_pred":nfail,
              "micro_P":round(miP,3),"micro_R":round(miR,3),"micro_F1":round(miF,3),
              "ex_P":round(statistics.mean(ps),3),"ex_R":round(statistics.mean(rs),3),
              "ex_F1":round(statistics.mean(fs),3),"jaccard":round(statistics.mean(js),3)}
            if has_fmt:out[name]["fmt_ok_pct"]=round(100*fmtok/has_fmt,1)
            print(name,out[name],flush=True)
    json.dump(cache,open(CACHE,"w"))
    json.dump(out,open(HERE+"/eval_full_"+"_".join(WANT)+".json","w"),indent=2)
    print("SAVED",flush=True)
asyncio.run(main())
