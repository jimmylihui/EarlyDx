import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, json
def from_detail(fn):
    rows=[]
    for l in open(fn):
        try:rows.append(json.loads(l))
        except:pass
    return 100*sum(1 for r in rows if r.get('F1',0)==1)/len(rows)
def from_pred(fns):
    rows=[]
    for f in fns:
        for l in open(f):
            try:rows.append(json.loads(l))
            except:pass
    n=len(rows);c=0
    for r in rows:
        m=r.get('matched_pairs',0);g=len(r['gold']);p=len(r['pred'])
        if g and p and m==g and m==p:c+=1
    return 100*c/n
M=[
 ("GPT-5.5",from_pred(["gpt55_pred.jsonl"]),"general"),
 ("Claude Opus 4.8",from_pred(["opus_pred.jsonl"]),"general"),
 ("GLM-5.2",from_detail("eval_detail_owl.jsonl"),"general"),
 ("Nemotron-550B",from_detail("eval_detail_nemotron.jsonl"),"general"),
 ("MedGemma-4B",from_detail("eval_detail_medgemma.jsonl"),"medical"),
 ("OpenBioLLM-8B",from_detail("eval_detail_openbio.jsonl"),"medical"),
 ("HuatuoGPT-o1-8B",from_detail("eval_detail_huatuo.jsonl"),"medical"),
 ("Qwen3.5-2B (CoT)",from_detail("eval_detail_qwen.jsonl"),"ours"),
 ("Qwen3.5-4B (CoT)",from_detail("eval_detail_qwen4b_sft.jsonl"),"ours"),
]
colors={"general":"#4C72B0","medical":"#DD8452","ours":"#55A868"}
M=sorted(M,key=lambda x:x[1])
names=[m[0] for m in M];vals=[m[1] for m in M];cats=[m[2] for m in M]
fig,ax=plt.subplots(figsize=(6.4,4.2))
bars=ax.barh(names,vals,color=[colors[c] for c in cats],edgecolor="black",linewidth=0.4)
for b,v in zip(bars,vals):
    ax.text(v+0.4,b.get_y()+b.get_height()/2,f"{v:.1f}%",va="center",fontsize=9)
ax.set_xlabel("Complete-match rate (% of encounters)",fontsize=11)
ax.set_xlim(0,42)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(facecolor=colors[k],edgecolor="black",label=l) for k,l in
  [("general","zero-shot general"),("medical","zero-shot medical"),("ours","post-trained (ours)")]],
  fontsize=8,loc="lower right",frameon=True)
ax.tick_params(labelsize=9); plt.tight_layout()
plt.savefig("complete_match.pdf",bbox_inches="tight"); plt.savefig("complete_match.png",dpi=200,bbox_inches="tight")
cp=dict((n,round(v,1)) for n,v,_ in M); print("vals:",cp); print("SAVED")
