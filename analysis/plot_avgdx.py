import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
# (model, avg_dx, category)  — exclude Qwen3.5-2B (direct)
data=[
 ("GPT-5.5",3.43,"general"),
 ("Claude Opus 4.8",5.78,"general"),
 ("GLM-5.2",2.94,"general"),
 ("Nemotron-550B",3.59,"general"),
 ("MedGemma-4B",1.76,"medical"),
 ("OpenBioLLM-8B",6.00,"medical"),
 ("HuatuoGPT-o1-8B",3.20,"medical"),
 ("Qwen3.5-2B (CoT)",1.20,"ours"),
 ("Qwen3.5-4B (CoT)",1.21,"ours"),
]
GOLD=1.48
colors={"general":"#4C72B0","medical":"#DD8452","ours":"#55A868"}
data=sorted(data,key=lambda x:x[1])  # ascending
names=[d[0] for d in data];vals=[d[1] for d in data];cats=[d[2] for d in data]
fig,ax=plt.subplots(figsize=(6.4,4.2))
bars=ax.barh(names,vals,color=[colors[c] for c in cats],edgecolor="black",linewidth=0.4)
for b,v in zip(bars,vals):
    ax.text(v+0.08,b.get_y()+b.get_height()/2,f"{v:.2f}",va="center",fontsize=9)
ax.axvline(GOLD,color="red",ls="--",lw=1.3)
ax.text(GOLD+0.05,len(names)-0.4,f"gold ≈{GOLD}",color="red",fontsize=9)
ax.set_xlabel("Average # diagnoses per encounter",fontsize=11)
ax.set_xlim(0,6.6)
# legend
from matplotlib.patches import Patch
ax.legend(handles=[Patch(facecolor=colors[k],edgecolor="black",label=l) for k,l in
          [("general","zero-shot general"),("medical","zero-shot medical"),("ours","post-trained (ours)")]],
          fontsize=8,loc="lower right",frameon=True)
ax.tick_params(labelsize=9)
plt.tight_layout()
plt.savefig("avg_dx.pdf",bbox_inches="tight")
plt.savefig("avg_dx.png",dpi=200,bbox_inches="tight")
print("SAVED avg_dx.pdf / avg_dx.png")
