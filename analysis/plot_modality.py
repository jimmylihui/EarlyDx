import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, json
from collections import Counter
c=Counter()
for l in open("modality_attr.jsonl"):
    p=json.loads(l)["primary"]
    if p.lower().startswith("labor"):p="Laboratory"
    c[p]+=1
tot=sum(c.values())
# 规范顺序 + 合并<1.5%到Other
order=["Imaging (CT/X-ray/US)","Prior history/records","Laboratory","Symptoms/chief complaint","Medications","ECG"]
labels=[];sizes=[]
for k in order:labels.append(k);sizes.append(c.get(k,0))
other=tot-sum(sizes)
labels.append("Other (vitals, exam, echo)");sizes.append(other)
colors=["#4C72B0","#DD8452","#55A868","#C44E52","#8172B3","#937860","#CCCCCC"]
fig,ax=plt.subplots(figsize=(7,5.2))
w,t,a=ax.pie(sizes,labels=None,autopct=lambda p:f"{p:.1f}%",startangle=90,counterclock=False,
    colors=colors,pctdistance=0.78,wedgeprops=dict(width=0.45,edgecolor="white"))
for x in a:x.set_fontsize(9)
ax.legend(w,[f"{l}" for l in labels],loc="center left",bbox_to_anchor=(1.0,0.5),fontsize=9,frameon=False)
ax.set_aspect("equal")
plt.tight_layout()
plt.savefig("modality_pie.pdf",bbox_inches="tight");plt.savefig("modality_pie.png",dpi=200,bbox_inches="tight")
print("SAVED",{l:round(100*s/tot,1) for l,s in zip(labels,sizes)})
