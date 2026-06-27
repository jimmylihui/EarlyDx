import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, json, numpy as np
data=json.load(open("modality_by_disease.json"))
order=["Pneumonia","Sepsis","Heart failure","MI / CAD","Stroke / ICH","Atrial fib","AKI / renal","Fracture","Diabetes"]
mods=["Imaging (CT/X-ray/US)","Laboratory","ECG","Prior history/records","Symptoms/chief complaint","Medications","Other"]
colors={"Imaging (CT/X-ray/US)":"#4C72B0","Laboratory":"#55A868","ECG":"#937860",
        "Prior history/records":"#DD8452","Symptoms/chief complaint":"#C44E52",
        "Medications":"#8172B3","Other":"#CCCCCC"}
def row(d):
    c=data[d];tot=sum(c.values())
    main={m:c.get(m,0) for m in mods[:-1]};main["Other"]=tot-sum(main.values())
    return {m:100*main[m]/tot for m in mods}
fig,ax=plt.subplots(figsize=(8,5))
bottom=np.zeros(len(order))
for m in mods:
    vals=[row(d)[m] for d in order]
    ax.bar(order,vals,bottom=bottom,color=colors[m],label=m,edgecolor="white",linewidth=0.5)
    for i,(v,b) in enumerate(zip(vals,bottom)):
        if v>=8:ax.text(i,b+v/2,f"{v:.0f}",va="center",ha="center",fontsize=8,color="white")
    bottom+=vals
ax.set_ylim(0,100);ax.set_ylabel("Share by primary evidence modality (%)",fontsize=11)
ax.set_xticklabels(order,rotation=35,ha="right",fontsize=9)
ax.legend(loc="center left",bbox_to_anchor=(1.0,0.5),fontsize=8,frameon=False)
ax.tick_params(axis='y',labelsize=9)
plt.tight_layout()
plt.savefig("disease_modality.pdf",bbox_inches="tight");plt.savefig("disease_modality.png",dpi=200,bbox_inches="tight")
print("SAVED")
