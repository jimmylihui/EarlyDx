"""One-time: build cohort index of all qualifying ED->admission stays across full MIMIC.
Applies the full label keep() filter; keeps stays with >=1 surviving diagnosis.
Saves ordered list of [subject_id, hadm_id, stay_id] to cohort_index.json.
"""
import pandas as pd, glob, re, json
SYMPT=["pain","ache","lumbago","backache","effusion","dizz","vertigo","syncope","nausea","vomit",
       "malaise","fatigue","weakness","hemoptysis","palpitation","tenderness","numbness","tingling",
       "fever","cough","epistaxis","hematuria","dysuria","colic"]
VAGUE=[r"\bcirculatory disease",r"disease of (the )?[\w ]+?(tract|system|organ)s?,? unspecified",
       r"\bunspecified disease\b",r"ill[- ]defined",r"other and unspecified disorders? of",
       r"disorder of [\w ]+?(system|tract), unspecified",
       r"\b(condition|dis|disorder|disease)s?,? (nec|nos)\b"]
def keep(c,vr,t):
    c=str(c)
    if vr==10:
        if c[0] in 'VWXYZR':return False
    else:
        if c.startswith(('E','V')):return False
        try:
            if 780<=int(c[:3])<=799:return False
        except:pass
    tl=str(t).lower()
    if any(k in tl for k in SYMPT):return False
    if any(re.search(p,tl) for p in VAGUE):return False
    return True
adm=pd.read_csv("mimic-iv/3.1/hosp/admissions.csv.gz",usecols=["hadm_id"]);admset=set(adm.hadm_id)
eds=pd.read_csv(glob.glob('mimic-iv-ed/*/ed/edstays.csv.gz')[0],usecols=["subject_id","hadm_id","stay_id"])
edl=eds[eds.hadm_id.notna()].copy();edl['hadm_id']=edl.hadm_id.astype(int)
edl=edl[edl.hadm_id.isin(admset)]
print("ED->admission stays:",len(edl),flush=True)
eddx=pd.read_csv(glob.glob('mimic-iv-ed/*/ed/diagnosis.csv.gz')[0])
eddx['k']=eddx.apply(lambda r:keep(r.icd_code,r.icd_version,r.icd_title),axis=1)
good=set(eddx[eddx.k].stay_id)
edl=edl[edl.stay_id.isin(good)].drop_duplicates('stay_id').sort_values('stay_id')
idx=[[int(r.subject_id),int(r.hadm_id),int(r.stay_id)] for r in edl.itertuples()]
json.dump(idx,open("MedTVT-R1_repo/ceiling/cohort_index.json","w"))
print("cohort index saved:",len(idx),"stays",flush=True)
print("DONE",flush=True)
