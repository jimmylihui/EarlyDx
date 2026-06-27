"""Plan A admission-time diagnosis sample builder (optimized, pre-grouped O(1) lookups).
Usage: python build_planA.py START COUNT OUTFILE   (set COHORT_INDEX env for full cohort)
All validated fixes integrated (see git history / prior notes). Judge is separate (judge_labels.py).
"""
import sys, os, pandas as pd, glob, re, json
from transformers import AutoTokenizer
START=int(sys.argv[1]) if len(sys.argv)>1 else 0
COUNT=int(sys.argv[2]) if len(sys.argv)>2 else 10
OUT=sys.argv[3] if len(sys.argv)>3 else f"planA_{START}_{START+COUNT}.json"
tok=AutoTokenizer.from_pretrained("/data1/jiahui/hf_llama32_1b_instruct")
g=lambda f,**k:pd.read_csv(f,**k)
WIN_H=48
SYMPT=["pain","ache","lumbago","backache","effusion","dizz","vertigo","syncope","nausea","vomit",
       "malaise","fatigue","weakness","hemoptysis","palpitation","tenderness","numbness","tingling",
       "fever","cough","epistaxis","hematuria","dysuria","colic"]
VAGUE=[r"\bcirculatory disease",r"disease of (the )?[\w ]+?(tract|system|organ)s?,? unspecified",
       r"\bunspecified disease\b",r"ill[- ]defined",r"other and unspecified disorders? of",
       r"disorder of [\w ]+?(system|tract), unspecified",
       r"\b(condition|dis|disorder|disease)s?,? (nec|nos)\b"]
def _chap(c,vr):
    c=str(c)
    if vr==10:return not (c[0] in 'VWXYZR')
    if c.startswith(('E','V')):return False
    try:return not (780<=int(c[:3])<=799)
    except:return True
def keep(c,vr,t):
    if not _chap(c,vr):return False
    tl=str(t).lower()
    if any(k in tl for k in SYMPT):return False
    if any(re.search(p,tl) for p in VAGUE):return False
    return True
def cln(s):return re.sub(r'\s+',' ',s).strip()
def labval(v,vn):
    v=str(v)
    if v in ('','nan') or '___' in v:
        return (f"{vn:g}" if pd.notna(vn) else "?")
    return v
def radtext(t):
    f=re.search(r"FINDINGS:(.*?)(IMPRESSION:|$)",t,re.S);i=re.search(r"IMPRESSION:(.*)",t,re.S);parts=[]
    if f and f.group(1).strip():parts.append("F: "+cln(f.group(1))[:230])
    if i and i.group(1).strip():parts.append("I: "+cln(i.group(1))[:180])
    if not parts:b=re.sub(r'^.*?:','',t,count=1,flags=re.S);parts.append(cln(b)[:230])
    return " ".join(parts)
def extract_pmh(text):
    m=re.search(r'Past Medical History:\s*(.*?)(\n\s*\n(?:Social History|Family History|Physical Exam|Medications|Brief Hospital Course|Past Surgical|PSH|Allergies)|\Z)',text,re.S|re.I)
    if not m:return None
    s=cln(m.group(1));return s[:400] if s else None
def grp(df,k):
    return {key:sub for key,sub in df.groupby(k)}

# ---- load tables ----
adm=g("mimic-iv/3.1/hosp/admissions.csv.gz");adm['admittime']=pd.to_datetime(adm['admittime']);admset=set(adm.hadm_id)
eds=g(glob.glob('mimic-iv-ed/*/ed/edstays.csv.gz')[0]);eds['intime']=pd.to_datetime(eds['intime'])
eds2=eds[eds.hadm_id.notna()].copy();eds2['hadm_id']=eds2.hadm_id.astype(int)
eddx=g(glob.glob('mimic-iv-ed/*/ed/diagnosis.csv.gz')[0]);tri=g(glob.glob('mimic-iv-ed/*/ed/triage.csv.gz')[0])
vsg=g(glob.glob('mimic-iv-ed/*/ed/vitalsign.csv.gz')[0]);vsg['charttime']=pd.to_datetime(vsg['charttime']);mrc=g(glob.glob('mimic-iv-ed/*/ed/medrecon.csv.gz')[0])
pt=g("mimic-iv/3.1/hosp/patients.csv.gz");te=g("MedTVT-R1_repo/QA/test_dip.csv")
lblf=lambda stay:(lambda e:e[e.apply(lambda x:keep(x.icd_code,x.icd_version,x.icd_title),axis=1)])(eddx[eddx.stay_id==stay])

CIDX=os.environ.get("COHORT_INDEX")
if CIDX:
    cand=[tuple(x) for x in json.load(open(CIDX))][START:START+COUNT]
else:
    allc=[]
    for sid in te.subject_id.unique():
        for _,r in eds2[eds2.subject_id==sid].iterrows():
            if r.hadm_id not in admset:continue
            if len(lblf(r.stay_id))==0:continue
            allc.append((int(sid),int(r.hadm_id),int(r.stay_id)));break
        if len(allc)>=START+COUNT:break
    cand=allc[START:START+COUNT]
SUBS={c[0] for c in cand};HADMS={c[1] for c in cand}
print(f"building {len(cand)} samples [{START}:{START+COUNT}]",flush=True)

# ---- labevents (single scan) ----
base=[];winlab=[]
for ch in pd.read_csv("mimic-iv/3.1/hosp/labevents.csv.gz",usecols=["subject_id","hadm_id","itemid","charttime","value","valuenum","flag"],chunksize=3_000_000,low_memory=False):
    b=ch[(ch.subject_id.isin(SUBS))&(ch.itemid.isin([50912,51222]))]
    if len(b):b=b.copy();b['charttime']=pd.to_datetime(b['charttime']);base.append(b)
    w=ch[ch.hadm_id.isin(HADMS)]
    if len(w):w=w.copy();w['charttime']=pd.to_datetime(w['charttime']);winlab.append(w)
BASE=pd.concat(base) if base else pd.DataFrame(columns=["subject_id","itemid","charttime","valuenum"])
WIN=pd.concat(winlab) if winlab else pd.DataFrame(columns=["hadm_id","itemid","charttime","value","valuenum","flag"])
print("labs scanned",flush=True)

# ---- discharge notes (prior admissions only) ----
prior_hadms=set()
adm_sub_full=grp(adm[adm.subject_id.isin(SUBS)],'subject_id')
admit_of={int(r.hadm_id):r.admittime for r in adm[adm.hadm_id.isin(HADMS)].itertuples()}
for SID,HID,_ in cand:
    sub=adm_sub_full.get(SID)
    if sub is not None:prior_hadms|=set(sub[sub.admittime<admit_of[HID]]['hadm_id'])
disch={}
if prior_hadms:
    for ch in pd.read_csv("mimic-iv-note/2.2/note/discharge.csv.gz",usecols=["hadm_id","text"],chunksize=50000):
        s=ch[ch.hadm_id.isin(prior_hadms)]
        for _,x in s.iterrows():disch[int(x.hadm_id)]=x['text']
print(f"prior discharge notes: {len(disch)}",flush=True)

# ---- other tables, restricted to cohort subjects then pre-grouped ----
dlab=g("mimic-iv/3.1/hosp/d_labitems.csv.gz")[["itemid","label"]].set_index("itemid")["label"].to_dict()
dx=g("mimic-iv/3.1/hosp/diagnoses_icd.csv.gz");dx=dx[dx.subject_id.isin(SUBS)]
dicdf=g("mimic-iv/3.1/hosp/d_icd_diagnoses.csv.gz");DICD={(r.icd_code,r.icd_version):r.long_title for r in dicdf.itertuples()}
prc=g("mimic-iv/3.1/hosp/procedures_icd.csv.gz");prc=prc[prc.subject_id.isin(SUBS)]
dprf=g("mimic-iv/3.1/hosp/d_icd_procedures.csv.gz");DPR={(r.icd_code,r.icd_version):r.long_title for r in dprf.itertuples()}
omr=g("mimic-iv/3.1/hosp/omr.csv.gz");omr=omr[omr.subject_id.isin(SUBS)].copy();omr['chartdate']=pd.to_datetime(omr['chartdate'])
mm=g("mimic-iv-ecg/1.0/machine_measurements.csv",low_memory=False);mm=mm[mm.subject_id.isin(SUBS)].copy();mm['ecg_time']=pd.to_datetime(mm['ecg_time'])
sm=g("mimic-iv-echo/structured-measurement.csv.gz",usecols=["subject_id","measurement_datetime","measurement","result","unit"]);sm=sm[sm.subject_id.isin(SUBS)].copy();sm['dt']=pd.to_datetime(sm['measurement_datetime'])
rad=g("mimic-iv-note/2.2/note/radiology.csv.gz",usecols=["subject_id","hadm_id","charttime","text"]);rad=rad[rad.subject_id.isin(SUBS)].copy();rad['charttime']=pd.to_datetime(rad['charttime'])
print("tables loaded, grouping...",flush=True)
PT={r.subject_id:r for r in pt[pt.subject_id.isin(SUBS)].itertuples()}
ADM_H={int(r.hadm_id):r for r in adm[adm.hadm_id.isin(HADMS)].itertuples()}
ADM_SUB=grp(adm[adm.subject_id.isin(SUBS)],'subject_id')
EDS2_ST={int(r.stay_id):r for r in eds2[eds2.stay_id.isin({c[2] for c in cand})].itertuples()}
EDS_ALL_SUB=grp(eds[eds.subject_id.isin(SUBS)],'subject_id')
TRI_ST={int(r.stay_id):r for r in tri[tri.stay_id.isin({c[2] for c in cand})].itertuples()}
VSG_ST=grp(vsg[vsg.stay_id.isin({c[2] for c in cand})],'stay_id')
MRC_ST=grp(mrc[mrc.stay_id.isin({c[2] for c in cand})],'stay_id')
EDDX_ST=grp(eddx[eddx.stay_id.isin({c[2] for c in cand})],'stay_id')
DX_SUB=grp(dx,'subject_id');PRC_SUB=grp(prc,'subject_id');OMR_SUB=grp(omr,'subject_id')
MM_SUB=grp(mm,'subject_id');SM_SUB=grp(sm,'subject_id');RAD_SUB=grp(rad,'subject_id')
BASE_SUB=grp(BASE,'subject_id') if len(BASE) else {}
WIN_HD=grp(WIN,'hadm_id') if len(WIN) else {}
print("grouped. building...",flush=True)
RPT=[f'report_{i}' for i in range(18)]
EMPTY=pd.DataFrame()
def lbl_titles(stay):
    e=EDDX_ST.get(stay)
    if e is None:return []
    return [r.icd_title for r in e.itertuples() if keep(r.icd_code,r.icd_version,r.icd_title)]
def build(SID,HID,stay):
    a=ADM_H[HID];admit=a.admittime;T0=admit+pd.Timedelta(f'{WIN_H}h')
    est=EDS2_ST[stay];cur_intime=est.intime;p=PT[SID];S=[]
    S.append(f"Demographics: {p.gender}, {p.anchor_age}yo, {a.race}. From {a.admission_location}, arrival {est.arrival_transport}.")
    t=TRI_ST.get(stay)
    S.append(f"Chief complaint: {t.chiefcomplaint if (t is not None and str(t.chiefcomplaint)!='nan') else 'None'}")
    if t is not None:S.append(f"Triage: T{t.temperature} HR{t.heartrate} RR{t.resprate} SpO2{t.o2sat} BP{t.sbp}/{t.dbp} acuity{t.acuity}")
    else:S.append("Triage: None")
    v=VSG_ST.get(stay,EMPTY)
    if len(v):v=v[v.charttime<=T0].sort_values("charttime")
    S.append("ED serial vitals: "+(" | ".join(f"HR{r.heartrate} SpO2{r.o2sat}" for r in v.iloc[[0,-1]].itertuples()) if len(v) else "None"))
    mr=MRC_ST.get(stay,EMPTY);mrn=mr['name'].dropna().unique()[:12] if len(mr) else []
    S.append("Home meds: "+("; ".join(mrn) if len(mrn) else "None"))
    o=OMR_SUB.get(SID,EMPTY)
    if len(o):o=o[o.chartdate<=admit].sort_values('chartdate')
    def lo(*ns):
        for n in ns:
            s=o[o.result_name==n] if len(o) else EMPTY
            if len(s):return s.iloc[-1]['result_value']
        return "None"
    S.append(f"Baseline (OMR): BP {lo('Blood Pressure')}, Wt {lo('Weight (Lbs)','Weight')}, Ht {lo('Height (Inches)','Height')}, BMI {lo('BMI (kg/m2)','BMI')}")
    bb=BASE_SUB.get(SID,EMPTY);bcr=bhgb='None'
    if len(bb):
        bb=bb[bb.charttime<admit]
        b1=bb[bb.itemid==50912];b2=bb[bb.itemid==51222]
        if len(b1):bcr=b1.sort_values('charttime').iloc[-1]['valuenum']
        if len(b2):bhgb=b2.sort_values('charttime').iloc[-1]['valuenum']
    S.append(f"Baseline labs (prior): creatinine {bcr}, hemoglobin {bhgb}")
    asub=ADM_SUB.get(SID,EMPTY)
    prior_h=list(asub[asub.admittime<admit].sort_values('admittime')['hadm_id']) if len(asub) else []
    dxs=DX_SUB.get(SID,EMPTY);pht=[]
    if len(dxs) and prior_h:
        for r in dxs[dxs.hadm_id.isin(prior_h)].itertuples():
            ti=DICD.get((r.icd_code,r.icd_version))
            if ti and keep(r.icd_code,r.icd_version,ti):pht.append(ti)
    S.append("Past medical history (prior dx): "+("; ".join(list(dict.fromkeys(pht))[:8]) if pht else "None"))
    edsub=EDS_ALL_SUB.get(SID,EMPTY)
    ped_stays=list(edsub[(edsub.intime<cur_intime)&(edsub.stay_id!=stay)].sort_values('intime')['stay_id']) if len(edsub) else []
    pedt=[]
    for st in ped_stays:
        e=EDDX_ST.get(st)
        if e is None:
            e=eddx[eddx.stay_id==st]
        for r in e.itertuples():
            if keep(r.icd_code,r.icd_version,r.icd_title):pedt.append(r.icd_title)
    S.append("Past ED diagnoses: "+("; ".join(list(dict.fromkeys(pedt))[:8]) if pedt else "None"))
    pmhtext=None
    for h in reversed(prior_h):
        if h in disch:
            x=extract_pmh(disch[h])
            if x:pmhtext=x;break
    S.append("Past medical history (from prior discharge note): "+(pmhtext if pmhtext else "None"))
    em=MM_SUB.get(SID,EMPTY)
    if len(em):em=em[em.ecg_time<=T0]
    if len(em):
        r=em.sort_values('ecg_time').iloc[-1];stm="; ".join(str(r[c]) for c in RPT if pd.notna(r[c]))
        try:
            hr=round(60000/r['rr_interval']);qrs=r['qrs_end']-r['qrs_onset'];qt=r['t_end']-r['qrs_onset'];pr=r['qrs_onset']-r['p_onset']
            if ("trial fib" in stm.lower()) or (not pd.notna(pr)) or pr<=0:pr="n/a"
        except:hr=qrs=qt=pr='?'
        S.append(f"ECG: {stm} | HR{hr} PR{pr} QRS{qrs} QT{qt}ms")
    else:S.append("ECG: None")
    se=SM_SUB.get(SID,EMPTY)
    if len(se):se=se[se.dt<=T0].dropna(subset=['result'])
    S.append("Echocardiogram: "+("; ".join(f"{r.measurement} {r.result}{r.unit}" for r in se.head(8).itertuples()) if len(se) else "None"))
    rr=RAD_SUB.get(SID,EMPTY)
    if len(rr):rr=rr[(rr.charttime>=cur_intime)&(rr.charttime<=T0)].sort_values("charttime")
    seen=set();rl=[]
    for x in (rr.itertuples() if len(rr) else []):
        em=re.search(r"EXAMINATION:\s*(.*)",x.text);ex=em.group(1).strip()[:40] if em else None
        body=radtext(x.text)
        sig=(ex or "")+"|"+body[:80]   # 真重复(同exam同内容)才去重; 无头的不同报告不塌缩
        if sig in seen:continue
        seen.add(sig);rl.append(f"  [{ex or 'study'}] {body}")
        if len(rl)>=8:break           # 上限8份控token
    S.append(f"Radiology ({len(rl)}): "+("\n".join(rl) if rl else "None"))
    wl=WIN_HD.get(HID,EMPTY)
    if len(wl):
        wl=wl[wl.charttime<=T0].drop_duplicates('itemid').copy();wl['_ab']=wl['flag'].notna();wl=wl.sort_values('_ab',ascending=False)
        # NOTE: 已交付数据用 [:700]。审计发现化验多(>~45项)的case文本被截断,~20-40正常值不可见;
        # 异常值优先排序已保留关键证据,判官质量未受损,故暂不重建。如重建可改 [:1400] 覆盖~80项。
        lab_s="; ".join(f"{dlab.get(r.itemid,r.itemid)} {labval(r.value,r.valuenum)}{'*' if pd.notna(r.flag) else ''}" for r in wl.itertuples())[:700]
    else:lab_s="None"
    S.append(f"Initial labs ({len(wl)}): "+lab_s)
    ans="<answer>"+"; ".join(lbl_titles(stay))+"</answer>"
    return "\n".join(S)+"\n\nBased on this presentation, what are the patient's diagnoses?",ans

samples=[]
for i,(SID,HID,stay) in enumerate(cand):
    try:
        U,A=build(SID,HID,stay)
        samples.append({"subject_id":SID,"hadm_id":HID,"messages":[{"role":"user","content":U},{"role":"assistant","content":A}]})
    except Exception as e:
        print(f"  skip {HID}: {type(e).__name__} {str(e)[:80]}",flush=True)
    if (i+1)%500==0:print(f"  {i+1}/{len(cand)}",flush=True)
json.dump(samples,open(f"MedTVT-R1_repo/ceiling/{OUT}","w"),indent=2)
print(f"saved {len(samples)} -> {OUT}",flush=True);print("DONE",flush=True)
