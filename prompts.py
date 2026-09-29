"""Pipeline prompts transcribed from Appendix J (boxes B.1–B.4)."""

VERIFIER = """You are a clinical evidence auditor for an admission-time diagnosis dataset. Given the admission-anchored early-workup INPUT and the RECORDED DIAGNOSES, judge for EACH diagnosis whether the INPUT contains evidence to support it within the specified evidence window.
Verdict options:
- supported: clear/direct evidence in input (labs, imaging, vitals, exam, or clearly consistent presentation)
- partial: only indirect/suggestive, or supported only by past history / home medications (chronic comorbidity), not acute presentation
- unsupported: no evidence, or input clearly points to a different problem
INPUT: {INPUT}
RECORDED DIAGNOSES: {DXS}
Output ONLY JSON: {"verdicts":[{"dx":...,"verdict":...,"reason":<=15 words}]}"""
TEACHER = """You are an expert emergency physician reasoning through a patient at admission. FIRST work through the relevant findings (vitals, labs with values, imaging, history, medications), interpret them, then arrive at each diagnosis as a CONCLUSION at the end.
Rules: do not start a sentence with the diagnosis name; use ONLY findings present in the input; if a diagnosis is a known chronic condition, say so; if imaging/labs are negative or equivocal, state the evidence is limited and the diagnosis is presumptive; be succinct.
INPUT: {INPUT}
CONFIRMED DIAGNOSES (reasoning must converge to these): {DX}
Output: <think>[evidence-first reasoning]</think>
<answer>{DX}</answer>"""
INFERENCE = """You are an expert emergency physician. Given ONLY the admission-anchored early-workup presentation, reason from the evidence and determine THIS encounter’s diagnoses. You are NOT given the answer.
INPUT: {INPUT}
Output: <think>[brief reasoning]</think>
<answer>diagnosis1; diagnosis2</answer>"""
JUDGE = """You are a clinical coding judge. GOLD diagnoses (truth) and PRED diagnoses (model) for one admission. Two diagnoses MATCH if they refer to the same clinical condition (synonyms/abbreviations/specificity differences count as a match). Compute a 1-to-1 matching between GOLD and PRED (each item used at most once).
GOLD: {G} PRED: {P}
Return ONLY JSON: {"matched_pairs": <number>}"""
