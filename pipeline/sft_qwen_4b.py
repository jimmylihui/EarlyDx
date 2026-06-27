"""CoT-SFT of Qwen/Qwen3.5-4B (VLM text backbone) with DeepSpeed ZeRO-2.
Same data/recipe as 2B (effective batch 48). Completion-only loss (mask prompt).
Usage:
  torchrun --nproc_per_node=3 sft_qwen_4b.py            # full train (ZeRO-2)
  python sft_qwen_4b.py --smoke                          # 1-GPU smoke (few steps)
"""
import os,sys,json,torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForImageTextToText, Trainer, TrainingArguments
os.environ.setdefault("HF_HOME","/data1/jiahui/hf_cache")
MODEL="Qwen/Qwen3.5-4B"
DATA="/data1/jiahui/MedTVT-R1_repo/ceiling/sft_train.jsonl"
OUT="/data1/jiahui/MedTVT-R1_repo/ceiling/qwen35_4b_cotsft"
DS_CFG="/data1/jiahui/MedTVT-R1_repo/ceiling/zero2_offload.json"
MAXLEN=3072
SMOKE="--smoke" in sys.argv

tok=AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None: tok.pad_token=tok.eos_token

class DS(Dataset):
    def __init__(self,path,limit=None):
        self.rows=[json.loads(l) for l in open(path)]
        if limit:self.rows=self.rows[:limit]
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):
        m=self.rows[i]["messages"]
        full_t=tok.apply_chat_template(m,tokenize=False,add_generation_prompt=False)
        MARK="<|im_start|>assistant\n"
        j=full_t.rfind(MARK); prompt_t=full_t[:j+len(MARK)]
        full=tok(full_t,add_special_tokens=False)["input_ids"]
        prompt=tok(prompt_t,add_special_tokens=False)["input_ids"]
        lp=min(len(prompt),len(full))
        labels=[-100]*lp+full[lp:]
        full=full[:MAXLEN];labels=labels[:MAXLEN]
        return {"input_ids":full,"labels":labels}

def collate(batch):
    mx=max(len(b["input_ids"]) for b in batch)
    pid=tok.pad_token_id
    ids=[b["input_ids"]+[pid]*(mx-len(b["input_ids"])) for b in batch]
    lbl=[b["labels"]+[-100]*(mx-len(b["labels"])) for b in batch]
    att=[[1]*len(b["input_ids"])+[0]*(mx-len(b["input_ids"])) for b in batch]
    return {"input_ids":torch.tensor(ids),"labels":torch.tensor(lbl),"attention_mask":torch.tensor(att)}

model=AutoModelForImageTextToText.from_pretrained(MODEL,dtype=torch.bfloat16)
model.config.use_cache=False
nfz=0
for n,p in model.named_parameters():
    if any(k in n.lower() for k in ["visual","vision","image","patch_embed","merger"]):
        p.requires_grad=False;nfz+=1
print(f"frozen {nfz} vision params; trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e9:.2f}B",flush=True)
if hasattr(model,"gradient_checkpointing_enable"):model.gradient_checkpointing_enable()

ds=DS(DATA,limit=64 if SMOKE else None)
args=TrainingArguments(
    output_dir=OUT, per_device_train_batch_size=1, gradient_accumulation_steps=16,
    num_train_epochs=1, learning_rate=1e-5, lr_scheduler_type="cosine",
    warmup_ratio=0.03, bf16=True, logging_steps=5, save_strategy="steps", save_steps=500,
    save_total_limit=2, report_to=("none" if SMOKE else "wandb"), run_name="qwen35_4b_cotsft",
    max_steps=(3 if SMOKE else -1),
    gradient_checkpointing=True, dataloader_num_workers=4,
    deepspeed=(None if SMOKE else DS_CFG))
tr=Trainer(model=model,args=args,train_dataset=ds,data_collator=collate)
_resume=os.environ.get("RESUME")
tr.train(resume_from_checkpoint=(_resume if _resume and _resume!="1" else (True if _resume=="1" else None)))
if not SMOKE:
    tr.save_model(OUT);tok.save_pretrained(OUT)
    print("SAVED",OUT)
print("DONE_TRAIN")
