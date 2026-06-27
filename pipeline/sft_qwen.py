"""Text-only CoT-SFT of Qwen/Qwen3.5-2B (VLM text backbone) with plain HF Trainer.
Completion-only loss (mask prompt). Usage:
  torchrun --nproc_per_node=3 sft_qwen.py            # full train
  python sft_qwen.py --smoke                          # 1-GPU smoke test (few steps)
"""
import os,sys,json,torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForImageTextToText, Trainer, TrainingArguments
os.environ.setdefault("HF_HOME","/data1/jiahui/hf_cache")
MODEL="Qwen/Qwen3.5-2B"
DATA="/data1/jiahui/MedTVT-R1_repo/ceiling/sft_train.jsonl"
OUT="/data1/jiahui/MedTVT-R1_repo/ceiling/qwen35_2b_cotsft"
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
        # 按 assistant 标记切分(避免Qwen3.5模板注入空<think>导致的mask错位)
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
# 纯文本SFT: 冻结vision塔(避免DDP未用参数错+省显存), 只训语言主干
nfz=0
for n,p in model.named_parameters():
    if any(k in n.lower() for k in ["visual","vision","image","patch_embed","merger"]):
        p.requires_grad=False;nfz+=1
print(f"frozen {nfz} vision params; trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e9:.2f}B")
if hasattr(model,"gradient_checkpointing_enable"):model.gradient_checkpointing_enable()

ds=DS(DATA,limit=64 if SMOKE else None)
args=TrainingArguments(
    output_dir=OUT, per_device_train_batch_size=2, gradient_accumulation_steps=8,
    num_train_epochs=1, learning_rate=1e-5, lr_scheduler_type="cosine",
    warmup_ratio=0.03, bf16=True, logging_steps=5, save_strategy="steps", save_steps=500,
    save_total_limit=2, report_to=("none" if SMOKE else "wandb"), run_name="qwen35_2b_cotsft",
    max_steps=(3 if SMOKE else -1),
    gradient_checkpointing=True, dataloader_num_workers=4, ddp_find_unused_parameters=False)
tr=Trainer(model=model,args=args,train_dataset=ds,data_collator=collate)
_resume=os.environ.get("RESUME")  # 路径 或 "1"=自动找最新
tr.train(resume_from_checkpoint=(_resume if _resume and _resume!="1" else (True if _resume=="1" else None)))
if not SMOKE:
    tr.save_model(OUT);tok.save_pretrained(OUT)
    print("SAVED",OUT)
print("DONE_TRAIN")
