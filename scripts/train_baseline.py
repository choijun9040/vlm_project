"""
Baseline: Student Direct Fine-tuning (증류 없음)
=================================================
Qwen2.5-VL-3B를 Teacher 없이 DriveLM + NuScenes-QA로 직접 LoRA FT.

L_total = L_task (Cross-Entropy only)

ablation study에서 "No Distillation" baseline으로 사용.

실행:
    python scripts/train_baseline.py

체크포인트:
    checkpoints/student_baseline/epoch_1/
"""

import math
import sys
from datetime import datetime
from pathlib import Path

import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from peft import LoraConfig, get_peft_model, TaskType
from accelerate import Accelerator

sys.path.append(str(Path(__file__).parent))
from dataloader import create_unified_dataloader


# =============================================================================
# 설정
# =============================================================================

CONFIG = {
    # 모델
    "model_name":         "Qwen/Qwen2.5-VL-3B-Instruct",
    "output_dir":         "checkpoints/student_baseline",

    # Resume
    "resume_from":        None,  # 체크포인트 경로 또는 None

    # LoRA
    "lora_rank":          16,
    "lora_alpha":         32,
    "lora_dropout":       0.05,

    # 학습
    "num_epochs":         1,
    "batch_size":         4,      # Teacher 없으므로 배치 늘림
    "grad_accum_steps":   4,      # effective batch = 4 × 4 = 16
    "learning_rate":      2e-5,
    "warmup_ratio":       0.05,

    # 데이터
    "drivelm_json":       "data/QA_dataset_nus/v1_0_train_nus.json",
    "nuscenesqa_json":    "data/nuscenes_qa/NuScenes_train_questions.json",
    "hazard_labels_path": "data/hazard_labels.json",  # 샘플링 비율용으로만 사용
    "drivelm_ratio":      0.4,
    "num_workers":        4,

    # 로깅/저장
    "log_every":          50,
    "save_every_steps":   1000,
}


# =============================================================================
# 모델 빌드
# =============================================================================

def build_model(config):
    print(f"\n[Model] 로드: {config['model_name']}")

    processor = AutoProcessor.from_pretrained(
        config["model_name"],
        max_pixels=256 * 28 * 28,
        min_pixels=64  * 28 * 28,
    )

    resume_from = config.get("resume_from")
    ckpt_path   = Path(resume_from) if resume_from else None

    if ckpt_path and ckpt_path.exists() and (ckpt_path / "adapter_config.json").exists():
        from peft import PeftModel
        print(f"  체크포인트에서 로드: {ckpt_path}")
        base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            config["model_name"],
            torch_dtype=torch.bfloat16,
        )
        model = PeftModel.from_pretrained(base, str(ckpt_path), is_trainable=True)
        print("  LoRA 어댑터 로드 완료 (이어서 학습)")
    else:
        print("  처음부터 학습")
        base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            config["model_name"],
            torch_dtype=torch.bfloat16,
        )
        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=config["lora_rank"],
            lora_alpha=config["lora_alpha"],
            lora_dropout=config["lora_dropout"],
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            bias="none",
        )
        model = get_peft_model(base, lora_cfg)

    model.print_trainable_parameters()
    return model, processor


# =============================================================================
# 학습 루프
# =============================================================================

def train(config):

    accelerator = Accelerator(
        gradient_accumulation_steps=config["grad_accum_steps"],
        mixed_precision="bf16",
    )

    # hazard_labels는 DataLoader 샘플링 비율용으로만 사용
    import json
    with open(config["hazard_labels_path"]) as f:
        hazard_labels = json.load(f)

    model, processor = build_model(config)

    print("\nDataLoader 구성 중...")
    dataloader = create_unified_dataloader(
        drivelm_json=config["drivelm_json"],
        nuscenesqa_json=config["nuscenesqa_json"],
        hazard_labels=hazard_labels,
        processor=processor,
        drivelm_ratio=config["drivelm_ratio"],
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
    )

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config["learning_rate"],
        weight_decay=0.01,
    )

    total_steps  = math.ceil(len(dataloader) / config["grad_accum_steps"]) * config["num_epochs"]
    warmup_steps = int(total_steps * config["warmup_ratio"])

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    model, optimizer, dataloader, scheduler = accelerator.prepare(
        model, optimizer, dataloader, scheduler
    )

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    resume_info = config.get("resume_from") or "처음부터"

    print("\n" + "=" * 60)
    print("Baseline: Student Direct Fine-tuning")
    print(f"  Model:          {config['model_name']} + LoRA")
    print(f"  시작:           {resume_info}")
    print(f"  epochs:         {config['num_epochs']}")
    print(f"  batch_size:     {config['batch_size']}")
    print(f"  effective_batch:{config['batch_size'] * config['grad_accum_steps']}")
    print(f"  total_steps:    {total_steps}")
    print(f"  warmup_steps:   {warmup_steps}")
    print(f"  learning_rate:  {config['learning_rate']}")
    print(f"  loss:           L_task only (No Distillation)")
    print("=" * 60 + "\n")

    global_step = 0

    for epoch in range(1, config["num_epochs"] + 1):
        model.train()
        epoch_loss  = 0.0
        epoch_steps = 0

        for step, batch in enumerate(dataloader):

            with accelerator.accumulate(model):

                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    pixel_values=batch["pixel_values"],
                    image_grid_thw=batch["image_grid_thw"],
                    labels=batch["labels"],
                )

                loss = outputs.loss

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            epoch_loss  += loss.item()
            epoch_steps += 1

            if accelerator.sync_gradients:
                global_step += 1

                if global_step % config["log_every"] == 0:
                    avg_loss    = epoch_loss / epoch_steps
                    lr_now      = scheduler.get_last_lr()[0]
                    drivelm_cnt = batch["source"].count("drivelm")
                    print(
                        f"  [{datetime.now().strftime('%H:%M:%S')}] "
                        f"[Epoch {epoch}/{config['num_epochs']}] "
                        f"step {global_step}/{total_steps} | "
                        f"loss {avg_loss:.4f} | "
                        f"lr {lr_now:.2e} | "
                        f"DriveLM {drivelm_cnt}/{config['batch_size']}"
                    )

                if global_step % config["save_every_steps"] == 0:
                    _save_checkpoint(
                        accelerator, model, processor,
                        output_dir, global_step, epoch,
                        epoch_loss, epoch_steps, config,
                    )

        avg_loss = epoch_loss / max(1, epoch_steps)
        print(f"\n[Epoch {epoch} 완료] avg_loss={avg_loss:.4f}\n")

        _save_checkpoint(
            accelerator, model, processor,
            output_dir, global_step, epoch,
            epoch_loss, epoch_steps, config,
            name=f"epoch_{epoch}",
        )

    print("\nBaseline Fine-tuning 완료!")
    last_ckpt = output_dir / ("epoch_" + str(config["num_epochs"]))
    print("최종 체크포인트: " + str(last_ckpt))


# =============================================================================
# 체크포인트 저장
# =============================================================================

def _save_checkpoint(
    accelerator, model, processor,
    output_dir, global_step, epoch,
    epoch_loss, epoch_steps, config,
    name=None,
):
    save_name = name or f"step_{global_step}"
    save_path = Path(output_dir) / save_name
    save_path.mkdir(parents=True, exist_ok=True)

    unwrapped = accelerator.unwrap_model(model)
    unwrapped.save_pretrained(save_path)
    processor.save_pretrained(save_path)

    torch.save({
        "global_step": global_step,
        "epoch":       epoch,
        "avg_loss":    epoch_loss / max(1, epoch_steps),
        "config":      config,
    }, save_path / "training_state.pt")

    print(f"  체크포인트 저장: {save_path}")


# =============================================================================
# 실행
# =============================================================================

if __name__ == "__main__":
    train(CONFIG)