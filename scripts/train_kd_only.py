"""
KD only: 균등 가중 Knowledge Distillation
==========================================
L_total = L_task + lambda * L_hazard (w=1 균등 가중)

모든 샘플에 동일한 가중치를 부여하는 일반 KD.
ablation study에서 "Standard KD" baseline으로 사용.

본 연구 Ours(위험도 가중)와 비교하여
위험도 가중의 효과를 검증하는 실험.

실행:
    python scripts/train_kd_only.py

체크포인트:
    checkpoints/student_kd_only/epoch_1/
"""

import json
import math
import sys
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from accelerate import Accelerator

sys.path.append(str(Path(__file__).parent))
from dataloader import create_unified_dataloader


# =============================================================================
# 설정
# =============================================================================

CONFIG = {
    # 모델
    "teacher_base":       "Qwen/Qwen2.5-VL-7B-Instruct",
    "teacher_lora":       "checkpoints/teacher_lora/epoch_1",
    "student_base":       "Qwen/Qwen2.5-VL-3B-Instruct",
    "output_dir":         "checkpoints/student_kd_only",

    # Resume
    "resume_from":        None,

    # Student LoRA
    "lora_rank":          16,
    "lora_alpha":         32,
    "lora_dropout":       0.05,

    # 학습
    "num_epochs":         1,
    "batch_size":         2,
    "grad_accum_steps":   8,
    "learning_rate":      2e-5,
    "warmup_ratio":       0.05,

    # 손실 가중치
    "lambda_kd":          1.0,

    # KD 설정
    "kd_temperature":     4.0,

    # 데이터
    "drivelm_json":       "data/QA_dataset_nus/v1_0_train_nus.json",
    "nuscenesqa_json":    "data/nuscenes_qa/NuScenes_train_questions.json",
    "hazard_labels_path": "data/hazard_labels.json",
    "drivelm_ratio":      0.4,
    "num_workers":        4,

    # 로깅/저장
    "log_every":          50,
    "save_every_steps":   1000,
}


# =============================================================================
# 1. Loss 함수 — 균등 가중 KD (w=1)
# =============================================================================

class UniformKDLoss(nn.Module):
    """
    균등 가중 KD Loss.
    모든 샘플에 동일한 가중치 1.0 부여.
    위험도와 무관하게 Teacher를 모방.

    Ours(HazardWeightedKDLoss)와 비교하여
    위험도 가중의 효과를 측정하는 ablation용.
    """
    def __init__(self, temperature=4.0):
        super().__init__()
        self.T = temperature

    def forward(self, student_logits, teacher_logits, labels):
        T = self.T

        # vocab 크기 불일치 처리
        min_vocab = min(student_logits.shape[-1], teacher_logits.shape[-1])
        student_logits = student_logits[..., :min_vocab]
        teacher_logits = teacher_logits[..., :min_vocab]

        mask = (labels != -100).float()
        if mask.sum() == 0:
            return torch.tensor(0.0, device=student_logits.device)

        s_log  = F.log_softmax(student_logits / T, dim=-1)
        t_prob = F.softmax(teacher_logits / T, dim=-1)

        kl = F.kl_div(s_log, t_prob, reduction="none").sum(dim=-1)
        kl_per_sample = (kl * mask).sum(dim=-1) / (mask.sum(dim=-1) + 1e-8)

        # 모든 샘플 동일 가중치 (w=1)
        return kl_per_sample.mean() * (T ** 2)


# =============================================================================
# 2. 모델 빌드
# =============================================================================

def build_teacher(config):
    print(f"\n[Teacher] 로드: {config['teacher_base']}")

    processor = AutoProcessor.from_pretrained(
        config["teacher_base"],
        max_pixels=256 * 28 * 28,
        min_pixels=64  * 28 * 28,
    )
    base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        config["teacher_base"],
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    teacher = PeftModel.from_pretrained(base, config["teacher_lora"])
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    print("  Teacher frozen")
    return teacher, processor


def build_student(config):
    print("  Student processor 다운로드 중...")
    AutoProcessor.from_pretrained(
        config["student_base"],
        max_pixels=256 * 28 * 28,
        min_pixels=64  * 28 * 28,
    )
    print("  Student processor 준비 완료")

    resume_from = config.get("resume_from")
    ckpt_path   = Path(resume_from) if resume_from else None

    if ckpt_path and ckpt_path.exists() and (ckpt_path / "adapter_config.json").exists():
        print(f"\n[Student] 체크포인트에서 로드: {ckpt_path}")
        base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            config["student_base"],
            torch_dtype=torch.bfloat16,
        )
        student = PeftModel.from_pretrained(base, str(ckpt_path), is_trainable=True)
        print("  LoRA 어댑터 로드 완료")
    else:
        print(f"\n[Student] 처음부터 로드: {config['student_base']}")
        base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            config["student_base"],
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
        student = get_peft_model(base, lora_cfg)

    student.print_trainable_parameters()
    return student


# =============================================================================
# 3. 학습 루프
# =============================================================================

def train(config):

    accelerator = Accelerator(
        gradient_accumulation_steps=config["grad_accum_steps"],
        mixed_precision="bf16",
    )

    with open(config["hazard_labels_path"]) as f:
        hazard_labels = json.load(f)
    print(f"\n위험도 라벨 로드: {len(hazard_labels)}개 (샘플링 비율용)")

    teacher, processor = build_teacher(config)
    student = build_student(config)

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

    criterion_kd = UniformKDLoss(temperature=config["kd_temperature"])

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, student.parameters()),
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

    student, optimizer, dataloader, scheduler = accelerator.prepare(
        student, optimizer, dataloader, scheduler
    )

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("KD only: 균등 가중 Knowledge Distillation")
    print(f"  Teacher:        {config['teacher_base']} + LoRA")
    print(f"  Student:        {config['student_base']} + LoRA")
    print(f"  epochs:         {config['num_epochs']}")
    print(f"  batch_size:     {config['batch_size']}")
    print(f"  effective_batch:{config['batch_size'] * config['grad_accum_steps']}")
    print(f"  total_steps:    {total_steps}")
    print(f"  warmup_steps:   {warmup_steps}")
    print(f"  learning_rate:  {config['learning_rate']}")
    print(f"  loss:           L_task + {config['lambda_kd']} * L_kd (w=1 균등)")
    print(f"  kd_temperature: {config['kd_temperature']}")
    print("=" * 60 + "\n")

    global_step = 0

    for epoch in range(1, config["num_epochs"] + 1):
        student.train()
        epoch_losses = {"total": 0.0, "task": 0.0, "kd": 0.0}
        epoch_steps  = 0

        for step, batch in enumerate(dataloader):

            with accelerator.accumulate(student):

                input_ids      = batch["input_ids"]
                attention_mask = batch["attention_mask"]
                pixel_values   = batch["pixel_values"]
                image_grid_thw = batch["image_grid_thw"]
                labels         = batch["labels"]

                with torch.no_grad():
                    t_out = teacher(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        pixel_values=pixel_values,
                        image_grid_thw=image_grid_thw,
                    )

                s_out = student(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                    image_grid_thw=image_grid_thw,
                    labels=labels,
                )

                loss_task = s_out.loss
                loss_kd   = criterion_kd(
                    student_logits=s_out.logits,
                    teacher_logits=t_out.logits,
                    labels=labels,
                )
                loss_total = loss_task + config["lambda_kd"] * loss_kd

                accelerator.backward(loss_total)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(student.parameters(), 1.0)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            epoch_losses["total"] += loss_total.item()
            epoch_losses["task"]  += loss_task.item()
            epoch_losses["kd"]    += loss_kd.item()
            epoch_steps += 1

            if accelerator.sync_gradients:
                global_step += 1

                if global_step % config["log_every"] == 0:
                    avg    = {k: v / epoch_steps for k, v in epoch_losses.items()}
                    lr_now = scheduler.get_last_lr()[0]
                    drivelm_cnt = batch["source"].count("drivelm")
                    print(
                        f"  [{datetime.now().strftime('%H:%M:%S')}] "
                        f"[Epoch {epoch}/{config['num_epochs']}] "
                        f"step {global_step}/{total_steps} | "
                        f"loss {avg['total']:.4f} "
                        f"(task={avg['task']:.3f} kd={avg['kd']:.3f}) | "
                        f"lr {lr_now:.2e} | "
                        f"DriveLM {drivelm_cnt}/{config['batch_size']}"
                    )

                if global_step % config["save_every_steps"] == 0:
                    _save_checkpoint(
                        accelerator, student, processor,
                        output_dir, global_step, epoch,
                        epoch_losses, epoch_steps, config,
                    )

        avg = {k: v / max(1, epoch_steps) for k, v in epoch_losses.items()}
        print(
            f"\n[Epoch {epoch} 완료] "
            f"avg_loss={avg['total']:.4f} "
            f"(task={avg['task']:.3f} kd={avg['kd']:.3f})\n"
        )
        _save_checkpoint(
            accelerator, student, processor,
            output_dir, global_step, epoch,
            epoch_losses, epoch_steps, config,
            name=f"epoch_{epoch}",
        )

    print("\nKD only Fine-tuning 완료!")
    last_ckpt = output_dir / ("epoch_" + str(config["num_epochs"]))
    print("최종 체크포인트: " + str(last_ckpt))


# =============================================================================
# 4. 체크포인트 저장
# =============================================================================

def _save_checkpoint(
    accelerator, model, processor,
    output_dir, global_step, epoch,
    epoch_losses, epoch_steps, config,
    name=None,
):
    save_name = name or f"step_{global_step}"
    save_path = Path(output_dir) / save_name
    save_path.mkdir(parents=True, exist_ok=True)

    unwrapped = accelerator.unwrap_model(model)
    unwrapped.save_pretrained(save_path)
    processor.save_pretrained(save_path)

    avg = {k: v / max(1, epoch_steps) for k, v in epoch_losses.items()}
    torch.save({
        "global_step": global_step,
        "epoch":       epoch,
        "avg_loss":    avg,
        "config":      config,
    }, save_path / "training_state.pt")

    print(f"  체크포인트 저장: {save_path}")


# =============================================================================
# 실행
# =============================================================================

if __name__ == "__main__":
    train(CONFIG)