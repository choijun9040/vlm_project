"""
Teacher LoRA Fine-tuning — Qwen2.5-VL-7B
==========================================
DriveLM + NuScenes-QA 데이터셋으로 Qwen2.5-VL-7B를 LoRA Fine-tuning.

실행:
    python scripts/train_teacher.py

체크포인트:
    checkpoints/teacher_lora/epoch_1/
    checkpoints/teacher_lora/epoch_2/
    ...
"""

import json
import os
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from peft import LoraConfig, get_peft_model, TaskType
from accelerate import Accelerator

# 같은 scripts/ 폴더의 dataloader.py에서 import
import sys
sys.path.append(str(Path(__file__).parent))
from dataloader import create_unified_dataloader, build_token_to_images

from datetime import datetime

# =============================================================================
# 설정
# =============================================================================

CONFIG = {
    # 모델
    "model_name":    "Qwen/Qwen2.5-VL-7B-Instruct",
    "output_dir":    "checkpoints/teacher_lora",

    # LoRA
    "lora_rank":     16,
    "lora_alpha":    32,
    "lora_dropout":  0.05,

    # 학습
    "num_epochs":        1,
    "batch_size":        4,
    "grad_accum_steps":  4,       # effective batch = 4 × 4 = 16
    "learning_rate":     2e-5,
    "warmup_ratio":      0.03,
    "max_input_length":  512,
    "max_label_length":  128,
    "drivelm_ratio":     0.4,
    "num_workers":       4,

    # 데이터
    "drivelm_json":    "data/QA_dataset_nus/v1_0_train_nus.json",
    "nuscenesqa_json": "data/nuscenes_qa/NuScenes_train_questions.json",

    # 로깅
    "log_every":   50,    # steps
    "save_every":  1,     # epochs
}


# =============================================================================
# 1. Processor를 사용하는 DataLoader 구성
# =============================================================================

def build_dataloader(processor, config: dict) -> DataLoader:
    """
    dataloader.py의 create_unified_dataloader를 processor와 함께 호출.
    processor가 있으면 tensor 모드로 동작.
    """
    return create_unified_dataloader(
        drivelm_json=config["drivelm_json"],
        nuscenesqa_json=config["nuscenesqa_json"],
        processor=processor,
        drivelm_ratio=config["drivelm_ratio"],
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
    )


# =============================================================================
# 2. 모델 및 LoRA 설정
# =============================================================================

def build_model_and_processor(config: dict):
    """
    Qwen2.5-VL-7B 로드 후 LoRA 적용.
    """
    print(f"\n모델 로드: {config['model_name']}")

    processor = AutoProcessor.from_pretrained(
        config["model_name"],
        max_pixels=256 * 28 * 28,   # visual token 수 절감 (OOM 방지)
        min_pixels=64  * 28 * 28,
    )

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        config["model_name"],
        torch_dtype=torch.bfloat16,   # A100은 bfloat16 네이티브 지원
        device_map="auto",
    )

    # LoRA 설정
    # target_modules: Qwen2.5-VL의 attention projection 레이어
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config["lora_rank"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"],
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",  # attention
            "gate_proj", "up_proj", "down_proj",       # FFN
        ],
        bias="none",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    # 예: trainable params: 83,886,080 || all params: 8,286,371,840 || trainable%: 1.01

    return model, processor


# =============================================================================
# 3. 학습 루프
# =============================================================================

def train(config: dict):

    # Accelerator 초기화 (단일 GPU)
    accelerator = Accelerator(
        gradient_accumulation_steps=config["grad_accum_steps"],
        mixed_precision="bf16",
        log_with=None,
    )

    # 모델 + processor 빌드
    model, processor = build_model_and_processor(config)

    # DataLoader 빌드
    print("\nDataLoader 구성 중...")
    dataloader = build_dataloader(processor, config)

    # Optimizer
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config["learning_rate"],
        weight_decay=0.01,
        betas=(0.9, 0.999),
    )

    # LR Scheduler — Warmup + Cosine
    total_steps   = math.ceil(len(dataloader) / config["grad_accum_steps"]) \
                    * config["num_epochs"]
    warmup_steps  = int(total_steps * config["warmup_ratio"])

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return current_step / max(1, warmup_steps)
        progress = (current_step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Accelerator로 wrapping
    model, optimizer, dataloader, scheduler = accelerator.prepare(
        model, optimizer, dataloader, scheduler
    )

    # 출력 디렉토리
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # 학습 시작
    print("\n" + "=" * 60)
    print("Teacher LoRA Fine-tuning 시작")
    print(f"  epochs:         {config['num_epochs']}")
    print(f"  batch_size:     {config['batch_size']}")
    print(f"  grad_accum:     {config['grad_accum_steps']}")
    print(f"  effective_batch:{config['batch_size'] * config['grad_accum_steps']}")
    print(f"  total_steps:    {total_steps}")
    print(f"  warmup_steps:   {warmup_steps}")
    print(f"  learning_rate:  {config['learning_rate']}")
    print("=" * 60 + "\n")

    global_step = 0

    for epoch in range(1, config["num_epochs"] + 1):
        model.train()

        epoch_loss  = 0.0
        epoch_steps = 0

        for step, batch in enumerate(dataloader):

            with accelerator.accumulate(model):

                # forward
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    pixel_values=batch["pixel_values"],
                    image_grid_thw=batch["image_grid_thw"],
                    labels=batch["labels"],
                )

                loss = outputs.loss

                # backward
                accelerator.backward(loss)

                # gradient clipping
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            epoch_loss  += loss.item()
            epoch_steps += 1

            # sync 시점에서만 global_step 증가
            if accelerator.sync_gradients:
                global_step += 1

                if global_step % config["log_every"] == 0:
                    avg_loss = epoch_loss / epoch_steps
                    lr_now   = scheduler.get_last_lr()[0]
                    print(
                        f"  [{datetime.now().strftime('%H:%M:%S')}] "
                        f"  [Epoch {epoch}/{config['num_epochs']}] "
                        f"step {global_step}/{total_steps} | "
                        f"loss: {avg_loss:.4f} | "
                        f"lr: {lr_now:.2e} | "
                        f"source: DriveLM {batch['source'].count('drivelm')}"
                        f"/{config['batch_size']}"
                    )

        # Epoch 종료 — 평균 손실 출력
        avg_epoch_loss = epoch_loss / max(1, epoch_steps)
        print(f"\n[Epoch {epoch} 완료] avg_loss: {avg_epoch_loss:.4f}\n")

        # 체크포인트 저장
        if epoch % config["save_every"] == 0:
            save_path = output_dir / f"epoch_{epoch}"
            save_path.mkdir(parents=True, exist_ok=True)

            # LoRA adapter만 저장 (전체 모델 대신 ~300MB)
            unwrapped = accelerator.unwrap_model(model)
            unwrapped.save_pretrained(save_path)
            processor.save_pretrained(save_path)

            # 학습 상태 저장
            torch.save({
                "epoch":       epoch,
                "global_step": global_step,
                "loss":        avg_epoch_loss,
                "config":      config,
            }, save_path / "training_state.pt")

            print(f"  체크포인트 저장: {save_path}")

    last_epoch = config["num_epochs"]
    last_ckpt = output_dir / ("epoch_" + str(last_epoch))
    print("\nTeacher LoRA Fine-tuning 완료!")
    print("최종 체크포인트: " + str(last_ckpt))


# =============================================================================
# 4. 실행
# =============================================================================

if __name__ == "__main__":
    train(CONFIG)