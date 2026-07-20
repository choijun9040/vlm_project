"""
Stage 2: 위험도 자동 태깅
==========================
Teacher VLM(Fine-tuned Qwen2.5-VL-7B)을 사용하여
DriveLM 키프레임 4,072개에 위험도(1~5)를 자동 부여.

실행:
    python scripts/generate_hazard_labels.py

출력:
    data/hazard_labels.json  ← {frame_token: hazard_score} 딕셔너리
"""

import json
import re
import time
from pathlib import Path
from collections import Counter

import torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from peft import PeftModel


# =============================================================================
# 설정
# =============================================================================

CONFIG = {
    # Teacher 체크포인트 (Stage 1 완료 결과)
    "base_model":      "Qwen/Qwen2.5-VL-7B-Instruct",
    "lora_checkpoint": "checkpoints/teacher_lora/epoch_1",

    # 데이터
    "drivelm_json":    "data/QA_dataset_nus/v1_0_train_nus.json",
    "output_path":     "data/hazard_labels.json",

    # 생성 설정
    "max_new_tokens":  128,
    "temperature":     0.1,    # 낮을수록 결정적 출력
    "use_camera":      "CAM_FRONT",

    # 검증
    "manual_verify_count": 20,  # 수동 검증할 샘플 수 (로그에 출력)
}

# 위험도 판단 프롬프트
HAZARD_PROMPT = """You are an autonomous driving safety expert analyzing a front camera image.

Rate the hazard level of this driving scene from 1 to 5:
1: Very safe - clear road, no obstacles, normal conditions
2: Low risk - normal traffic, predictable environment
3: Moderate - intersections, lane changes, moderate traffic
4: High risk - pedestrians near road, sudden stops needed, complex situation
5: Very dangerous - imminent collision risk, emergency situation

Respond ONLY in this JSON format:
{"score": <1-5>, "reason": "<brief one-sentence reason>"}"""


# =============================================================================
# 메인 함수
# =============================================================================

def generate_hazard_labels(config: dict):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # =========================================
    # 1. Teacher 모델 로드 (LoRA 체크포인트)
    # =========================================
    print(f"\n모델 로드: {config['base_model']}")
    print(f"LoRA 체크포인트: {config['lora_checkpoint']}")

    processor = AutoProcessor.from_pretrained(
        config["base_model"],
        max_pixels=256 * 28 * 28,
        min_pixels=64  * 28 * 28,
    )

    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        config["base_model"],
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    model = PeftModel.from_pretrained(base_model, config["lora_checkpoint"])
    model.eval()
    print("모델 로드 완료")

    # =========================================
    # 2. DriveLM 키프레임 목록 구성
    # =========================================
    json_path = Path(config["drivelm_json"])
    json_dir  = json_path.parent

    with open(json_path) as f:
        drivelm = json.load(f)

    # frame_token → image_path 매핑 (중복 제거)
    frame_to_image = {}
    for scene_token, scene_data in drivelm.items():
        for frame_token, frame_data in scene_data["key_frames"].items():
            raw_paths = frame_data.get("image_paths", {})
            cam_path  = raw_paths.get(config["use_camera"], "")
            if cam_path:
                abs_path = (json_dir / cam_path).resolve()
                if abs_path.exists():
                    frame_to_image[frame_token] = str(abs_path)

    total_frames = len(frame_to_image)
    print(f"\n총 키프레임 수: {total_frames}")

    # =========================================
    # 3. 위험도 자동 태깅
    # =========================================
    hazard_labels  = {}
    failed_frames  = []
    manual_samples = []  # 수동 검증용 샘플

    print("\n위험도 태깅 시작...")
    print("=" * 60)

    start_time = time.time()

    for i, (frame_token, img_path) in enumerate(frame_to_image.items()):

        try:
            # 이미지 로드
            image = Image.open(img_path).convert("RGB")

            # chat template 구성
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image",  "image": img_path},
                        {"type": "text",   "text": HAZARD_PROMPT},
                    ],
                }
            ]
            text = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            inputs = processor(
                text=[text],
                images=[image],
                return_tensors="pt",
                padding=False,
            ).to(device)

            # 생성
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=config["max_new_tokens"],
                    temperature=config["temperature"],
                    do_sample=False,
                )

            # 입력 토큰 제외하고 생성 부분만 디코딩
            input_len = inputs["input_ids"].shape[1]
            generated = processor.tokenizer.decode(
                output_ids[0][input_len:],
                skip_special_tokens=True,
            )

            # JSON 파싱
            score, reason = parse_hazard_response(generated)
            hazard_labels[frame_token] = float(score)

            # 수동 검증용 샘플 수집
            if len(manual_samples) < config["manual_verify_count"]:
                manual_samples.append({
                    "frame_token": frame_token,
                    "img_path":    img_path,
                    "score":       score,
                    "reason":      reason,
                    "raw_output":  generated,
                })

        except Exception as e:
            print(f"  [ERROR] frame {frame_token}: {e}")
            hazard_labels[frame_token] = 1.0  # 실패 시 기본값
            failed_frames.append(frame_token)

        # 진행 상황 로그
        if (i + 1) % 100 == 0:
            elapsed   = time.time() - start_time
            remaining = elapsed / (i + 1) * (total_frames - i - 1)
            print(
                f"  [{i+1}/{total_frames}] "
                f"경과: {elapsed/3600:.1f}h | "
                f"예상 남은 시간: {remaining/3600:.1f}h"
            )

    # =========================================
    # 4. 결과 저장
    # =========================================
    output_path = Path(config["output_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(hazard_labels, f, indent=2)

    print(f"\n위험도 라벨 저장: {output_path}")

    # =========================================
    # 5. 통계 출력
    # =========================================
    scores = [int(v) for v in hazard_labels.values()]
    dist   = Counter(scores)

    print("\n" + "=" * 60)
    print("위험도 분포")
    print("=" * 60)
    for score in sorted(dist.keys()):
        count = dist[score]
        pct   = count / total_frames * 100
        bar   = "█" * int(pct / 2)
        label = {1: "매우 안전", 2: "낮은 위험", 3: "보통", 4: "높은 위험", 5: "매우 위험"}
        print(f"  {score}점 ({label[score]:8s}): {count:4d}개 ({pct:5.1f}%) {bar}")

    print(f"\n실패한 프레임: {len(failed_frames)}개")
    if failed_frames:
        print(f"  {failed_frames[:5]}...")

    # =========================================
    # 6. 수동 검증용 샘플 출력
    # =========================================
    print("\n" + "=" * 60)
    print(f"수동 검증 샘플 (상위 {len(manual_samples)}개)")
    print("=" * 60)
    for j, sample in enumerate(manual_samples[:10]):
        print(f"\n[{j+1}] frame: {sample['frame_token'][:8]}...")
        print(f"     이미지: {Path(sample['img_path']).name}")
        print(f"     점수:   {sample['score']}점")
        print(f"     이유:   {sample['reason']}")

    print(f"\n[완료] 총 {total_frames}개 키프레임 태깅 완료")
    return hazard_labels


def parse_hazard_response(response: str) -> tuple:
    """
    Teacher 모델의 응답에서 위험도 점수와 이유를 파싱.

    Returns:
        (score: int, reason: str)
    """
    # JSON 파싱 시도
    try:
        # JSON 블록 추출
        json_match = re.search(r'\{[^}]+\}', response, re.DOTALL)
        if json_match:
            data   = json.loads(json_match.group())
            score  = int(data.get("score", 1))
            reason = data.get("reason", "")
            score  = max(1, min(5, score))  # 1~5 범위 클램핑
            return score, reason
    except Exception:
        pass

    # JSON 파싱 실패 시 숫자만 추출
    numbers = re.findall(r'\b[1-5]\b', response)
    if numbers:
        score = int(numbers[0])
        return score, response[:100]

    # 완전 실패 시 기본값
    return 1, "parsing failed"


# =============================================================================
# 실행
# =============================================================================

if __name__ == "__main__":
    hazard_labels = generate_hazard_labels(CONFIG)
    print(f"\n최종 라벨 수: {len(hazard_labels)}개")