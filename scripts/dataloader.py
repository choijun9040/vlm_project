"""
통합 DataLoader — DriveLM + NuScenes-QA
=========================================
실제 확인된 JSON 구조 기반:

DriveLM:
  - frame["key_object_infos"], frame["QA"], frame["image_paths"]
  - frame["QA"].keys() = ['perception', 'prediction', 'planning', 'behavior']
  - image_paths: {'CAM_FRONT': '../nuscenes/samples/CAM_FRONT/xxx.jpg', ...}

NuScenes-QA:
  - qa_data["info"], qa_data["questions"]
  - questions[i].keys() = ['split', 'sample_token', 'question', 'answer',
                            'num_hop', 'template_type']
  - 이미지: token_to_images[sample_token]["CAM_FRONT"] 로 연결
"""

import json
from pathlib import Path
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset, WeightedRandomSampler


# =============================================================================
# 1. DriveLM Dataset
# =============================================================================

class DriveLMDataset(Dataset):
    """
    DriveLM-nuScenes 데이터셋.

    JSON 구조:
      data[scene_token]["key_frames"][frame_token] = {
          "key_object_infos": {...},
          "QA": {
              "perception": [{"Q":..., "A":..., "C":None, ...}, ...],
              "prediction":  [...],
              "planning":    [...],
              "behavior":    [...],
          },
          "image_paths": {
              "CAM_FRONT":       "../nuscenes/samples/CAM_FRONT/xxx.jpg",
              "CAM_FRONT_LEFT":  "../nuscenes/samples/CAM_FRONT_LEFT/xxx.jpg",
              ...
          }
      }
    """

    TASKS = ["perception", "prediction", "planning", "behavior"]

    def __init__(
        self,
        json_path: str,
        hazard_labels: dict = None,  # {frame_token: score} Stage 2 이후 사용
        processor=None,
        use_camera: str = "CAM_FRONT",  # 단일 뷰 사용 시
    ):
        self.json_path = Path(json_path)
        self.json_dir  = self.json_path.parent   # data/QA_dataset_nus/
        self.hazard_labels = hazard_labels or {}
        self.processor = processor
        self.use_camera = use_camera

        # JSON 로드
        with open(self.json_path) as f:
            raw = json.load(f)

        # (frame_token, task, qa_index) 단위로 샘플 목록 구성
        self.samples = []
        for scene_token, scene_data in raw.items():
            for frame_token, frame_data in scene_data["key_frames"].items():
                # 이미지 경로 절대경로 변환
                image_paths = {
                    cam: str((self.json_dir / rel).resolve())
                    for cam, rel in frame_data.get("image_paths", {}).items()
                }

                for task in self.TASKS:
                    for qa in frame_data["QA"].get(task, []):
                        self.samples.append({
                            "frame_token": frame_token,
                            "question":    qa["Q"],
                            "answer":      qa["A"],
                            "task":        task,
                            "image_paths": image_paths,
                        })

        print(f"[DriveLMDataset] 총 샘플 수: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]

        # 이미지 로드 (CAM_FRONT 단일 뷰)
        img_path = s["image_paths"].get(self.use_camera, "")
        image = Image.open(img_path).convert("RGB") if img_path else None

        # 위험도 점수 (Stage 2 이후 hazard_labels 있으면 사용, 없으면 1.0)
        hazard_score = float(
            self.hazard_labels.get(s["frame_token"], 1.0)
        )

        # Processor 있으면 토큰화, 없으면 raw 반환
        if self.processor is not None:
            messages_q = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img_path},
                        {"type": "text",  "text": s["question"]},
                    ],
                }
            ]
            # 질문 부분 텍스트 (답변 없이)
            text_q = self.processor.apply_chat_template(
                messages_q,
                tokenize=False,
                add_generation_prompt=True,
            )
            # 질문 + 답변 full text
            text_full = text_q + s["answer"]

            # full sequence 토큰화 (이미지 포함, 1번만 호출)
            inputs = self.processor(
                text=[text_full],
                images=[image],
                return_tensors="pt",
                padding="max_length",
                max_length=640,
                truncation=True,
            )

            # q_len 계산 — 이미지 없이 텍스트만 토큰화 (빠름)
            q_text_ids = self.processor.tokenizer(
                text_q,
                return_tensors="pt",
                padding=False,
                add_special_tokens=True,
            ).input_ids
            q_text_len = q_text_ids.shape[1]

            # visual token 수 계산 (image_pad 토큰 개수)
            image_token_id = self.processor.tokenizer.convert_tokens_to_ids(
                "<|image_pad|>"
            )
            visual_len = (
                inputs["input_ids"][0] == image_token_id
            ).sum().item()

            # 최종 q_len = visual tokens + 텍스트 질문 tokens
            q_len = visual_len + q_text_len - 1

            # labels = input_ids 복사 후 질문 부분 -100 마스킹
            labels = inputs["input_ids"].squeeze(0).clone()
            labels[:q_len] = -100
            labels[labels == self.processor.tokenizer.pad_token_id] = -100

            return {
                "input_ids":       inputs["input_ids"].squeeze(0),
                "attention_mask":  inputs["attention_mask"].squeeze(0),
                "pixel_values":    inputs["pixel_values"].squeeze(0),
                "image_grid_thw":  inputs["image_grid_thw"].squeeze(0),
                "labels":          labels,
                "hazard_score":    torch.tensor(hazard_score, dtype=torch.float32),
                "source":          "drivelm",
                "task":            s["task"],
                "frame_token":     s["frame_token"],
            }

        # Processor 없음 → 구조 확인용 raw 반환
        return {
            "question":    s["question"],
            "answer":      s["answer"],
            "task":        s["task"],
            "image_path":  img_path,
            "hazard_score": hazard_score,
            "source":      "drivelm",
            "frame_token": s["frame_token"],
        }


# =============================================================================
# 2. NuScenes-QA Dataset
# =============================================================================

class NuScenesQADataset(Dataset):
    """
    NuScenes-QA 데이터셋.

    JSON 구조:
      qa_data["info"]      = {'split': 'train', 'version': '1.0', ...}
      qa_data["questions"] = [
          {
              "split":         "train",
              "sample_token":  "e93e98b63d3b40209056d129dc53ceee",
              "question":      "There is a car to the back right...",
              "answer":        "moving",
              "num_hop":       1,
              "template_type": "status",
          },
          ...
      ]

    이미지 연결:
      DriveLM의 token_to_images 딕셔너리로 연결
      (nuScenes 메타데이터 없이 DriveLM 서브셋 이미지만 사용)
    """

    def __init__(
        self,
        json_path: str,
        token_to_images: dict,  # {sample_token: {cam: abs_path}}
        processor=None,
        use_camera: str = "CAM_FRONT",
    ):
        self.token_to_images = token_to_images
        self.processor = processor
        self.use_camera = use_camera

        with open(json_path) as f:
            qa_data = json.load(f)

        all_questions = qa_data["questions"]

        # DriveLM 이미지 서브셋과 매핑 가능한 샘플만 필터링
        self.samples = [
            q for q in all_questions
            if q["sample_token"] in token_to_images
        ]

        print(f"[NuScenesQADataset] 전체: {len(all_questions)}, "
              f"매핑 가능: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]

        img_path = str(
            self.token_to_images[s["sample_token"]].get(self.use_camera, "")
        )
        image = Image.open(img_path).convert("RGB") if img_path else None

        if self.processor is not None:
            messages_q = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img_path},
                        {"type": "text",  "text": s["question"]},
                    ],
                }
            ]
            text_q = self.processor.apply_chat_template(
                messages_q,
                tokenize=False,
                add_generation_prompt=True,
            )
            text_full = text_q + s["answer"]

            # full sequence 토큰화 (이미지 포함, 1번만 호출)
            inputs = self.processor(
                text=[text_full],
                images=[image],
                return_tensors="pt",
                padding="max_length",
                max_length=640,
                truncation=True,
            )

            # q_len 계산 — 이미지 없이 텍스트만 토큰화 (빠름)
            q_text_ids = self.processor.tokenizer(
                text_q,
                return_tensors="pt",
                padding=False,
                add_special_tokens=True,
            ).input_ids
            q_text_len = q_text_ids.shape[1]

            # visual token 수 계산
            image_token_id = self.processor.tokenizer.convert_tokens_to_ids(
                "<|image_pad|>"
            )
            visual_len = (
                inputs["input_ids"][0] == image_token_id
            ).sum().item()

            q_len = visual_len + q_text_len - 1

            labels = inputs["input_ids"].squeeze(0).clone()
            labels[:q_len] = -100
            labels[labels == self.processor.tokenizer.pad_token_id] = -100

            return {
                "input_ids":       inputs["input_ids"].squeeze(0),
                "attention_mask":  inputs["attention_mask"].squeeze(0),
                "pixel_values":    inputs["pixel_values"].squeeze(0),
                "image_grid_thw":  inputs["image_grid_thw"].squeeze(0),
                "labels":          labels,
                "hazard_score":    torch.tensor(1.0, dtype=torch.float32),
                "source":          "nuscenes_qa",
                "task":            s["template_type"],
                "frame_token":     s["sample_token"],
            }

        return {
            "question":    s["question"],
            "answer":      s["answer"],
            "task":        s["template_type"],
            "image_path":  img_path,
            "hazard_score": 1.0,
            "source":      "nuscenes_qa",
            "frame_token": s["sample_token"],
        }


# =============================================================================
# 3. token_to_images 딕셔너리 생성 헬퍼
# =============================================================================

def build_token_to_images(drivelm_json_path: str) -> dict:
    """
    DriveLM JSON에서 {frame_token: {cam: abs_path}} 딕셔너리 생성.
    NuScenes-QA가 이 딕셔너리로 이미지 경로를 연결.
    """
    json_path = Path(drivelm_json_path)
    json_dir  = json_path.parent

    with open(json_path) as f:
        drivelm = json.load(f)

    token_to_images = {}
    for scene_token, scene_data in drivelm.items():
        for frame_token, frame_data in scene_data["key_frames"].items():
            raw_paths = frame_data.get("image_paths", {})
            token_to_images[frame_token] = {
                cam: (json_dir / rel).resolve()
                for cam, rel in raw_paths.items()
            }

    print(f"[build_token_to_images] 총 토큰 수: {len(token_to_images)}")
    return token_to_images


# =============================================================================
# 4. 통합 DataLoader 생성
# =============================================================================

def create_unified_dataloader(
    drivelm_json:    str,
    nuscenesqa_json: str,
    hazard_labels:   dict = None,
    processor=None,
    drivelm_ratio:   float = 0.4,   # 배치 내 DriveLM 비율
    batch_size:      int   = 8,
    num_workers:     int   = 4,
    use_camera:      str   = "CAM_FRONT",
) -> DataLoader:
    """
    DriveLM과 NuScenes-QA를 WeightedRandomSampler로 통합.

    DriveLM   : ~수만 개 QA 쌍
    NuScenes-QA: 54,607개 (매핑 가능 샘플)

    drivelm_ratio=0.4 → 배치의 40%가 DriveLM, 60%가 NuScenes-QA
    """
    # token_to_images 공유
    token_to_images = build_token_to_images(drivelm_json)

    # 각 데이터셋 생성
    ds_drivelm = DriveLMDataset(
        json_path=drivelm_json,
        hazard_labels=hazard_labels,
        processor=processor,
        use_camera=use_camera,
    )
    ds_nuscenesqa = NuScenesQADataset(
        json_path=nuscenesqa_json,
        token_to_images=token_to_images,
        processor=processor,
        use_camera=use_camera,
    )

    n_d = len(ds_drivelm)
    n_n = len(ds_nuscenesqa)

    # WeightedRandomSampler 가중치 계산
    # DriveLM 샘플당 가중치 = drivelm_ratio / n_d
    # NuScenes-QA 샘플당 가중치 = (1 - drivelm_ratio) / n_n
    w_d = drivelm_ratio       / n_d
    w_n = (1 - drivelm_ratio) / n_n

    weights = [w_d] * n_d + [w_n] * n_n

    combined = ConcatDataset([ds_drivelm, ds_nuscenesqa])
    sampler  = WeightedRandomSampler(
        weights=weights,
        num_samples=n_d + n_n,
        replacement=True,
    )

    dataloader = DataLoader(
        combined,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )

    print(f"\n[DataLoader 구성]")
    print(f"  DriveLM:     {n_d:,}개  (배치 내 목표 비율: {drivelm_ratio*100:.0f}%)")
    print(f"  NuScenes-QA: {n_n:,}개  (배치 내 목표 비율: {(1-drivelm_ratio)*100:.0f}%)")
    print(f"  batch_size:  {batch_size}")
    print(f"  전체 스텝/epoch: {(n_d + n_n) // batch_size:,}")

    return dataloader


# =============================================================================
# 5. Collate Function
# =============================================================================

def collate_fn(batch):
    """
    processor 없는 raw 모드(구조 확인용)와
    processor 있는 tensor 모드 모두 처리.
    """
    # tensor 필드와 string 필드 분리
    tensor_keys = ["input_ids", "attention_mask", "pixel_values",
                   "image_grid_thw", "labels"]
    string_keys = ["source", "task", "frame_token",
                   "question", "answer", "image_path"]

    collated = {}

    for key in tensor_keys:
        if key in batch[0]:
            collated[key] = torch.stack([item[key] for item in batch])

    # hazard_score는 float 또는 tensor 모두 처리
    if "hazard_score" in batch[0]:
        collated["hazard_score"] = torch.tensor(
            [item["hazard_score"] if not isinstance(item["hazard_score"], torch.Tensor)
             else item["hazard_score"].item()
             for item in batch],
            dtype=torch.float32,
        )

    for key in string_keys:
        if key in batch[0]:
            collated[key] = [item[key] for item in batch]

    return collated


# =============================================================================
# 6. 동작 확인 (processor 없는 raw 모드)
# =============================================================================

if __name__ == "__main__":

    DRIVELM_JSON    = "data/QA_dataset_nus/v1_0_train_nus.json"
    NUSCENESQA_JSON = "data/nuscenes_qa/NuScenes_train_questions.json"

    print("=" * 60)
    print("Step 1: 개별 데이터셋 구조 확인 (processor 없음)")
    print("=" * 60)

    # --- DriveLM 단독 확인 ---
    ds_d = DriveLMDataset(json_path=DRIVELM_JSON)
    sample_d = ds_d[0]
    print("\n[DriveLM 샘플]")
    for k, v in sample_d.items():
        print(f"  {k}: {v}")

    # --- NuScenes-QA 단독 확인 ---
    token_to_images = build_token_to_images(DRIVELM_JSON)
    ds_n = NuScenesQADataset(
        json_path=NUSCENESQA_JSON,
        token_to_images=token_to_images,
    )
    sample_n = ds_n[0]
    print("\n[NuScenes-QA 샘플]")
    for k, v in sample_n.items():
        print(f"  {k}: {v}")

    print("\n" + "=" * 60)
    print("Step 2: 통합 DataLoader 확인")
    print("=" * 60)

    dataloader = create_unified_dataloader(
        drivelm_json=DRIVELM_JSON,
        nuscenesqa_json=NUSCENESQA_JSON,
        batch_size=8,
        num_workers=0,  # 디버깅 시 0으로 설정
    )

    # 첫 번째 배치 확인
    batch = next(iter(dataloader))
    print("\n[첫 번째 배치]")
    print(f"  source 분포: {batch['source']}")
    print(f"  task   분포: {batch['task']}")
    print(f"  hazard_score: {batch['hazard_score']}")
    print(f"\n  질문 예시: {batch['question'][0]}")
    print(f"  답변 예시: {batch['answer'][0]}")

    # DriveLM / NuScenes-QA 비율 확인 (100 배치)
    print("\n" + "=" * 60)
    print("Step 3: 100 배치 샘플링 비율 확인")
    print("=" * 60)

    drivelm_count = 0
    total_count   = 0
    for i, batch in enumerate(dataloader):
        if i >= 100:
            break
        drivelm_count += batch["source"].count("drivelm")
        total_count   += len(batch["source"])

    print(f"  DriveLM 비율: {drivelm_count}/{total_count} "
          f"= {drivelm_count/total_count*100:.1f}% "
          f"(목표: 40%)")