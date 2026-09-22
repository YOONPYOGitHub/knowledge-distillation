"""Gemma 3 파이프라인 단계 실행기.

main.py 와 달리 두 가지를 더 한다.

1. 앞 단계가 실패해 체크포인트가 없으면 가능한 조건으로 낮춰서 진행한다
   (Teacher FT 어댑터 없음 -> pretrained Teacher, Student SFT 없음 -> pretrained Student).
2. evaluate 단계는 Teacher 를 양자화 없이 bf16 으로 두 장에 분할해 측정한다.
   KD 단계의 int8 Teacher 는 학습용 배치일 뿐, 보고하는 Teacher PPL 은 양자화 오차가 없어야 한다.

사용법:
    python scripts/gemma3_stage.py <config.yaml> <step>
"""

import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from main import run_pipeline
from src.config import from_yaml
from src.distributed import cleanup_distributed, is_main_process, setup_distributed


def _announce(message: str) -> None:
    if is_main_process():
        print(message, flush=True)


def build_config(config_path: str, step: str):
    if step in {"evaluate", "compare"}:
        # Teacher 는 bf16 분할로 측정한다. Student 는 cuda:0, Teacher 는 cuda:2/3 을 쓴다.
        return from_yaml(
            config_path,
            teacher_quantization="",
            teacher_devices=[],
            teacher_device_map="auto",
            teacher_max_memory={"2": "22GiB", "3": "22GiB"},
            device="cuda:0",
            num_workers=0,
            # 평가는 단일 프로세스다. global_batch_size 는 DDP rank 수와 묶여 있어
            # WORLD_SIZE=1 에서는 batch_size 와 맞지 않아 설정 검증에 걸린다.
            global_batch_size=0,
        )
    return from_yaml(config_path)


def apply_fallbacks(config, step: str) -> None:
    """앞 단계 산출물이 없으면 조건을 낮춰 진행한다."""
    if step in {"distill", "evaluate"} and config.teacher_checkpoint:
        if not Path(config.teacher_checkpoint).exists():
            _announce(
                f"⚠️  Teacher FT 어댑터 없음 → pretrained Teacher 로 진행합니다: "
                f"{config.teacher_checkpoint}"
            )
            config.teacher_checkpoint = ""

    if step == "distill" and config.distill_student_checkpoint:
        if not Path(config.distill_student_checkpoint).exists():
            _announce(
                f"⚠️  Student SFT 체크포인트 없음 → pretrained Student 에서 KD 를 시작합니다: "
                f"{config.distill_student_checkpoint}"
            )
            config.distill_student_checkpoint = ""


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2

    config_path, step = sys.argv[1], sys.argv[2]
    config = build_config(config_path, step)

    setup_distributed(config)
    try:  # noqa: PLR1702
        apply_fallbacks(config, step)
        if is_main_process():
            print(f"📄 {config_path} | step={step} | run_id={config.run_id}")
            print(f"   Teacher: {config.teacher_model} (quant={config.teacher_quantization or 'bf16'})")
            print(f"   Student: {config.student_model}")
            print(f"   kd_top_k={config.kd_top_k}, T={config.temperature}, α={config.alpha}, "
                  f"{config.kd_divergence}/{config.kd_reduction}")
        run_pipeline(config, [step])
    except BaseException:
        # process group 을 먼저 정리하면 torch 의 excepthook 이 rank 를 못 읽어
        # 원래 예외가 가려진다. 정리 전에 트레이스백을 남긴다.
        traceback.print_exc()
        raise
    finally:
        cleanup_distributed()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
