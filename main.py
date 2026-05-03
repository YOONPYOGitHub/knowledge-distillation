"""전체 파이프라인 실행 — YAML config 파일 기반

사용법:
  uv run python main.py configs/exp02_local_longer.yaml
  uv run python main.py configs/exp02_local_longer.yaml --step distill
  uv run python main.py configs/exp02_local_longer.yaml --step evaluate
"""

import argparse
import sys

from src.config import from_yaml, local_config
from src.train_teacher import train_teacher
from src.distill import distill
from src.train_baseline import train_baseline
from src.evaluate import evaluate_all
from src.compare import compare


STEPS = ["train_teacher", "distill", "baseline", "evaluate", "compare"]


def run_pipeline(config, steps=None):
    """파이프라인 실행. steps가 None이면 전체 실행."""
    steps = steps or STEPS

    if "train_teacher" in steps:
        print("\n" + "=" * 50)
        print("STEP 1/5: Teacher Fine-tuning")
        print("=" * 50)
        train_teacher(config)
        # teacher_checkpoint 자동 설정 (이후 distill에서 사용)
        if not config.teacher_checkpoint:
            config.teacher_checkpoint = str(
                config.checkpoint_dir / "teacher_ft_best.pt"
            )

    if "distill" in steps:
        print("\n" + "=" * 50)
        print("STEP 2/5: Knowledge Distillation")
        print("=" * 50)
        distill(config)

    if "baseline" in steps:
        print("\n" + "=" * 50)
        print("STEP 3/5: Baseline Fine-tuning")
        print("=" * 50)
        train_baseline(config)

    if "evaluate" in steps:
        print("\n" + "=" * 50)
        print("STEP 4/5: Evaluation")
        print("=" * 50)
        evaluate_all(config)

    if "compare" in steps:
        print("\n" + "=" * 50)
        print("STEP 5/5: Comparison")
        print("=" * 50)
        compare(config)

    print("\n✅ 파이프라인 완료!")
    print(f"   결과: {config.log_dir}")
    print(f"   차트: {config.figure_dir}")


def main():
    parser = argparse.ArgumentParser(description="KD 실험 파이프라인")
    parser.add_argument(
        "config",
        nargs="?",
        help="YAML config 파일 경로 (예: configs/exp02_local_longer.yaml)",
    )
    parser.add_argument(
        "--step",
        choices=STEPS,
        action="append",
        help="특정 단계만 실행 (여러 번 지정 가능: --step distill --step evaluate)",
    )
    args = parser.parse_args()

    # config 로드
    if args.config:
        print(f"📄 Config: {args.config}")
        config = from_yaml(args.config)
    else:
        print("📄 Config: local_config() (기본값)")
        config = local_config()

    # 설정 출력
    print(f"   Teacher:  {config.teacher_model}")
    print(f"   Student:  {config.student_model}")
    print(f"   T={config.temperature}, α={config.alpha}, epochs={config.epochs}")
    print(f"   Teacher FT: epochs={config.teacher_epochs}, lr={config.teacher_learning_rate}")
    if config.teacher_checkpoint:
        print(f"   Teacher checkpoint: {config.teacher_checkpoint}")
    print(f"   seq={config.max_seq_length}, batch={config.batch_size}")
    print(f"   Device:   {config.device}")
    print(f"   Run ID:   {config.run_id}")

    run_pipeline(config, steps=args.step)


if __name__ == "__main__":
    main()
