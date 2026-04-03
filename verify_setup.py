"""환경 검증 스크립트 — 프로젝트 실행 전 필수 의존성 및 디바이스 확인"""
import sys


def check_python():
    print(f"[1] Python: {sys.version}")
    assert sys.version_info >= (3, 10), "Python 3.10 이상 필요"
    print("    ✅ OK")


def check_pytorch():
    import torch

    print(f"\n[2] PyTorch: {torch.__version__}")
    print(f"    CUDA 사용 가능: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"    GPU: {torch.cuda.get_device_name(0)}")
        vram = torch.cuda.get_device_properties(0).total_mem / 1e9
        print(f"    VRAM: {vram:.1f} GB")
    print(f"    MPS 사용 가능: {torch.backends.mps.is_available()}")
    if torch.backends.mps.is_available():
        print("    Apple Silicon MPS 백엔드 활성화됨")
    print("    ✅ OK")


def check_libraries():
    import transformers
    import datasets
    import accelerate
    import numpy
    import tqdm
    import matplotlib

    print(f"\n[3] 라이브러리:")
    print(f"    transformers: {transformers.__version__}")
    print(f"    datasets:     {datasets.__version__}")
    print(f"    accelerate:   {accelerate.__version__}")
    print(f"    numpy:        {numpy.__version__}")
    print(f"    matplotlib:   {matplotlib.__version__}")
    print("    ✅ OK")


def check_model_load():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("\n[4] 모델 로드 테스트 (gpt2)...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    param_count = sum(p.numel() for p in model.parameters())
    print(f"    파라미터: {param_count:,}")
    print("    ✅ OK")
    return tokenizer, model


def check_device_inference(tokenizer, model):
    import torch

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"\n[5] 디바이스 추론 테스트 ({device})...")
    model = model.to(device)
    inputs = tokenizer("Hello, world!", return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    print(f"    Output shape: {outputs.logits.shape}")
    print("    ✅ OK")


def main():
    print("=" * 50)
    print("LLM Knowledge Distillation - 환경 검증")
    print("=" * 50)

    check_python()
    check_pytorch()
    check_libraries()
    tokenizer, model = check_model_load()
    check_device_inference(tokenizer, model)

    print("\n" + "=" * 50)
    print("모든 검증 통과!")
    print("=" * 50)


if __name__ == "__main__":
    main()
