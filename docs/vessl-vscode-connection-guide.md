# VESSL GPU 서버 VS Code 원격 접속 가이드

VESSL Cloud 워크스페이스에 VS Code Remote-SSH로 접속하는 방법을 정리한 문서입니다.

> **핵심 개념**: SSH 키는 **내 로컬 컴퓨터(Mac)에서 내가 직접** 만듭니다. GPU 센터나 VESSL이 만들어 주지 않습니다.
> - **개인키** (`~/.ssh/id_ed25519`): 내 Mac에만 보관, 절대 공유 금지 (신분증 원본)
> - **공개키** (`~/.ssh/id_ed25519.pub`): VESSL 서버에 등록, 공유 가능 (자물쇠)

---

## 사전 준비

### 1. VESSL CLI 설치
프로젝트 가상환경(`.venv`)에 설치합니다.

```bash
python -m pip install vessl
# 설치 확인
.venv/bin/vessl --version   # 예: vessl, version 0.1.199
```

### 2. VS Code Remote-SSH 확장 설치
VS Code의 Extensions 화면에서 `Remote - SSH`를 검색해 설치하거나 CLI를 사용합니다.

```bash
code --install-extension ms-vscode-remote.remote-ssh
```

> macOS에서 `code` 명령을 찾지 못하면 VS Code 명령 팔레트에서 **Shell Command: Install 'code' command in PATH**를 먼저 실행합니다.

> 다음 확장이 함께 설치됩니다: `remote-ssh`, `remote-ssh-edit`, `remote-explorer`

---

## 접속 절차 (최초 1회 설정)

### ① SSH 키 만들기 — *이미 있으면 건너뛰기*
기존 키가 있는지 먼저 확인합니다.

```bash
ls -al ~/.ssh/ | grep -E "id_|\.pub"
```

- 이미 `id_ed25519` / `id_ed25519.pub`가 있으면 **재사용**합니다. (새로 만들 필요 없음)
- 없다면 새로 생성:

```bash
ssh-keygen -t ed25519 -C "vessl-<사용자명>"
# 저장 경로: 기본값(~/.ssh/id_ed25519)으로 Enter
# passphrase: 비워도 됨 (입력 시 접속할 때마다 물어봄)
```

### ② VESSL 로그인
```bash
.venv/bin/vessl configure
```

1. 터미널에 표시된 URL(`https://app.vessl.ai/cli/grant-access?token=...`)을 **브라우저에서 열기**
2. **해당 워크스페이스에 접근 권한이 있는 계정**으로 로그인
3. **Grant access(접근 승인)** 클릭
4. 터미널에 `Welcome, <username>!`이 뜨면 완료

로그인 확인:
```bash
.venv/bin/vessl whoami
# Username / Email / Default organization 확인

.venv/bin/vessl workspace list
# 선택된 조직의 워크스페이스 이름과 상태 확인
```

### ③ 공개키를 VESSL에 등록
```bash
.venv/bin/vessl ssh-key add
```

- `SSH public key path`: 기본값 `~/.ssh/id_ed25519.pub` → **Enter**
- `SSH public key name`: 키를 구분할 이름 (예: `vessl-<사용자명>`)
- `SSH Key '<name>' created.` 가 뜨면 완료

> SSH 키 등록은 VESSL 계정별로 관리됩니다. 계정을 변경했다면 같은 로컬 공개키라도 새 계정에 다시 등록해야 합니다.

### ④ VS Code 접속 설정 생성
```bash
.venv/bin/vessl workspace vscode
```

- `~/.ssh/config`에 접속용 Host 항목이 **자동으로 추가**됩니다.

추가된 항목 확인:
```bash
grep -A6 -i "<워크스페이스 이름>" ~/.ssh/config
```

예시 출력:
```
Host <워크스페이스 이름>-<고유 ID>
    User root
    Hostname <서버 IP 또는 호스트명>
    Port <SSH 포트>
    StrictHostKeyChecking accept-new
    CheckHostIP no
    IdentityFile <사용자 홈>/.ssh/id_ed25519
```

아래 명령에서 사용할 Host 이름을 변수로 저장합니다.

```bash
WORKSPACE_HOST="<위에서 확인한 Host 이름>"
```

### ⑤ VS Code 원격 창 열기
```bash
code --remote "ssh-remote+$WORKSPACE_HOST"
```

- 새 VS Code 창이 원격 서버에 연결됩니다.
- 연결 후 **File → Open Folder...**에서 작업할 원격 폴더를 선택합니다.
- 첫 연결 시 원격에 VS Code 서버를 설치하느라 수십 초 걸립니다.
- 왼쪽 아래에 `SSH: <Host 이름>`이 뜨면 접속 완료.

---

## 접속 확인

터미널에서 직접 SSH로 확인할 수도 있습니다.

```bash
ssh -o BatchMode=yes -o ConnectTimeout=15 "$WORKSPACE_HOST" \
  "hostname; nvidia-smi --query-gpu=name,memory.total --format=csv"
```

정상 출력 예시:
```
workspace-xxxxxxxx-0
name, memory.total [MiB]
<GPU 모델>, <메모리 용량> MiB
```

---

## 다시 접속할 때 (2회차 이후)

워크스페이스를 재시작하면 **Host 이름/포트가 바뀔 수 있으므로** 설정을 갱신합니다.

**접속 정보 갱신**

```bash
.venv/bin/vessl workspace vscode
grep -A6 -i "<워크스페이스 이름>" ~/.ssh/config   # 새 Host 이름 확인
```

**VS Code에서 접속**

- 왼쪽 아래 **`><`** 아이콘 클릭 → **Connect to Host...** → 갱신된 Host 선택
- 또는 위 ⑤번 명령을 새 Host 이름으로 실행

---

## 내 환경 정보 확인

계정, 조직, 워크스페이스, 클러스터, 리소스는 사용자마다 다를 수 있습니다.

```bash
.venv/bin/vessl whoami
.venv/bin/vessl workspace list
```

SSH Host, 서버 주소, 포트, 사용자, 키 경로는 자동 생성된 설정에서 확인합니다.

```bash
grep -A6 -i "<워크스페이스 이름>" ~/.ssh/config
```

---

## 실행 시간 자동 연장

VESSL CLI에는 실행 중인 워크스페이스를 연장하는 전용 명령이 없습니다. 대신 이 저장소의 감시 스크립트가 VESSL SDK로 서버의 종료 시각을 조회하고, 설정한 임계 시간 전에 연장 API를 호출합니다.

먼저 `workspace list`에서 조직과 워크스페이스 ID를 확인합니다.

```bash
.venv/bin/vessl workspace list
```

실제 변경 없이 현재 상태와 연장 판단을 확인합니다.

```bash
.venv/bin/python scripts/vessl_auto_extend.py \
  --organization "<조직 이름>" \
  --workspace-id <워크스페이스 ID>
```

종료 30분 전부터 5분 간격으로 확인하고, 한 번에 24시간씩 VESSL 서버가 허용하는 동안 계속 자동 연장합니다.

```bash
caffeinate -i .venv/bin/python scripts/vessl_auto_extend.py \
  --organization "<조직 이름>" \
  --workspace-id <워크스페이스 ID> \
  --extend-hours 24 \
  --threshold-minutes 30 \
  --poll-seconds 300 \
  --watch \
  --apply
```

상태와 연장 결과는 기본적으로 `results/logs/vessl-auto-extend-<워크스페이스 ID>.log`에 기록됩니다. 이 로그의 최초 생성 시각, 누적 실행 시간, 연장 거부 시점을 비교하면 실제 최대 세션 시간을 확인할 수 있습니다.

> 기본값에는 로컬 최대 시간이 없습니다. 매 연장 전에 VESSL 서버의 연장 가능 여부를 확인하며, 나중에 실제 한도를 확인한 뒤 `--max-total-hours <시간>`으로 안전 상한을 설정할 수 있습니다. `caffeinate -i`는 감시 중 macOS의 유휴 절전을 막지만, 터미널을 닫거나 컴퓨터를 종료하면 자동 연장도 중단됩니다.

---

## 문제 해결 (Troubleshooting)

| 증상 | 원인 / 해결 |
|------|------------|
| `command not found: code` | VS Code 명령 팔레트에서 **Shell Command: Install 'code' command in PATH** 실행 |
| `Please run vessl configure first.` | 미로그인 → `②` 로그인 단계 진행 |
| 브라우저 인증 페이지가 로그인만 요구 | 워크스페이스 발급 계정으로 로그인해야 승인 가능 |
| VS Code가 Host 목록에 안 뜸 | `.venv/bin/vessl workspace vscode` 재실행 후 `~/.ssh/config` 확인 |
| 접속은 되나 포트 오류 | 워크스페이스 재시작으로 포트 변경됨 → 설정 갱신 (2회차 절차) |
| 조직/프로젝트 변경 필요 | `.venv/bin/vessl configure --reset` |
