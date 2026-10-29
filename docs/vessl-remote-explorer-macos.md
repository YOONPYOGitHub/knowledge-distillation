# macOS에서 VESSL에 VS Code Remote Explorer로 접속하기

일반 macOS 사용자를 위한 독립 안내입니다. 특정 프로젝트나 기존 설치를 전제로 하지 않습니다.
계정·조직·워크스페이스는 사용자가 선택합니다. **본인 소유로 제한하지 않으며, 접근 권한이 있는 공유 워크스페이스도 선택할 수 있습니다.**
실제 접속 성공, GPU 할당, 파일 준비 또는 테스트 완료를 의미하는 문서가 아닙니다.

## 1. [Mac LOCAL · 로컬] 준비 사항 확인

- [VS Code](https://code.visualstudio.com/)를 Mac에 설치합니다.
- Python **3.10 이상**, macOS 기본 **OpenSSH**, VESSL 로그인 및 대상 워크스페이스 접근 권한이 필요합니다.
- VESSL 웹 콘솔에서 사용할 조직과 워크스페이스를 직접 선택하고 실행 상태가 `running`인지 확인합니다.
- `[Mac LOCAL]` 명령은 Mac 터미널 또는 **SSH 연결 표시가 없는 VS Code 창**에서 실행합니다.
- `[VESSL LINUX · 원격]` 명령은 연결 후 **SSH 표시가 있는 원격 창**의 터미널에서만 실행합니다.

```bash
python3 --version
ssh -V
```

Python이 없거나 3.10 미만이면 [Python 공식 macOS 설치 프로그램](https://www.python.org/downloads/macos/)을 설치하고 터미널을 다시 열어 확인합니다.
Homebrew 설치를 전제로 하지 않습니다. `ssh -V`에는 OpenSSH 버전이 표시되어야 합니다.
SSH를 찾지 못하면 macOS 기본 OpenSSH와 PATH 상태를 점검합니다. 별도 SSH 구현으로 임의 교체하지 않습니다.

## 2. [Mac LOCAL · 로컬] VS Code 확장 설치

1. Mac의 VS Code에서 **Cmd+Shift+X**로 Extensions를 엽니다.
2. Microsoft의 **Remote - SSH** (`ms-vscode-remote.remote-ssh`)를 찾아 **로컬에 설치**합니다.
3. Microsoft의 **Remote Explorer** (`ms-vscode.remote-explorer`)도 **로컬에 설치**합니다.
4. 재시작/Reload 요청이 있으면 적용합니다. 이 단계는 원격 서버에 확장을 설치하는 단계가 아닙니다.

## 3. [Mac LOCAL · 로컬] 프로젝트와 분리된 VESSL CLI 설치

프로젝트 폴더로 이동할 필요가 없습니다. CLI 전용 가상환경을 만들고 그 Python으로 설치합니다.
각 명령이 성공한 뒤 다음 명령을 실행합니다. 설치 실패 시 이후 단계로 넘어가지 않습니다.

```bash
CLI_ENV="$HOME/.venvs/vessl-cli"
if [ ! -x "$CLI_ENV/bin/python" ]; then
  python3 -m venv "$CLI_ENV"
fi
"$CLI_ENV/bin/python" -m pip install vessl
VESSL="$CLI_ENV/bin/vessl"
"$VESSL" --version
```

가상환경 활성화는 필요하지 않습니다. 시스템 Python이나 프로젝트 가상환경에 CLI를 설치하지 않습니다.
**이후 로컬 명령은 같은 터미널 세션에서 실행합니다.** 새 터미널에서는 `CLI_ENV`, `VESSL`, `KEY_PATH`, `SSH_HOST`를 다시 지정해야 합니다(10단계 참고).

## 4. [Mac LOCAL · 로컬] SSH 키 확인 및 안전한 생성

키는 **Mac에 보관하는 클라이언트 인증 키**입니다. 원격 서버에서 생성하는 호스트 키와 다릅니다.
기존 키를 재사용하며, 아래는 개인키와 공개키가 **둘 다 없을 때만** 생성합니다.

```bash
KEY_PATH="$HOME/.ssh/id_ed25519"
mkdir -p "$HOME/.ssh"
chmod 700 "$HOME/.ssh"
if [ -f "$KEY_PATH" ] && [ -f "$KEY_PATH.pub" ]; then
  printf '%s\n' '기존 키 쌍을 사용합니다. 덮어쓰지 마세요.'
elif [ ! -e "$KEY_PATH" ] && [ ! -L "$KEY_PATH" ] && \
     [ ! -e "$KEY_PATH.pub" ] && [ ! -L "$KEY_PATH.pub" ]; then
  ssh-keygen -t ed25519 -f "$KEY_PATH" -C vessl-client
else
  printf '%s\n' 'STOP: 키가 한쪽만 있거나 경로가 비정상입니다. 점검·백업 전 진행하지 마세요.'
fi
```

- **STOP이면 여기서 중단**합니다. 남아 있는 키와 경로를 점검하고 안전하게 백업한 뒤 복구 방법을 결정합니다.
- 키가 하나만 있어도 새 키로 덮어쓰지 않습니다. 덮어쓰기 질문이 뜻밖에 나오면 취소합니다.
- 다른 기존 키를 사용하려면 `KEY_PATH`를 해당 개인키 경로로 바꾸고 공개키도 함께 확인합니다.
- passphrase는 **터미널의 비밀 입력 프롬프트에 직접** 입력합니다. 채팅·문서·명령 인자로 보내지 않습니다.
- **개인키 내용을 출력하거나 업로드하지 않습니다.** 원격 서버에 복사하지 않으며, 등록 대상은 공개키뿐입니다.

두 파일이 정상적으로 준비된 경우에만 공개키 지문을 확인합니다. 기존 공개키도 등록 전에 확인합니다.

```bash
ssh-keygen -lf "$KEY_PATH.pub"
```

## 5. [Mac LOCAL · 로컬] 로그인·조직 선택·공개키 등록

```bash
"$VESSL" configure
"$VESSL" whoami
"$VESSL" workspace list
"$VESSL" ssh-key list
```

1. 인증 브라우저에서 **원하는 계정**인지 확인하고, CLI 설정 과정에서 **사용할 조직**을 선택합니다.
2. `whoami`와 워크스페이스 목록을 대조합니다. 잘못된 계정/조직이면 `"$VESSL" configure --reset`으로 다시 설정하고 재확인합니다.
3. 선택할 워크스페이스가 없으면 조직 선택, 멤버십, 접근 권한을 확인합니다. 소유자와 사용자가 같을 필요는 없습니다.
4. 등록된 공개키와 앞에서 확인한 지문을 대조합니다. 목록에 지문이 없으면 콘솔의 키 상세 정보를 확인합니다.
5. 해당 공개키가 등록되지 않았을 때만 아래 대화형 등록을 실행합니다.

```bash
"$VESSL" ssh-key add
```

공개키 경로 질문에는 **`KEY_PATH` 값 뒤에 `.pub`를 붙인 실제 경로**를 입력하고, 키 이름은 직접 정합니다.
대화형 입력은 셸 변수를 자동 확장하지 않을 수 있으므로 `$KEY_PATH.pub` 문자열을 그대로 입력하지 않습니다.
개인키 경로를 등록하지 않습니다. 등록 후 `"$VESSL" ssh-key list`로 확인하고, 인증 URL·토큰은 공유하지 않습니다.

## 6. [Mac LOCAL · 로컬] SSH 설정 생성과 별칭 선택

```bash
"$VESSL" workspace vscode --key-path "$KEY_PATH"
grep -Ei '^[[:space:]]*(Host|HostName|User|Port|IdentityFile|ProxyCommand|ProxyJump)[[:space:]]' "$HOME/.ssh/config"
```

CLI 버전에 따라 현재 조직의 여러 워크스페이스 항목을 자동 생성하거나 선택을 물을 수 있습니다. 질문이 없으면 생성된 SSH 설정에서 원하는 워크스페이스의 `Host`를 찾습니다.
VS Code 자동 실행 안내/시도가 있어도 이 안내에서는 다음 단계의 **GUI 연결**을 사용합니다.
여러 `Host`가 나오면 방금 CLI가 안내한 항목과 대조합니다. **`HostName`이나 IP가 아닌 `Host` 별칭**을 사용합니다.
아래 `YOUR_SSH_HOST_ALIAS`를 **방금 생성된 실제 별칭으로 교체한 뒤** 실행합니다. 그대로 접속하지 않습니다.

```bash
SSH_HOST="YOUR_SSH_HOST_ALIAS"
ssh -o ConnectTimeout=15 "$SSH_HOST" 'hostname; whoami'
```

최초 연결의 **서버 호스트 키 지문**은 신뢰할 수 있는 VESSL 접속 정보 또는 관리자에게 확인한 값과 대조한 뒤 승인합니다.
4단계의 사용자 공개키 지문과는 다른 지문입니다. 확인할 근거가 없으면 승인하지 말고 문의합니다.
passphrase는 터미널에 직접 입력합니다. 이 테스트는 대화형이며 원격 호스트명·사용자 출력으로 응답을 확인합니다.
passphrase/에이전트 문제로 인증이 반복 실패하는 경우에만, Mac의 SSH 에이전트에 키를 추가하고 다시 시도합니다.

```bash
ssh-add "$KEY_PATH"
```

에이전트에 연결할 수 없다면 로컬 에이전트 상태부터 점검합니다. 비밀 값을 채팅에 입력하지 않습니다.

## 7. [Mac LOCAL → VESSL LINUX · 연결] Remote Explorer에서 열기

1. VS Code 왼쪽 **Remote Explorer**를 엽니다. 없으면 **View → Open View… → Remote Explorer**를 선택합니다.
2. 위쪽 드롭다운에서 **SSH Targets** 또는 **Remotes → SSH**를 선택합니다(버전에 따라 표기가 다름).
3. **새로고침(↻)** 후 방금 확인한 별칭을 찾습니다.
4. 해당 항목을 우클릭하여 **Connect to Host in New Window**를 선택합니다.
5. 원격 운영체제를 물으면 **Linux**를 선택합니다. 서버 지문을 다시 물으면 6단계와 동일하게 검증합니다.
6. 첫 접속에서는 **VS Code Server 다운로드·설치**를 기다립니다. 네트워크·프록시·디스크·쓰기 권한이 필요합니다.
7. passphrase 요청은 VS Code 입력창이나 터미널에 직접 응답합니다.
8. 새 창 왼쪽 아래의 **SSH: 선택한 별칭** 표시를 확인합니다. 이 창에서 연 터미널은 원격 Linux입니다.

## 8. [VESSL LINUX · 원격] 원하는 폴더 열기

**File → Open Folder…**에서 사용할 **원격 Linux 절대 경로**를 직접 선택합니다.
특정 프로젝트 폴더를 강제하지 않습니다. 폴더는 원격에 존재하고 현재 사용자에게 접근 권한이 있어야 합니다.
Mac의 사용자 폴더 경로를 입력하지 않습니다. 로컬 파일이 자동 복사되는 기능도 아닙니다.
**Terminal → New Terminal**을 열고 위치와 사용자를 확인합니다.

```bash
pwd
whoami
hostname
```

원격 호스트명이 SSH 별칭과 같을 필요는 없습니다. 예상한 원격 폴더·사용자인지 확인합니다.
없는 프로젝트는 조직에서 승인한 방법으로 별도 준비하며, Mac 가상환경·개인키·인증 파일은 복사하지 않습니다.

## 9. [VESSL LINUX · 원격, 선택] Python 인터프리터 선택

Python 작업을 할 때만 Microsoft **Python** (`ms-python.python`) 확장을 **SSH 대상에 설치**합니다.
**Cmd+Shift+P → Python: Select Interpreter**에서 원격 프로젝트의 Python을 선택합니다.
원격 프로젝트에 `.venv`가 준비되어 있다면 그 안의 Linux Python을 선택합니다.
없으면 프로젝트 요구사항에 맞게 **원격에서 별도로 생성**합니다. Mac의 CLI 전용 환경이나 Mac `.venv`를 선택/복사하지 않습니다.
이 안내는 원격 Python·학습 패키지·GPU가 준비되었다고 가정하지 않습니다.

## 10. [Mac LOCAL · 로컬] 재접속과 보존 주의

새 로컬 터미널에서는 변수를 먼저 다시 지정합니다. 다른 키를 사용했다면 같은 키 경로로 바꿉니다.

```bash
CLI_ENV="$HOME/.venvs/vessl-cli"
VESSL="$CLI_ENV/bin/vessl"
KEY_PATH="$HOME/.ssh/id_ed25519"
"$VESSL" workspace list
"$VESSL" workspace vscode --key-path "$KEY_PATH"
grep -Ei '^[[:space:]]*(Host|HostName|Port)[[:space:]]' "$HOME/.ssh/config"
SSH_HOST="YOUR_SSH_HOST_ALIAS"
```

마지막 줄의 자리표시자는 **현재 생성된 별칭으로 교체**합니다. 재시작 후 별칭·포트가 달라질 수 있습니다.
6단계 SSH 테스트 → Remote Explorer 새로고침 → 현재 별칭으로 새 창 연결 → 원격 폴더 확인 순으로 진행합니다.
워크스페이스 중지·만료·삭제 시 저장소 보존은 보장되지 않습니다. 마운트/보존 정책을 확인하고 중요한 파일은 별도 백업합니다.
SSH 재접속은 학습 재개가 아니며, 창을 닫아도 프로세스가 계속 실행된다고 가정하지 않습니다.

## 11. 문제 해결

| 증상 | 확인 및 조치 |
|---|---|
| 워크스페이스가 안 보임 | [Mac LOCAL] 올바른 계정·조직인지 `whoami`/`workspace list`로 확인하고 공유 워크스페이스 접근 권한을 점검합니다. |
| `Permission denied (publickey)` | [Mac LOCAL] 현재 계정의 공개키 등록, 키 지문, `KEY_PATH`와 생성된 `IdentityFile` 일치를 확인합니다. |
| 연결 timeout | [Mac LOCAL] 실행 상태·만료 여부, 최신 별칭/포트, 네트워크·VPN·조직 방화벽 정책을 확인합니다. |
| Remote Explorer가 안 보임 | [Mac LOCAL] 두 확장이 로컬에서 활성화되었는지 확인하고 **View → Open View…**로 연 뒤 SSH 드롭다운·새로고침을 확인합니다. |
| `code` 명령을 찾지 못함 | [Mac LOCAL] 이 GUI 안내에는 VS Code CLI의 PATH 등록이 필요 없습니다. SSH 설정이 생성되었으면 Remote Explorer로 연결합니다. |
| 호스트 키 변경 경고 | [Mac LOCAL] 재생성 여부와 새 서버 지문을 먼저 검증합니다. 확인 후 해당 호스트·포트의 오래된 기록만 정리하고 재접속합니다. 무작정 전체 기록 삭제나 호스트 키 검증 비활성화를 하지 않습니다. |
| Server 설치/폴더 열기 실패 | [Mac LOCAL / VESSL LINUX] **Output → Remote - SSH** 로그와 워크스페이스 권한·원격 디스크·프록시/다운로드 정책을 점검합니다. 비밀 정보를 가리고 공유합니다. |

## 관련 안내

- 접속 이후 공통 학습 절차: [./vessl-3090x4-remote-explorer-guide.md](./vessl-3090x4-remote-explorer-guide.md). 해당 문서의 환경별 값은 그대로 복사하지 말고 선택한 환경에 맞게 확인합니다.
- Windows 접속 안내: [./vessl-remote-explorer-windows.md](./vessl-remote-explorer-windows.md).