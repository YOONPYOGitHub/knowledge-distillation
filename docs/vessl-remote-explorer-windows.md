# Windows 10/11: VESSL을 VS Code Remote Explorer로 연결하기

임의의 VESSL 워크스페이스에 접속하는 독립 안내서입니다. 특정 계정·조직·GPU·프로젝트를 전제로 하지 않습니다.
**`<YOUR_...>` 등의 자리표시자는 꺾쇠까지 지우고 실제 값으로 바꾸세요.** 출력 예시는 성공 판단 기준이며, 접속이나 학습을 완료했다는 보고가 아닙니다.

## 1. 준비물과 터미널 구분

- Windows 10/11 PC, 인터넷, [VS Code](https://code.visualstudio.com/download), Python 3.10 이상, Windows OpenSSH Client가 필요합니다.
- VESSL 웹 콘솔에서 **사용할 계정·조직**, 접근 권한, 대상 워크스페이스의 **Running** 상태를 직접 확인합니다.
- **[로컬 Windows · PowerShell]**: Windows 시작 메뉴의 PowerShell 또는 SSH 연결 전 VS Code 터미널입니다. 기본 프로필도 PowerShell로 선택합니다.
- **[원격 Linux]**: 연결된 VS Code 창에서 여는 Linux 터미널입니다. 아래 원격 명령만 여기서 실행합니다.
- 로컬 명령은 **같은 PowerShell 세션**에서 순서대로 실행합니다. 창을 닫거나 새 터미널을 열면 변수는 사라지므로 8절의 정의를 다시 실행합니다.
- WSL·Git Bash는 이 안내서의 대상이 아닙니다. Windows와 WSL의 홈·SSH 키·설정을 혼용하지 마세요.

**[로컬 Windows · PowerShell]** 준비 상태 확인:
```powershell
py -3 --version
Get-Command ssh
ssh -V
```
기대 결과: `Python 3.10` 이상, Windows 네이티브 `ssh.exe`의 경로, OpenSSH 버전이 표시됩니다.
Python이 없거나 버전이 낮으면 [Python 공식 Windows 다운로드](https://www.python.org/downloads/windows/)에서 3.10 이상을 수동 설치합니다.
공식 설치 프로그램의 Python 실행 환경과 `py` 실행기 옵션을 확인하고, 새 PowerShell에서 위 명령을 다시 확인합니다.
SSH가 없으면 **설정 → 선택적 기능(Optional features)**을 검색해 **기능 추가/기능 보기 → OpenSSH Client → 설치**를 선택합니다.
Windows 버전에 따라 **앱** 또는 **시스템** 아래에 있습니다. OpenSSH **Server**는 필요하지 않습니다.
관리자 승인이 필요하면 장치 관리자에게 요청합니다. 관리자 자동 설정이나 PowerShell 실행 정책 완화는 하지 않습니다.

## 2. [로컬 Windows] VS Code 확장 설치

1. SSH 연결이 없는 로컬 VS Code 창에서 **Ctrl+Shift+X**로 확장 화면을 엽니다.
2. Microsoft의 **Remote - SSH**(`ms-vscode-remote.remote-ssh`)를 검색해 설치합니다.
3. Microsoft의 **Remote Explorer**(`ms-vscode.remote-explorer`)도 설치합니다.
4. 다시 로드 안내가 있으면 실행합니다. 두 확장은 원격 서버가 아닌 **로컬 Windows**에 설치되어야 합니다.

## 3. [로컬 Windows · PowerShell] 프로젝트와 분리된 VESSL CLI 설치

현재 폴더와 관계없이 실행합니다. 프로젝트 다운로드나 기존 프로젝트 가상환경은 필요하지 않습니다.
```powershell
$CliEnv = Join-Path $env:USERPROFILE '.venvs\vessl-cli'
if (-not (Test-Path $CliEnv)) {
    py -3 -m venv $CliEnv
}
& "$CliEnv\Scripts\python.exe" -m pip install vessl
$Vessl = "$CliEnv\Scripts\vessl.exe"
& $Vessl --version
```
기대 결과: 설치가 완료되고 VESSL CLI 버전이 표시됩니다. 중간 오류가 있으면 다음 단계로 넘어가지 않습니다.
기존 경로에 Python 실행 파일이 없다면 불완전한 환경인지 확인하고, 기존 파일을 무작정 삭제하지 않습니다.
가상환경 **활성화는 필요 없습니다**. `Activate.ps1`이나 `Set-ExecutionPolicy`를 실행하지 마세요.

## 4. [로컬 Windows · PowerShell] SSH 키를 안전하게 준비

아래는 기존 키를 덮어쓰지 않습니다. 개인키와 공개키 중 하나만 있으면 **중단**하고 기존 키 위치·백업부터 확인합니다.
```powershell
$SshDir = Join-Path $env:USERPROFILE '.ssh'
if (-not (Test-Path $SshDir)) {
    New-Item -ItemType Directory -Path $SshDir | Out-Null
}
$KeyPath = Join-Path $SshDir 'id_ed25519'
$PrivateExists = Test-Path $KeyPath
$PublicExists = Test-Path "$KeyPath.pub"
if ($PrivateExists -xor $PublicExists) {
    throw '키 쌍 중 하나만 있습니다. 재생성하지 말고 기존 키를 확인하세요.'
}
if (-not $PrivateExists -and -not $PublicExists) {
    ssh-keygen -t ed25519 -f $KeyPath -C 'vessl-client'
    if ($LASTEXITCODE -ne 0) { throw 'SSH 키 생성 실패: 여기서 중단하세요.' }
}
```
기대 결과: 두 파일이 이미 있으면 그대로 사용하고, 둘 다 없을 때만 새 키 쌍을 생성합니다.
예상과 달리 덮어쓰기 질문이 나오면 취소합니다. 다른 이름의 기존 키를 쓰려면 `$KeyPath`를 실제 개인키 경로로 바꾸고 쌍을 확인합니다.
키 암호(passphrase)는 안전하게 정해 **로컬 터미널/VS Code 보안 입력창에 직접 입력**하며 채팅에 보내지 않습니다.
공개키는 `$KeyPath` 뒤에 **`.pub`가 붙은 파일**입니다. 개인키는 출력·업로드·원격 복사하지 않습니다.

## 5. [로컬 Windows · PowerShell] 로그인·조직 확인·공개키 등록

```powershell
& $Vessl configure
& $Vessl whoami
& $Vessl workspace list
& $Vessl ssh-key list
```
1. `configure`가 안내하는 브라우저 인증에서 **실제로 사용할 계정**으로 승인하고 조직을 선택·확인합니다.
2. `whoami`의 계정·조직과 웹 콘솔을 대조합니다. 다르면 `& $Vessl configure --reset`으로 다시 설정한 뒤 재확인합니다.
3. `workspace list`에서 원하는 워크스페이스가 보이고 Running인지 확인합니다. 없으면 조직·권한부터 확인합니다.
4. `ssh-keygen -lf "$KeyPath.pub"`로 확인한 지문을 `ssh-key list` 또는 콘솔의 등록 키와 대조합니다. **이 PC의 공개키가 이미 등록되어 있으면 건너뜁니다**. 이름만 같다고 같은 키는 아닙니다.

등록되어 있지 않을 때만 아래를 실행합니다. 질문에 공개키 경로와 구별하기 쉬운 등록 이름을 입력합니다.
```powershell
"$KeyPath.pub"
& $Vessl ssh-key add
```
첫 줄은 **경로만** 표시합니다. 공개키 경로 질문에 그 경로를 넣으세요. `.pub` 없는 개인키를 선택하지 않습니다.
등록 후 `& $Vessl ssh-key list`로 확인합니다. 인증 URL·토큰·키 암호는 문서나 채팅에 공유하지 않습니다.

## 6. [로컬 Windows · PowerShell] SSH 설정 생성과 연결 확인

```powershell
& $Vessl workspace vscode --key-path $KeyPath
Get-Content (Join-Path $SshDir 'config')
```
이 명령은 로컬 홈의 `.ssh/config`에 SSH 연결 항목을 생성·갱신합니다. CLI 버전에 따라 현재 조직의 여러 워크스페이스 항목을 자동으로 만들거나 선택을 물을 수 있습니다. 선택 질문이 없다면 다음 단계에서 원하는 `Host` 항목을 찾으면 됩니다.
두 번째 명령은 **SSH 설정만** 읽으며 개인키를 읽지 않습니다. 설정의 주소·사용자 정보도 공개 공유하지 마세요.
방금 생성된 항목의 `Host` 뒤 **SSH 별칭**을 선택합니다. `HostName` 주소나 웹 콘솔의 표시 이름을 그대로 쓰는 것이 아닙니다.
여러 항목이면 CLI 출력과 대조하고 `HostName`, `Port`, `User`, `IdentityFile`이 대상과 키에 맞는지 확인합니다.

아래 자리표시자를 실제 별칭으로 바꿉니다. PowerShell의 `$Host`는 예약된 자동 변수이므로 **대입하지 말고 `$SshHost`를 사용**합니다.
```powershell
$SshHost = '<YOUR_SSH_HOST_ALIAS>'
ssh -o ConnectTimeout=15 $SshHost 'hostname; whoami'
```
처음 보이는 서버 지문은 VESSL의 신뢰할 수 있는 접속 정보 또는 관리자에게 확인한 지문과 대조한 뒤에만 승인합니다.
확인할 수 없거나 다르면 중단합니다. `StrictHostKeyChecking=no`로 검증을 끄지 마세요.
키 암호 요청은 정상일 수 있습니다. 첫 연결에 `BatchMode=yes`를 강제하면 암호 입력이 막혀 실패할 수 있습니다.
기대 결과: **원격 Linux 호스트 이름과 원격 사용자 이름**이 출력되고 PowerShell로 돌아옵니다. 특정 사용자 이름을 가정하지 않습니다.

## 7. [로컬 VS Code → 원격 Linux] Remote Explorer에서 새 창 연결

1. 왼쪽 **Remote Explorer** 아이콘을 클릭합니다. 안 보이면 **View → Open View… → Remote Explorer**를 선택합니다.
2. 위쪽 드롭다운을 **SSH Targets** 또는 **Remotes → SSH**로 바꿉니다. WSL·Containers·Tunnels 목록이 아닙니다.
3. **새로고침(↻)** 후 6절에서 확인한 새 별칭을 찾습니다.
4. 해당 항목을 우클릭하여 **Connect to Host in New Window**를 선택합니다.
5. 원격 운영체제를 물으면 **Linux**를 선택합니다. 지문은 확인 후 승인하고 키 암호는 로컬 보안 입력창에 입력합니다.
6. VS Code Server 설치를 기다린 뒤 새 창 왼쪽 아래 **SSH: 해당 별칭** 표시를 확인합니다.
7. 새 창의 **File → Open Folder…**에서 원하는 **원격 Linux 절대 경로**를 입력합니다. 로컬 Windows의 C: 경로를 넣지 않습니다.
8. 예를 들어 `/<YOUR_REMOTE_DIRECTORY>`를 실제 존재하고 접근 가능한 폴더로 바꿉니다. 프로젝트 폴더가 이미 있다고 가정하지 마세요.
9. Workspace Trust는 신뢰하는 코드에만 허용합니다. **Terminal → New Terminal**을 열어 아래를 실행합니다.

**[원격 Linux · 셸]** — 로컬 PowerShell에서 실행하지 않습니다.
```bash
hostname
pwd
whoami
```
기대 결과: 6절과 같은 원격 호스트·사용자, 방금 연 원격 폴더 경로입니다. 호스트 이름은 SSH 별칭과 달라도 정상입니다.
**상태 표시줄의 SSH 표시와 터미널 결과를 함께 확인**하세요. 로컬 PC 이름이나 Windows 경로가 나오면 잘못된 창입니다.
Python 작업이 필요할 때만 확장 화면에서 Microsoft Python 확장을 **Install in SSH: 해당 별칭**으로 설치합니다.
**Ctrl+Shift+P → Python: Select Interpreter**에서 원격에 설치된 Python/가상환경을 선택합니다. 로컬 CLI 환경은 원격 실행 환경이 아닙니다.

## 8. [로컬 Windows · PowerShell] 재접속

워크스페이스를 재시작하면 SSH 포트나 별칭이 바뀔 수 있습니다. 웹 콘솔에서 올바른 조직과 Running 상태를 다시 확인합니다.
PowerShell을 새로 열었다면 먼저 다음 변수를 복원합니다. 다른 키를 사용했다면 `$KeyPath`도 실제 경로로 수정합니다.
```powershell
$CliEnv = Join-Path $env:USERPROFILE '.venvs\vessl-cli'
$Vessl = "$CliEnv\Scripts\vessl.exe"
$SshDir = Join-Path $env:USERPROFILE '.ssh'
$KeyPath = Join-Path $SshDir 'id_ed25519'
& $Vessl workspace vscode --key-path $KeyPath
Get-Content (Join-Path $SshDir 'config')
```
6절에서 `$SshHost`를 **새 별칭으로 다시 정의**하고 확인 → Remote Explorer 새로고침 → 새 창 연결 → 원격 폴더 열기를 반복합니다.
연결 종료는 상태 표시줄의 원격 메뉴 → **Close Remote Connection**을 사용합니다. 창 종료와 워크스페이스 중지는 별개입니다.
저장소 보존 여부는 워크스페이스·볼륨 정책에 따라 다릅니다. 영구 보존을 가정하지 말고 중지·삭제 전에 보존 정책 확인과 중요 파일 백업을 진행합니다.
SSH 재접속은 학습 재개가 아니며, 창을 닫아도 실행 중인 작업이 유지된다고 보장하지 않습니다.

## 9. 문제 해결

| 증상 | 확인과 조치 |
|---|---|
| Remote Explorer 메뉴가 없음 | Ctrl+Shift+X에서 두 확장의 로컬 설치·활성화 확인 후 View → Open View… 또는 Ctrl+Shift+P에서 Remote Explorer를 찾습니다. |
| SSH 별칭이 목록에 없음 | 상단을 SSH로 전환하고 새로고침합니다. **Remote-SSH: Open SSH Configuration File…**에서 CLI가 갱신한 Windows 사용자 설정을 선택합니다. `Remote.SSH: Config File`이 다른 파일을 가리키지 않는지 확인합니다. |
| PowerShell에서는 되지만 VS Code는 실패 | **Remote-SSH: Show Log**에서 사용하는 SSH 실행 파일·설정을 확인합니다. Windows 네이티브 OpenSSH를 사용하고 WSL의 홈·키·설정과 섞지 않습니다. 필요하면 `Remote.SSH: Path`를 `Get-Command ssh`의 Windows 실행 파일 경로로 지정합니다. |
| 사용자 프로필 경로에 공백이 있음 | PowerShell에서는 위와 같이 경로를 따옴표로 감쌉니다. SSH 설정의 `IdentityFile`도 `IdentityFile "C:/Users/<WINDOWS_USER>/.ssh/id_ed25519"`처럼 실제 경로를 슬래시와 큰따옴표로 적습니다. 공백을 임의로 삭제하지 않습니다. |
| `Permission denied (publickey)` | 현재 계정·조직, 해당 공개키 등록, 선택한 워크스페이스 접근 권한, `IdentityFile`과 키 쌍의 대응을 확인합니다. 개인키 업로드로 해결하지 않습니다. |
| 시간 초과/연결 거부 | 워크스페이스 중지·만료 여부, Running 전환 완료, 최신 포트·별칭, 인터넷·조직 방화벽/VPN 정책을 확인합니다. 중지 상태라면 권한·비용을 확인한 뒤 콘솔에서 시작하고 설정을 재생성합니다. |
| 서버 키 변경 경고 | 재생성 여부와 새 지문을 관리자에게 먼저 확인합니다. 확인 전 `known_hosts`를 통째로 지우거나 검증을 끄지 않습니다. 확인 후에만 해당 호스트·포트 항목 갱신을 진행합니다. |
| VS Code Server 설치 실패 | 로컬 VS Code의 **Remote-SSH: Show Log**를 확인합니다. 원격 설치 위치의 쓰기 권한·디스크 공간, 로컬/원격 다운로드 경로의 인터넷·프록시 제한을 확인하고 관리자와 해결합니다. |
| Python이 로컬에서 실행됨 | SSH 상태 표시, 원격 터미널, Python 확장의 SSH 대상 설치, 원격 인터프리터 선택을 다시 확인합니다. Windows 가상환경을 Linux로 복사하지 않습니다. |

## 10. 선택 사항: 프로젝트 작업으로 이어가기

- 프로젝트 학습 절차: [vessl-3090x4-remote-explorer-guide.md](./vessl-3090x4-remote-explorer-guide.md). 그 문서의 특정 자원·경로·실험 조건은 별도 사례이므로 자신의 환경과 적용 가능성을 먼저 확인합니다.
- macOS 사용자용 안내서: [vessl-remote-explorer-macos.md](./vessl-remote-explorer-macos.md).
- 코드·데이터·체크포인트 준비는 프로젝트 담당자와 별도로 확인합니다. 저장소가 이미 있거나 최신 코드가 push되어 있다고 가정하지 않습니다.
- 이 문서는 원격 편집기 접속까지 다룹니다. 특정 GPU 할당, 프로젝트 설치, 학습·테스트 성공을 보장하지 않습니다.