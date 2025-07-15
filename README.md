## 경구약제 이미지 객체 검출(Object Detection) 프로젝트

2팀 화이팅!

## 프로젝트 설치 및 데이터 다운로드 가이드

이 프로젝트를 로컬 환경에서 실행하고 Kaggle 데이터를 다운로드하기 위한 단계별 가이드입니다.

### 1. Python 및 Poetry 설치

프로젝트는 [Poetry](https://python-poetry.org/)를 사용하여 의존성을 관리합니다. 먼저 시스템에 Python 3.11.9 버전과 Poetry를 설치해야 합니다.

**Python 설치:**

-   **Windows:** [Python 공식 웹사이트](https://www.python.org/downloads/windows/)에서 설치 프로그램을 다운로드하여 실행합니다. 설치 시 "Add Python to PATH" 옵션을 반드시 선택합니다.
-   **macOS:** Homebrew를 사용하는 것이 가장 편리합니다. 터미널을 열고 다음 명령어를 실행합니다:
    ```bash
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    brew install python@3.11 # 또는 원하는 Python 버전
    ```
-   **Linux:** 대부분의 Linux 배포판에는 Python이 기본으로 설치되어 있습니다. 필요한 경우 패키지 관리자를 통해 설치합니다:
    ```bash
    # Debian/Ubuntu 기반
    sudo apt update
    sudo apt install python3.9 python3.9-venv

    # Fedora/RHEL 기반
    sudo dnf install python3.9 python3.9-venv
    ```

**Poetry 설치:**

Python 설치 후, 터미널/명령 프롬프트에서 다음 명령어를 실행하여 Poetry를 설치합니다:

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

설치 완료 후, `poetry --version` 명령어를 실행하여 Poetry가 올바르게 설치되었는지 확인합니다.

### 2. 프로젝트 의존성 설치

프로젝트 루트 디렉토리로 이동하여 다음 명령어를 실행하여 필요한 라이브러리들을 설치합니다:

```bash
poetry install
```

이 명령어는 `pyproject.toml` 파일에 명시된 모든 의존성을 가상 환경에 설치합니다.

### 3. Kaggle API 인증 설정

Kaggle 데이터를 다운로드하려면 Kaggle API 인증이 필요합니다. 다음 단계를 따릅니다:

1.  [Kaggle 웹사이트](https://www.kaggle.com/)에 로그인합니다.
2.  우측 상단의 프로필 아이콘을 클릭하고 "My Account"로 이동합니다.
3.  "API" 섹션에서 "Create New API Token" 버튼을 클릭합니다. `kaggle.json` 파일이 다운로드됩니다.
4.  다운로드된 `kaggle.json` 파일을 다음 경로로 이동합니다:
    -   **Windows:** `C:\Users\<Your Username>\.kaggle\kaggle.json`
    -   **macOS/Linux:** `~/.kaggle/kaggle.json`

    `.kaggle` 디렉토리가 없다면 직접 생성해야 합니다.

### 4. Kaggle 데이터셋 다운로드 및 압축 해제

프로젝트는 `ai03-level1-project` Kaggle 대회 데이터를 사용합니다. 다음 명령어를 사용하여 데이터를 다운로드하고 압축을 해제합니다:

```bash
poetry run kaggle competitions download -c ai03-level1-project -p data/ai03-level1-project
poetry run python -c "import zipfile; import os; with zipfile.ZipFile('data/ai03-level1-project/ai03-level1-project.zip', 'r') as zip_ref: zip_ref.extractall('data/ai03-level1-project')"
```

첫 번째 명령어는 Kaggle 데이터를 `data/ai03-level1-project` 디렉토리에 `ai03-level1-project.zip` 파일로 다운로드합니다. 두 번째 명령어는 다운로드된 zip 파일의 압축을 해제합니다.

### 5. 프로젝트 실행

모든 설정이 완료되면, `main.py` 파일을 실행하여 프로젝트를 시작할 수 있습니다:

```bash
poetry run python main.py
```
