# Flask를 활용한 은행 웹서버 제작

Flask 프레임워크를 사용하여 사용자의 로그인, 입출금, 잔액 조회 등 기본적인 은행 기능을 갖춘 웹 애플리케이션을 개발하는 프로젝트입니다.

<br>

## 📂 프로젝트 개요

* [cite_start]**목표**: 웹 프레임워크를 이용해 동적인 웹 페이지를 구축하고, 서버와 클라이언트 간의 데이터 처리 및 세션 관리 등 백엔드 시스템의 핵심 흐름을 학습합니다. [cite: 316, 317]
* **주요 기능**:
    * [cite_start]사용자 로그인 및 로그아웃 기능 [cite: 264, 267]
    * [cite_start]사용자별 계좌 잔액 확인 기능 [cite: 265]
    * [cite_start]입금 및 출금 처리 기능 [cite: 266]
    * [cite_start]잔액 부족, 음수 입력 등 예외 처리 로직 구현 [cite: 300]

<br>

## 🛠️ 개발 환경

* [cite_start]**언어**: Python 3.12.3 [cite: 269]
* [cite_start]**프레임워크**: Flask 3.1.1 [cite: 270]
* [cite_start]**개발 도구**: Spyder, Chrome [cite: 271]

<br>

## ⚙️ 실행 방법

1.  이 레퍼지토리를 로컬 환경에 복제(Clone)합니다.
2.  Flask를 설치합니다.
    ```bash
    pip install Flask
    ```
3.  메인 파이썬 스크립트(`app.py`)를 실행합니다.
    ```bash
    python app.py
    ```
4.  [cite_start]웹 브라우저에서 `http://127.0.0.1:5000` 주소로 접속합니다. [cite: 303]

<br>

## 📖 주요 구현 내용

#### 1. 시스템 흐름

[cite_start]사용자는 로그인 페이지를 통해 아이디와 비밀번호를 인증하고, 성공 시 입출금 및 잔액 조회가 가능한 대시보드로 이동합니다. [cite: 273, 275, 277]

#### 2. 서버 구조 및 라우팅
* [cite_start]Flask 앱을 초기화하고, 세션 관리를 위한 `secret_key`를 설정합니다. [cite: 283, 284, 289]
* [cite_start]`@app.route()` 데코레이터를 사용하여 URL 경로에 따라 다른 함수가 실행되도록 라우팅을 구현합니다. [cite: 285]

#### 3. 핵심 기능 로직
* [cite_start]**로그인**: `POST` 요청으로 받은 사용자 정보를 확인하여 세션에 아이디를 저장합니다. [cite: 291, 297]
* [cite_start]**대시보드**: 로그인된 사용자의 잔액 정보를 화면에 표시합니다. [cite: 294, 298]
* [cite_start]**입출금**: `POST` 요청으로 금액을 받아 사용자의 잔액을 업데이트하고, 다양한 오류 상황을 처리합니다. [cite: 299, 300]

<br>

## 🖥️ 실행 화면 예시

#### 로그인 및 대시보드
<p align="center">
  <img src="https://github.com/songgongho/Industrial_AI/assets/174919318/eb4b05b3-c15b-4de0-8e10-9b3780c88301" width="300"/>
  <img src="https://github.com/songgongho/Industrial_AI/assets/174919318/1e73fcb5-8854-478a-a4df-1e3532f84b39" width="300"/>
</p>

#### 입출금 및 오류 처리
<p align="center">
  <img src="https://github.com/songgongho/Industrial_AI/assets/174919318/38a0631f-4b0d-4566-a612-4aa3500d0752" width="300"/>
  <img src="https://github.com/songgongho/Industrial_AI/assets/174919318/8892697b-3c35-442c-a29d-47c34d4a84fd" width="300"/>
</p>
<br>

## 💡 결론

* [cite_start]Flask 프레임워크를 활용하여 웹 서버의 기본 동작 원리와 백엔드 로직을 성공적으로 구현했습니다. [cite: 316]
* [cite_start]사용자 인증, 세션 처리, 데이터베이스 연동(간이) 등 웹 애플리케이션의 핵심 요소를 직접 다루며 실무적인 개발 역량을 향상시켰습니다. [cite: 317, 318]
