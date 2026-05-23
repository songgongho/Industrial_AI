"""논문 기획과 운영 원칙을 UI에 제공하기 위한 공용 콘텐츠."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProjectDirection:
    title: str
    one_liner: str
    problem: str
    why_it_matters: str
    proposed_solution: str
    expected_contribution: str
    scope_for_now: str
    security_notes: list[str]
    next_ideas: list[str]


@dataclass(frozen=True)
class PaperInsight:
    title: str
    authors: str
    year: int
    venue: str
    what_it_does: str
    technique: str
    strength: str
    weakness: str
    our_response: str


@dataclass(frozen=True)
class GlossaryTerm:
    term: str
    short_definition: str
    easy_example: str


@dataclass(frozen=True)
class SecurityPrinciple:
    title: str
    description: str
    practical_rule: str


@dataclass(frozen=True)
class BuildUpItem:
    title: str
    why_needed: str
    deliverable: str


def project_direction() -> ProjectDirection:
    return ProjectDirection(
        title="라벨 희소 산업 환경에서의 멀티모달 불량 전파 예측",
        one_liner="Press 시계열, AOI 이미지, 이벤트 로그를 합친 멀티모달 모델로 불량이 어디서 시작되고 어떻게 퍼지는지 설명하는 방향입니다.",
        problem=(
            "현장에서는 데이터가 여러 시스템에 흩어져 있고, 불량 라벨은 적으며, "
            "불량 원인을 나중에 설명해야 합니다. 단순 분류만으로는 현장 의사결정에 부족합니다."
        ),
        why_it_matters=(
            "불량이 한 번 나면 원인 추적과 재발 방지가 중요합니다. 특히 FN(놓침) 비용이 크기 때문에 "
            "정확도만 보는 모델보다 비용과 설명까지 보는 모델이 필요합니다."
        ),
        proposed_solution=(
            "공개 데이터로 먼저 사전학습·베이스라인을 만들고, 반도체 PCB 적층 공정 실데이터가 오면 "
            "멀티모달 인코더 + 전파 예측 + 설명 가능한 출력으로 확장합니다."
        ),
        expected_contribution="1) 멀티모달 융합 2) 인과/전파 관점의 해석 3) 비용민감 평가 4) 실무에서 바로 읽을 수 있는 대시보드",
        scope_for_now=(
            "현재는 공개 데이터(SECOM, DeepPCB 등)로 구조를 안정화하고, 실제 반도체 PCB 적층 공정 데이터가 오면 "
            "라벨 정합과 성능 검증으로 넘어갑니다."
        ),
        security_notes=[
            "보안상 원본 데이터는 `data/raw/` 아래에서만 관리하고 Git에 올리지 않습니다.",
            "업로드 파일도 분석 후 필요 시 삭제하거나 `data/interim/uploads/`에서 관리합니다.",
            "비밀번호, 토큰, API Key는 저장하지 않고 환경변수 또는 로컬 비밀 저장소만 사용합니다.",
            "NDA/비밀유지 범위를 넘는 내용은 대시보드나 보고서에 노출하지 않습니다.",
        ],
        next_ideas=[
            "전처리 결과를 자동으로 학습 후보와 연결",
            "공개 데이터별 베이스라인 비교표 구축",
            "실데이터 매칭 키가 들어오면 cycle/panel/lot 정합",
            "불량 비용표를 반영한 임계값 정책 수립",
        ],
    )


def paper_insights() -> list[PaperInsight]:
    return [
        PaperInsight(
            title="Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting",
            authors="Lim et al.",
            year=2021,
            venue="International Journal of Forecasting",
            what_it_does="해석 가능한 시계열 예측 구조를 제안합니다.",
            technique="변수 선택 + attention 기반 시계열 인코더",
            strength="시계열에서 중요한 변수를 설명하기 좋습니다.",
            weakness="이미지나 이벤트처럼 다른 모달과의 결합은 별도 설계가 필요합니다.",
            our_response="Press 시계열의 기본 뼈대로 쓰고, AOI/이벤트는 별도 인코더로 붙입니다.",
        ),
        PaperInsight(
            title="Anomaly Transformer: Time Series Anomaly Detection with Association Discrepancy",
            authors="Xu et al.",
            year=2022,
            venue="ICLR",
            what_it_does="시계열 이상탐지에서 비정상 구간을 찾습니다.",
            technique="attention association discrepancy",
            strength="정상과 이상 사이 차이를 찾는 데 유리합니다.",
            weakness="이상 점수는 잘 주지만 현장 원인 설명은 제한적일 수 있습니다.",
            our_response="이상 구간 탐지 후, 전파 경로와 원인 변수 설명까지 연결합니다.",
        ),
        PaperInsight(
            title="Gated Multimodal Units for Information Fusion",
            authors="Arevalo et al.",
            year=2017,
            venue="ICLR Workshop",
            what_it_does="여러 모달을 안정적으로 합치는 기본 구조입니다.",
            technique="gated multimodal fusion",
            strength="모달 하나가 약해도 다른 모달이 보완할 수 있습니다.",
            weakness="복잡한 상호작용을 세밀하게 보여주기에는 단순할 수 있습니다.",
            our_response="초기 베이스라인으로 사용하고, 이후 cross-attention으로 고도화합니다.",
        ),
        PaperInsight(
            title="An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale",
            authors="Dosovitskiy et al.",
            year=2021,
            venue="ICLR",
            what_it_does="이미지를 transformer 방식으로 표현합니다.",
            technique="Vision Transformer",
            strength="이미지 패치 단위로 결함을 넓게 볼 수 있습니다.",
            weakness="라벨이 적으면 학습이 불안정할 수 있습니다.",
            our_response="DINOv2 같은 사전학습 표현을 먼저 쓰고, 필요할 때만 미세조정합니다.",
        ),
        PaperInsight(
            title="DINOv2: Learning Robust Visual Features without Supervision",
            authors="Oquab et al.",
            year=2024,
            venue="TMLR",
            what_it_does="라벨이 부족한 상황에서 쓸 수 있는 강한 이미지 표현을 만듭니다.",
            technique="self-supervised visual representation",
            strength="산업 이미지처럼 라벨이 적은 상황에 잘 맞습니다.",
            weakness="도메인 특수성은 여전히 데이터 적응이 필요합니다.",
            our_response="DeepPCB와 향후 AOI 이미지에 맞게 부분 미세조정합니다.",
        ),
        PaperInsight(
            title="SHAP: A Unified Approach to Interpreting Model Predictions",
            authors="Lundberg & Lee",
            year=2017,
            venue="NeurIPS",
            what_it_does="모델 예측에 어떤 변수가 영향을 줬는지 설명합니다.",
            technique="Shapley value 기반 설명",
            strength="비개발자도 이해하기 쉬운 설명 자료를 만들 수 있습니다.",
            weakness="대규모 고차원 데이터에서는 계산 비용이 큽니다.",
            our_response="중요 변수 중심으로 요약하고, 실시간 화면에는 핵심 변수만 보여줍니다.",
        ),
        PaperInsight(
            title="Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization",
            authors="Selvaraju et al.",
            year=2017,
            venue="ICCV",
            what_it_does="이미지에서 모델이 어디를 봤는지 보여줍니다.",
            technique="gradient-based localization",
            strength="AOI 결함 위치 설명에 바로 쓸 수 있습니다.",
            weakness="시계열/이벤트 설명에는 직접 적용하기 어렵습니다.",
            our_response="이미지 설명 도구로 쓰고, 시계열은 attention/importance로 보완합니다.",
        ),
        PaperInsight(
            title="Measuring classifier performance: a coherent alternative to AUC",
            authors="Hand",
            year=2009,
            venue="Machine Learning",
            what_it_does="AUC만으로는 부족한 상황에서 다른 평가 관점을 제시합니다.",
            technique="성능 측정/비용 관점",
            strength="불량처럼 비용이 비대칭인 문제에 잘 맞습니다.",
            weakness="현장별 비용표를 따로 정의해야 합니다.",
            our_response="FN 비용을 크게 두는 비용민감 지표를 함께 사용합니다.",
        ),
        PaperInsight(
            title="Machine Learning for Predictive Maintenance",
            authors="Susto et al.",
            year=2014,
            venue="IEEE Transactions on Industrial Informatics",
            what_it_does="산업 예지보전의 기본 흐름을 정리합니다.",
            technique="제조 데이터 기반 PdM 개요",
            strength="논문 서론과 관련연구의 기본 축으로 쓰기 좋습니다.",
            weakness="최근 멀티모달/해석가능성까지는 다루지 못합니다.",
            our_response="이 기본 틀 위에 멀티모달·인과·설명가능성을 추가합니다.",
        ),
    ]


def glossary_terms() -> list[GlossaryTerm]:
    return [
        GlossaryTerm(
            term="Cycle",
            short_definition="한 번의 공정 반복 단위입니다.",
            easy_example="Press 한 사이클이 하나의 샘플이 됩니다.",
        ),
        GlossaryTerm(
            term="Panel",
            short_definition="기판 한 장 또는 한 묶음 단위입니다.",
            easy_example="같은 Panel에서 나온 여러 Cycle을 묶어 봅니다.",
        ),
        GlossaryTerm(
            term="LOT",
            short_definition="같은 조건으로 묶인 생산 묶음입니다.",
            easy_example="LOT 단위로 불량률을 비교합니다.",
        ),
        GlossaryTerm(
            term="FAR",
            short_definition="False Alarm Rate, 오탐 비율입니다.",
            easy_example="정상인데 이상이라고 잘못 잡는 정도입니다.",
        ),
        GlossaryTerm(
            term="FN",
            short_definition="False Negative, 놓친 불량입니다.",
            easy_example="불량인데 정상으로 놓치는 경우입니다.",
        ),
        GlossaryTerm(
            term="FP",
            short_definition="False Positive, 잘못 잡은 정상입니다.",
            easy_example="정상인데 불량으로 표시하는 경우입니다.",
        ),
        GlossaryTerm(
            term="AUROC",
            short_definition="분류 성능의 기본 요약 지표입니다.",
            easy_example="임계값을 바꿔도 전반적으로 잘 구분되는지 봅니다.",
        ),
        GlossaryTerm(
            term="Cost-aware",
            short_definition="오탐과 미탐의 비용을 다르게 보는 평가 방식입니다.",
            easy_example="미탐이 더 비싸면 FN을 더 강하게 벌점 줍니다.",
        ),
        GlossaryTerm(
            term="Cross-attention",
            short_definition="한 모달이 다른 모달을 참고하는 attention입니다.",
            easy_example="시계열이 이미지 정보에 주의를 주는 식입니다.",
        ),
        GlossaryTerm(
            term="XAI",
            short_definition="Explainable AI, 모델 설명 기법입니다.",
            easy_example="어떤 변수 때문에 이상으로 판단했는지 보여줍니다.",
        ),
        GlossaryTerm(
            term="DVC",
            short_definition="데이터/모델 버전 관리 도구입니다.",
            easy_example="대용량 데이터는 Git 대신 DVC로 추적합니다.",
        ),
        GlossaryTerm(
            term="MLflow",
            short_definition="실험 기록 및 비교 도구입니다.",
            easy_example="어떤 설정이 성능이 좋았는지 자동으로 남깁니다.",
        ),
        GlossaryTerm(
            term="Hydra",
            short_definition="실험 설정을 파일로 관리하는 도구입니다.",
            easy_example="데이터/모델/학습 설정을 YAML로 나눠 둡니다.",
        ),
        GlossaryTerm(
            term="Weak supervision",
            short_definition="정확한 라벨 대신 약한 신호로 학습하는 방식입니다.",
            easy_example="정확한 cycle 라벨이 없어도 구간 단위 신호로 학습합니다.",
        ),
    ]


def security_principles() -> list[SecurityPrinciple]:
    return [
        SecurityPrinciple(
            title="원본 데이터는 외부로 내보내지 않기",
            description="NDA가 있는 데이터와 원본 로그는 Git에 올리지 않고 로컬 또는 DVC로만 관리합니다.",
            practical_rule="`data/raw/`는 추적 제외, 공유 파일은 익명화 후 별도 저장",
        ),
        SecurityPrinciple(
            title="비밀 정보는 코드에 쓰지 않기",
            description="API Key, 토큰, 비밀번호는 코드/문서/채팅에 적지 않습니다.",
            practical_rule="환경변수 또는 로컬 비밀 저장소만 사용",
        ),
        SecurityPrinciple(
            title="업로드 파일도 민감 데이터로 본다",
            description="웹 UI에 올린 파일은 분석 목적이 끝나면 삭제 또는 제한된 폴더에 보관합니다.",
            practical_rule="`data/interim/uploads/` 사용 후 필요 시 정리",
        ),
        SecurityPrinciple(
            title="로그에는 필요한 것만 남기기",
            description="대시보드와 CLI 로그에 원본값 전체를 남기지 않습니다.",
            practical_rule="샘플, 통계, 요약 중심으로 출력",
        ),
        SecurityPrinciple(
            title="권한은 최소한으로",
            description="실험용 계정과 공유 폴더는 필요한 사람만 접근합니다.",
            practical_rule="읽기/쓰기 폴더를 분리하고, 민감 폴더는 별도 관리",
        ),
    ]


def build_up_items() -> list[BuildUpItem]:
    return [
        BuildUpItem(
            title="참고 논문 노트",
            why_needed="논문 배경과 방법론을 한 곳에서 관리해야 합니다.",
            deliverable="`paper/notes/{author_year}.md`",
        ),
        BuildUpItem(
            title="참고문헌 관리",
            why_needed="인용 형식이 흔들리지 않게 해야 합니다.",
            deliverable="`paper/references.bib`",
        ),
        BuildUpItem(
            title="실험 결과 표준화",
            why_needed="모델을 바꿔도 같은 기준으로 비교해야 합니다.",
            deliverable="`reports/analysis/`와 `mlruns/`",
        ),
        BuildUpItem(
            title="용어 사전",
            why_needed="기획자/협업자가 같은 말을 같은 뜻으로 써야 합니다.",
            deliverable="대시보드의 용어 사전 페이지",
        ),
        BuildUpItem(
            title="보안 체크리스트",
            why_needed="NDA와 개인정보/영업비밀을 지켜야 합니다.",
            deliverable="대시보드의 보안 운영 원칙 페이지",
        ),
    ]



