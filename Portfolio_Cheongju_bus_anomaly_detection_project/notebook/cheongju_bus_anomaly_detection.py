#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
패턴인식_청주_버스노선_이상발생_감지_실습_최종_API키제거.py

청주시 버스노선 이상발생 감지 실습 (Tago API 키 완전 제거 버전)
작성일: 2025-12-28
작성자: R&D Planning Manager (Smart Factory DX Strategy)

사용법:
1. 환경변수 설정: export TAGO_KEY=실제_API_키
2. 실행: python 이파일명.py
"""

import os
import sys
import time
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib
matplotlib.use('Agg')  # 서버 환경에서 실행 가능
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
import warnings
warnings.filterwarnings('ignore')

print("=== 청주 버스노선 이상발생 감지 분석 시작 ===")

# =============================================================================
# 1. 환경 설정 및 한글 폰트 적용
# =============================================================================

def setup_environment():
    """환경 설정 및 한글 폰트 설치"""
    print("1. 환경 설정 중...")
    
    # 전역 상수 (API 키 완전 제거)
    os.environ.setdefault('TAGO_KEY', 'YOUR_TAGO_API_KEY_HERE')
    global TAGO_KEY, CHEONGJU_CITY_CODE
    TAGO_KEY = os.getenv('TAGO_KEY', 'YOUR_TAGO_API_KEY_HERE')
    CHEONGJU_CITY_CODE = "33010"
    
    if TAGO_KEY == 'YOUR_TAGO_API_KEY_HERE':
        print("⚠️  [경고] TAGO_KEY 환경변수가 설정되지 않았습니다.")
        print("   export TAGO_KEY=실제키값 명령어로 설정하세요.")
        print("   시뮬레이션 데이터만 사용됩니다.")
    
    # 한글 폰트 설정
    try:
        # Ubuntu/Debian 한글 폰트 설치
        os.system('sudo apt-get update -qq > /dev/null 2>&1')
        os.system('sudo apt-get install -y fonts-nanum-extra -qq > /dev/null 2>&1')
        os.system('fc-cache -fv > /dev/null 2>&1')
        
        font_candidates = ["NanumGothic", "NanumBarunGothic", "Malgun Gothic", "DejaVu Sans"]
        
        # NanumGothic 직접 추가
        try:
            fm.fontManager.addfont("/usr/share/fonts/truetype/nanum/NanumGothic.ttf")
        except:
            pass
        
        # 폰트 설정
        nanum_path = fm.findfont("NanumGothic")
        if nanum_path:
            plt.rcParams["font.family"] = "NanumGothic"
            print(f"✅ 한글 폰트 설정 완료: NanumGothic")
        else:
            for font_name in font_candidates:
                if font_name in [f.name for f in fm.fontManager.ttflist]:
                    plt.rcParams["font.family"] = font_name
                    print(f"✅ 한글 폰트 설정 완료: {font_name}")
                    break
            else:
                print("⚠️  기본 폰트(DejaVu Sans) 사용")
        
        plt.rcParams["axes.unicode_minus"] = False
        sns.set(style="whitegrid")
        
    except Exception as e:
        print(f"폰트 설정 중 오류: {e}")

# =============================================================================
# 2. Tago API 유틸리티 함수
# =============================================================================

def tago_get(url, params):
    """Tago API 호출 래퍼"""
    if TAGO_KEY == 'YOUR_TAGO_API_KEY_HERE':
        print("❌ API 키 없음. 더미 데이터 반환")
        return None
    
    processed_url = url.strip()
    try:
        r = requests.get(processed_url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        
        if not isinstance(data, dict):
            return None
        
        resp = data.get("response", {})
        header = resp.get("header", {})
        code = str(header.get("resultCode", ""))
        
        if code not in ("0", "00", "0000"):
            return None
        
        body = resp.get("body", {})
        if not isinstance(body, dict):
            return None
        
        return body
    except requests.exceptions.RequestException:
        return None

def get_city_codes():
    """도시코드 조회"""
    url = "https://apis.data.go.kr/1613000/BusRouteInfoInqireService/getCtyCodeList"
    params = {
        "serviceKey": TAGO_KEY,
        "_type": "json",
        "numOfRows": 200,
        "pageNo": 1,
    }
    body = tago_get(url, params)
    if body is None:
        return pd.DataFrame()
    
    items = body.get("items", {}).get("item", [])
    if isinstance(items, dict):
        items = [items]
    return pd.DataFrame(items)

def get_cheongju_routes_all(page_size=200):
    """청주시 전체 버스 노선 조회"""
    print("   청주 버스 노선 데이터 수집 중...")
    url = "https://apis.data.go.kr/1613000/BusRouteInfoInqireService/getRouteNoList"
    all_items = []
    page = 1
    
    while True:
        params = {
            "serviceKey": TAGO_KEY,
            "cityCode": CHEONGJU_CITY_CODE,
            "_type": "json",
            "numOfRows": page_size,
            "pageNo": page,
            "routeNo": ""
        }
        
        body = tago_get(url, params)
        if body is None:
            break
        
        items_from_body = body.get("items", {})
        items = items_from_body.get("item", []) if isinstance(items_from_body, dict) else []
        
        if not items:
            break
        
        if isinstance(items, dict):
            all_items.append(items)
        else:
            all_items.extend(items)
        
        if len(items) < page_size:
            break
        page += 1
    
    if not all_items:
        print("   ⚠️  API 데이터 없음. 더미 노선 데이터 생성")
        # 더미 데이터 생성
        dummy_routes = []
        for i in range(50):
            dummy_routes.append({
                'routeid': f'CJB{i:03d}0001',
                'routeno': f'{100+i:03d}',
                'routetpnm': np.random.choice(['간선', '지선', '순환']),
                'regionnm': '청주시'
            })
        df = pd.DataFrame(dummy_routes)
    else:
        df = pd.DataFrame(all_items)
        if 'routeid' in df.columns:
            df['routeid'] = df['routeid'].astype(str)
    
    df.to_csv('cheongju_bus_routes.csv', index=False, encoding='utf-8-sig')
    print(f"   ✅ 청주 노선 {len(df)}개 수집 완료 -> cheongju_bus_routes.csv")
    return df

# =============================================================================
# 3. 시뮬레이션 데이터 생성
# =============================================================================

def collect_snapshots(route_df, n_routes=10, start_date="2024-01-01", num_days=7):
    """버스 운행 시뮬레이션 데이터 생성"""
    print(f"2. 시뮬레이션 데이터 생성 중... ({n_routes}노선 x {num_days}일)")
    
    snapshot_list = []
    current_date_dt = datetime.strptime(start_date, '%Y-%m-%d')
    target_routes_df = route_df.head(n_routes)
    
    for day_offset in range(num_days):
        date_to_simulate = current_date_dt + timedelta(days=day_offset)
        is_weekend = date_to_simulate.weekday() >= 5
        
        for hour_of_day in range(24):
            ts = date_to_simulate.replace(hour=hour_of_day, minute=0, second=0).strftime("%Y-%m-%d %H:%M:%S")
            
            for _, row in target_routes_df.iterrows():
                rid = row["routeid"]
                routenm = row.get("routeno", "000")
                
                # 시간대별 버스 수 시뮬레이션
                if is_weekend:
                    num_buses = np.random.randint(0, 4)
                else:
                    if 6 <= hour_of_day <= 20:
                        num_buses = np.random.randint(2, 8)
                    else:
                        num_buses = np.random.randint(0, 3)
                
                # 위치 데이터 생성
                positions = []
                for i in range(num_buses):
                    positions.append({
                        'gpslati': 36.63 + np.random.rand() * 0.05,
                        'gpslong': 127.49 + np.random.rand() * 0.05,
                        'vehicleno': f'BUS_{rid}_{hour_of_day}_{i}',
                        'routeid': rid,
                        'routenm': routenm
                    })
                
                df_pos = pd.DataFrame(positions)
                if df_pos.empty:
                    df_pos = pd.DataFrame({'routeid': [rid]})
                snapshot_list.append((ts, df_pos))
    
    print(f"   ✅ {len(snapshot_list)}개 스냅샷 생성 완료")
    return snapshot_list

def build_route_hour_matrix(snapshot_list):
    """노선-시간 행렬 생성"""
    records = []
    for ts, df_pos in snapshot_list:
        dt = pd.to_datetime(ts)
        rid = df_pos['routeid'].iloc[0]
        num_veh = df_pos['vehicleno'].nunique() if 'vehicleno' in df_pos.columns else 0
        records.append({'routeid': rid, 'date': dt.date(), 'hour': dt.hour, 'num_veh': num_veh})
    
    if not records:
        return None, None
    
    df_data = pd.DataFrame(records)
    matrix = df_data.pivot_table(
        index=['routeid', 'date'], columns='hour', values='num_veh', aggfunc='first'
    ).fillna(0).reindex(columns=range(24), fill_value=0)
    
    return matrix, df_data

# =============================================================================
# 4. 머신러닝 분석
# =============================================================================

def analyze_anomalies(usage_matrix, n_clusters=8, contamination=0.1):
    """KMeans + IsolationForest 분석"""
    print("3. 머신러닝 분석 실행 중...")
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(usage_matrix.values)
    
    # KMeans 클러스터링
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    
    # IsolationForest 이상감지
    iso = IsolationForest(n_estimators=200, contamination=contamination, random_state=42)
    anomalies = iso.predict(X_scaled)
    scores = iso.score_samples(X_scaled)
    
    result = pd.DataFrame({
        'routeid_date': usage_matrix.index.map(lambda x: f"{x[0]}_{x[1]}"),
        'cluster': clusters,
        'anomaly': anomalies,
        'score': scores
    }).set_index('routeid_date')
    
    return result

def add_insights(usage_matrix, result):
    """인사이트 컬럼 추가"""
    hour_cols = [col for col in usage_matrix.columns if isinstance(col, int)]
    insights = []
    
    for idx, row in result.iterrows():
        rid_date = idx.split('_')
        if len(rid_date) == 2:
            rid, date_str = rid_date
            date = pd.to_datetime(date_str).date()
            if (rid, date) in usage_matrix.index:
                row_data = usage_matrix.loc[(rid, date)]
                peak_ratio = row_data[hour_cols].max() / row_data[hour_cols].sum() if row_data.sum() > 0 else 0
            else:
                peak_ratio = 0
        
        insights.append({
            'cluster': row['cluster'],
            'anomaly': '이상' if row['anomaly'] == -1 else '정상',
            'score': row['score'],
            'peak_ratio': peak_ratio
        })
    
    insight_df = pd.DataFrame(insights, index=result.index)
    
    def get_comment(row):
        if row['anomaly'] == '이상':
            if row['peak_ratio'] > 0.5:
                return "특정시간대_수요급증_증차검토"
            return "패턴이상_운영점검필요"
        return "정상"
    
    insight_df['comment'] = insight_df.apply(get_comment, axis=1)
    return insight_df

# =============================================================================
# 5. 시각화
# =============================================================================

def create_plots(usage_matrix, insights, n_clusters=8):
    """분석 결과 시각화"""
    print("4. 시각화 생성 중...")
    
    # 1. 클러스터 패턴
    plt.figure(figsize=(12, 6))
    hours = range(24)
    for c in range(n_clusters):
        cluster_routes = insights[insights['cluster'] == c].index
        if len(cluster_routes) > 0:
            routes_idx = []
            for rt in cluster_routes:
                rid, date = rt.split('_')
                routes_idx.append((rid, pd.to_datetime(date).date()))
            if routes_idx:
                mean_pattern = usage_matrix.loc[routes_idx].mean()
                plt.plot(hours, mean_pattern, marker='o', label=f'Cluster {c}')
    plt.title('클러스터별 24시간 운행 패턴')
    plt.xlabel('시간대')
    plt.ylabel('평균 차량수')
    plt.legend()
    plt.grid(True)
    plt.savefig('cluster_patterns.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. 이상 점수 분포
    plt.figure(figsize=(8, 5))
    sns.histplot(insights['score'], bins=20, kde=True)
    plt.title('이상 점수 분포')
    plt.xlabel('이상점수 (낮을수록 이상)')
    plt.savefig('anomaly_scores.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. 상위 이상 노선
    top_anomalies = insights[insights['anomaly'] == '이상'].nsmallest(10, 'score')
    plt.figure(figsize=(12, 6))
    sns.barplot(data=top_anomalies.reset_index(), x='routeid_date', y='score')
    plt.title('상위 10개 이상 노선')
    plt.xticks(rotation=45)
    plt.savefig('top_anomalies.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("   ✅ 차트 저장 완료: cluster_patterns.png, anomaly_scores.png, top_anomalies.png")

# =============================================================================
# 6. 메인 실행
# =============================================================================

def main():
    """메인 실행 함수"""
    try:
        # 환경 설정
        setup_environment()
        
        # 1. 노선 데이터 수집
        routes = get_cheongju_routes_all()
        
        # 2. 시뮬레이션 데이터 생성
        snapshots = collect_snapshots(routes)
        usage_matrix, raw_data = build_route_hour_matrix(snapshots)
        
        if usage_matrix is None or usage_matrix.empty:
            print("❌ 데이터 생성 실패")
            return
        
        # 3. 머신러닝 분석
        result = analyze_anomalies(usage_matrix)
        insights = add_insights(usage_matrix, result)
        
        # 4. 결과 저장
        insights.to_csv('cheongju_bus_anomaly_results.csv', encoding='utf-8-sig')
        usage_matrix.to_csv('usage_matrix.csv')
        raw_data.to_csv('raw_snapshots.csv', encoding='utf-8-sig')
        
        # 5. 시각화
        create_plots(usage_matrix, insights)
        
        print("\n" + "="*60)
        print("✅ 분석 완료!")
        print("="*60)
        print("생성된 파일:")
        print("  📄 cheongju_bus_routes.csv      - 노선 기본정보")
        print("  📄 cheongju_bus_anomaly_results.csv - 이상감지 결과")
        print("  📄 usage_matrix.csv             - 사용 행렬")
        print("  📄 raw_snapshots.csv            - 원본 시뮬레이션 데이터")
        print("  🖼️  cluster_patterns.png        - 클러스터 패턴")
        print("  🖼️  anomaly_scores.png          - 이상점수 분포")
        print("  🖼️  top_anomalies.png           - 상위 이상 노선")
        print("\n사용법: export TAGO_KEY=실제키값 후 재실행")
        
        # 요약 통계
        anomalies_count = (insights['anomaly'] == '이상').sum()
        print(f"\n📊 분석 요약:")
        print(f"   총 노선-일자: {len(insights):,}")
        print(f"   이상 감지: {anomalies_count:,}건 ({anomalies_count/len(insights)*100:.1f}%)")
        
    except Exception as e:
        print(f"❌ 실행 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
