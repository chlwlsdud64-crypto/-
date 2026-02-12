# EV RSM 모델 (실데이터용)

이 프로젝트는 보존 온도(`temp_c`)와 보존 시간(`time_d`)에 따라 아래 반응변수 변화를 예측하는 2요인 2차 RSM(반응표면모델) 파이프라인입니다.

- `particle_n` (입자 수)  ⬆️ maximize
- `pdi` ⬇️ minimize
- `zeta_mV` (기본적으로 -20mV 근처 선호)
- `rna_ng_uL` ⬆️ maximize

## 1) 실험 데이터 준비

CSV 컬럼(헤더 고정):

`temp_c,time_d,particle_n,pdi,zeta_mV,rna_ng_uL`

최소 12행 이상을 권장합니다(2차식 6개 계수 × 안정적 추정).

## 2) 실행

```bash
python3 rsm_ev_model.py --input your_data.csv
```

코딩 범위를 직접 지정하고 싶다면:

```bash
python3 rsm_ev_model.py --input your_data.csv --temp-min -80 --temp-max 4 --time-min 1 --time-max 90
```

## 3) 출력 해석

- `results/rsm_report.txt`
  - 코딩식(`x1`, `x2`) 정의
  - 반응별 계수(2차식) + R²
  - 다중반응 desirability 기반 추천 저장 조건
- `results/rsm_predictions.csv`
  - 코딩 좌표(-1~1)의 예측 그리드

## 4) 실제 RSM 구축 체크리스트

1. 실험설계: CCD 또는 Box-Behnken + 중심점 반복(최소 3회)
2. 데이터 QC: 이상치/측정 실패 제거, 단위 통일
3. 모델 적합: 2차식 + 교호작용
4. 진단: R², 잔차패턴, 중심점 재현성
5. 최적화: 목표(입자수↑, RNA↑, PDI↓, zeta 목표범위) 기반 desirability
6. 검증실험: 추천 조건에서 실제 반복실험으로 재확인

> 참고: `--input`을 생략하면 합성 데이터로 파이프라인만 점검합니다.
