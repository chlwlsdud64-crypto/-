# EV RSM 모델 데모

보존 온도(`temp_c`)와 보존 시간(`time_d`)을 입력으로 하여 아래 4개 반응변수에 대한 2차 반응표면모델(RSM)을 적합합니다.

- `particle_n` (입자 수)
- `pdi`
- `zeta_mV`
- `rna_ng_uL`

## 실행

```bash
python3 rsm_ev_model.py
```

## 출력

- `results/rsm_report.txt`: 모델 계수, R², 예측 예시
- `results/rsm_predictions.csv`: 예측 그리드 결과

입력 CSV를 사용할 경우:

```bash
python3 rsm_ev_model.py --input your_data.csv
```

CSV 컬럼:

`temp_c,time_d,particle_n,pdi,zeta_mV,rna_ng_uL`
