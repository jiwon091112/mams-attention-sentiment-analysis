# MAMS Attention Sentiment Analysis

## 학습 결과 사이트

[학습 결과 대시보드](https://jiwon091112.github.io/mams-attention-sentiment-analysis/)에서 정확도, Macro-F1, 학습 곡선과 실제 예측 사례를 볼 수 있습니다. 사이트 소스는 `docs/` 폴더에 있습니다.

GitHub Pages 최초 설정: 저장소 **Settings → Pages → Deploy from a branch → main /docs → Save**. 활성화 후 위 주소로 접속할 수 있습니다. 사이트는 실제 실험 결과를 담은 정적 페이지이며, 모델 재학습이나 실시간 추론은 실행하지 않습니다.

MAMS 데이터로 LSTM의 평균 풀링과 대상 정보를 이용하는 Attention 풀링을 비교한 연구입니다.
문장과 평가 대상을 입력받아 **부정(0), 중립(1), 긍정(2)** 중 하나를 예측합니다. 평가 대상을 자동으로 찾는 과제는 아닙니다.

## 실제 실험 결과

동일한 데이터 분할과 학습 조건에서 시드 17, 42, 83으로 총 6개 모델을 학습했습니다.

| 세 시드 평균 | 평균 풀링 | Attention 풀링 |
|---|---:|---:|
| 정확도 | 65.1946% | 73.8273% |
| Macro-F1 | 0.634083 | 0.723159 |
| 문장의 모든 대상을 맞힌 비율 | 37.7333% | 48.4000% |

정확도 차이는 약 **+8.63%p**입니다. 이 조건에서의 제한된 비교 결과이며, 일반적인 우수성이나 원 논문 점수의 재현을 입증하지 않습니다.
시드 17·42가 최초 비교이고, 시드 83은 기존 테스트 결과를 본 뒤 조건 변경 없이 추가한 반복입니다. 새로운 맹검 검증은 아닙니다.
세 시드 보고서는 [recorded/followup_83/report_ko.txt](recorded/followup_83/report_ko.txt), 최초 두 시드 보고서는 [recorded/results/report_ko.txt](recorded/results/report_ko.txt)에 있습니다.

## 조건

- 공식 분할: 학습 4,297문장/11,186사례, 검증 500문장/1,332사례, 테스트 500문장/1,336사례.
- 두 모델 모두 단방향 1층 LSTM(은닉 64), 고정 GloVe 100차원, 동일한 평가 대상 벡터 및 위치 표식을 사용합니다.
- 최대 12 epoch, 검증 Macro-F1이 3회 연속 개선되지 않으면 종료합니다. 가장 좋은 검증 체크포인트를 선택합니다.
- AdamW, 학습률 0.001, 배치 32. 상세 조건은 [원래 계획](recorded/protocol.json)을 참고하세요.
- Attention에는 추가 학습 파라미터가 있습니다. 모델 용량의 효과까지 분리한 실험은 아닙니다.
- 불확실성 구간은 학습 모델을 고정한 문장 단위 부트스트랩 결과이며, 모든 학습·분할 변동을 포함하지 않습니다.

## 처음부터 실행

Python 3.11 환경에서 실행한 코드를 보존했습니다. 원래 연구 코드 세 파일은 수정하지 않았습니다.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python setup_data.py
python experiment.py prepare
python experiment.py run
python report.py
python repeat_one.py
```

`setup_data.py`는 공개 원자료 XML 3개와 GloVe(약 134MB)를 내려받아 원래 데이터 해시와 대조합니다.
처음 네 번의 학습 이후 `repeat_one.py`가 시드 83에서 두 번만 추가 학습합니다.
현재 폴더에서 이미 실행했다면 `run`은 결과를 덮어쓸 수 있습니다. 별도의 새 폴더에서 실행하세요.
새 실행 결과는 저장된 수치와 차이가 날 수 있습니다.

## 파일

- `experiment.py`: 전처리, 모델, 학습, 예측.
- `report.py`: 최초 두 시드의 결과 분석.
- `repeat_one.py`: 고정 조건의 세 번째 시드 반복 및 집계.
- `recorded/`: 실제 실행 당시 계획, 지표, 로그, 검증 기록. 원문과 대상 문자열은 예측 CSV에서 제외했으며 사례 ID·정답·예측·확률은 보존했습니다.
- 모델 가중치와 대용량 원자료는 Git에 포함하지 않았습니다. 기존 기록은 `recorded/`에 보존되어 새 실행으로 덮어쓰지 않습니다.

## 자료 출처

- [MAMS-for-ABSA](https://github.com/siat-nlp/MAMS-for-ABSA), MAMS-ATSA 공식 XML.
- [MAMS 논문](https://aclanthology.org/D19-1654/), Jiang et al. (2019), A Challenge Dataset and Effective Models for Aspect-Based Sentiment Analysis.
- [GloVe](https://nlp.stanford.edu/projects/glove/), [Gensim-data 배포본](https://github.com/RaRe-Technologies/gensim-data).

기존 Attention 방식을 비교·분석하는 연구이며 새로운 Attention 구조를 제안하지 않습니다. 원자료와 사전학습 벡터에는 각각의 배포 조건이 적용됩니다.
