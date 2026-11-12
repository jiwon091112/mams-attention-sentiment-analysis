MAMS 속성별 감성 분류의 제한된 비교 실험

자료: https://github.com/siat-nlp/MAMS-for-ABSA (원저자 공개 ATSA XML)
단어 벡터: GloVe Wikipedia 2014+Gigaword 5, 100d; Gensim-data 배포본
https://github.com/RaRe-Technologies/gensim-data
벡터 ZIP MD5: 40ec481866001177b8cd4cb0df92924f
다운로드 URL:
https://github.com/RaRe-Technologies/gensim-data/releases/download/glove-wiki-gigaword-100/glove-wiki-gigaword-100.gz

예산: 두 시드 x 두 모델 = 4회, 최대 12 epoch, 조기 종료 3회.
설정 탐색은 하지 않는다. 원래 제안한 30 epoch/5개 시드보다 축소한 예비 비교다.
두 모델에 동일한 대상 정보/위치 표식/사전 벡터를 제공해 공정한 기준을 사용한다.
벡터는 고정해 학습 파라미터와 계산량을 줄인다.
정답 속성을 주는 분류 과제이며 속성 자동 추출은 하지 않는다.
원래 MAMS 논문의 특정 모델 점수를 재현한 실험이라고 주장하지 않는다.

현재 준비 자료에서 재현:
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -u experiment.py run > results/run.log 2>&1
.venv/bin/python report.py
주의: run은 네 모델을 처음부터 재학습해 결과를 덮어쓴다. 기존 결과는 다른 폴더에 보존한다.

원자료에서 재현하려면 raw의 XML과 위 gzip 파일을 배치하고 새 폴더에서 prepare,run 순으로 실행한다.
완료된 protocol.json이 있으면 prepare는 덮어쓰지 않는다.
배포 ZIP에는 골라낸 embeddings.npy가 있어 134MB 원본 벡터 파일 없이 학습할 수 있다.
134MB 원본 벡터 gzip은 배포에서 생략한다.
prepared.pkl은 직접 만든 로컬 자료다. 신뢰하는 파일만 사용한다.

주 지표는 Macro-F1, 보조 지표는 정확도와 문장의 모든 속성을 맞힌 비율이다.
예측 파일은 각 속성별 정답/예측/세 감성 확률을 포함한다.
테스트는 네 실행을 모두 마친 뒤 평가하고 결과에 맞춘 추가 학습은 하지 않는다.
문장 단위 부트스트랩은 같은 문장의 여러 속성을 함께 뽑는다.
이 구간은 학습된 모델을 고정한 평가 표본 불확실성이며 학습 시드 변동 전체를 담지 않는다.

후속 고정 조건 반복:
사용자의 추가 학습 요청에 따라 새 시드 83에서 두 모델을 각 한 번만 추가 학습했다.
원래 코드/조건/자료는 그대로이며 계획과 결과는 followup_83에 별도로 저장한다.
repeat_one.py는 새 반복을 한 번 실행하는 코드다. 이미 plan.json이 있으면 중복 실행하지 않는다.
후속 결과는 기존 테스트를 본 뒤의 고정 조건 재현이며 새로운 맹검 검증이라고 주장하지 않는다.
현재 합계는 시드 17,42,83의 6개 모델이다. 최신 집계는 followup_83/report_ko.txt를 본다.
원래 results/report_ko.txt는 최초 두 시드 결과를 보존한 문서다.
