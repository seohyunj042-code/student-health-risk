# Student Health Risk — EDA to FT-Transformer

Kaggle **Predicting Student Health Risk (Playground Series S6E7)** 프로젝트입니다.

이 저장소는 단순 EDA 기록이 아니라, **데이터 이해 → 클래스 불균형 확인 → 베이스라인 모델 비교 → FT-Transformer 모델링 → inverse-prior 보정 → Kaggle 제출**까지 하나의 흐름으로 정리합니다.

## Project Flow

1. **EDA & Problem Understanding**
   - train/test 구조 확인
   - 숫자형 7개, 범주형 6개 변수 구분
   - `health_condition` 클래스 불균형 확인
   - 숫자형·범주형 변수와 건강 상태 관계 분석
   - Cramér's V 등으로 범주형 변수 관계 확인

2. **Baseline Model Comparison**
   - 동일한 Stratified holdout에서 HistGradientBoosting과 LightGBM 비교
   - LightGBM이 더 높은 Balanced Accuracy를 기록해 초기 주 모델로 선정

3. **FT-Transformer v2**
   - 원본 13개 feature 사용
   - 13개 feature 각각에 3개 클래스별 exact-value multiclass Target Encoding 추가
   - 39개 TE feature + 원본 13개 = **총 52개 input**
   - 7-fold Stratified CV
   - fold별 4-member FT-Transformer ensemble
   - Target Encoding은 outer fold 내부에서 5-fold cross-fitting으로 생성해 leakage 방지

4. **Inverse-Prior Decision Correction**
   - 데이터의 약 86%가 `at-risk`인 심한 클래스 불균형을 고려
   - raw probability를 그대로 argmax하지 않고, 각 class probability를 class prior로 나눠 decision boundary를 보정
   - Balanced Accuracy에서 minority class recall을 더 잘 반영하도록 조정

## Key Results

- `at-risk`: 약 **85.87%**로 클래스 불균형이 매우 심함
- EDA에서 수면 시간, 스트레스 수준, 수면의 질, 활동 수준 등에서 클래스별 차이가 확인됨
- HistGradientBoosting Balanced Accuracy: **0.9101**
- 초기 LightGBM Balanced Accuracy: 약 **0.9498**
- 개선 LightGBM Kaggle Private Score: **0.95012**
- **FT-Transformer v2 + inverse prior**
  - Private Balanced Accuracy: **0.95066**
  - Public Balanced Accuracy: **0.95054**
- 4-model class-wise ensemble Private Score: **0.95065**

현재 단일 모델 기준 최고 성능은 **FT-Transformer v2 + inverse-prior correction**입니다.

## Interpretation

이번 결과는 단순히 "Transformer가 LightGBM보다 좋았다"고 보기보다는 다음 요소들이 함께 작동한 결과로 해석합니다.

- class-specific Target Encoding으로 target과 feature의 관계를 추가 제공
- FT-Transformer가 feature interaction을 attention 기반으로 학습
- 7-fold CV와 fold별 4-member ensemble로 variance 감소
- Balanced Accuracy에 맞춰 inverse-prior correction으로 다수 클래스 쏠림 보정

## Notebooks

- [01. EDA & Baseline Model Plan](notebooks/01_seohyun_eda_model_plan.ipynb)
- [02. FT-Transformer v2 + Inverse-Prior Correction](notebooks/02_ft_transformer_v2_inverse_prior.ipynb)

## Final Modeling Pipeline

```text
EDA
↓
Class imbalance 확인
↓
Baseline model comparison
↓
Exact-value multiclass Target Encoding
↓
FT-Transformer
↓
7-fold CV + fold-wise ensemble
↓
Inverse-prior correction
↓
Kaggle submission
```
