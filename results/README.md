# Experiment Results

이 디렉터리는 FT-Transformer의 Target Encoding 축소 실험 결과를 정리합니다. 모든 Balanced Accuracy는 별도 표기가 없으면 fixed inverse-prior decision rule 기준입니다.

## 1. TE probability-sum audit

`te_probability_sum_audit.csv`에서 범주형 6개 feature의 세 클래스 TE 합은 사실상 1이었지만, 연속형 7개 feature는 exact-value mapping과 fallback 때문에 같은 성질이 항상 성립하지 않았습니다. 따라서 모든 feature에서 기준 클래스 TE를 일괄 삭제하는 대신 범주형과 연속형을 구분했습니다.

## 2. LOFO screening

`lofo_screening.csv`는 26개 TE 후보(각 원본 feature의 `fit`, `unhealthy`)에서 source-feature 단위로 하나씩 제외한 screening 결과입니다.

- `sleep_duration` TE 제거 시 성능 하락이 가장 컸습니다.
- `calorie_expenditure`, `step_count`, `heart_rate`도 유지 가치가 확인됐습니다.
- `bmi` TE 그룹 제거는 screening Balanced Accuracy를 소폭 높였습니다.
- 단일 screening 결과의 작은 차이를 확정적 중요도로 해석하지 않고, 후속 full CV 후보 선정에만 사용했습니다.

## 3. Candidate screening

`candidate_screening.csv`에서 다음 네 구성을 비교했습니다.

| Candidate | Raw | TE | Total | Screening BA |
|---|---:|---:|---:|---:|
| baseline52 | 13 | 39 | 52 | 0.949574 |
| safe46 | 13 | 33 | 46 | 0.950009 |
| aggressive39 | 13 | 26 | 39 | 0.949530 |
| aggressive37 | 13 | 24 | 37 | 0.950012 |

Screening 상위 후보인 Safe46과 Aggressive37을 full CV로 재평가했습니다.

## 4. Full CV finalist comparison

`finalist_full_cv_comparison.csv`는 7-fold, 16-epoch, `n_ens=1` 조건에서 모든 후처리 후보를 비교한 결과이며, `finalist_fixed_inverse_prior_comparison.csv`는 최종 fixed inverse-prior 행만 추린 파일입니다.

| Candidate | Total features | OOF BA | OOF LogLoss |
|---|---:|---:|---:|
| Safe46 | 46 | **0.950520** | **0.087019** |
| Aggressive37 | 37 | 0.950479 | 0.087085 |

두 후보 모두 fixed inverse-prior가 cross-fitted prior beta 및 unrestricted multiplier보다 높았습니다. CV 차이는 0.000041로 매우 작아 사실상 동률에 가깝습니다.

## 5. Kaggle comparison

`kaggle_leaderboard_comparison.csv`에 최종 제출 결과를 기록했습니다.

| Model | Features | Private | Public |
|---|---:|---:|---:|
| Original FT-Transformer v2 | 52 | 0.95066 | 0.95054 |
| Safe46 | 46 | 0.95066 | **0.95061** |
| Aggressive37 | **37** | **0.95067** | 0.95044 |

Aggressive37의 Private score가 수치상 가장 높지만 기존 모델과의 차이는 0.00001로, 유의미한 성능 향상으로 해석하기 어렵습니다. 핵심 결론은 전체 feature를 52개에서 37개로 28.8%, TE feature를 39개에서 24개로 38.5% 줄이면서 성능을 유지했다는 것입니다.
