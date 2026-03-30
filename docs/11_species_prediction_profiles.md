# Species Prediction Profiles

> 74 prediction candidates with full dimension profiles.
> Generated from `scripts/eda_prediction_candidates.py` + origin/spec analysis.
> **Pick species worth predicting by reviewing the profiles below.**

## How to Read

Each species card shows:
- **Signal**: Lag-1 autocorrelation (raw daily / 7d smoothed). Higher = more predictable.
- **State**: Whether state (활/선/냉) matters for price. If partition needed, each state is a separate prediction target.
- **Origin**: Whether domestic vs foreign (imported) prices diverge. Ratio >1.3x or <0.7x = origin-sensitive.
- **Spec Ladder**: Price by size/count. Ratio >1.5x = spec-sensitive, needs per-spec-class models.
- **Flags**: `STATE` = needs state partition, `ORIGIN` = origin-sensitive, `SPEC` = spec-sensitive.

## Quick Reference

| # | Species | Tier | Rows | Days | Lag1(7d) | Price | State | Origin | Spec | Flags |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 전복 | A | 189,626 | 6,076 | 0.9943 | 32,253 | 활(99.3%) | for=0.0% (1.12x) | 5.6x | SPEC |
| 2 | 병어 | C | 104,736 | 5,649 | 0.9837 | 97,867 | 선(97.7%) | for=0.1% (0.75x) | 4.2x | SPEC |
| 3 | 삼치 | B | 94,176 | 5,983 | 0.9935 | 40,394 | 선(93.9%) | for=0.0% (0.67x) | 4.5x | SPEC |
| 4 | 아귀 | B | 88,909 | 5,924 | 0.9592 | 35,010 | 선(81.8%) | for=27.5% (2.4x) | 7.4x | STATE, ORIGIN, SPEC |
| 5 | 은갈치 | B | 86,923 | 6,013 | 0.9928 | 94,356 | 선(95.4%) | for=0.2% (0.71x) | 3.0x | SPEC |
| 6 | 대구 | B | 86,737 | 5,261 | 0.9926 | 30,363 | 선(96.3%) | for=15.4% (1.06x) | 3.4x | SPEC |
| 7 | 낙지 | A | 71,741 | 5,852 | 0.9869 | 26,079 | 선(85.0%) | for=92.0% (0.64x) | 3.7x | STATE, ORIGIN, SPEC |
| 8 | 오징어 | A | 58,888 | 5,890 | 0.9956 | 32,993 | 선(68.0%) | for=7.9% (2.26x) | 3.9x | STATE, ORIGIN, SPEC |
| 9 | 넙치 | A | 56,129 | 6,071 | 0.9904 | 14,930 | 활(62.4%) | for=0.7% (0.94x) | 1.5x | STATE |
| 10 | 고등어 | A | 55,446 | 5,667 | 0.9633 | 28,964 | 선(60.1%) | for=12.6% (1.43x) | 3.5x | STATE, ORIGIN, SPEC |
| 11 | 참돔 | A | 47,424 | 6,067 | 0.9785 | 15,770 | 활(66.3%) | for=34.8% (0.58x) | 1.3x | STATE, ORIGIN |
| 12 | 수꽃게 | A | 46,160 | 5,048 | 0.9885 | 13,889 | 활(68.9%) | for=5.1% (1.25x) | 1.6x | STATE, SPEC |
| 13 | 가자미 | C | 43,994 | 5,842 | 0.9762 | 37,516 | 선(90.3%) | for=3.4% (0.8x) | 4.2x | SPEC |
| 14 | 민어 | B | 42,411 | 5,920 | 0.9791 | 68,267 | 선(78.4%) | for=4.5% (0.41x) | 2.1x | STATE, SPEC |
| 15 | 잡어 | C | 42,206 | 5,997 | 0.891 | 5,021 | 선(81.1%) | for=7.4% (0.32x) | 1.3x | STATE, ORIGIN |
| 16 | 암꽃게 | A | 41,423 | 5,035 | 0.9941 | 22,628 | 활(77.8%) | for=2.0% (1.2x) | 1.5x | STATE |
| 17 | 농어 | A | 40,060 | 6,075 | 0.9763 | 13,977 | 활(66.3%) | for=29.9% (0.7x) | 1.5x | STATE |
| 18 | 백조기 | C | 34,591 | 5,526 | 0.969 | 32,999 | 선(99.1%) | for=0.1% (0.52x) | 3.8x | SPEC |
| 19 | 갑오징어 | B | 34,100 | 4,965 | 0.9823 | 39,782 | 선(98.5%) | for=0.9% (0.9x) | 3.7x | SPEC |
| 20 | 쭈꾸미 | B | 32,378 | 5,046 | 0.9816 | 29,452 | 선(79.1%) | for=48.2% (0.57x) | 1.6x | STATE, ORIGIN, SPEC |
| 21 | 방어 | A | 31,445 | 5,474 | 0.9562 | 5,697 | 선(56.4%) | for=14.4% (0.88x) | 1.5x | STATE |
| 22 | 소라 | C | 31,215 | 5,898 | 0.9873 | 76,042 | 활(98.3%) | for=4.0% (1.18x) | 2.4x | SPEC |
| 23 | 우럭 | B | 30,363 | 6,041 | 0.9621 | 10,308 | 선(50.2%) | for=2.2% (0.72x) | 1.7x | STATE, SPEC |
| 24 | 금태 | C | 29,993 | 5,552 | 0.99 | 133,952 | 선(99.9%) | for=0.1% (1.15x) | 5.8x | SPEC |
| 25 | 갈치 | C | 28,260 | 4,971 | 0.9762 | 69,473 | 선(72.0%) | for=19.3% (0.9x) | 4.0x | STATE, SPEC |
| 26 | 깐바지락 | B | 27,798 | 6,039 | 0.9961 | 33,154 | 선(99.3%) | for=6.6% (0.67x) | 6.0x | ORIGIN, SPEC |
| 27 | 만디 | B | 26,960 | 6,047 | 0.9956 | 7,619 | 선(99.2%) | for=0.7% (1.07x) | 4.2x | SPEC |
| 28 | 홍어 | C | 25,570 | 5,391 | 0.9785 | 60,593 | 선(86.0%) | for=55.3% (2.25x) | 2.7x | STATE, ORIGIN, SPEC |
| 29 | 문어 | C | 24,913 | 5,580 | 0.9801 | 60,671 | 선(71.8%) | for=0.7% (0.68x) | 2.3x | STATE, SPEC |
| 30 | 감숭어 | A | 23,160 | 5,794 | 0.9768 | 4,357 | 활(96.0%) | for=0.1% (0.78x) | 2.1x | SPEC |
| 31 | 참숭어 | A | 21,700 | 5,577 | 0.9806 | 5,428 | 활(98.5%) | for=0.1% (0.74x) | 1.7x | SPEC |
| 32 | 참가자미 | C | 20,483 | 5,315 | 0.9739 | 61,947 | 선(98.2%) | for=0.2% (0.73x) | 2.2x | SPEC |
| 33 | 새꼬막 | B | 20,338 | 5,614 | 0.9952 | 43,381 | 활(99.9%) | for=2.4% (1.05x) | 3.9x | SPEC |
| 34 | 깐굴 | A | 20,194 | 5,915 | 0.9947 | 16,724 | 선(99.7%) | for=0.1% (1.54x) | 2.7x | SPEC |
| 35 | 진주담치 | B | 19,625 | 5,761 | 0.9865 | 12,924 | 활(100.0%) | for=0.2% (2.67x) | 1.9x | SPEC |
| 36 | 간재미 | B | 19,070 | 5,457 | 0.8934 | 2,508 | 선(93.2%) | for=1.5% (1.07x) | 1.4x | - |
| 37 | 왕게 | A | 17,889 | 6,556 | 0.9957 | 34,314 | 활(50.4%) | for=97.8% (1.47x) | - | STATE, ORIGIN |
| 38 | 가무락 | C | 17,796 | 5,037 | 0.984 | 128,959 | 활(100.0%) | for=6.8% (0.58x) | 3.5x | ORIGIN, SPEC |
| 39 | 돔 | C | 17,121 | 5,047 | 0.8958 | 37,427 | 선(82.8%) | for=20.2% (0.76x) | 2.9x | STATE, SPEC |
| 40 | 물바지락 | C | 17,046 | 4,963 | 0.993 | 32,773 | 활(100.0%) | for=2.9% (0.76x) | 3.0x | SPEC |
| 41 | 도다리 | C | 16,093 | 4,277 | 0.9641 | 13,916 | 활(52.0%) | for=1.4% (0.51x) | 1.7x | STATE, SPEC |
| 42 | 피꼬막(피조개) | C | 15,471 | 4,554 | 0.9891 | 22,321 | 활(100.0%) | for=0.5% (1.56x) | 1.4x | - |
| 43 | 키조개 | C | 14,851 | 3,707 | 0.9788 | 41,702 | 선(60.4%) | for=0.3% (1.39x) | 2.3x | STATE, SPEC |
| 44 | 대게 | A | 14,332 | 6,424 | 0.9891 | 9,090 | 선(49.9%) | for=94.5% (1.14x) | - | STATE |
| 45 | 깐홍합 | C | 12,378 | 6,051 | 0.991 | 14,836 | 선(99.2%) | for=0.7% (0.88x) | 4.5x | SPEC |
| 46 | 청어 | C | 12,229 | 4,636 | 0.9834 | 6,457 | 선(94.3%) | for=4.9% (2.99x) | 2.3x | SPEC |
| 47 | 감성돔 | C | 12,183 | 4,908 | 0.9798 | 20,587 | 활(86.9%) | for=44.3% (0.67x) | 1.6x | STATE, ORIGIN, SPEC |
| 48 | 복어 | C | 11,847 | 4,427 | 0.9414 | 17,636 | 선(89.5%) | for=1.8% (2.89x) | 2.1x | STATE, SPEC |
| 49 | 개조개 | C | 11,780 | 4,228 | 0.9916 | 48,705 | 활(99.3%) | for=29.9% (0.59x) | 1.6x | ORIGIN, SPEC |
| 50 | 봉바지락 | C | 11,712 | 5,933 | 0.979 | 11,529 | 활(99.9%) | for=35.0% (0.78x) | 1.7x | SPEC |
| 51 | 미더덕 | C | 10,710 | 6,030 | 0.988 | 14,812 | 선(98.9%) | for=0.0% (0.58x) | 5.7x | SPEC |
| 52 | 토바지락 | C | 10,646 | 6,043 | 0.9758 | 5,766 | 활(100.0%) | for=60.4% (0.8x) | 1.4x | - |
| 53 | 해삼 | C | 10,633 | 3,764 | 0.9703 | 32,431 | 활(98.8%) | for=0.3% (2.44x) | 2.7x | SPEC |
| 54 | 연자돔 | C | 10,431 | 3,763 | 0.973 | 77,223 | 선(93.1%) | for=8.2% (0.64x) | 2.4x | ORIGIN, SPEC |
| 55 | 분홍새우 | C | 10,313 | 5,589 | 0.9972 | 21,642 | 선(98.3%) | for=0.8% (0.79x) | 2.5x | SPEC |
| 56 | 바다가재 | C | 10,206 | 4,091 | 0.9729 | 12,519 | 선(52.4%) | for=99.5% (1.16x) | - | STATE |
| 57 | 다랑어 | C | 10,007 | 3,182 | 0.9272 | 52,351 | 선(100.0%) | for=0.3% (0.71x) | 6.3x | SPEC |
| 58 | 칼바지락 | B | 9,760 | 6,054 | 0.9815 | 8,424 | 활(100.0%) | for=69.2% (0.82x) | 1.2x | - |
| 59 | 잿방어 | C | 9,259 | 3,444 | 0.9403 | 18,193 | 선(71.9%) | for=25.7% (1.0x) | 3.2x | STATE, SPEC |
| 60 | 가오리 | C | 8,840 | 4,157 | 0.926 | 23,865 | 선(78.3%) | for=18.5% (1.37x) | 2.6x | STATE, ORIGIN, SPEC |
| 61 | 점성어 | B | 8,572 | 5,925 | 0.99 | 7,294 | 활(81.9%) | for=98.2% (1.35x) | 1.5x | STATE, ORIGIN |
| 62 | 능성어 | C | 7,287 | 4,217 | 0.955 | 31,849 | 활(90.6%) | for=42.4% (0.83x) | 1.0x | - |
| 63 | 동죽 | C | 7,262 | 4,028 | 0.995 | 16,924 | 활(92.4%) | for=5.8% (0.6x) | 7.2x | ORIGIN, SPEC |
| 64 | 우럭조개 | C | 7,020 | 2,117 | 0.9913 | 59,951 | 활(99.6%) | for=0.1% (0.35x) | 2.5x | SPEC |
| 65 | 백합 | C | 6,800 | 3,620 | 0.992 | 106,329 | 활(99.8%) | for=32.6% (0.62x) | 2.3x | ORIGIN, SPEC |
| 66 | 꼴뚜기 | C | 6,612 | 2,915 | 0.9907 | 29,206 | 선(99.0%) | for=0.1% (0.36x) | 3.3x | SPEC |
| 67 | 줄돔 | C | 5,124 | 3,084 | 0.9689 | 54,264 | 활(90.8%) | for=50.8% (0.91x) | 1.5x | - |
| 68 | 염고등어 | C | 4,986 | 2,908 | 0.9957 | 33,083 | 냉(74.5%) | for=59.9% (1.45x) | 1.4x | STATE, ORIGIN |
| 69 | 부시리 | C | 4,399 | 1,698 | 0.9614 | 9,314 | 활(62.4%) | for=3.3% (0.65x) | 1.2x | STATE |
| 70 | 줄무늬전갱이 | C | 2,319 | 620 | 0.9837 | 32,070 | 활(75.0%) | for=99.7% (1.12x) | - | STATE |
| 71 | 자바리 | C | 2,123 | 1,260 | 0.9713 | 23,344 | 활(77.5%) | for=94.2% (0.41x) | - | STATE, ORIGIN |
| 72 | 강도다리 | C | 2,072 | 1,566 | 0.9695 | 17,413 | 활(97.3%) | for=7.2% (0.47x) | - | ORIGIN |
| 73 | 말백합 | C | 1,610 | 1,094 | 0.9739 | 117,242 | 활(100.0%) | for=0.1% (1.02x) | 2.2x | SPEC |
| 74 | 홍가리비 | C | 757 | 363 | 0.9916 | 19,321 | 활(100.0%) | for=0.5% (0.78x) | - | - |

---

## 1. 전복 (Tier A)

| Attribute | Value |
|---|---|
| Rows | 189,626 |
| Trading Days | 6,076 |
| Date Range | 2006.03.21 ~ 2026.01.03 |
| Mean Price | 32,253 KRW |
| CV (volatility) | 0.2785 |
| Lag-1 (daily) | 0.8299 |
| Lag-1 (7d smoothed) | 0.9943 |
| Dominant Packaging | kg (99.7%) |
| Flags | `SPEC` |

**State Distribution:**

| State | Rows | % | Mean Price |
|---|---|---|---|
| 활 | 188,362 | 99.3% | 32,663 |
| 냉 | 666 | 0.4% | 13,075 |
| 선 | 597 | 0.3% | 15,145 |
| 가공 | 1 | 0.0% | 18,000 |

**Origin:** Foreign 0.0%
, price ratio (foreign/domestic) = 1.12x

Top origins: 완도(175,951), 고흥(4,886), 대천(1,754), 태안(1,359), 안흥(1,094)

**Spec Price Ladder** (within 활/kg):

| Spec Class | Mean Price | Rows |
|---|---|---|
| 소 | 18,026 | 2,267 |
| 중 | 32,503 | 4,125 |
| 대 | 79,063 | 1,104 |
| 특대 | 101,453 | 89 |
| count_1-5 | 85,448 | 2,993 |
| count_6-10 | 45,636 | 19,766 |
| count_11-20 | 28,287 | 35,136 |
| count_21+ | 17,749 | 20,957 |
| other | 32,909 | 101,491 |

Size ratio: 5.6x

> Spec-sensitive — needs per-spec-class prediction.

---

## 2. 병어 (Tier C)

| Attribute | Value |
|---|---|
| Rows | 104,736 |
| Trading Days | 5,649 |
| Date Range | 2006.03.18 ~ 2026.01.02 |
| Mean Price | 97,867 KRW |
| CV (volatility) | 0.6734 |
| Lag-1 (daily) | 0.5807 |
| Lag-1 (7d smoothed) | 0.9837 |
| Dominant Packaging | S/P (91.8%) |
| Flags | `SPEC` |

**State Distribution:**

| State | Rows | % | Mean Price |
|---|---|---|---|
| 선 | 102,355 | 97.7% | 99,731 |
| 냉 | 2,367 | 2.3% | 118,874 |
| 활 | 14 | 0.0% | 22,200 |

**Origin:** Foreign 0.1%
, price ratio (foreign/domestic) = 0.75x

Top origins: 삼천포(24,674), 통영(21,746), 목포(14,732), 부산(기장)(11,322), 나로도(10,698)

**Spec Price Ladder** (within 선/S/P):

| Spec Class | Mean Price | Rows |
|---|---|---|
| 소 | 22,240 | 1,149 |
| 중 | 53,278 | 9,420 |
| 대 | 92,444 | 2,202 |
| 특대 | 87,267 | 42 |
| count_1-5 | 41,032 | 6,476 |
| count_6-10 | 31,074 | 16,214 |
| count_11-20 | 76,969 | 13,181 |
| count_21+ | 183,722 | 33,165 |
| other | 98,631 | 13,107 |

Size ratio: 4.2x

> Spec-sensitive — needs per-spec-class prediction.

---

## 3. 삼치 (Tier B)

| Attribute | Value |
|---|---|
| Rows | 94,176 |
| Trading Days | 5,983 |
| Date Range | 2006.03.21 ~ 2026.01.03 |
| Mean Price | 40,394 KRW |
| CV (volatility) | 0.8461 |
| Lag-1 (daily) | 0.8082 |
| Lag-1 (7d smoothed) | 0.9935 |
| Dominant Packaging | S/P (85.3%) |
| Flags | `SPEC` |

**State Distribution:**

| State | Rows | % | Mean Price |
|---|---|---|---|
| 선 | 88,391 | 93.9% | 32,452 |
| 냉 | 5,784 | 6.1% | 39,671 |
| 가공 | 1 | 0.0% | 67,000 |

**Origin:** Foreign 0.0%
, price ratio (foreign/domestic) = 0.67x

Top origins: 부산(기장)(13,153), 여수(12,037), 삼천포(10,399), 제주도(7,424), 통영(6,954)

**Spec Price Ladder** (within 선/S/P):

| Spec Class | Mean Price | Rows |
|---|---|---|
| 소 | 13,365 | 624 |
| 중 | 31,834 | 17,158 |
| 대 | 51,206 | 7,707 |
| 특대 | 60,404 | 157 |
| count_1-5 | 53,374 | 15,026 |
| count_6-10 | 26,720 | 28,836 |
| count_11-20 | 19,958 | 8,347 |
| count_21+ | 35,576 | 392 |
| other | 20,841 | 764 |

Size ratio: 4.5x

> Spec-sensitive — needs per-spec-class prediction.

---

## 4. 아귀 (Tier B)

| Attribute | Value |
|---|---|
| Rows | 88,909 |
| Trading Days | 5,924 |
| Date Range | 2006.03.21 ~ 2026.01.03 |
| Mean Price | 35,010 KRW |
| CV (volatility) | 0.7977 |
| Lag-1 (daily) | 0.3329 |
| Lag-1 (7d smoothed) | 0.9592 |
| Dominant Packaging | S/P (62.7%) |
| Flags | `STATE` `ORIGIN` `SPEC` |

**State Distribution:**

| State | Rows | % | Mean Price |
|---|---|---|---|
| 선 | 72,700 | 81.8% | 27,729 |
| 냉 | 16,189 | 18.2% | 69,813 |
| 활 | 20 | 0.0% | 5,495 |

> Needs state partition — price diverges >1.5x across states.

**Origin:** Foreign 27.5%
, price ratio (foreign/domestic) = 2.4x

> Origin-sensitive — must filter by domestic/foreign.

Top origins: 중국(19,185), 목포(10,417), 제주도(5,355), 방어진(4,979), 포항(4,784)

**Spec Price Ladder** (within 선/S/P):

| Spec Class | Mean Price | Rows |
|---|---|---|
| 소 | 17,905 | 1,258 |
| 중 | 29,595 | 13,048 |
| 대 | 50,798 | 4,206 |
| 특대 | 132,480 | 348 |
| count_1-5 | 31,702 | 19,929 |
| count_6-10 | 26,570 | 10,707 |
| count_11-20 | 27,575 | 2,646 |
| count_21+ | 40,720 | 169 |
| other | 25,279 | 2,010 |

Size ratio: 7.4x

> Spec-sensitive — needs per-spec-class prediction.

---

## 5. 은갈치 (Tier B)

| Attribute | Value |
|---|---|
| Rows | 86,923 |
| Trading Days | 6,013 |
| Date Range | 2006.03.25 ~ 2026.01.03 |
| Mean Price | 94,356 KRW |
| CV (volatility) | 0.4069 |
| Lag-1 (daily) | 0.7962 |
| Lag-1 (7d smoothed) | 0.9928 |
| Dominant Packaging | S/P (94.6%) |
| Flags | `SPEC` |

**State Distribution:**

| State | Rows | % | Mean Price |
|---|---|---|---|
| 선 | 82,938 | 95.4% | 88,782 |
| 냉 | 3,985 | 4.6% | 144,090 |

**Origin:** Foreign 0.2%
, price ratio (foreign/domestic) = 0.71x

Top origins: 제주도(83,912), 부산(기장)(682), 남해(656), 거문도(614), 추자도(323)

**Spec Price Ladder** (within 선/S/P):

| Spec Class | Mean Price | Rows |
|---|---|---|
| 소 | 20,252 | 524 |
| 중 | 57,691 | 4,230 |
| 대 | 60,303 | 941 |
| 특대 | 44,198 | 328 |
| count_1-5 | 128,459 | 15,543 |
| count_6-10 | 123,974 | 26,581 |
| count_11-20 | 55,417 | 16,583 |
| count_21+ | 42,412 | 13,962 |
| other | 46,806 | 2,646 |

Size ratio: 3.0x

> Spec-sensitive — needs per-spec-class prediction.

---

## 6. 대구 (Tier B)

| Attribute | Value |
|---|---|
| Rows | 86,737 |
| Trading Days | 5,261 |
| Date Range | 2006.03.14 ~ 2026.01.03 |
| Mean Price | 30,363 KRW |
| CV (volatility) | 0.7507 |
| Lag-1 (daily) | 0.8074 |
| Lag-1 (7d smoothed) | 0.9926 |
| Dominant Packaging | S/P (78.6%) |
| Flags | `SPEC` |

**State Distribution:**

| State | Rows | % | Mean Price |
|---|---|---|---|
| 선 | 83,504 | 96.3% | 32,743 |
| 냉 | 2,770 | 3.2% | 44,007 |
| 가공 | 316 | 0.4% | 49,735 |
| 활 | 147 | 0.2% | 8,755 |

**Origin:** Foreign 15.4%
, price ratio (foreign/domestic) = 1.06x

Top origins: 속초(13,393), 보령(10,422), 죽변(8,717), 중국(8,321), 방어진(7,804)

**Spec Price Ladder** (within 선/S/P):

| Spec Class | Mean Price | Rows |
|---|---|---|
| 소 | 14,511 | 517 |
| 중 | 42,120 | 13,848 |
| 대 | 49,746 | 6,657 |
| 특대 | 37,257 | 89 |
| count_1-5 | 29,615 | 28,489 |
| count_6-10 | 27,867 | 14,949 |
| count_11-20 | 23,978 | 2,882 |
| count_21+ | 36,508 | 109 |
| other | 28,245 | 399 |

Size ratio: 3.4x

> Spec-sensitive — needs per-spec-class prediction.

---

## 7. 낙지 (Tier A)

| Attribute | Value |
|---|---|
| Rows | 71,741 |
| Trading Days | 5,852 |
| Date Range | 2006.04.03 ~ 2026.01.03 |
| Mean Price | 26,079 KRW |
| CV (volatility) | 0.3284 |
| Lag-1 (daily) | 0.7739 |
| Lag-1 (7d smoothed) | 0.9869 |
| Dominant Packaging | box (84.4%) |
| Flags | `STATE` `ORIGIN` `SPEC` |

**State Distribution:**

| State | Rows | % | Mean Price |
|---|---|---|---|
| 선 | 61,004 | 85.0% | 24,982 |
| 활 | 5,631 | 7.8% | 40,853 |
| 냉 | 5,106 | 7.1% | 29,093 |

> Needs state partition — price diverges >1.5x across states.

**Origin:** Foreign 92.0%
, price ratio (foreign/domestic) = 0.64x

> Origin-sensitive — must filter by domestic/foreign.

Top origins: 중국(65,651), 태안(948), 대천(889), 인천(781), 대부도(513)

**Spec Price Ladder** (within 선/box):

| Spec Class | Mean Price | Rows |
|---|---|---|
| 소 | 18,200 | 26 |
| 중 | 27,264 | 387 |
| 대 | 29,421 | 1,073 |
| 특대 | 67,255 | 11 |
| count_1-5 | 21,218 | 580 |
| count_6-10 | 24,166 | 11,102 |
| count_11-20 | 26,032 | 26,706 |
| count_21+ | 23,351 | 18,224 |
| other | 29,843 | 2,172 |

Size ratio: 3.7x

> Spec-sensitive — needs per-spec-class prediction.

---

## 8. 오징어 (Tier A)

| Attribute | Value |
|---|---|
| Rows | 58,888 |
| Trading Days | 5,890 |
| Date Range | 2006.03.13 ~ 2026.01.03 |
| Mean Price | 32,993 KRW |
| CV (volatility) | 0.7041 |
| Lag-1 (daily) | 0.8674 |
| Lag-1 (7d smoothed) | 0.9956 |
| Dominant Packaging | S/P (73.9%) |
| Flags | `STATE` `ORIGIN` `SPEC` |

**State Distribution:**

| State | Rows | % | Mean Price |
|---|---|---|---|
| 선 | 40,039 | 68.0% | 25,488 |
| 냉 | 15,976 | 27.1% | 38,748 |
| 활 | 2,376 | 4.0% | 1,024 |
| 가공 | 367 | 0.6% | 27,610 |
| 냉건 | 130 | 0.2% | 177,856 |

> Needs state partition — price diverges >1.5x across states.

**Origin:** Foreign 7.9%
, price ratio (foreign/domestic) = 2.26x

> Origin-sensitive — must filter by domestic/foreign.

Top origins: 부산(기장)(6,698), 후포(6,046), 죽변(5,304), 속초(5,092), 구룡포(3,180)

**Spec Price Ladder** (within 선/S/P):

| Spec Class | Mean Price | Rows |
|---|---|---|
| 소 | 12,494 | 2,463 |
| 중 | 23,279 | 4,705 |
| 대 | 48,718 | 422 |
| count_1-5 | 21,747 | 62 |
| count_6-10 | 23,242 | 256 |
| count_11-20 | 29,351 | 22,383 |
| count_21+ | 21,569 | 6,794 |
| other | 20,437 | 530 |

Size ratio: 3.9x

> Spec-sensitive — needs per-spec-class prediction.

---

## 9. 넙치 (Tier A)

| Attribute | Value |
|---|---|
| Rows | 56,129 |
| Trading Days | 6,071 |
| Date Range | 2006.03.22 ~ 2026.01.03 |
| Mean Price | 14,930 KRW |
| CV (volatility) | 0.3088 |
| Lag-1 (daily) | 0.756 |
| Lag-1 (7d smoothed) | 0.9904 |
| Dominant Packaging | kg (76.2%) |
| Flags | `STATE` |

**State Distribution:**

| State | Rows | % | Mean Price |
|---|---|---|---|
| 활 | 35,029 | 62.4% | 14,721 |
| 선 | 21,064 | 37.5% | 11,655 |
| 냉 | 36 | 0.1% | 19,783 |

> Needs state partition — price diverges >1.5x across states.

**Origin:** Foreign 0.7%
, price ratio (foreign/domestic) = 0.94x

Top origins: 제주도(11,877), 군산(7,081), 완도(5,682), 태안(4,455), 안흥(4,325)

**Spec Price Ladder** (within 활/kg):

| Spec Class | Mean Price | Rows |
|---|---|---|
| 소 | 9,729 | 1,450 |
| 중 | 14,454 | 26,707 |
| 대 | 14,076 | 89 |
| count_1-5 | 16,225 | 4,604 |
| count_6-10 | 18,300 | 465 |
| count_11-20 | 16,093 | 29 |
| other | 18,432 | 1,588 |

Size ratio: 1.5x

---

## 10. 고등어 (Tier A)

| Attribute | Value |
|---|---|
| Rows | 55,446 |
| Trading Days | 5,667 |
| Date Range | 2006.03.16 ~ 2026.01.03 |
| Mean Price | 28,964 KRW |
| CV (volatility) | 0.4654 |
| Lag-1 (daily) | 0.4357 |
| Lag-1 (7d smoothed) | 0.9633 |
| Dominant Packaging | S/P (64.8%) |
| Flags | `STATE` `ORIGIN` `SPEC` |

**State Distribution:**

| State | Rows | % | Mean Price |
|---|---|---|---|
| 선 | 33,349 | 60.1% | 26,912 |
| 냉 | 22,034 | 39.7% | 36,238 |
| 활 | 43 | 0.1% | 8,907 |
| 가공 | 20 | 0.0% | 23,645 |

> Needs state partition — price diverges >1.5x across states.

**Origin:** Foreign 12.6%
, price ratio (foreign/domestic) = 1.43x

> Origin-sensitive — must filter by domestic/foreign.

Top origins: 부산(기장)(36,151), 제주도(6,599), 노르웨이(2,343), 일본(1,997), 중국(1,451)

**Spec Price Ladder** (within 선/S/P):

| Spec Class | Mean Price | Rows |
|---|---|---|
| 소 | 10,357 | 529 |
| 중 | 23,609 | 2,949 |
| 대 | 29,242 | 491 |
| 특대 | 36,286 | 14 |
| count_1-5 | 23,707 | 87 |
| count_6-10 | 27,659 | 1,672 |
| count_11-20 | 38,547 | 9,882 |
| count_21+ | 21,388 | 16,829 |
| other | 16,282 | 342 |

Size ratio: 3.5x

> Spec-sensitive — needs per-spec-class prediction.

---

## 11. 참돔 (Tier A)

| Attribute | Value |
|---|---|
| Rows | 47,424 |
| Trading Days | 6,067 |
| Date Range | 2006.03.21 ~ 2026.01.03 |
| Mean Price | 15,770 KRW |
| CV (volatility) | 0.2682 |
| Lag-1 (daily) | 0.5331 |
| Lag-1 (7d smoothed) | 0.9785 |
| Dominant Packaging | kg (87.1%) |
| Flags | `STATE` `ORIGIN` |

**State Distribution:**

| State | Rows | % | Mean Price |
|---|---|---|---|
| 활 | 31,438 | 66.3% | 15,510 |
| 선 | 15,778 | 33.3% | 21,440 |
| 냉 | 208 | 0.4% | 39,171 |

> Needs state partition — price diverges >1.5x across states.

**Origin:** Foreign 34.8%
, price ratio (foreign/domestic) = 0.58x

> Origin-sensitive — must filter by domestic/foreign.

Top origins: 일본(14,417), 통영(6,867), 완도(3,520), 제주도(3,068), 태안(2,065)

**Spec Price Ladder** (within 활/kg):

| Spec Class | Mean Price | Rows |
|---|---|---|
| 소 | 12,634 | 1,928 |
| 중 | 15,823 | 22,244 |
| 대 | 16,690 | 392 |
| count_1-5 | 15,658 | 4,904 |
| count_6-10 | 10,353 | 410 |
| count_11-20 | 12,636 | 11 |
| other | 14,725 | 1,477 |

Size ratio: 1.3x

---

## 12. 수꽃게 (Tier A)

| Attribute | Value |
|---|---|
| Rows | 46,160 |
| Trading Days | 5,048 |
| Date Range | 2004.04.06 ~ 2026.01.03 |
| Mean Price | 13,889 KRW |
| CV (volatility) | 0.378 |
| Lag-1 (daily) | 0.7537 |
| Lag-1 (7d smoothed) | 0.9885 |
| Dominant Packaging | kg (69.9%) |
| Flags | `STATE` `SPEC` |

**State Distribution:**

| State | Rows | % | Mean Price |
|---|---|---|---|
| 활 | 31,816 | 68.9% | 12,360 |
| 선 | 8,810 | 19.1% | 43,259 |
| 냉 | 5,534 | 12.0% | 47,420 |

> Needs state partition — price diverges >1.5x across states.

**Origin:** Foreign 5.1%
, price ratio (foreign/domestic) = 1.25x

Top origins: 태안(15,040), 진도(5,647), 서산(4,730), 인천(2,915), 안흥(2,449)

**Spec Price Ladder** (within 활/kg):

| Spec Class | Mean Price | Rows |
|---|---|---|
| 소 | 9,905 | 5,919 |
| 중 | 11,570 | 9,581 |
| 대 | 14,892 | 10,436 |
| 특대 | 16,258 | 971 |
| other | 10,111 | 4,695 |

Size ratio: 1.6x

> Spec-sensitive — needs per-spec-class prediction.

---

## 13. 가자미 (Tier C)

| Attribute | Value |
|---|---|
| Rows | 43,994 |
| Trading Days | 5,842 |
| Date Range | 2006.03.14 ~ 2026.01.03 |
| Mean Price | 37,516 KRW |
| CV (volatility) | 0.917 |
| Lag-1 (daily) | 0.4837 |
| Lag-1 (7d smoothed) | 0.9762 |
| Dominant Packaging | S/P (71.0%) |
| Flags | `SPEC` |

**State Distribution:**

| State | Rows | % | Mean Price |
|---|---|---|---|
| 선 | 39,708 | 90.3% | 32,162 |
| 냉건 | 1,469 | 3.3% | 14,352 |
| 냉 | 1,390 | 3.2% | 26,735 |
| 활 | 962 | 2.2% | 12,254 |
| 가공 | 465 | 1.1% | 20,239 |

**Origin:** Foreign 3.4%
, price ratio (foreign/domestic) = 0.8x

Top origins: 방어진(7,106), 보령(6,143), 속초(4,314), 제주도(2,549), 군산(2,215)

**Spec Price Ladder** (within 선/S/P):

| Spec Class | Mean Price | Rows |
|---|---|---|
| 소 | 15,807 | 417 |
| 중 | 34,746 | 11,193 |
| 대 | 61,809 | 3,184 |
| 특대 | 66,075 | 110 |
| count_1-5 | 18,005 | 1,062 |
| count_6-10 | 14,859 | 4,647 |
| count_11-20 | 20,965 | 4,405 |
| count_21+ | 40,878 | 2,367 |
| other | 77,666 | 2,373 |

Size ratio: 4.2x

> Spec-sensitive — needs per-spec-class prediction.

---

## 14. 민어 (Tier B)

| Attribute | Value |
|---|---|
| Rows | 42,411 |
| Trading Days | 5,920 |
| Date Range | 2006.03.21 ~ 2026.01.03 |
| Mean Price | 68,267 KRW |
| CV (volatility) | 0.7825 |
| Lag-1 (daily) | 0.5519 |
| Lag-1 (7d smoothed) | 0.9791 |
| Dominant Packaging | S/P (63.3%) |
| Flags | `STATE` `SPEC` |

**State Distribution:**

| State | Rows | % | Mean Price |
|---|---|---|---|
| 선 | 33,242 | 78.4% | 64,115 |
| 활 | 7,386 | 17.4% | 28,419 |
| 냉 | 1,783 | 4.2% | 76,966 |

> Needs state partition — price diverges >1.5x across states.

**Origin:** Foreign 4.5%
, price ratio (foreign/domestic) = 0.41x

Top origins: 제주도(10,335), 목포(9,955), 완도(5,604), 나로도(2,354), 신안(1,770)

**Spec Price Ladder** (within 선/S/P):

| Spec Class | Mean Price | Rows |
|---|---|---|
| 소 | 26,291 | 197 |
| 중 | 53,973 | 8,642 |
| 대 | 55,628 | 3,780 |
| 특대 | 48,168 | 75 |
| count_1-5 | 90,761 | 6,482 |
| count_6-10 | 108,084 | 4,114 |
| count_11-20 | 98,995 | 2,100 |
| count_21+ | 91,893 | 289 |
| other | 60,708 | 258 |

Size ratio: 2.1x

> Spec-sensitive — needs per-spec-class prediction.

---

## 15. 잡어 (Tier C)

| Attribute | Value |
|---|---|
| Rows | 42,206 |
| Trading Days | 5,997 |
| Date Range | 2005.10.01 ~ 2026.01.03 |
| Mean Price | 5,021 KRW |
| CV (volatility) | 1.9488 |
| Lag-1 (daily) | 0.0289 |
| Lag-1 (7d smoothed) | 0.891 |
| Dominant Packaging | kg (47.6%) |
| Flags | `STATE` `ORIGIN` |

**State Distribution:**

| State | Rows | % | Mean Price |
|---|---|---|---|
| 선 | 34,218 | 81.1% | 13,567 |
| 활 | 7,868 | 18.6% | 11,671 |
| 냉 | 120 | 0.3% | 24,814 |

> Needs state partition — price diverges >1.5x across states.

**Origin:** Foreign 7.4%
, price ratio (foreign/domestic) = 0.32x

> Origin-sensitive — must filter by domestic/foreign.

Top origins: 안흥(5,218), 통영(4,431), 목포(3,822), 제주도(3,170), 완도(2,759)

**Spec Price Ladder** (within 선/kg):

| Spec Class | Mean Price | Rows |
|---|---|---|
| 소 | 4,502 | 92 |
| 중 | 4,669 | 8,929 |
| 대 | 5,677 | 172 |
| count_1-5 | 6,450 | 1,750 |
| count_6-10 | 7,110 | 451 |
| count_11-20 | 14,598 | 133 |
| count_21+ | 44,640 | 45 |
| other | 5,815 | 696 |

Size ratio: 1.3x

---

## 16. 암꽃게 (Tier A)

| Attribute | Value |
|---|---|
| Rows | 41,423 |
| Trading Days | 5,035 |
| Date Range | 2004.04.06 ~ 2026.01.02 |
| Mean Price | 22,628 KRW |
| CV (volatility) | 0.481 |
| Lag-1 (daily) | 0.8797 |
| Lag-1 (7d smoothed) | 0.9941 |
| Dominant Packaging | kg (79.0%) |
| Flags | `STATE` |

**State Distribution:**

| State | Rows | % | Mean Price |
|---|---|---|---|
| 활 | 32,226 | 77.8% | 22,111 |
| 선 | 5,237 | 12.6% | 73,795 |
| 냉 | 3,960 | 9.6% | 77,021 |

> Needs state partition — price diverges >1.5x across states.

**Origin:** Foreign 2.0%
, price ratio (foreign/domestic) = 1.2x

Top origins: 태안(9,753), 진도(7,373), 인천(4,334), 서산(3,973), 격포(2,656)

**Spec Price Ladder** (within 활/kg):

| Spec Class | Mean Price | Rows |
|---|---|---|
| 소 | 17,624 | 5,527 |
| 중 | 21,313 | 10,270 |
| 대 | 26,214 | 12,076 |
| 특대 | 25,129 | 541 |
| other | 15,728 | 3,668 |

Size ratio: 1.5x

---

## 17. 농어 (Tier A)

| Attribute | Value |
|---|---|
| Rows | 40,060 |
| Trading Days | 6,075 |
| Date Range | 2006.03.21 ~ 2026.01.03 |
| Mean Price | 13,977 KRW |
| CV (volatility) | 0.2646 |
| Lag-1 (daily) | 0.511 |
| Lag-1 (7d smoothed) | 0.9763 |
| Dominant Packaging | kg (79.8%) |
| Flags | `STATE` |

**State Distribution:**

| State | Rows | % | Mean Price |
|---|---|---|---|
| 활 | 26,546 | 66.3% | 14,339 |
| 선 | 13,397 | 33.4% | 11,011 |
| 냉 | 117 | 0.3% | 20,435 |

> Needs state partition — price diverges >1.5x across states.

**Origin:** Foreign 29.9%
, price ratio (foreign/domestic) = 0.7x

Top origins: 중국(10,647), 완도(3,636), 통영(3,617), 목포(3,577), 부산(기장)(1,916)

**Spec Price Ladder** (within 활/kg):

| Spec Class | Mean Price | Rows |
|---|---|---|
| 소 | 9,889 | 1,003 |
| 중 | 14,433 | 20,269 |
| 대 | 13,351 | 118 |
| count_1-5 | 15,045 | 4,247 |
| count_6-10 | 8,528 | 140 |
| count_11-20 | 7,577 | 13 |
| other | 13,999 | 687 |

Size ratio: 1.5x

---

## 18. 백조기 (Tier C)

| Attribute | Value |
|---|---|
| Rows | 34,591 |
| Trading Days | 5,526 |
| Date Range | 2006.03.23 ~ 2026.01.02 |
| Mean Price | 32,999 KRW |
| CV (volatility) | 0.7993 |
| Lag-1 (daily) | 0.3676 |
| Lag-1 (7d smoothed) | 0.969 |
| Dominant Packaging | S/P (94.4%) |
| Flags | `SPEC` |

**State Distribution:**

| State | Rows | % | Mean Price |
|---|---|---|---|
| 선 | 34,282 | 99.1% | 28,808 |
| 냉 | 299 | 0.9% | 36,036 |
| 냉건 | 7 | 0.0% | 15,086 |
| 활 | 3 | 0.0% | 7,000 |

**Origin:** Foreign 0.1%
, price ratio (foreign/domestic) = 0.52x

Top origins: 나로도(13,526), 삼천포(4,533), 제주도(4,510), 통영(3,481), 여수(2,368)

**Spec Price Ladder** (within 선/S/P):

| Spec Class | Mean Price | Rows |
|---|---|---|
| 소 | 14,501 | 645 |
| 중 | 34,294 | 6,905 |
| 대 | 54,615 | 1,294 |
| 특대 | 41,737 | 188 |
| count_1-5 | 31,050 | 1,738 |
| count_6-10 | 16,778 | 7,575 |
| count_11-20 | 17,034 | 7,430 |
| count_21+ | 30,767 | 3,945 |
| other | 71,128 | 2,731 |

Size ratio: 3.8x

> Spec-sensitive — needs per-spec-class prediction.

---

## 19. 갑오징어 (Tier B)

| Attribute | Value |
|---|---|
| Rows | 34,100 |
| Trading Days | 4,965 |
| Date Range | 2006.03.21 ~ 2026.01.03 |
| Mean Price | 39,782 KRW |
| CV (volatility) | 1.1936 |
| Lag-1 (daily) | 0.593 |
| Lag-1 (7d smoothed) | 0.9823 |
| Dominant Packaging | S/P (54.7%) |
| Flags | `SPEC` |

**State Distribution:**

| State | Rows | % | Mean Price |
|---|---|---|---|
| 선 | 33,589 | 98.5% | 27,589 |
| 냉 | 322 | 0.9% | 39,220 |
| 활 | 188 | 0.6% | 11,923 |
| 가공 | 1 | 0.0% | 15,000 |

**Origin:** Foreign 0.9%
, price ratio (foreign/domestic) = 0.9x

Top origins: 군산(7,488), 보령(3,698), 포항(3,109), 속초(2,488), 여수(2,271)

**Spec Price Ladder** (within 선/S/P):

| Spec Class | Mean Price | Rows |
|---|---|---|
| 소 | 17,156 | 284 |
| 중 | 25,098 | 5,037 |
| 대 | 63,806 | 1,153 |
| 특대 | 62,818 | 11 |
| count_1-5 | 23,582 | 4,986 |
| count_6-10 | 34,539 | 4,590 |
| count_11-20 | 31,487 | 1,982 |
| count_21+ | 92,703 | 291 |
| other | 43,094 | 188 |

Size ratio: 3.7x

> Spec-sensitive — needs per-spec-class prediction.

---

## 20. 쭈꾸미 (Tier B)

| Attribute | Value |
|---|---|
| Rows | 32,378 |
| Trading Days | 5,046 |
| Date Range | 2006.03.28 ~ 2025.12.27 |
| Mean Price | 29,452 KRW |
| CV (volatility) | 0.4074 |
| Lag-1 (daily) | 0.6387 |
| Lag-1 (7d smoothed) | 0.9816 |
| Dominant Packaging | box (74.8%) |
| Flags | `STATE` `ORIGIN` `SPEC` |

**State Distribution:**

| State | Rows | % | Mean Price |
|---|---|---|---|
| 선 | 25,624 | 79.1% | 29,450 |
| 활 | 3,713 | 11.5% | 52,522 |
| 냉 | 3,041 | 9.4% | 18,918 |

> Needs state partition — price diverges >1.5x across states.

**Origin:** Foreign 48.2%
, price ratio (foreign/domestic) = 0.57x

> Origin-sensitive — must filter by domestic/foreign.

Top origins: 중국(13,159), 군산(6,105), 장항(5,280), 베트남(1,842), 인천(1,705)

**Spec Price Ladder** (within 선/box):

| Spec Class | Mean Price | Rows |
|---|---|---|
| 소 | 20,857 | 437 |
| 중 | 34,093 | 3,602 |
| 대 | 26,118 | 2,941 |
| count_1-5 | 31,521 | 151 |
| count_6-10 | 18,337 | 457 |
| count_11-20 | 21,998 | 3,411 |
| count_21+ | 21,512 | 5,427 |
| other | 34,061 | 7,648 |

Size ratio: 1.6x

> Spec-sensitive — needs per-spec-class prediction.

---

## 21. 방어 (Tier A)

| Attribute | Value |
|---|---|
| Rows | 31,445 |
| Trading Days | 5,474 |
| Date Range | 2006.03.21 ~ 2026.01.03 |
| Mean Price | 5,697 KRW |
| CV (volatility) | 0.8543 |
| Lag-1 (daily) | 0.3379 |
| Lag-1 (7d smoothed) | 0.9562 |
| Dominant Packaging | kg (50.3%) |
| Flags | `STATE` |

**State Distribution:**

| State | Rows | % | Mean Price |
|---|---|---|---|
| 선 | 17,720 | 56.4% | 14,109 |
| 활 | 13,652 | 43.4% | 12,144 |
| 냉 | 73 | 0.2% | 7,881 |

> Needs state partition — price diverges >1.5x across states.

**Origin:** Foreign 14.4%
, price ratio (foreign/domestic) = 0.88x

Top origins: 일본(4,429), 속초(3,745), 통영(3,021), 제주도(2,725), 동해시(2,537)

**Spec Price Ladder** (within 선/kg):

| Spec Class | Mean Price | Rows |
|---|---|---|
| 소 | 3,694 | 18 |
| 중 | 5,594 | 2,713 |
| 대 | 5,232 | 104 |
| count_1-5 | 6,272 | 569 |
| other | 3,718 | 28 |

Size ratio: 1.5x

---

## 22. 소라 (Tier C)

| Attribute | Value |
|---|---|
| Rows | 31,215 |
| Trading Days | 5,898 |
| Date Range | 2005.03.18 ~ 2026.01.03 |
| Mean Price | 76,042 KRW |
| CV (volatility) | 0.4212 |
| Lag-1 (daily) | 0.7142 |
| Lag-1 (7d smoothed) | 0.9873 |
| Dominant Packaging | 그물망 (47.7%) |
| Flags | `SPEC` |

**State Distribution:**

| State | Rows | % | Mean Price |
|---|---|---|---|
| 활 | 30,689 | 98.3% | 75,732 |
| 선 | 393 | 1.3% | 24,568 |
| 냉 | 112 | 0.4% | 72,994 |
| 가공 | 21 | 0.1% | 50,438 |

**Origin:** Foreign 4.0%
, price ratio (foreign/domestic) = 1.18x

Top origins: 군산(5,586), 인천(4,182), 장항(3,301), 순천(2,933), 대천(2,252)

**Spec Price Ladder** (within 활/그물망):

| Spec Class | Mean Price | Rows |
|---|---|---|
| 소 | 38,892 | 1,770 |
| 중 | 71,630 | 9,613 |
| 대 | 82,105 | 2,449 |
| 특대 | 92,112 | 92 |
| other | 60,358 | 965 |

Size ratio: 2.4x

> Spec-sensitive — needs per-spec-class prediction.

---

## 23. 우럭 (Tier B)

| Attribute | Value |
|---|---|
| Rows | 30,363 |
| Trading Days | 6,041 |
| Date Range | 2006.03.23 ~ 2026.01.03 |
| Mean Price | 10,308 KRW |
| CV (volatility) | 0.5285 |
| Lag-1 (daily) | 0.2787 |
| Lag-1 (7d smoothed) | 0.9621 |
| Dominant Packaging | kg (70.1%) |
| Flags | `STATE` `SPEC` |

**State Distribution:**

| State | Rows | % | Mean Price |
|---|---|---|---|
| 선 | 15,239 | 50.2% | 16,343 |
| 활 | 14,960 | 49.3% | 11,191 |
| 냉 | 163 | 0.5% | 24,969 |

> Needs state partition — price diverges >1.5x across states.

**Origin:** Foreign 2.2%
, price ratio (foreign/domestic) = 0.72x

Top origins: 통영(9,607), 충무(3,795), 삼천포(2,349), 완도(1,798), 흑산도(1,631)

**Spec Price Ladder** (within 선/kg):

| Spec Class | Mean Price | Rows |
|---|---|---|
| 소 | 6,103 | 39 |
| 중 | 10,278 | 6,007 |
| 대 | 6,611 | 18 |
| count_1-5 | 8,306 | 90 |
| count_6-10 | 9,997 | 39 |
| count_11-20 | 8,865 | 23 |
| count_21+ | 15,600 | 28 |
| other | 8,745 | 117 |

Size ratio: 1.7x

> Spec-sensitive — needs per-spec-class prediction.

---

## 24. 금태 (Tier C)

| Attribute | Value |
|---|---|
| Rows | 29,993 |
| Trading Days | 5,552 |
| Date Range | 2006.03.21 ~ 2026.01.03 |
| Mean Price | 133,952 KRW |
| CV (volatility) | 0.8157 |
| Lag-1 (daily) | 0.6827 |
| Lag-1 (7d smoothed) | 0.99 |
| Dominant Packaging | S/P (94.4%) |
| Flags | `SPEC` |

**State Distribution:**

| State | Rows | % | Mean Price |
|---|---|---|---|
| 선 | 29,969 | 99.9% | 107,689 |
| 냉 | 24 | 0.1% | 47,004 |

**Origin:** Foreign 0.1%
, price ratio (foreign/domestic) = 1.15x

Top origins: 제주도(9,553), 부산(기장)(9,501), 목포(5,439), 방어진(2,254), 삼천포(1,210)

**Spec Price Ladder** (within 선/S/P):

| Spec Class | Mean Price | Rows |
|---|---|---|
| 소 | 26,972 | 267 |
| 중 | 99,758 | 7,507 |
| 대 | 155,743 | 1,684 |
| 특대 | 70,790 | 40 |
| count_1-5 | 58,810 | 770 |
| count_6-10 | 47,066 | 2,307 |
| count_11-20 | 106,917 | 6,006 |
| count_21+ | 159,646 | 6,303 |
| other | 97,591 | 3,416 |

Size ratio: 5.8x

> Spec-sensitive — needs per-spec-class prediction.

---

## 25. 갈치 (Tier C)

| Attribute | Value |
|---|---|
| Rows | 28,260 |
| Trading Days | 4,971 |
| Date Range | 2006.03.21 ~ 2026.01.02 |
| Mean Price | 69,473 KRW |
| CV (volatility) | 0.8922 |
| Lag-1 (daily) | 0.5273 |
| Lag-1 (7d smoothed) | 0.9762 |
| Dominant Packaging | S/P (67.0%) |
| Flags | `STATE` `SPEC` |

**State Distribution:**

| State | Rows | % | Mean Price |
|---|---|---|---|
| 선 | 20,357 | 72.0% | 77,271 |
| 냉 | 7,897 | 27.9% | 61,472 |
| 활 | 6 | 0.0% | 59,567 |

> Needs state partition — price diverges >1.5x across states.

**Origin:** Foreign 19.3%
, price ratio (foreign/domestic) = 0.9x

Top origins: 부산(기장)(3,568), 목포(2,848), 중국(2,824), 안흥(2,322), 남해(2,017)

**Spec Price Ladder** (within 선/S/P):

| Spec Class | Mean Price | Rows |
|---|---|---|
| 소 | 21,587 | 127 |
| 중 | 50,570 | 4,063 |
| 대 | 56,795 | 1,234 |
| 특대 | 85,381 | 110 |
| count_1-5 | 86,677 | 1,220 |
| count_6-10 | 108,120 | 2,536 |
| count_11-20 | 110,694 | 3,968 |
| count_21+ | 83,853 | 3,760 |
| other | 68,672 | 1,080 |

Size ratio: 4.0x

> Spec-sensitive — needs per-spec-class prediction.

---

## 26. 깐바지락 (Tier B)

| Attribute | Value |
|---|---|
| Rows | 27,798 |
| Trading Days | 6,039 |
| Date Range | 2006.04.05 ~ 2026.01.03 |
| Mean Price | 33,154 KRW |
| CV (volatility) | 0.5006 |
| Lag-1 (daily) | 0.8925 |
| Lag-1 (7d smoothed) | 0.9961 |
| Dominant Packaging | box (99.3%) |
| Flags | `ORIGIN` `SPEC` |

**State Distribution:**

| State | Rows | % | Mean Price |
|---|---|---|---|
| 선 | 27,615 | 99.3% | 36,411 |
| 냉 | 168 | 0.6% | 34,781 |
| 가공 | 15 | 0.1% | 13,200 |

**Origin:** Foreign 6.6%
, price ratio (foreign/domestic) = 0.67x

> Origin-sensitive — must filter by domestic/foreign.

Top origins: 삼천포(10,294), 여수(8,359), 사천(2,917), 순천(1,303), 남해(1,206)

**Spec Price Ladder** (within 선/box):

| Spec Class | Mean Price | Rows |
|---|---|---|
| 소 | 14,823 | 17,212 |
| 중 | 25,496 | 263 |
| 대 | 89,092 | 7,599 |
| other | 26,078 | 2,503 |

Size ratio: 6.0x

> Spec-sensitive — needs per-spec-class prediction.

---

## 27. 만디 (Tier B)

| Attribute | Value |
|---|---|
| Rows | 26,960 |
| Trading Days | 6,047 |
| Date Range | 2006.04.03 ~ 2026.01.03 |
| Mean Price | 7,619 KRW |
| CV (volatility) | 0.6374 |
| Lag-1 (daily) | 0.8701 |
| Lag-1 (7d smoothed) | 0.9956 |
| Dominant Packaging | box (99.8%) |
| Flags | `SPEC` |

**State Distribution:**

| State | Rows | % | Mean Price |
|---|---|---|---|
| 선 | 26,743 | 99.2% | 7,396 |
| 냉 | 204 | 0.8% | 13,814 |
| 활 | 13 | 0.0% | 6,108 |

**Origin:** Foreign 0.7%
, price ratio (foreign/domestic) = 1.07x

Top origins: 마산(12,444), 통영(7,006), 여수(5,273), 고성(1,204), 삼천포(534)

**Spec Price Ladder** (within 선/box):

| Spec Class | Mean Price | Rows |
|---|---|---|
| 소 | 5,704 | 15,605 |
| 중 | 8,918 | 176 |
| 대 | 24,176 | 2,054 |
| other | 6,448 | 8,855 |

Size ratio: 4.2x

> Spec-sensitive — needs per-spec-class prediction.

---

## 28. 홍어 (Tier C)

| Attribute | Value |
|---|---|
| Rows | 25,570 |
| Trading Days | 5,391 |
| Date Range | 2004.09.18 ~ 2026.01.03 |
| Mean Price | 60,593 KRW |
| CV (volatility) | 0.8435 |
| Lag-1 (daily) | 0.4765 |
| Lag-1 (7d smoothed) | 0.9785 |
| Dominant Packaging | S/P (64.1%) |
| Flags | `STATE` `ORIGIN` `SPEC` |

**State Distribution:**

| State | Rows | % | Mean Price |
|---|---|---|---|
| 선 | 21,992 | 86.0% | 75,168 |
| 냉 | 3,346 | 13.1% | 68,608 |
| 활 | 232 | 0.9% | 10,282 |

> Needs state partition — price diverges >1.5x across states.

**Origin:** Foreign 55.3%
, price ratio (foreign/domestic) = 2.25x

> Origin-sensitive — must filter by domestic/foreign.

Top origins: 중국(8,863), 일본(2,809), 인천(2,647), 목포(2,014), 방어진(1,460)

**Spec Price Ladder** (within 선/S/P):

| Spec Class | Mean Price | Rows |
|---|---|---|
| 소 | 37,138 | 58 |
| 중 | 46,347 | 4,527 |
| 대 | 24,025 | 1,363 |
| 특대 | 17,488 | 17 |
| count_1-5 | 99,475 | 5,437 |
| count_6-10 | 115,067 | 3,930 |
| count_11-20 | 89,718 | 462 |
| count_21+ | 68,461 | 28 |
| other | 43,513 | 252 |

Size ratio: 2.7x

> Spec-sensitive — needs per-spec-class prediction.

---

## 29. 문어 (Tier C)

| Attribute | Value |
|---|---|
| Rows | 24,913 |
| Trading Days | 5,580 |
| Date Range | 2006.03.21 ~ 2026.01.03 |
| Mean Price | 60,671 KRW |
| CV (volatility) | 0.8607 |
| Lag-1 (daily) | 0.5224 |
| Lag-1 (7d smoothed) | 0.9801 |
| Dominant Packaging | S/P (65.2%) |
| Flags | `STATE` `SPEC` |

**State Distribution:**

| State | Rows | % | Mean Price |
|---|---|---|---|
| 선 | 17,878 | 71.8% | 62,569 |
| 활 | 6,814 | 27.4% | 20,947 |
| 냉 | 208 | 0.8% | 37,187 |
| 가공 | 13 | 0.1% | 123,077 |

> Needs state partition — price diverges >1.5x across states.

**Origin:** Foreign 0.7%
, price ratio (foreign/domestic) = 0.68x

Top origins: 속초(5,319), 포항(5,147), 완도(1,421), 목포(1,371), 죽변(1,063)

**Spec Price Ladder** (within 선/S/P):

| Spec Class | Mean Price | Rows |
|---|---|---|
| 소 | 31,727 | 843 |
| 중 | 62,677 | 9,169 |
| 대 | 72,317 | 4,833 |
| 특대 | 72,922 | 37 |
| count_1-5 | 62,408 | 1,049 |
| count_6-10 | 37,277 | 97 |
| count_11-20 | 40,016 | 31 |
| count_21+ | 19,600 | 15 |
| other | 56,876 | 37 |

Size ratio: 2.3x

> Spec-sensitive — needs per-spec-class prediction.

---

## 30. 감숭어 (Tier A)

| Attribute | Value |
|---|---|
| Rows | 23,160 |
| Trading Days | 5,794 |
| Date Range | 2006.04.04 ~ 2026.01.03 |
| Mean Price | 4,357 KRW |
| CV (volatility) | 0.5224 |
| Lag-1 (daily) | 0.6149 |
| Lag-1 (7d smoothed) | 0.9768 |
| Dominant Packaging | kg (98.4%) |
| Flags | `SPEC` |

**State Distribution:**

| State | Rows | % | Mean Price |
|---|---|---|---|
| 활 | 22,242 | 96.0% | 4,110 |
| 선 | 917 | 4.0% | 2,469 |
| 냉 | 1 | 0.0% | 2,000 |

**Origin:** Foreign 0.1%
, price ratio (foreign/domestic) = 0.78x

Top origins: 여수(3,569), 군산(2,441), 거제도(2,418), 완도(1,947), 격포(1,594)

**Spec Price Ladder** (within 활/kg):

| Spec Class | Mean Price | Rows |
|---|---|---|
| 소 | 2,255 | 1,096 |
| 중 | 3,894 | 18,682 |
| 대 | 4,750 | 20 |
| count_1-5 | 6,860 | 682 |
| count_6-10 | 6,818 | 394 |
| count_11-20 | 6,239 | 284 |
| count_21+ | 8,075 | 36 |
| other | 6,418 | 1,018 |

Size ratio: 2.1x

> Spec-sensitive — needs per-spec-class prediction.

---

## 31. 참숭어 (Tier A)

| Attribute | Value |
|---|---|
| Rows | 21,700 |
| Trading Days | 5,577 |
| Date Range | 2006.04.17 ~ 2026.01.03 |
| Mean Price | 5,428 KRW |
| CV (volatility) | 0.4591 |
| Lag-1 (daily) | 0.6461 |
| Lag-1 (7d smoothed) | 0.9806 |
| Dominant Packaging | kg (99.6%) |
| Flags | `SPEC` |

**State Distribution:**

| State | Rows | % | Mean Price |
|---|---|---|---|
| 활 | 21,382 | 98.5% | 5,685 |
| 선 | 317 | 1.5% | 3,359 |
| 냉 | 1 | 0.0% | 30,000 |

**Origin:** Foreign 0.1%
, price ratio (foreign/domestic) = 0.74x

Top origins: 인천(4,192), 여수(2,284), 군산(1,994), 완도(1,330), 격포(1,288)

**Spec Price Ladder** (within 활/kg):

| Spec Class | Mean Price | Rows |
|---|---|---|
| 소 | 3,228 | 693 |
| 중 | 5,349 | 16,983 |
| 대 | 5,452 | 23 |
| count_1-5 | 8,005 | 1,459 |
| count_6-10 | 7,713 | 877 |
| count_11-20 | 7,137 | 289 |
| count_21+ | 8,671 | 48 |
| other | 7,399 | 991 |

Size ratio: 1.7x

> Spec-sensitive — needs per-spec-class prediction.

---

## 32. 참가자미 (Tier C)

| Attribute | Value |
|---|---|
| Rows | 20,483 |
| Trading Days | 5,315 |
| Date Range | 2006.03.21 ~ 2026.01.03 |
| Mean Price | 61,947 KRW |
| CV (volatility) | 0.7173 |
| Lag-1 (daily) | 0.4725 |
| Lag-1 (7d smoothed) | 0.9739 |
| Dominant Packaging | S/P (86.7%) |
| Flags | `SPEC` |

**State Distribution:**

| State | Rows | % | Mean Price |
|---|---|---|---|
| 선 | 20,124 | 98.2% | 53,488 |
| 냉 | 330 | 1.6% | 33,648 |
| 활 | 29 | 0.1% | 28,634 |

**Origin:** Foreign 0.2%
, price ratio (foreign/domestic) = 0.73x

Top origins: 군산(5,290), 제주도(2,691), 보령(2,596), 목포(2,506), 부산(기장)(1,677)

**Spec Price Ladder** (within 선/S/P):

| Spec Class | Mean Price | Rows |
|---|---|---|
| 소 | 33,343 | 1,364 |
| 중 | 54,212 | 9,805 |
| 대 | 72,894 | 3,809 |
| 특대 | 62,213 | 236 |
| count_1-5 | 51,022 | 162 |
| count_6-10 | 30,469 | 372 |
| count_11-20 | 60,924 | 675 |
| count_21+ | 65,888 | 348 |
| other | 56,050 | 703 |

Size ratio: 2.2x

> Spec-sensitive — needs per-spec-class prediction.

---

## 33. 새꼬막 (Tier B)

| Attribute | Value |
|---|---|
| Rows | 20,338 |
| Trading Days | 5,614 |
| Date Range | 2006.06.12 ~ 2026.01.03 |
| Mean Price | 43,381 KRW |
| CV (volatility) | 0.4549 |
| Lag-1 (daily) | 0.878 |
| Lag-1 (7d smoothed) | 0.9952 |
| Dominant Packaging | 그물망 (57.1%) |
| Flags | `SPEC` |

**State Distribution:**

| State | Rows | % | Mean Price |
|---|---|---|---|
| 활 | 20,315 | 99.9% | 35,717 |
| 냉 | 23 | 0.1% | 69,713 |

**Origin:** Foreign 2.4%
, price ratio (foreign/domestic) = 1.05x

Top origins: 순천(11,671), 벌교(5,890), 남해(571), 여수(546), 고흥(404)

**Spec Price Ladder** (within 활/그물망):

| Spec Class | Mean Price | Rows |
|---|---|---|
| 소 | 20,202 | 120 |
| 중 | 44,769 | 9,201 |
| 대 | 40,167 | 2,233 |
| 특대 | 77,930 | 27 |
| other | 33,476 | 38 |

Size ratio: 3.9x

> Spec-sensitive — needs per-spec-class prediction.

---

## 34. 깐굴 (Tier A)

| Attribute | Value |
|---|---|
| Rows | 20,194 |
| Trading Days | 5,915 |
| Date Range | 2006.04.04 ~ 2026.01.03 |
| Mean Price | 16,724 KRW |
| CV (volatility) | 0.4707 |
| Lag-1 (daily) | 0.9146 |
| Lag-1 (7d smoothed) | 0.9947 |
| Dominant Packaging | box (99.7%) |
| Flags | `SPEC` |

**State Distribution:**

| State | Rows | % | Mean Price |
|---|---|---|---|
| 선 | 20,134 | 99.7% | 19,740 |
| 냉 | 60 | 0.3% | 25,183 |

**Origin:** Foreign 0.1%
, price ratio (foreign/domestic) = 1.54x

Top origins: 통영(6,375), 삼천포(3,213), 마산(2,603), 남해(2,526), 사천(2,189)

**Spec Price Ladder** (within 선/box):

| Spec Class | Mean Price | Rows |
|---|---|---|
| 소 | 18,644 | 18,878 |
| 중 | 27,119 | 436 |
| 대 | 49,527 | 460 |
| other | 30,717 | 298 |

Size ratio: 2.7x

> Spec-sensitive — needs per-spec-class prediction.

---

## 35. 진주담치 (Tier B)

| Attribute | Value |
|---|---|
| Rows | 19,625 |
| Trading Days | 5,761 |
| Date Range | 2006.04.21 ~ 2026.01.03 |
| Mean Price | 12,924 KRW |
| CV (volatility) | 0.4097 |
| Lag-1 (daily) | 0.6759 |
| Lag-1 (7d smoothed) | 0.9865 |
| Dominant Packaging | 그물망 (94.6%) |
| Flags | `SPEC` |

**State Distribution:**

| State | Rows | % | Mean Price |
|---|---|---|---|
| 활 | 19,622 | 100.0% | 13,534 |
| 선 | 2 | 0.0% | 15,000 |
| 냉 | 1 | 0.0% | 18,100 |

**Origin:** Foreign 0.2%
, price ratio (foreign/domestic) = 2.67x

Top origins: 마산(8,120), 여수(7,893), 통영(1,721), 고성(712), 태안(480)

**Spec Price Ladder** (within 활/그물망):

| Spec Class | Mean Price | Rows |
|---|---|---|
| 소 | 8,173 | 432 |
| 중 | 11,247 | 8,926 |
| 대 | 15,769 | 8,309 |
| 특대 | 11,357 | 877 |

Size ratio: 1.9x

> Spec-sensitive — needs per-spec-class prediction.

---

## 36. 간재미 (Tier B)

| Attribute | Value |
|---|---|
| Rows | 19,070 |
| Trading Days | 5,457 |
| Date Range | 2006.03.16 ~ 2026.01.02 |
| Mean Price | 2,508 KRW |
| CV (volatility) | 1.6774 |
| Lag-1 (daily) | 0.0463 |
| Lag-1 (7d smoothed) | 0.8934 |
| Dominant Packaging | kg (42.8%) |
| Flags | None |

**State Distribution:**

| State | Rows | % | Mean Price |
|---|---|---|---|
| 선 | 17,773 | 93.2% | 14,997 |
| 활 | 1,219 | 6.4% | 2,884 |
| 냉 | 78 | 0.4% | 69,292 |

**Origin:** Foreign 1.5%
, price ratio (foreign/domestic) = 1.07x

Top origins: 목포(3,133), 인천(3,037), 태안(2,617), 안흥(2,607), 서산(1,499)

**Spec Price Ladder** (within 선/kg):

| Spec Class | Mean Price | Rows |
|---|---|---|
| 소 | 1,934 | 59 |
| 중 | 2,541 | 6,060 |
| 대 | 1,876 | 25 |
| count_1-5 | 2,113 | 601 |
| count_6-10 | 4,131 | 187 |
| count_11-20 | 1,735 | 82 |
| count_21+ | 7,982 | 49 |
| other | 1,743 | 21 |

Size ratio: 1.4x

---

## 37. 왕게 (Tier A)

| Attribute | Value |
|---|---|
| Rows | 17,889 |
| Trading Days | 6,556 |
| Date Range | 2004.01.02 ~ 2026.01.03 |
| Mean Price | 34,314 KRW |
| CV (volatility) | 0.59 |
| Lag-1 (daily) | 0.8933 |
| Lag-1 (7d smoothed) | 0.9957 |
| Dominant Packaging | kg (99.5%) |
| Flags | `STATE` `ORIGIN` |

**State Distribution:**

| State | Rows | % | Mean Price |
|---|---|---|---|
| 활 | 9,012 | 50.4% | 36,775 |
| 선 | 7,866 | 44.0% | 14,750 |
| 냉 | 1,011 | 5.7% | 13,570 |

> Needs state partition — price diverges >1.5x across states.

**Origin:** Foreign 97.8%
, price ratio (foreign/domestic) = 1.47x

> Origin-sensitive — must filter by domestic/foreign.

Top origins: 러시아(11,903), 노르웨이(4,005), 일본(900), 미국(523), 캄보디아(217)

**Spec Price Ladder** (within 활/kg):

| Spec Class | Mean Price | Rows |
|---|---|---|
| 중 | 35,104 | 8,400 |
| count_1-5 | 56,425 | 120 |
| count_6-10 | 62,393 | 14 |
| other | 62,249 | 449 |

---

## 38. 가무락 (Tier C)

| Attribute | Value |
|---|---|
| Rows | 17,796 |
| Trading Days | 5,037 |
| Date Range | 2007.04.06 ~ 2026.01.03 |
| Mean Price | 128,959 KRW |
| CV (volatility) | 0.3747 |
| Lag-1 (daily) | 0.6909 |
| Lag-1 (7d smoothed) | 0.984 |
| Dominant Packaging | 그물망 (88.5%) |
| Flags | `ORIGIN` `SPEC` |

**State Distribution:**

| State | Rows | % | Mean Price |
|---|---|---|---|
| 활 | 17,790 | 100.0% | 123,764 |
| 선 | 6 | 0.0% | 124,983 |

**Origin:** Foreign 6.8%
, price ratio (foreign/domestic) = 0.58x

> Origin-sensitive — must filter by domestic/foreign.

Top origins: 인천(4,995), 부안(2,783), 고창(1,429), 태안(1,380), 화성(1,166)

**Spec Price Ladder** (within 활/그물망):

| Spec Class | Mean Price | Rows |
|---|---|---|
| 소 | 44,463 | 1,308 |
| 중 | 99,481 | 5,090 |
| 대 | 155,804 | 9,326 |
| 특대 | 84,429 | 14 |

Size ratio: 3.5x

> Spec-sensitive — needs per-spec-class prediction.

---

## 39. 돔 (Tier C)

| Attribute | Value |
|---|---|
| Rows | 17,121 |
| Trading Days | 5,047 |
| Date Range | 2006.03.22 ~ 2026.01.02 |
| Mean Price | 37,427 KRW |
| CV (volatility) | 1.5056 |
| Lag-1 (daily) | 0.0686 |
| Lag-1 (7d smoothed) | 0.8958 |
| Dominant Packaging | S/P (73.7%) |
| Flags | `STATE` `SPEC` |

**State Distribution:**

| State | Rows | % | Mean Price |
|---|---|---|---|
| 선 | 14,178 | 82.8% | 34,393 |
| 활 | 2,173 | 12.7% | 17,683 |
| 냉 | 760 | 4.4% | 52,261 |
| 가공 | 10 | 0.1% | 85,500 |

> Needs state partition — price diverges >1.5x across states.

**Origin:** Foreign 20.2%
, price ratio (foreign/domestic) = 0.76x

Top origins: 부산(기장)(3,267), 제주도(2,762), 목포(2,512), 일본(1,905), 통영(1,075)

**Spec Price Ladder** (within 선/S/P):

| Spec Class | Mean Price | Rows |
|---|---|---|
| 소 | 16,299 | 77 |
| 중 | 39,201 | 4,661 |
| 대 | 41,123 | 1,175 |
| 특대 | 47,994 | 32 |
| count_1-5 | 20,495 | 1,902 |
| count_6-10 | 32,693 | 2,091 |
| count_11-20 | 46,435 | 1,729 |
| count_21+ | 44,398 | 523 |
| other | 33,878 | 243 |

Size ratio: 2.9x

> Spec-sensitive — needs per-spec-class prediction.

---

## 40. 물바지락 (Tier C)

| Attribute | Value |
|---|---|
| Rows | 17,046 |
| Trading Days | 4,963 |
| Date Range | 2006.04.08 ~ 2026.01.03 |
| Mean Price | 32,773 KRW |
| CV (volatility) | 0.4997 |
| Lag-1 (daily) | 0.8255 |
| Lag-1 (7d smoothed) | 0.993 |
| Dominant Packaging | box (99.3%) |
| Flags | `SPEC` |

**State Distribution:**

| State | Rows | % | Mean Price |
|---|---|---|---|
| 활 | 17,038 | 100.0% | 33,290 |
| 냉 | 6 | 0.0% | 8,333 |
| 선 | 2 | 0.0% | 24,650 |

**Origin:** Foreign 2.9%
, price ratio (foreign/domestic) = 0.76x

Top origins: 여수(6,626), 삼천포(5,105), 순천(1,606), 사천(1,464), 통영(789)

**Spec Price Ladder** (within 활/box):

| Spec Class | Mean Price | Rows |
|---|---|---|
| 소 | 12,669 | 1,015 |
| 중 | 20,469 | 3,161 |
| 대 | 38,593 | 10,786 |
| 특대 | 33,727 | 83 |
| other | 36,025 | 1,864 |

Size ratio: 3.0x

> Spec-sensitive — needs per-spec-class prediction.

---

## 41. 도다리 (Tier C)

| Attribute | Value |
|---|---|
| Rows | 16,093 |
| Trading Days | 4,277 |
| Date Range | 2006.03.30 ~ 2026.01.02 |
| Mean Price | 13,916 KRW |
| CV (volatility) | 0.5121 |
| Lag-1 (daily) | 0.3841 |
| Lag-1 (7d smoothed) | 0.9641 |
| Dominant Packaging | kg (71.5%) |
| Flags | `STATE` `SPEC` |

**State Distribution:**

| State | Rows | % | Mean Price |
|---|---|---|---|
| 활 | 8,376 | 52.0% | 12,466 |
| 선 | 7,713 | 47.9% | 12,827 |
| 냉 | 4 | 0.0% | 20,000 |

> Needs state partition — price diverges >1.5x across states.

**Origin:** Foreign 1.4%
, price ratio (foreign/domestic) = 0.51x

Top origins: 안흥(2,285), 군산(2,265), 태안(2,130), 보령(1,583), 대천(1,200)

**Spec Price Ladder** (within 활/kg):

| Spec Class | Mean Price | Rows |
|---|---|---|
| 소 | 7,713 | 128 |
| 중 | 13,363 | 6,133 |
| count_1-5 | 10,861 | 978 |
| count_6-10 | 8,462 | 352 |
| count_11-20 | 7,844 | 57 |
| other | 10,125 | 709 |

Size ratio: 1.7x

> Spec-sensitive — needs per-spec-class prediction.

---

## 42. 피꼬막(피조개) (Tier C)

| Attribute | Value |
|---|---|
| Rows | 15,471 |
| Trading Days | 4,554 |
| Date Range | 2007.11.10 ~ 2026.01.03 |
| Mean Price | 22,321 KRW |
| CV (volatility) | 0.4668 |
| Lag-1 (daily) | 0.7997 |
| Lag-1 (7d smoothed) | 0.9891 |
| Dominant Packaging | 그물망 (88.6%) |
| Flags | None |

**State Distribution:**

| State | Rows | % | Mean Price |
|---|---|---|---|
| 활 | 15,471 | 100.0% | 21,046 |

**Origin:** Foreign 0.5%
, price ratio (foreign/domestic) = 1.56x

Top origins: 마산(3,815), 여수(2,496), 순천(2,358), 군산(2,145), 장항(1,860)

**Spec Price Ladder** (within 활/그물망):

| Spec Class | Mean Price | Rows |
|---|---|---|
| 소 | 14,931 | 127 |
| 중 | 21,037 | 13,012 |
| 대 | 18,823 | 565 |

Size ratio: 1.4x

---

## 43. 키조개 (Tier C)

| Attribute | Value |
|---|---|
| Rows | 14,851 |
| Trading Days | 3,707 |
| Date Range | 2006.03.22 ~ 2026.01.02 |
| Mean Price | 41,702 KRW |
| CV (volatility) | 0.3795 |
| Lag-1 (daily) | 0.5809 |
| Lag-1 (7d smoothed) | 0.9788 |
| Dominant Packaging | box (93.1%) |
| Flags | `STATE` `SPEC` |

**State Distribution:**

| State | Rows | % | Mean Price |
|---|---|---|---|
| 선 | 8,971 | 60.4% | 37,728 |
| 활 | 5,817 | 39.2% | 24,601 |
| 냉 | 55 | 0.4% | 41,856 |
| 가공 | 8 | 0.1% | 98,750 |

> Needs state partition — price diverges >1.5x across states.

**Origin:** Foreign 0.3%
, price ratio (foreign/domestic) = 1.39x

Top origins: 보령(4,740), 군산(4,669), 장항(3,280), 여수(1,145), 대천(372)

**Spec Price Ladder** (within 선/box):

| Spec Class | Mean Price | Rows |
|---|---|---|
| 소 | 18,472 | 53 |
| 중 | 35,152 | 366 |
| 대 | 41,976 | 2,425 |
| count_1-5 | 34,176 | 29 |
| count_6-10 | 27,844 | 50 |
| count_11-20 | 23,727 | 44 |
| count_21+ | 36,539 | 1,171 |
| other | 37,339 | 4,643 |

Size ratio: 2.3x

> Spec-sensitive — needs per-spec-class prediction.

---

## 44. 대게 (Tier A)

| Attribute | Value |
|---|---|
| Rows | 14,332 |
| Trading Days | 6,424 |
| Date Range | 2004.01.02 ~ 2026.01.03 |
| Mean Price | 9,090 KRW |
| CV (volatility) | 0.5561 |
| Lag-1 (daily) | 0.746 |
| Lag-1 (7d smoothed) | 0.9891 |
| Dominant Packaging | kg (95.7%) |
| Flags | `STATE` |

**State Distribution:**

| State | Rows | % | Mean Price |
|---|---|---|---|
| 선 | 7,149 | 49.9% | 9,794 |
| 활 | 6,807 | 47.5% | 22,734 |
| 냉 | 376 | 2.6% | 5,522 |

> Needs state partition — price diverges >1.5x across states.

**Origin:** Foreign 94.5%
, price ratio (foreign/domestic) = 1.14x

Top origins: 러시아(12,632), 일본(403), 속초(338), 북한(207), 캐나다(139)

**Spec Price Ladder** (within 선/kg):

| Spec Class | Mean Price | Rows |
|---|---|---|
| 중 | 8,753 | 5,968 |
| count_1-5 | 12,811 | 150 |
| count_6-10 | 16,122 | 40 |
| other | 13,429 | 404 |

---

## 45. 깐홍합 (Tier C)

| Attribute | Value |
|---|---|
| Rows | 12,378 |
| Trading Days | 6,051 |
| Date Range | 2006.04.04 ~ 2026.01.03 |
| Mean Price | 14,836 KRW |
| CV (volatility) | 0.5752 |
| Lag-1 (daily) | 0.8092 |
| Lag-1 (7d smoothed) | 0.991 |
| Dominant Packaging | box (99.1%) |
| Flags | `SPEC` |

**State Distribution:**

| State | Rows | % | Mean Price |
|---|---|---|---|
| 선 | 12,284 | 99.2% | 16,946 |
| 냉 | 66 | 0.5% | 20,245 |
| 가공 | 28 | 0.2% | 19,679 |

**Origin:** Foreign 0.7%
, price ratio (foreign/domestic) = 0.88x

Top origins: 여수(7,120), 마산(4,076), 삼천포(404), 고성(351), 통영(250)

**Spec Price Ladder** (within 선/box):

| Spec Class | Mean Price | Rows |
|---|---|---|
| 소 | 11,118 | 10,228 |
| 중 | 20,185 | 141 |
| 대 | 49,515 | 1,803 |
| 특대 | 37,733 | 15 |
| other | 17,213 | 68 |

Size ratio: 4.5x

> Spec-sensitive — needs per-spec-class prediction.

---

## 46. 청어 (Tier C)

| Attribute | Value |
|---|---|
| Rows | 12,229 |
| Trading Days | 4,636 |
| Date Range | 2006.03.15 ~ 2026.01.02 |
| Mean Price | 6,457 KRW |
| CV (volatility) | 0.5324 |
| Lag-1 (daily) | 0.6462 |
| Lag-1 (7d smoothed) | 0.9834 |
| Dominant Packaging | S/P (90.7%) |
| Flags | `SPEC` |

**State Distribution:**

| State | Rows | % | Mean Price |
|---|---|---|---|
| 선 | 11,536 | 94.3% | 6,060 |
| 냉 | 674 | 5.5% | 17,539 |
| 가공 | 18 | 0.1% | 101,150 |
| 활 | 1 | 0.0% | 7,000 |

**Origin:** Foreign 4.9%
, price ratio (foreign/domestic) = 2.99x

Top origins: 속초(2,278), 임원(1,944), 통영(1,900), 포항(1,507), 죽변(1,167)

**Spec Price Ladder** (within 선/S/P):

| Spec Class | Mean Price | Rows |
|---|---|---|
| 소 | 4,422 | 161 |
| 중 | 5,613 | 6,468 |
| 대 | 10,237 | 425 |
| count_1-5 | 6,286 | 14 |
| count_6-10 | 7,488 | 88 |
| count_11-20 | 6,511 | 3,617 |
| count_21+ | 5,121 | 256 |
| other | 11,370 | 44 |

Size ratio: 2.3x

> Spec-sensitive — needs per-spec-class prediction.

---

## 47. 감성돔 (Tier C)

| Attribute | Value |
|---|---|
| Rows | 12,183 |
| Trading Days | 4,908 |
| Date Range | 2006.04.17 ~ 2026.01.03 |
| Mean Price | 20,587 KRW |
| CV (volatility) | 0.3933 |
| Lag-1 (daily) | 0.5428 |
| Lag-1 (7d smoothed) | 0.9798 |
| Dominant Packaging | kg (97.4%) |
| Flags | `STATE` `ORIGIN` `SPEC` |

**State Distribution:**

| State | Rows | % | Mean Price |
|---|---|---|---|
| 활 | 10,581 | 86.9% | 21,681 |
| 선 | 1,599 | 13.1% | 6,537 |
| 냉 | 3 | 0.0% | 6,500 |

> Needs state partition — price diverges >1.5x across states.

**Origin:** Foreign 44.3%
, price ratio (foreign/domestic) = 0.67x

> Origin-sensitive — must filter by domestic/foreign.

Top origins: 중국(5,101), 통영(3,427), 목포(1,200), 완도(742), 일본(300)

**Spec Price Ladder** (within 활/kg):

| Spec Class | Mean Price | Rows |
|---|---|---|
| 소 | 13,453 | 139 |
| 중 | 21,023 | 7,760 |
| 대 | 21,945 | 11 |
| count_1-5 | 25,317 | 2,131 |
| count_6-10 | 14,784 | 172 |
| other | 20,988 | 353 |

Size ratio: 1.6x

> Spec-sensitive — needs per-spec-class prediction.

---

## 48. 복어 (Tier C)

| Attribute | Value |
|---|---|
| Rows | 11,847 |
| Trading Days | 4,427 |
| Date Range | 2006.03.24 ~ 2026.01.03 |
| Mean Price | 17,636 KRW |
| CV (volatility) | 1.3327 |
| Lag-1 (daily) | 0.2984 |
| Lag-1 (7d smoothed) | 0.9414 |
| Dominant Packaging | S/P (75.2%) |
| Flags | `STATE` `SPEC` |

**State Distribution:**

| State | Rows | % | Mean Price |
|---|---|---|---|
| 선 | 10,606 | 89.5% | 15,444 |
| 활 | 950 | 8.0% | 14,242 |
| 냉 | 289 | 2.4% | 31,214 |
| 가공 | 2 | 0.0% | 125,000 |

> Needs state partition — price diverges >1.5x across states.

**Origin:** Foreign 1.8%
, price ratio (foreign/domestic) = 2.89x

Top origins: 통영(1,536), 안흥(1,411), 여수(955), 목포(831), 보령(660)

**Spec Price Ladder** (within 선/S/P):

| Spec Class | Mean Price | Rows |
|---|---|---|
| 소 | 10,826 | 300 |
| 중 | 17,266 | 5,549 |
| 대 | 17,695 | 1,922 |
| 특대 | 23,023 | 13 |
| count_1-5 | 14,357 | 382 |
| count_6-10 | 15,610 | 287 |
| count_11-20 | 12,862 | 80 |
| count_21+ | 31,674 | 50 |
| other | 13,588 | 139 |

Size ratio: 2.1x

> Spec-sensitive — needs per-spec-class prediction.

---

## 49. 개조개 (Tier C)

| Attribute | Value |
|---|---|
| Rows | 11,780 |
| Trading Days | 4,228 |
| Date Range | 2007.01.31 ~ 2026.01.03 |
| Mean Price | 48,705 KRW |
| CV (volatility) | 0.3916 |
| Lag-1 (daily) | 0.8123 |
| Lag-1 (7d smoothed) | 0.9916 |
| Dominant Packaging | box (87.4%) |
| Flags | `ORIGIN` `SPEC` |

**State Distribution:**

| State | Rows | % | Mean Price |
|---|---|---|---|
| 활 | 11,699 | 99.3% | 48,730 |
| 냉 | 81 | 0.7% | 37,448 |

**Origin:** Foreign 29.9%
, price ratio (foreign/domestic) = 0.59x

> Origin-sensitive — must filter by domestic/foreign.

Top origins: 여수(6,202), 중국(3,277), 통영(1,377), 북한(250), 보령(179)

**Spec Price Ladder** (within 활/box):

| Spec Class | Mean Price | Rows |
|---|---|---|
| 소 | 28,787 | 31 |
| 중 | 43,121 | 598 |
| 대 | 46,028 | 2,813 |
| count_6-10 | 48,429 | 163 |
| count_11-20 | 56,651 | 5,657 |
| count_21+ | 47,962 | 1,019 |
| other | 43,336 | 11 |

Size ratio: 1.6x

> Spec-sensitive — needs per-spec-class prediction.

---

## 50. 봉바지락 (Tier C)

| Attribute | Value |
|---|---|
| Rows | 11,712 |
| Trading Days | 5,933 |
| Date Range | 2006.04.04 ~ 2026.01.03 |
| Mean Price | 11,529 KRW |
| CV (volatility) | 0.3684 |
| Lag-1 (daily) | 0.6615 |
| Lag-1 (7d smoothed) | 0.979 |
| Dominant Packaging | box (99.8%) |
| Flags | `SPEC` |

**State Distribution:**

| State | Rows | % | Mean Price |
|---|---|---|---|
| 활 | 11,706 | 99.9% | 11,509 |
| 가공 | 6 | 0.1% | 14,133 |

**Origin:** Foreign 35.0%
, price ratio (foreign/domestic) = 0.78x

Top origins: 태안(5,071), 중국(3,461), 서산(786), 북한(637), 기타(국내)(624)

**Spec Price Ladder** (within 활/box):

| Spec Class | Mean Price | Rows |
|---|---|---|
| 소 | 10,442 | 73 |
| 중 | 11,692 | 3,972 |
| 대 | 11,207 | 7,340 |
| 특대 | 7,058 | 19 |
| other | 17,079 | 283 |

Size ratio: 1.7x

> Spec-sensitive — needs per-spec-class prediction.

---

## 51. 미더덕 (Tier C)

| Attribute | Value |
|---|---|
| Rows | 10,710 |
| Trading Days | 6,030 |
| Date Range | 2006.04.03 ~ 2026.01.03 |
| Mean Price | 14,812 KRW |
| CV (volatility) | 0.5419 |
| Lag-1 (daily) | 0.7058 |
| Lag-1 (7d smoothed) | 0.988 |
| Dominant Packaging | box (99.8%) |
| Flags | `SPEC` |

**State Distribution:**

| State | Rows | % | Mean Price |
|---|---|---|---|
| 선 | 10,596 | 98.9% | 15,588 |
| 냉 | 106 | 1.0% | 15,535 |
| 활 | 8 | 0.1% | 12,112 |

**Origin:** Foreign 0.0%
, price ratio (foreign/domestic) = 0.58x

Top origins: 마산(7,519), 통영(2,144), 고성(976), 여수(26), 삼천포(22)

**Spec Price Ladder** (within 선/box):

| Spec Class | Mean Price | Rows |
|---|---|---|
| 소 | 10,101 | 8,873 |
| 중 | 14,067 | 84 |
| 대 | 49,852 | 1,310 |
| 특대 | 58,029 | 62 |
| other | 20,559 | 248 |

Size ratio: 5.7x

> Spec-sensitive — needs per-spec-class prediction.

---

## 52. 토바지락 (Tier C)

| Attribute | Value |
|---|---|
| Rows | 10,646 |
| Trading Days | 6,043 |
| Date Range | 2006.04.04 ~ 2026.01.03 |
| Mean Price | 5,766 KRW |
| CV (volatility) | 0.3824 |
| Lag-1 (daily) | 0.6819 |
| Lag-1 (7d smoothed) | 0.9758 |
| Dominant Packaging | box (99.7%) |
| Flags | None |

**State Distribution:**

| State | Rows | % | Mean Price |
|---|---|---|---|
| 활 | 10,646 | 100.0% | 6,189 |

**Origin:** Foreign 60.4%
, price ratio (foreign/domestic) = 0.8x

Top origins: 중국(5,674), 태안(1,956), 북한(758), 기타(국내)(678), 서산(676)

**Spec Price Ladder** (within 활/box):

| Spec Class | Mean Price | Rows |
|---|---|---|
| 소 | 4,957 | 411 |
| 중 | 4,881 | 2,670 |
| 대 | 6,598 | 6,692 |
| 특대 | 5,800 | 32 |
| other | 7,699 | 811 |

Size ratio: 1.4x

---

## 53. 해삼 (Tier C)

| Attribute | Value |
|---|---|
| Rows | 10,633 |
| Trading Days | 3,764 |
| Date Range | 2006.03.28 ~ 2026.01.03 |
| Mean Price | 32,431 KRW |
| CV (volatility) | 0.7634 |
| Lag-1 (daily) | 0.5288 |
| Lag-1 (7d smoothed) | 0.9703 |
| Dominant Packaging | box (87.6%) |
| Flags | `SPEC` |

**State Distribution:**

| State | Rows | % | Mean Price |
|---|---|---|---|
| 활 | 10,507 | 98.8% | 32,276 |
| 선 | 110 | 1.0% | 45,641 |
| 냉건 | 13 | 0.1% | 30,462 |
| 가공 | 2 | 0.0% | 375,000 |
| 냉 | 1 | 0.0% | 50,000 |

**Origin:** Foreign 0.3%
, price ratio (foreign/domestic) = 2.44x

Top origins: 마산(3,254), 통영(2,590), 여수(2,052), 완도(825), 삼천포(556)

**Spec Price Ladder** (within 활/box):

| Spec Class | Mean Price | Rows |
|---|---|---|
| 소 | 25,980 | 7,397 |
| 중 | 43,905 | 254 |
| 대 | 70,501 | 990 |
| other | 35,843 | 642 |

Size ratio: 2.7x

> Spec-sensitive — needs per-spec-class prediction.

---

## 54. 연자돔 (Tier C)

| Attribute | Value |
|---|---|
| Rows | 10,431 |
| Trading Days | 3,763 |
| Date Range | 2006.04.12 ~ 2026.01.03 |
| Mean Price | 77,223 KRW |
| CV (volatility) | 0.4634 |
| Lag-1 (daily) | 0.4503 |
| Lag-1 (7d smoothed) | 0.973 |
| Dominant Packaging | S/P (93.5%) |
| Flags | `ORIGIN` `SPEC` |

**State Distribution:**

| State | Rows | % | Mean Price |
|---|---|---|---|
| 선 | 9,712 | 93.1% | 80,760 |
| 냉 | 717 | 6.9% | 101,072 |
| 활 | 2 | 0.0% | 62,500 |

**Origin:** Foreign 8.2%
, price ratio (foreign/domestic) = 0.64x

> Origin-sensitive — must filter by domestic/foreign.

Top origins: 제주도(8,446), 부산(기장)(861), 일본(847), 삼천포(125), 방어진(35)

**Spec Price Ladder** (within 선/S/P):

| Spec Class | Mean Price | Rows |
|---|---|---|
| 소 | 35,175 | 73 |
| 중 | 68,217 | 2,963 |
| 대 | 85,950 | 549 |
| count_1-5 | 30,810 | 104 |
| count_6-10 | 71,242 | 1,058 |
| count_11-20 | 95,549 | 3,467 |
| count_21+ | 84,616 | 712 |
| other | 114,329 | 394 |

Size ratio: 2.4x

> Spec-sensitive — needs per-spec-class prediction.

---

## 55. 분홍새우 (Tier C)

| Attribute | Value |
|---|---|
| Rows | 10,313 |
| Trading Days | 5,589 |
| Date Range | 2006.03.21 ~ 2026.01.03 |
| Mean Price | 21,642 KRW |
| CV (volatility) | 0.9983 |
| Lag-1 (daily) | 0.9116 |
| Lag-1 (7d smoothed) | 0.9972 |
| Dominant Packaging | S/P (91.6%) |
| Flags | `SPEC` |

**State Distribution:**

| State | Rows | % | Mean Price |
|---|---|---|---|
| 선 | 10,136 | 98.3% | 23,841 |
| 냉 | 171 | 1.7% | 96,287 |
| 활 | 6 | 0.1% | 25,717 |

**Origin:** Foreign 0.8%
, price ratio (foreign/domestic) = 0.79x

Top origins: 속초(6,217), 동해시(1,542), 죽변(1,013), 여수(665), 삼척(303)

**Spec Price Ladder** (within 선/S/P):

| Spec Class | Mean Price | Rows |
|---|---|---|
| 소 | 24,181 | 350 |
| 중 | 18,339 | 6,153 |
| 대 | 46,418 | 1,539 |
| count_1-5 | 14,848 | 21 |
| count_11-20 | 6,995 | 214 |
| count_21+ | 25,414 | 1,100 |
| other | 38,116 | 49 |

Size ratio: 2.5x

> Spec-sensitive — needs per-spec-class prediction.

---

## 56. 바다가재 (Tier C)

| Attribute | Value |
|---|---|
| Rows | 10,206 |
| Trading Days | 4,091 |
| Date Range | 2006.03.25 ~ 2026.01.03 |
| Mean Price | 12,519 KRW |
| CV (volatility) | 0.3035 |
| Lag-1 (daily) | 0.4085 |
| Lag-1 (7d smoothed) | 0.9729 |
| Dominant Packaging | kg (99.5%) |
| Flags | `STATE` |

**State Distribution:**

| State | Rows | % | Mean Price |
|---|---|---|---|
| 선 | 5,343 | 52.4% | 12,286 |
| 활 | 3,989 | 39.1% | 26,210 |
| 냉 | 874 | 8.6% | 5,708 |

> Needs state partition — price diverges >1.5x across states.

**Origin:** Foreign 99.5%
, price ratio (foreign/domestic) = 1.16x

Top origins: 캐나다(7,032), 미국(2,903), 러시아(120), 호주(84), 안흥(9)

**Spec Price Ladder** (within 선/kg):

| Spec Class | Mean Price | Rows |
|---|---|---|
| 중 | 12,268 | 4,873 |
| count_1-5 | 11,166 | 102 |
| count_6-10 | 12,870 | 27 |
| other | 12,635 | 294 |

---

## 57. 다랑어 (Tier C)

| Attribute | Value |
|---|---|
| Rows | 10,007 |
| Trading Days | 3,182 |
| Date Range | 2006.03.30 ~ 2026.01.03 |
| Mean Price | 52,351 KRW |
| CV (volatility) | 2.0564 |
| Lag-1 (daily) | 0.1659 |
| Lag-1 (7d smoothed) | 0.9272 |
| Dominant Packaging | S/P (88.5%) |
| Flags | `SPEC` |

**State Distribution:**

| State | Rows | % | Mean Price |
|---|---|---|---|
| 선 | 10,003 | 100.0% | 55,722 |
| 활 | 4 | 0.0% | 11,000 |

**Origin:** Foreign 0.3%
, price ratio (foreign/domestic) = 0.71x

Top origins: 속초(1,552), 강구(1,402), 죽변(1,265), 부산(기장)(911), 축산(831)

**Spec Price Ladder** (within 선/S/P):

| Spec Class | Mean Price | Rows |
|---|---|---|
| 소 | 19,450 | 28 |
| 중 | 41,558 | 2,789 |
| 대 | 69,439 | 1,863 |
| 특대 | 122,415 | 40 |
| count_1-5 | 62,476 | 3,352 |
| count_6-10 | 16,036 | 640 |
| count_11-20 | 15,015 | 92 |
| other | 29,330 | 43 |

Size ratio: 6.3x

> Spec-sensitive — needs per-spec-class prediction.

---

## 58. 칼바지락 (Tier B)

| Attribute | Value |
|---|---|
| Rows | 9,760 |
| Trading Days | 6,054 |
| Date Range | 2006.04.04 ~ 2026.01.03 |
| Mean Price | 8,424 KRW |
| CV (volatility) | 0.3978 |
| Lag-1 (daily) | 0.7362 |
| Lag-1 (7d smoothed) | 0.9815 |
| Dominant Packaging | box (99.6%) |
| Flags | None |

**State Distribution:**

| State | Rows | % | Mean Price |
|---|---|---|---|
| 활 | 9,760 | 100.0% | 8,708 |

**Origin:** Foreign 69.2%
, price ratio (foreign/domestic) = 0.82x

Top origins: 중국(5,940), 태안(1,745), 북한(810), 곰소(276), 기타(국내)(253)

**Spec Price Ladder** (within 활/box):

| Spec Class | Mean Price | Rows |
|---|---|---|
| 소 | 7,525 | 119 |
| 중 | 9,049 | 3,041 |
| 대 | 8,549 | 6,355 |
| 특대 | 7,319 | 27 |
| other | 9,409 | 180 |

Size ratio: 1.2x

---

## 59. 잿방어 (Tier C)

| Attribute | Value |
|---|---|
| Rows | 9,259 |
| Trading Days | 3,444 |
| Date Range | 2006.03.23 ~ 2025.12.27 |
| Mean Price | 18,193 KRW |
| CV (volatility) | 1.3473 |
| Lag-1 (daily) | 0.2277 |
| Lag-1 (7d smoothed) | 0.9403 |
| Dominant Packaging | S/P (57.7%) |
| Flags | `STATE` `SPEC` |

**State Distribution:**

| State | Rows | % | Mean Price |
|---|---|---|---|
| 선 | 6,661 | 71.9% | 16,611 |
| 활 | 2,583 | 27.9% | 16,998 |
| 냉 | 15 | 0.2% | 10,760 |

> Needs state partition — price diverges >1.5x across states.

**Origin:** Foreign 25.7%
, price ratio (foreign/domestic) = 1.0x

Top origins: 일본(2,281), 속초(1,273), 제주도(641), 통영(581), 포항(550)

**Spec Price Ladder** (within 선/S/P):

| Spec Class | Mean Price | Rows |
|---|---|---|
| 소 | 7,275 | 12 |
| 중 | 17,984 | 2,856 |
| 대 | 23,499 | 1,330 |
| 특대 | 8,977 | 48 |
| count_1-5 | 16,217 | 922 |
| count_6-10 | 17,693 | 119 |
| count_11-20 | 7,324 | 41 |

Size ratio: 3.2x

> Spec-sensitive — needs per-spec-class prediction.

---

## 60. 가오리 (Tier C)

| Attribute | Value |
|---|---|
| Rows | 8,840 |
| Trading Days | 4,157 |
| Date Range | 2006.03.29 ~ 2026.01.03 |
| Mean Price | 23,865 KRW |
| CV (volatility) | 0.9949 |
| Lag-1 (daily) | 0.141 |
| Lag-1 (7d smoothed) | 0.926 |
| Dominant Packaging | S/P (66.7%) |
| Flags | `STATE` `ORIGIN` `SPEC` |

**State Distribution:**

| State | Rows | % | Mean Price |
|---|---|---|---|
| 선 | 6,924 | 78.3% | 22,787 |
| 냉 | 1,637 | 18.5% | 34,976 |
| 활 | 279 | 3.2% | 21,803 |

> Needs state partition — price diverges >1.5x across states.

**Origin:** Foreign 18.5%
, price ratio (foreign/domestic) = 1.37x

> Origin-sensitive — must filter by domestic/foreign.

Top origins: 포항(1,045), 후포(645), 제주도(616), 나로도(604), 목포(552)

**Spec Price Ladder** (within 선/S/P):

| Spec Class | Mean Price | Rows |
|---|---|---|
| 소 | 17,735 | 26 |
| 중 | 24,116 | 2,966 |
| 대 | 22,754 | 1,147 |
| 특대 | 46,314 | 21 |
| count_1-5 | 24,228 | 1,313 |
| count_6-10 | 20,022 | 234 |
| count_11-20 | 28,600 | 41 |
| other | 29,870 | 37 |

Size ratio: 2.6x

> Spec-sensitive — needs per-spec-class prediction.

---

## 61. 점성어 (Tier B)

| Attribute | Value |
|---|---|
| Rows | 8,572 |
| Trading Days | 5,925 |
| Date Range | 2006.04.21 ~ 2026.01.03 |
| Mean Price | 7,294 KRW |
| CV (volatility) | 0.3022 |
| Lag-1 (daily) | 0.775 |
| Lag-1 (7d smoothed) | 0.99 |
| Dominant Packaging | kg (98.9%) |
| Flags | `STATE` `ORIGIN` |

**State Distribution:**

| State | Rows | % | Mean Price |
|---|---|---|---|
| 활 | 7,018 | 81.9% | 7,459 |
| 선 | 1,554 | 18.1% | 1,065 |

> Needs state partition — price diverges >1.5x across states.

**Origin:** Foreign 98.2%
, price ratio (foreign/domestic) = 1.35x

> Origin-sensitive — must filter by domestic/foreign.

Top origins: 중국(8,320), 일본(96), 통영(53), 제주도(48), 충무(14)

**Spec Price Ladder** (within 활/kg):

| Spec Class | Mean Price | Rows |
|---|---|---|
| 소 | 5,163 | 233 |
| 중 | 7,135 | 5,770 |
| 대 | 7,937 | 19 |
| count_1-5 | 9,793 | 805 |
| count_6-10 | 11,750 | 10 |
| other | 10,193 | 174 |

Size ratio: 1.5x

---

## 62. 능성어 (Tier C)

| Attribute | Value |
|---|---|
| Rows | 7,287 |
| Trading Days | 4,217 |
| Date Range | 2006.05.01 ~ 2026.01.03 |
| Mean Price | 31,849 KRW |
| CV (volatility) | 0.4206 |
| Lag-1 (daily) | 0.2857 |
| Lag-1 (7d smoothed) | 0.955 |
| Dominant Packaging | kg (98.9%) |
| Flags | None |

**State Distribution:**

| State | Rows | % | Mean Price |
|---|---|---|---|
| 활 | 6,599 | 90.6% | 31,414 |
| 선 | 684 | 9.4% | 12,550 |
| 냉 | 4 | 0.1% | 17,675 |

**Origin:** Foreign 42.4%
, price ratio (foreign/domestic) = 0.83x

Top origins: 통영(2,970), 일본(2,358), 중국(724), 제주도(406), 충무(267)

**Spec Price Ladder** (within 활/kg):

| Spec Class | Mean Price | Rows |
|---|---|---|
| 소 | 32,023 | 62 |
| 중 | 32,269 | 5,411 |
| count_1-5 | 28,355 | 832 |
| count_6-10 | 16,011 | 37 |
| count_11-20 | 6,964 | 11 |
| other | 26,124 | 229 |

Size ratio: 1.0x

---

## 63. 동죽 (Tier C)

| Attribute | Value |
|---|---|
| Rows | 7,262 |
| Trading Days | 4,028 |
| Date Range | 2006.04.03 ~ 2026.01.03 |
| Mean Price | 16,924 KRW |
| CV (volatility) | 0.8226 |
| Lag-1 (daily) | 0.8847 |
| Lag-1 (7d smoothed) | 0.995 |
| Dominant Packaging | box (54.8%) |
| Flags | `ORIGIN` `SPEC` |

**State Distribution:**

| State | Rows | % | Mean Price |
|---|---|---|---|
| 활 | 6,711 | 92.4% | 21,124 |
| 선 | 515 | 7.1% | 9,023 |
| 냉 | 36 | 0.5% | 90,014 |

**Origin:** Foreign 5.8%
, price ratio (foreign/domestic) = 0.6x

> Origin-sensitive — must filter by domestic/foreign.

Top origins: 태안(1,825), 장항(1,717), 인천(1,540), 군산(786), 고창(417)

**Spec Price Ladder** (within 활/box):

| Spec Class | Mean Price | Rows |
|---|---|---|
| 소 | 3,795 | 1,584 |
| 중 | 9,215 | 456 |
| 대 | 27,168 | 1,403 |
| other | 6,073 | 11 |

Size ratio: 7.2x

> Spec-sensitive — needs per-spec-class prediction.

---

## 64. 우럭조개 (Tier C)

| Attribute | Value |
|---|---|
| Rows | 7,020 |
| Trading Days | 2,117 |
| Date Range | 2008.05.20 ~ 2026.01.03 |
| Mean Price | 59,951 KRW |
| CV (volatility) | 0.8988 |
| Lag-1 (daily) | 0.7524 |
| Lag-1 (7d smoothed) | 0.9913 |
| Dominant Packaging | box (67.1%) |
| Flags | `SPEC` |

**State Distribution:**

| State | Rows | % | Mean Price |
|---|---|---|---|
| 활 | 6,990 | 99.6% | 40,124 |
| 선 | 25 | 0.4% | 11,040 |
| 냉 | 5 | 0.1% | 36,140 |

**Origin:** Foreign 0.1%
, price ratio (foreign/domestic) = 0.35x

Top origins: 여수(5,809), 통영(1,051), 거제도(53), 마산(24), 삼천포(18)

**Spec Price Ladder** (within 활/box):

| Spec Class | Mean Price | Rows |
|---|---|---|
| 소 | 37,814 | 14 |
| 중 | 67,164 | 166 |
| 대 | 94,068 | 148 |
| count_1-5 | 37,267 | 48 |
| count_6-10 | 46,215 | 3,076 |
| count_11-20 | 59,895 | 791 |
| count_21+ | 86,274 | 437 |

Size ratio: 2.5x

> Spec-sensitive — needs per-spec-class prediction.

---

## 65. 백합 (Tier C)

| Attribute | Value |
|---|---|
| Rows | 6,800 |
| Trading Days | 3,620 |
| Date Range | 2006.11.14 ~ 2026.01.03 |
| Mean Price | 106,329 KRW |
| CV (volatility) | 0.4688 |
| Lag-1 (daily) | 0.8052 |
| Lag-1 (7d smoothed) | 0.992 |
| Dominant Packaging | 그물망 (65.9%) |
| Flags | `ORIGIN` `SPEC` |

**State Distribution:**

| State | Rows | % | Mean Price |
|---|---|---|---|
| 활 | 6,786 | 99.8% | 88,670 |
| 냉 | 7 | 0.1% | 19,143 |
| 선 | 7 | 0.1% | 49,429 |

**Origin:** Foreign 32.6%
, price ratio (foreign/domestic) = 0.62x

> Origin-sensitive — must filter by domestic/foreign.

Top origins: 중국(2,148), 강화(1,533), 장항(1,373), 군산(599), 인천(485)

**Spec Price Ladder** (within 활/그물망):

| Spec Class | Mean Price | Rows |
|---|---|---|
| 소 | 49,524 | 370 |
| 중 | 105,229 | 3,682 |
| 대 | 112,092 | 419 |

Size ratio: 2.3x

> Spec-sensitive — needs per-spec-class prediction.

---

## 66. 꼴뚜기 (Tier C)

| Attribute | Value |
|---|---|
| Rows | 6,612 |
| Trading Days | 2,915 |
| Date Range | 2006.03.23 ~ 2026.01.03 |
| Mean Price | 29,206 KRW |
| CV (volatility) | 0.9214 |
| Lag-1 (daily) | 0.8023 |
| Lag-1 (7d smoothed) | 0.9907 |
| Dominant Packaging | box (69.9%) |
| Flags | `SPEC` |

**State Distribution:**

| State | Rows | % | Mean Price |
|---|---|---|---|
| 선 | 6,549 | 99.0% | 27,085 |
| 냉 | 29 | 0.4% | 16,203 |
| 활 | 26 | 0.4% | 29,338 |
| 가공 | 8 | 0.1% | 17,375 |

**Origin:** Foreign 0.1%
, price ratio (foreign/domestic) = 0.36x

Top origins: 여수(2,302), 군산(1,775), 장항(853), 인천(487), 보령(417)

**Spec Price Ladder** (within 선/box):

| Spec Class | Mean Price | Rows |
|---|---|---|
| 소 | 9,981 | 1,988 |
| 중 | 24,088 | 117 |
| 대 | 33,118 | 238 |
| other | 45,427 | 2,233 |

Size ratio: 3.3x

> Spec-sensitive — needs per-spec-class prediction.

---

## 67. 줄돔 (Tier C)

| Attribute | Value |
|---|---|
| Rows | 5,124 |
| Trading Days | 3,084 |
| Date Range | 2006.05.06 ~ 2026.01.02 |
| Mean Price | 54,264 KRW |
| CV (volatility) | 0.4689 |
| Lag-1 (daily) | 0.4147 |
| Lag-1 (7d smoothed) | 0.9689 |
| Dominant Packaging | kg (96.7%) |
| Flags | None |

**State Distribution:**

| State | Rows | % | Mean Price |
|---|---|---|---|
| 활 | 4,652 | 90.8% | 55,490 |
| 선 | 472 | 9.2% | 8,437 |

**Origin:** Foreign 50.8%
, price ratio (foreign/domestic) = 0.91x

Top origins: 일본(2,456), 제주도(635), 통영(612), 목포(350), 완도(338)

**Spec Price Ladder** (within 활/kg):

| Spec Class | Mean Price | Rows |
|---|---|---|
| 소 | 36,927 | 44 |
| 중 | 53,552 | 3,506 |
| count_1-5 | 66,261 | 911 |
| count_6-10 | 30,310 | 63 |
| other | 54,271 | 107 |

Size ratio: 1.5x

---

## 68. 염고등어 (Tier C)

| Attribute | Value |
|---|---|
| Rows | 4,986 |
| Trading Days | 2,908 |
| Date Range | 2007.04.30 ~ 2025.12.30 |
| Mean Price | 33,083 KRW |
| CV (volatility) | 0.3253 |
| Lag-1 (daily) | 0.8292 |
| Lag-1 (7d smoothed) | 0.9957 |
| Dominant Packaging | CT/(BT) (89.4%) |
| Flags | `STATE` `ORIGIN` |

**State Distribution:**

| State | Rows | % | Mean Price |
|---|---|---|---|
| 냉 | 3,714 | 74.5% | 34,335 |
| 가공 | 1,194 | 23.9% | 21,757 |
| 선 | 78 | 1.6% | 39,486 |

> Needs state partition — price diverges >1.5x across states.

**Origin:** Foreign 59.9%
, price ratio (foreign/domestic) = 1.45x

> Origin-sensitive — must filter by domestic/foreign.

Top origins: 노르웨이(2,803), 부산(기장)(1,492), 기타(국내)(487), 영국(80), 네덜란드(29)

**Spec Price Ladder** (within 냉/CT/(BT)):

| Spec Class | Mean Price | Rows |
|---|---|---|
| 소 | 32,200 | 80 |
| 중 | 34,370 | 391 |
| 대 | 40,641 | 691 |
| 특대 | 44,211 | 603 |
| other | 29,241 | 1,898 |

Size ratio: 1.4x

---

## 69. 부시리 (Tier C)

| Attribute | Value |
|---|---|
| Rows | 4,399 |
| Trading Days | 1,698 |
| Date Range | 2007.03.15 ~ 2026.01.03 |
| Mean Price | 9,314 KRW |
| CV (volatility) | 0.4234 |
| Lag-1 (daily) | 0.4434 |
| Lag-1 (7d smoothed) | 0.9614 |
| Dominant Packaging | kg (76.2%) |
| Flags | `STATE` |

**State Distribution:**

| State | Rows | % | Mean Price |
|---|---|---|---|
| 활 | 2,745 | 62.4% | 8,589 |
| 선 | 1,654 | 37.6% | 16,993 |

> Needs state partition — price diverges >1.5x across states.

**Origin:** Foreign 3.3%
, price ratio (foreign/domestic) = 0.65x

Top origins: 제주도(979), 통영(873), 속초(377), 여수(289), 완도(251)

**Spec Price Ladder** (within 활/kg):

| Spec Class | Mean Price | Rows |
|---|---|---|
| 소 | 7,650 | 14 |
| 중 | 8,983 | 1,682 |
| count_1-5 | 8,282 | 939 |
| count_6-10 | 3,560 | 25 |
| other | 5,595 | 79 |

Size ratio: 1.2x

---

## 70. 줄무늬전갱이 (Tier C)

| Attribute | Value |
|---|---|
| Rows | 2,319 |
| Trading Days | 620 |
| Date Range | 2023.07.19 ~ 2025.09.10 |
| Mean Price | 32,070 KRW |
| CV (volatility) | 0.1497 |
| Lag-1 (daily) | 0.6462 |
| Lag-1 (7d smoothed) | 0.9837 |
| Dominant Packaging | kg (99.9%) |
| Flags | `STATE` |

**State Distribution:**

| State | Rows | % | Mean Price |
|---|---|---|---|
| 활 | 1,740 | 75.0% | 33,904 |
| 선 | 579 | 25.0% | 9,316 |

> Needs state partition — price diverges >1.5x across states.

**Origin:** Foreign 99.7%
, price ratio (foreign/domestic) = 1.12x

Top origins: 일본(2,304), 중국(9), 통영(5), 인천(1)

**Spec Price Ladder** (within 활/kg):

| Spec Class | Mean Price | Rows |
|---|---|---|
| 중 | 30,528 | 398 |
| count_1-5 | 34,693 | 876 |
| count_6-10 | 36,568 | 127 |
| count_11-20 | 37,818 | 11 |
| other | 34,801 | 325 |

---

## 71. 자바리 (Tier C)

| Attribute | Value |
|---|---|
| Rows | 2,123 |
| Trading Days | 1,260 |
| Date Range | 2020.01.24 ~ 2026.01.03 |
| Mean Price | 23,344 KRW |
| CV (volatility) | 0.3904 |
| Lag-1 (daily) | 0.4185 |
| Lag-1 (7d smoothed) | 0.9713 |
| Dominant Packaging | kg (100.0%) |
| Flags | `STATE` `ORIGIN` |

**State Distribution:**

| State | Rows | % | Mean Price |
|---|---|---|---|
| 활 | 1,646 | 77.5% | 23,815 |
| 선 | 477 | 22.5% | 8,457 |

> Needs state partition — price diverges >1.5x across states.

**Origin:** Foreign 94.2%
, price ratio (foreign/domestic) = 0.41x

> Origin-sensitive — must filter by domestic/foreign.

Top origins: 중국(1,910), 일본(89), 통영(51), 제주도(40), 거문도(17)

**Spec Price Ladder** (within 활/kg):

| Spec Class | Mean Price | Rows |
|---|---|---|
| 중 | 22,631 | 994 |
| count_1-5 | 25,771 | 569 |
| other | 23,840 | 78 |

---

## 72. 강도다리 (Tier C)

| Attribute | Value |
|---|---|
| Rows | 2,072 |
| Trading Days | 1,566 |
| Date Range | 2015.03.25 ~ 2026.01.03 |
| Mean Price | 17,413 KRW |
| CV (volatility) | 0.2687 |
| Lag-1 (daily) | 0.45 |
| Lag-1 (7d smoothed) | 0.9695 |
| Dominant Packaging | kg (99.8%) |
| Flags | `ORIGIN` |

**State Distribution:**

| State | Rows | % | Mean Price |
|---|---|---|---|
| 활 | 2,016 | 97.3% | 17,792 |
| 선 | 56 | 2.7% | 2,732 |

**Origin:** Foreign 7.2%
, price ratio (foreign/domestic) = 0.47x

> Origin-sensitive — must filter by domestic/foreign.

Top origins: 제주도(1,795), 중국(149), 완도(58), 통영(31), 포항(7)

**Spec Price Ladder** (within 활/kg):

| Spec Class | Mean Price | Rows |
|---|---|---|
| 중 | 16,738 | 1,463 |
| count_1-5 | 22,662 | 130 |
| count_6-10 | 21,159 | 136 |
| count_11-20 | 19,859 | 128 |
| count_21+ | 19,372 | 32 |
| other | 19,280 | 115 |

---

## 73. 말백합 (Tier C)

| Attribute | Value |
|---|---|
| Rows | 1,610 |
| Trading Days | 1,094 |
| Date Range | 2020.07.08 ~ 2025.12.30 |
| Mean Price | 117,242 KRW |
| CV (volatility) | 0.2456 |
| Lag-1 (daily) | 0.5431 |
| Lag-1 (7d smoothed) | 0.9739 |
| Dominant Packaging | 그물망 (97.5%) |
| Flags | `SPEC` |

**State Distribution:**

| State | Rows | % | Mean Price |
|---|---|---|---|
| 활 | 1,610 | 100.0% | 114,082 |

**Origin:** Foreign 0.1%
, price ratio (foreign/domestic) = 1.02x

Top origins: 강화(1,255), 태안(238), 장항(67), 인천(45), 중국(2)

**Spec Price Ladder** (within 활/그물망):

| Spec Class | Mean Price | Rows |
|---|---|---|
| 소 | 56,397 | 243 |
| 중 | 124,629 | 1,319 |

Size ratio: 2.2x

> Spec-sensitive — needs per-spec-class prediction.

---

## 74. 홍가리비 (Tier C)

| Attribute | Value |
|---|---|
| Rows | 757 |
| Trading Days | 363 |
| Date Range | 2024.05.11 ~ 2026.01.03 |
| Mean Price | 19,321 KRW |
| CV (volatility) | 0.291 |
| Lag-1 (daily) | 0.8408 |
| Lag-1 (7d smoothed) | 0.9916 |
| Dominant Packaging | box (98.0%) |
| Flags | None |

**State Distribution:**

| State | Rows | % | Mean Price |
|---|---|---|---|
| 활 | 757 | 100.0% | 18,124 |

**Origin:** Foreign 0.5%
, price ratio (foreign/domestic) = 0.78x

Top origins: 삼천포(293), 통영(249), 마산(198), 고흥(9), 중국(4)

**Spec Price Ladder** (within 활/box):

| Spec Class | Mean Price | Rows |
|---|---|---|
| 중 | 18,402 | 734 |

---
