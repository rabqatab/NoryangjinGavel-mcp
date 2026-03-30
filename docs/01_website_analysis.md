# Website Analysis: Noryangjin Fish Market

## Overview

| Item                  | Details                                                |
| --------------------- | ------------------------------------------------------ |
| **Website**           | 노량진수산시장 (Noryangjin Fisheries Wholesale Market) |
| **URL**               | https://www.susansijang.co.kr                          |
| **Data Type**         | Fish auction prices (경락시세)                         |
| **Language**          | Korean (with EN/JA/CN versions available)              |
| **Data Availability** | **January 2000 ~ Present** (25+ years)                 |

> **Confirmed**: Data is available from January 2000. The user's target start date of January 2, 2004 is fully supported.

---

## Target Endpoint for Crawling

### Today's Auction Prices (오늘의 경락시세) - PRIMARY

- **URL**: `/nsis/miw/ko/info/miw3110`
- **Method**: POST
- **Features**:
  - Single date query (one date per request)
  - Optional fish species filter
  - Includes ALL fish types (76+ species)
  - Pagination support (10 items per page)
- **Crawling Strategy**: Daily iteration from Jan 1, 2004 to present (Jan 1, 2004 is vacant since it was holiday)

---

## Other Endpoints (NOT Used for Crawling)

| Endpoint           | URL                         | Reason Not Used                                 |
| ------------------ | --------------------------- | ----------------------------------------------- |
| Prices by Species  | `/nsis/miw/ko/info/miw3130` | Limited to 29 species, different data structure |
| Prices by Category | `/nsis/miw/ko/info/miw3120` | Category-based, not comprehensive               |
| Prices by Origin   | `/nsis/miw/ko/info/miw3140` | Origin-based filtering                          |
| Statistics         | `/nsis/miw/ko/info/miw3161` | Pre-aggregated, less granular                   |

---

## Data Structure (miw3110)

### Price Record Fields (8 columns)

| Field (Korean) | Field (English) | Data Type | Description                    |
| -------------- | --------------- | --------- | ------------------------------ |
| 어종           | Fish Species    | String    | Species name with state prefix |
| 산지           | Origin          | String    | Source location                |
| 규격           | Specification   | String    | Size or count per unit         |
| 포장           | Packaging       | String    | Packaging type                 |
| 수량           | Quantity        | Decimal   | Quantity (weight or count)     |
| 낙찰고가       | Highest Bid     | Integer   | Highest auction price (KRW)    |
| 낙찰저가       | Lowest Bid      | Integer   | Lowest auction price (KRW)     |
| 평균가         | Average Price   | Integer   | Average price (KRW)            |

> **Note**: miw3110 has 8 columns. The miw3130 endpoint has 9 columns (includes 중량/weight), but we are NOT using that endpoint.

### Fish State Prefixes

The fish species name includes a state prefix indicating the product condition:

| Prefix | Korean | English   | Description           |
| ------ | ------ | --------- | --------------------- |
| (선)   | 선어   | Fresh     | Fresh fish (not live) |
| (활)   | 활어   | Live      | Live fish/seafood     |
| (냉)   | 냉동   | Frozen    | Frozen products       |
| (가공) | 가공   | Processed | Processed products    |

**Example**: `(냉)고등어` = Frozen Mackerel

### Packaging Types

| Code      | Description        |
| --------- | ------------------ |
| kg        | Per kilogram       |
| S/P       | Styrofoam package  |
| box       | Box/crate          |
| CT/(BT)   | Carton/Basket      |
| C/S       | Box                |
| c/s(상자) | Box (older format) |
| PAN(펜)   | Pan                |

### Specification Formats

| Format   | Example | Meaning            |
| -------- | ------- | ------------------ |
| {N}미    | 22미    | 22 fish per unit   |
| 대/중/소 | 대      | Large/Medium/Small |
| M1/M2    | M2      | Size grade         |
| (empty)  |         | Not specified      |

---

## Fish Species (76+ types from miw3110)

Species list varies by day. Current dropdown includes:

```
가무락, 가오리, 가자미, 감성돔, 감숭어, 갑오징어, 강도다리, 개조개,
고등어, 금태, 까치복, 깐굴, 깐바지락, 깐홍합, 꼴뚜기, 낙지, 넙치,
농어, 능성어, 다랑어, 대게, 대구, 대하(새우), 돌김, 동죽, 만디,
매생이, 먹갈치, 명태, 무늬오징어, 문어, 물메기, 물미역, 물바지락,
미더덕, 민어, 바다가재, 방어, 백합, 복어, 볼락, 봉굴, 봉바지락,
부시리, 분홍새우, 삼치, 새꼬막, 소라, 수꽃게, 아귀, 연자돔, 오징어,
왕게, 우럭, 우럭조개, 은갈치, 자바리, 잡어, 적어, 전복, 절단게,
점성어, 준치, 진주담치, 참가자미, 참돔, 참복, 참숭어, 참조기,
칼바지락, 토바지락, 피꼬막(피조개), 해삼, 홍가리비, 홍어, 흑점줄전갱이
```

> **Note**: Historical data may include additional species no longer in the dropdown (e.g., 골뱅이, 돔, 새치, 백조기, 자주복, etc.)

---

## Form Submission Details

### Endpoint: `/nsis/miw/ko/info/miw3110`

**Method**: POST

**Parameters**:

| Parameter | Type    | Required | Description                       |
| --------- | ------- | -------- | --------------------------------- |
| pageIndex | Integer | Yes      | Page number (1-based)             |
| pageUnit  | Integer | Yes      | Items per page (default: 10)      |
| pageSize  | Integer | Yes      | Page size (default: 10)           |
| kdfshNm   | String  | No       | Fish species filter (empty = all) |
| searchDe  | String  | Yes      | Search date (YYYY.MM.DD)          |

**Example Request**:

```
POST /nsis/miw/ko/info/miw3110
Content-Type: application/x-www-form-urlencoded

pageIndex=1&pageUnit=10&pageSize=10&kdfshNm=&searchDe=2004.01.02
```

### Date Format

- Format: `YYYY.MM.DD`
- Example: `2004.01.02`

### Pagination

- Default: 10 items per page
- Total pages shown in pagination div
- Example: Today's data has ~39 pages (~390 records) or more
- Historical data (2004): ~1-5 pages per day

---

## Sample Data

### Raw HTML Table Structure (miw3110)

```html
<table>
  <thead>
    <tr>
      <th>어종</th>
      <th>산지</th>
      <th>규격</th>
      <th>포장</th>
      <th>수량</th>
      <th>낙찰고가</th>
      <th>낙찰저가</th>
      <th>평균가</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>(활)방어</td>
      <td>일본</td>
      <td>2미</td>
      <td>kg</td>
      <td class="a-r">203</td>
      <td class="a-r">30,000</td>
      <td class="a-r">5,000</td>
      <td class="a-r">13,900</td>
    </tr>
  </tbody>
</table>
```

### Sample Record (2004.01.02)

```
어종: (선)대게
산지: 러시아
규격: 중
포장: kg
수량: 35.3
낙찰고가: 6,000
낙찰저가: 1,500
평균가: 4,600
```

---

## Origin Locations (Sample)

Common origin locations found in the data:

| Region | Locations                  |
| ------ | -------------------------- |
| 부산   | 부산(기장), 부산           |
| 제주   | 제주도                     |
| 전남   | 여수, 목포, 완도, 순천     |
| 경남   | 통영, 거제, 삼천포         |
| 강원   | 강릉, 속초, 묵호           |
| 충남   | 안흥, 서천, 대천           |
| 경북   | 포항, 강구, 울진           |
| 전북   | 군산                       |
| 수입   | 일본, 중국, 러시아, 필리핀 |
| 원양   | (원양)포클랜드             |
| 기타   | 기타                       |

---

## Data Availability by Year

| Year         | Data Available | Estimated Records/Day |
| ------------ | -------------- | --------------------- |
| 2000         | Yes            | ~10-50                |
| 2001-2003    | Yes            | ~20-100               |
| 2004-2010    | Yes            | ~50-150               |
| 2011-2020    | Yes            | ~100-250              |
| 2021-Present | Yes            | ~200-400              |

> **Note**: Market closed on Sundays and major holidays (no data on those days)

---

## Technical Notes

### Session Management

- Website uses standard HTTP sessions
- No authentication required for public data
- Cookies may be used for session tracking

### Pagination Detection

- Standard server-side pagination
- Navigation via `fnList(pageNumber)` JavaScript function
- Last page number visible in pagination div: `fnList(N)` where N is the last page
- Check for `.arr.last` element to find total pages

### Rate Limiting

- No explicit rate limiting observed
- Recommended: 0.5-1 second delay between requests
- Recommended: Max 90 requests per minute

---

## Crawling Strategy Summary

| Aspect               | Value                                      |
| -------------------- | ------------------------------------------ |
| **Endpoint**         | `/nsis/miw/ko/info/miw3110`                |
| **Date Range**       | 2004.01.02 ~ Present                       |
| **Approach**         | Iterate each date, fetch all pages         |
| **Species Filter**   | None (fetch all species per day)           |
| **Expected Days**    | ~7,600 days (21 years, excluding holidays) |
| **Expected Records** | ~1-2 million total                         |
