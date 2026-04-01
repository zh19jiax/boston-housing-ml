# CS506-FINAL
## Project Description
This project intends to clean, analyze, and produce meaningful visualizations of data concerning housing violations within the City of Boston; it aims to identify and evaluate potential patterns involving building management, characteristics, locality, and the nature of these violations.

## Timeline
**Week 0 (2/11)** - Project outlining/planning  

**Week 1 & 2 (2/15)** - Data preprocessing, implement routines to import, clean, and compile data from various sources 

**Week 3 & 4 (3/1)** - Implement routines to calculate data statistics relevant to goals, scheduled check-in #1  

**Week 5 & 6 (3/15)** - Data analysis, implement initial data visualization routines, initial tests  

**Week 7 & 8 (3/29)** - Optimization of existing routines, scheduled check-in #2  

**Week 9 & 10 (4/5)** - Debugging/testing of optimized code, visualization clean up  

**Week 11 (4/26)** - Final touches, presentation prep  


## Project Goals
For this project, our primary objective is to understand and identify the main problems regarding Boston's housing compliance. We would like to pursue this objective in the following directions:

**Goal 1:** Find offending patterns in the city by identifying and ranking management companies or individual housing owners with the highest violation frequencies.

**Goal 2:** Determine if certain building characteristics, such as build year and property type, are associated with higher violation rates. 

**Goal 3:** Categorize and quantify the specific types of building complaints from the residents to understand the most prevalent housing violation issues in the city.

Achieving these goals will allow the government to focus on the pain points of the housing issues, effectively increasing the satisfaction level of Boston residents. 


## What Data/How we Collected it
For this project, we will use publicly available datasets provided by the City of Boston. Instead of collecting new data ourselves, we will use the existing city records and combine them to form a single, clean dataset that connects violations to specific properties, owners, and neighborhoods. All data is available for download from the city's website Analyze Boston.

The datasets include:

- **Building and Property Violations:** Our primary dataset, containing records of housing and building-related complaints and violations across the city.
  [Building and Property Violations](https://data.boston.gov/dataset/building-and-property-violations1)
- **Public Works Violations:** Additional violation data that provides more context about property-related issues.
  [Public Works Violations](https://data.boston.gov/dataset/public-works-violations)
- **Property Assessment Data:** Contains details about properties such as parcel IDs, ownership information, building type, and build year.
  [Property Assesment](https://data.boston.gov/dataset/property-assessment)
- **SAM (Street Address Management) Addresses Dataset:** Used to standardize and match addresses across all datasets.
  [SAM Database](https://data.boston.gov/dataset/live-street-address-management-sam-addresses)
- **Neighborhood Boundaries / SAM Neighborhood Boundaries:** Used to group properties by location and determine which communities are most affected.
  [Neighborhood Boundaries](https://data.boston.gov/dataset/bpda-neighborhood-boundaries) [SAM Neighborhood Boundaries](https://bostonopendata-boston.opendata.arcgis.com/datasets/boston::live-street-address-management-sam-addresses/about)

We plan to link the datasets together using shared identifiers. Specifically:

- The `sam_id` column in the building violations dataset corresponds to the `sam_address_id` column in the SAM Addresses dataset.
- The `parcel` column in the SAM dataset corresponds to the `PID` column in the Property Assessment dataset, which has information about building ownership.
- This same mapping can also be applied to the Public Works Violations dataset.

Using these relationships, we aim to connect each violation to a specific address, parcel, and property record. 

## Preliminary Findings

The following findings are drawn from our initial data exploration on the merged dataset of 17,075 violation records.

**Goal 1 — Identifying Offending Patterns**
Violations are geographically concentrated, with Dorchester accounting for nearly 27% of all cases (4,526 of 17,075). A small number of private LLCs and limited partnerships are disproportionate repeat offenders — the top 15 private owners each hold 20–43 violations, and most are structured as LLCs, which complicates direct accountability. Enforcement resolution is consistent citywide (~94% closed), meaning the problem is not inspection response but recurring non-compliance from the same actors.

**Goal 2 — Building Characteristics and Violation Rates**
Building age is the strongest signal: the median build year of violating properties is 1910, with the distribution heavily concentrated before 1930. Multi-family residential buildings (2-family, 3-family, condos) account for the majority of violations, suggesting that tenant density amplifies the impact of owner negligence. Overall building condition, however, is not a reliable predictor — most violations occur in "Average" or "Good" rated buildings, meaning structural decay is not necessary for non-compliance to occur.

**Goal 3 — Prevalent Complaint Types**
Two categories dominate: "Failure to Obtain Permit" (4,181 cases, 24.5%) and "Unsafe and Dangerous" (3,611 cases, 21.1%), together making up nearly half of all violations. This split suggests two distinct problem populations — one driven by regulatory non-compliance (permit failures, certification lapses) and one by genuine structural safety risk — which likely require different enforcement strategies to address.
