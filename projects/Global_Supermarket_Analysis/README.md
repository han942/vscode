## 📊 Global Supermarket Analysis
### 📌 Project Overview
This project performs an in-depth Exploratory Data Analysis (EDA) and Loss Analysis on a global retail dataset containing 51,290 transactions. Beyond standard visualization, this study utilizes advanced feature engineering and multivariate analytics to identify systemic revenue leakage—specifically uncovering that a small segment of high-discount trades accounts for 68% of total corporate losses. The final output provides data-backed strategic guidelines to stabilize profit margins and optimize global operations.


### 🛠️ Key Technical Implementation

High-Fidelity Data Cleaning: Processed 26 features across 51,290 rows with 0% missing values to ensure statistical reliability.

Advanced Feature Engineering: Designed custom economic indicators to deepen the analysis:


`pre_sales`: Reconstructed original revenue before discount application to measure the true impact of pricing policies.


`uni_cost`: Calculated unit cost per product to evaluate the health of margin structures.


`eta`: Modeled the lead time from order to shipment to assess logistics performance.


`Statistical Correlation`: Quantified the relationship between numeric variables using a heatmap, proving a negative correlation **(-0.32)** between Discount and Profit.


### 🔍 Core Insights (EDA & Loss Analysis)

Critical Loss Threshold: Identified that transactions with discount rates exceeding 40% represent only 13% of the data but contribute to 68% of total losses.

Segmented Risk Profiling:


High-Risk Categories: The "Tables" sub-category was identified as the primary driver of margin compression, especially from manufacturers such as Bevis, Barricks, and Lesro.



Regional Disparity: In APAC, US, and LATAM markets, the loss share from table sales is significantly higher than their overall market loss share.


Temporal Dynamics: Recorded a 34.2% surge in transaction volume in 2022, with distinct demand spikes during the mid-year and year-end seasons.

### 💡 Strategic Recommendations
Aggressive Discount Reform:

Implement a mandatory discount cap of 40% for high-volume categories like Phones and Bookcases to protect the bottom line.


Reduce average discount rates in the EU by 20 percentage points to correct regional profitability.

Portfolio & Market Optimization:

Downsize or discontinue table offerings from manufacturers with consistent high-loss profiles (Bevis, Barricks, Lesro).

Execute targeted pricing adjustments in high-loss cities like Istanbul and Lagos by restructuring discount bands below the 40% threshold.

### 🧰 Tech Stack
Language: Python

Libraries: Pandas (Data Wrangling), Matplotlib & Seaborn (Statistical Visualization), NumPy

Analysis Techniques: Multivariate EDA, Feature Engineering, Correlation Analysis, Risk Assessment, Business Intelligence Reporting
