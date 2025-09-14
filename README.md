# Comment Classification & Analysis Pipeline

This Python script processes and analyzes user comments related to beauty products, specifically in the categories of skincare, makeup, and fragrance. It includes steps for cleaning, translation, classification, metric calculation, and prepares the data for visualization in Power BI.

---

## Features

* Decodes emojis and translates non-English text into English
* Tokenizes and classifies comments into:

  * Skincare
  * Makeup
  * Fragrance
* Extracts brand and product names
* Calculates SOE (Share of Engagement) metrics
* Uses clustering and distancing to calculate relevance 
* Assigns comment quality scores and spam recognition using relevance and other metrics
* Exports final data to Excel for Power BI integration

---

## Input

* A CSV file containing raw user comments and related data

> **Important:**
> You must specify the full path to your CSV file on **lines 39 and 40** of the script.
> You must input k= optimal number of clusters on **lines 574** of the script

---

## Output

* An Excel file containing cleaned and categorized comment data

---

## Requirements

* Python 3.x
* Required libraries:

  * pandas
  * numpy
  * langdetect
  * emoji
  * openpyxl
  * (and any other dependencies used in the script)

---

## Usage

1. Download or clone the script.
2. Open the script and set your CSV file path on lines 39 and 40.
3. Run the script:

   ```bash
   python databrixkcode.py
   ```
4. The final Excel file will be saved and ready for Power BI.

---

## Power BI Integration

* Import the exported Excel file into Power BI.
* Upload the data into the Power BI template.
* Relevant fields (category, brand, product, SOE metrics, quality scores) will be updated on the dashboard accordingly.
