# Customer Segmentation — RFM Analysis & K-means

**English** | [Français](README.fr.md)

Segmenting the customers of a real online retailer using RFM analysis (Recency, Frequency, Monetary) and K-means clustering, to answer a question every store owner asks: *who are my best customers, and who am I about to lose?*

**Key finding:** about **20% of customers generate 74% of revenue** — and nearly a quarter of customers are valuable buyers who have stopped coming back.

**🚀 [Live demo](https://isadora-customer-segmentation.streamlit.app)** — explore the segments in 3D and classify a customer yourself.

---

## Context

Before moving into IT, I managed a fashion retail business. Knowing which customers to reward, which to win back and which to let go was a daily question — answered by intuition. This project answers it with data, using techniques from the *Unsupervised Learning* course of Andrew Ng's Machine Learning Specialization.

## Data

[Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii) (UCI Machine Learning Repository): **1,067,371 transactions** from a UK-based online gift-ware retailer, December 2009 to December 2011. Many of its customers are wholesalers.

> Chen, D. (2012). *Online Retail II* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5CG6D — licensed under CC BY 4.0.

The data is not included in this repository (45 MB). See [How to run](#how-to-run).

## Results

K-means found **four natural segments** among 5,839 customers:

| Segment | Customers | Last purchase | Orders | Typical spend | Share of revenue |
|---|---|---|---|---|---|
| **Champions** | 20.2% | ~2 weeks ago | 13 | £4,863 | **73.7%** |
| **At risk** | 24.7% | ~6 months ago | 4 | £1,443 | 16.2% |
| **Promising** | 21.6% | ~3 weeks ago | 3 | £713 | 6.4% |
| **Lost** | 33.5% | over a year ago | 1 | £269 | 3.7% |

*Typical values are medians.*

**What a business could do with this:**
- **Champions** — protect them: loyalty perks, priority service. Losing a few would hurt badly.
- **At risk** — the highest-return target: they were good customers, they haven't returned in six months (not even for the 2011 holiday season), and they are still recoverable. Winning back a customer costs less than finding a new one.
- **Promising** — encourage the next purchase to build the habit.
- **Lost** — low-cost reactivation at most; a third of customers, but under 4% of revenue.

## Challenges & solutions

**1. Cancelled orders hiding in the data.** After the first cleaning pass, one customer ranked among the ten biggest buyers with only 2 orders — an inconsistent profile. Investigating revealed an order of **80,995 units** of a single item, cancelled the same day. Removing cancellation rows wasn't enough: the original order stayed in the data, as if the sale had happened.
→ Each cancellation is now matched to its original purchase (same customer, product and quantity, placed before the cancellation) and both are removed: **6,332 orders and £673,413 of "phantom" revenue** taken out.

**2. Purchases without a customer.** 235,151 transactions (22%) had no customer ID and could not be segmented — likely guest checkouts. They were removed and documented as a limitation. Every cleaning step is logged; in total, **72.2% of rows were kept**.

**3. A few giants distorting everything.** The top 1% of customers account for **31% of revenue**. Since K-means works with distances, these wholesalers would have dominated the clustering. Rather than removing them — they are real, and the most valuable — a **log transformation** compresses large values, followed by **standardization** so that no metric outweighs the others because of its units.

**4. Choosing the number of segments.** The silhouette score is highest at k = 2, but two segments are too coarse to act on. The score then shows a **local peak at k = 4** (0.367, above k = 3 and k = 5), and the elbow curve bends in the same area — four segments is supported by the data *and* actionable.

**5. Checking a hypothesis.** The recency distribution showed an unusual peak around 400 days. Hypothesis: holiday shoppers who never came back. Checking the 769 customers concerned: their last purchase was in fall 2010 — mostly **October and November**, not December. For a gift-ware supplier with many wholesale customers, this points to **retailers stocking up before Christmas** who did not reorder the following year.

**6. Classic RFM scores vs. K-means.** The traditional method (scores from 1 to 5 by quintile) was computed for comparison. It cuts customers into equal-sized groups whether or not those groups make sense — the score distribution comes out almost flat. K-means finds boundaries from the data itself.

## Limitations

- The data shows **that** *At risk* customers left, not **why** (a competitor? dissatisfaction?). Answering that would need other sources: surveys, customer service records.
- **Partial returns** (e.g. 3 units returned out of 10) can't be matched to an exact order and are not netted out.
- *Promising* customers are active, but without their first purchase date we can't say they are **new**.

## Project structure

```
customer-segmentation/
├── notebooks/
│   ├── 01_cleaning.ipynb     # cleaning, with a log of every step
│   ├── 02_rfm.ipynb          # RFM metrics, log transformation, classic scores
│   └── 03_clustering.ipynb   # K-means, choice of k, segment profiles
├── app/
│   ├── streamlit_app.py      # interactive demo (Streamlit)
│   ├── rfm_segments.csv      # segment results used by the demo
│   └── requirements.txt
├── requirements.txt
└── data/                     # not versioned — see below
```

## How to run

```bash
git clone https://github.com/IsadoraEaston/customer-segmentation.git
cd customer-segmentation
python -m venv .venv
.venv\Scripts\activate        # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
```

Download the dataset from the [UCI page](https://archive.ics.uci.edu/dataset/502/online+retail+ii), extract it, and place `online_retail_II.xlsx` in a `data/` folder. Then run the notebooks in order (01 → 02 → 03) with `jupyter notebook`.

## Next steps

- Check whether *At risk* customers made more partial returns before leaving (a possible sign of dissatisfaction)
- Compare scikit-learn's K-means with my own implementation from the ML Specialization labs

## Tech stack

Python · pandas · NumPy · scikit-learn · Matplotlib · Plotly · Streamlit · Jupyter

---

**Isabelle D. Easton** — [Portfolio](https://isadoraeaston.github.io) · [GitHub](https://github.com/IsadoraEaston)
