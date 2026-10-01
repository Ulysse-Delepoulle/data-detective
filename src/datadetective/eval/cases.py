"""Benchmark cases with known answers (ground truth).

Each case is a question over a dataset, plus the tokens that must appear in the
agent's report for the answer to count as correct. Answers were computed by
hand or directly from the files, independently of the agent, which is what
makes them a valid ground truth. Scoring removes commas before matching, so
"650000" matches "650,000".

Cases are split into two tiers so the headline number is honest:
- "smoke": tiny synthetic datasets, one operation each. Fast, exact, and good
  for catching regressions, but trivial. Not a quality measure.
- "benchmark": real datasets (Titanic, Superstore, Telco), including a few
  harder multi-step questions. This is the meaningful quality signal.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BenchmarkCase:
    name: str
    dataset: str             # filename under data/sample_datasets
    question: str
    expect: tuple[str, ...]  # every token must appear in the report
    tier: str = "benchmark"  # "smoke" or "benchmark"


CASES: tuple[BenchmarkCase, ...] = (
    # ----------------------------------------------------------------------
    # SMOKE: synthetic datasets, one operation each. Fast regression coverage.
    # ----------------------------------------------------------------------
    BenchmarkCase(
        "sales_top_region", "sample_sales.csv",
        "Which region has the highest total revenue, where revenue is units times price?",
        ("West",), tier="smoke",
    ),
    BenchmarkCase(
        "sales_west_revenue", "sample_sales.csv",
        "What is the total revenue (units times price) for the West region?",
        ("95.5",), tier="smoke",
    ),
    BenchmarkCase(
        "emp_top_avg_salary_dept", "employees.csv",
        "Which department has the highest average salary?",
        ("Engineering",), tier="smoke",
    ),
    BenchmarkCase(
        "emp_oldest", "employees.csv",
        "Who is the oldest employee?",
        ("Eve",), tier="smoke",
    ),
    BenchmarkCase(
        "emp_total_salary", "employees.csv",
        "What is the total salary across all employees?",
        ("650000",), tier="smoke",
    ),
    BenchmarkCase(
        "visits_total", "monthly_visits.csv",
        "What is the total number of visits across all months?",
        ("9300",), tier="smoke",
    ),
    BenchmarkCase(
        "visits_peak_month", "monthly_visits.csv",
        "Which month had the highest number of visits?",
        ("2024-05",), tier="smoke",
    ),

    # ----------------------------------------------------------------------
    # BENCHMARK: real datasets, single-step questions.
    # ----------------------------------------------------------------------
    # titanic.csv
    BenchmarkCase(
        "titanic_total_passengers", "titanic.csv",
        "How many passengers are in the dataset?",
        ("891",),
    ),
    BenchmarkCase(
        "titanic_survivors", "titanic.csv",
        "How many passengers survived (Survived equals 1)?",
        ("342",),
    ),
    BenchmarkCase(
        "titanic_sex_higher_survival", "titanic.csv",
        "Which sex had the higher survival rate, male or female?",
        ("female",),
    ),
    BenchmarkCase(
        "titanic_third_class_count", "titanic.csv",
        "How many passengers traveled in third class (Pclass equals 3)?",
        ("491",),
    ),
    BenchmarkCase(
        "titanic_southampton_count", "titanic.csv",
        "How many passengers embarked from Southampton (Embarked equals S)?",
        ("644",),
    ),
    # superstore.csv
    BenchmarkCase(
        "super_top_sales_region", "superstore.csv",
        "Which region has the highest total Sales?",
        ("West",),
    ),
    BenchmarkCase(
        "super_top_category_sales", "superstore.csv",
        "Which product Category has the highest total Sales?",
        ("Technology",),
    ),
    BenchmarkCase(
        "super_most_profitable_region", "superstore.csv",
        "Which region has the highest total Profit?",
        ("West",),
    ),
    BenchmarkCase(
        "super_top_segment", "superstore.csv",
        "Which customer Segment appears most frequently in the data?",
        ("Consumer",),
    ),
    BenchmarkCase(
        "super_unique_orders", "superstore.csv",
        "How many unique orders are there, counting distinct Order ID values?",
        ("5015",),
    ),
    # telco_churn.csv
    BenchmarkCase(
        "telco_total_customers", "telco_churn.csv",
        "How many customers are in the dataset?",
        ("7043",),
    ),
    BenchmarkCase(
        "telco_churned_count", "telco_churn.csv",
        "How many customers churned, where Churn equals Yes?",
        ("1869",),
    ),
    BenchmarkCase(
        "telco_top_contract", "telco_churn.csv",
        "What is the most common Contract type?",
        ("Month-to-month",),
    ),
    BenchmarkCase(
        "telco_male_count", "telco_churn.csv",
        "How many customers have gender equal to Male?",
        ("3555",),
    ),

    # ----------------------------------------------------------------------
    # BENCHMARK: harder multi-step questions (filter then aggregate).
    # ----------------------------------------------------------------------
    BenchmarkCase(
        "titanic_firstclass_women_survived", "titanic.csv",
        "How many first-class women survived (Pclass equals 1, Sex is female, Survived equals 1)?",
        ("91",),
    ),
    BenchmarkCase(
        "super_west_top_subcategory", "superstore.csv",
        "Within the West region only, which Sub-Category has the highest total Profit?",
        ("Copiers",),
    ),
    BenchmarkCase(
        "telco_m2m_churned", "telco_churn.csv",
        "How many Month-to-month contract customers churned (Contract is Month-to-month and Churn equals Yes)?",
        ("1655",),
    ),

    # ----------------------------------------------------------------------
    # HARD: multi-step reasoning, shared by both backends so local and cloud
    # can be compared where there is headroom to differ.
    # ----------------------------------------------------------------------
    BenchmarkCase(
        "titanic_children_3rd_survived", "titanic.csv",
        "How many children (Age under 18) in third class (Pclass equals 3) survived?",
        ("29",), tier="hard",
    ),
    BenchmarkCase(
        "super_lowest_profit_subcategory", "superstore.csv",
        "Which Sub-Category has the lowest (most negative) total Profit?",
        ("Tables",), tier="hard",
    ),
    BenchmarkCase(
        "super_negative_profit_subcats", "superstore.csv",
        "Which Sub-Categories have a negative total Profit? List all of them.",
        ("Tables", "Bookcases", "Supplies"), tier="hard",
    ),
    BenchmarkCase(
        "telco_fiber_churned", "telco_churn.csv",
        "How many customers with Fiber optic internet service churned (Churn equals Yes)?",
        ("1297",), tier="hard",
    ),

    # Harder still: data cleaning, a ratio, a correlation sign, a filtered
    # median. These probe the agent's limits and may not always pass.
    BenchmarkCase(
        "telco_totalcharges_sum", "telco_churn.csv",
        "What is the sum of the TotalCharges column? The column may contain blank "
        "or non-numeric values; treat those as missing. Round to the nearest whole number.",
        ("16056169",), tier="hard",
    ),
    BenchmarkCase(
        "super_profit_margin", "superstore.csv",
        "What is the overall profit margin, that is total Profit as a percentage of "
        "total Sales, rounded to one decimal place?",
        ("12.5",), tier="hard",
    ),
    BenchmarkCase(
        "super_discount_profit_corr", "superstore.csv",
        "Is the correlation between Discount and Profit positive or negative?",
        ("negative",), tier="hard",
    ),
    BenchmarkCase(
        "titanic_firstclass_median_age", "titanic.csv",
        "What is the median age of first-class passengers (Pclass equals 1)?",
        ("37",), tier="hard",
    ),
)

