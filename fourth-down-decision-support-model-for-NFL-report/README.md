# Fourth-Down Decision Support Model for NFL Play-Calling

## Project Overview

This project builds a probability model for NFL fourth-down conversion attempts. The goal is not simply to classify whether a play will succeed, but to estimate the probability of success well enough for the model to be useful inside a football decision-support workflow.

The notebook develops a complete modeling pipeline: exploratory data analysis, football-specific feature engineering, baseline modeling, model comparison, time-aware tuning, calibration analysis, and final model selection. The report was also structured for clean Quarto rendering so it can be read as a polished final project report.

## Research Question

Can game-state information such as yards to go, field position, play type, and football-specific engineered features be used to estimate the probability that a fourth-down attempt will be converted?

## Why This Matters

Fourth-down decisions are among the highest-leverage choices in football. Coaches must decide whether to punt, kick a field goal, or go for the conversion. A model that produces reasonable conversion probabilities can support these decisions by adding a consistent, data-driven estimate of risk and reward.

This project focuses only on conversion probability. It is intended as one component of a broader decision system, not as a complete fourth-down recommendation engine.

## Data

- **Source:** NFL play-by-play data obtained through `nflreadpy`, which draws from the `nflverse` data ecosystem.
- **Unit of observation:** One fourth-down conversion attempt.
- **Sample size:** 4,553 plays.
- **Date range:** September 5, 2019 to February 9, 2025.
- **Response variable:** `converted` (`Yes` / `No`).
- **Core predictors:** `togo`, `yardline`, `play_type`, `posteam`, `defteam`.

The notebook uses a **time-based train/test split**:

- Training set: plays before `2024-01-01`
- Test set: plays on or after `2024-01-01`

This setup better reflects real deployment, where a model is trained on past seasons and evaluated on future games.

## Exploratory Findings

The exploratory analysis shows that:

- The response is reasonably balanced: about **52.9%** of attempts were converted.
- **Yards to go** is the clearest single driver of conversion success.
- Fourth-and-1 situations convert far more often than long-yardage situations.
- In the grouped EDA summary:
  - `1 yard`: **67.3%** conversion rate
  - `11+ yards`: **18.5%** conversion rate
- Run plays convert at a higher observed rate than pass plays in this sample.

These findings motivate the use of grouped distance, field-context indicators, and interaction terms in the final model.

## Feature Engineering

To make the model more football-aware, the notebook creates several engineered features:

- `red_zone`: indicator for plays snapped inside the opponent's 20-yard line
- `goal_to_go`: indicator for goal-to-go situations
- `distance_group`: grouped version of yards to go
- `field_position_group`: grouped field-position context
- `togo_yardline_ratio`: relationship between needed yards and field position
- `pass_play`: pass indicator
- `pass_togo`: interaction between play type and distance

These features are designed to reflect how coaches and analysts think about fourth-down context rather than relying only on raw columns.

## Modeling Approach

The project compares several candidate models:

1. **Baseline Logistic Regression**
   - Uses only the original variables.
   - Numeric features are standardized.
   - Categorical features are one-hot encoded.

2. **Feature-Engineered Logistic Regression**
   - Uses the engineered football features.
   - Evaluated both **with** and **without** team identifiers.

3. **Random Forest with Target Encoding**
   - Tests whether a non-linear ensemble can improve probability estimates.

4. **HistGradientBoosting with Target Encoding**
   - Tests a second non-linear approach with regularization and early stopping.

5. **Time-Series Cross-Validation for Tuning**
   - `GridSearchCV` for logistic regression
   - `RandomizedSearchCV` for random forest
   - Uses `TimeSeriesSplit` to preserve temporal ordering

## Model Selection Logic

Because this project is about **probability estimation**, the most important metrics are:

- **Log Loss**
- **Brier Score**

ROC-AUC and accuracy are still reported, but they are secondary to probability quality. A model with a slightly higher ROC-AUC but worse log loss is not preferred if the main goal is reliable conversion probabilities.

## Final Results

### Baseline Logistic Regression

- Accuracy: **0.6282**
- ROC-AUC: **0.6699**
- Log Loss: **0.6365**
- Brier Score: **0.2235**

### Best Overall Model

The selected model is the **feature-engineered logistic regression without team identifiers**.

Held-out test performance:

- Accuracy: **0.6593**
- ROC-AUC: **0.6912**
- Log Loss: **0.6265**
- Brier Score: **0.2186**
- Precision: **0.6907**
- Recall: **0.7271**
- F1: **0.7085**

### Why This Model Was Chosen

It produced the best out-of-sample **probability metrics** among all candidate models. Even though a tuned random forest achieved a slightly higher ROC-AUC in one comparison, its log loss and Brier score were worse than the logistic model. For this problem, the simpler logistic model was the stronger and more defensible choice.

## Calibration and Baseline Comparison

The final model was also evaluated as a probability model:

- Naive baseline Brier Score: **0.2477**
- Final model Brier Score: **0.2186**
- Brier Skill Score: **0.1175**
- Expected Calibration Error (ECE): **0.0414**
- Maximum Calibration Error (MCE): **0.1208**

Interpretation:

- The final model reduces probability error by about **11.8%** relative to a naive baseline that always predicts the training-set average conversion rate.
- Calibration is reasonably good overall, although some probability bins still show noticeable error.

## Main Takeaways

- Football-specific feature engineering improved performance over the baseline model.
- Team identifiers did **not** improve generalization on future-season data.
- The final model is interpretable, stable, and reasonably calibrated.
- The model is useful as a **decision-support prototype**, but not as a full fourth-down recommendation system.

## Repository Structure

```text
.
├── README.md
├── football.parquet
├── Football – CS 307 Lab.pdf
├── fourth-down-decision-support-model-for-NFL-notebook.ipynb
└── fourth-down-decision-support-model-for-NFL-report.html
```

## Limitations

This model estimates only conversion probability. It does **not** account for:

- score differential
- quarter and time remaining
- timeout situation
- weather
- player quality
- personnel groupings
- play design
- win probability or expected points

Because of that, the model should not be used alone to decide whether an offense should go for it on fourth down.

## Future Improvements

Possible next steps include:

- adding richer game-context variables
- incorporating player- and team-strength information that updates over time
- combining conversion probability with expected points or win probability
- testing calibration on additional future seasons
- building a full fourth-down decision framework that compares go-for-it, punt, and field-goal options
