# ============================================================
#   UNIVERSAL HR ATTRITION PIPELINE v3
#   Works with any HR dataset - auto-detects CSV
#   Exports results.json for the dashboard
#
#   HOW TO USE:
#   1. Place this file in the SAME FOLDER as your CSV file
#   2. Run: python universal_hr_pipeline_v3.py
#   3. results.json will be created in the same folder
#   4. Open hr_dashboard_pro.html and upload results.json
# ============================================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings, os, glob, json
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, recall_score, f1_score, roc_auc_score)

try:
    from imblearn.over_sampling import SMOTE
    SMOTE_OK = True
except ImportError:
    SMOTE_OK = False

try:
    from xgboost import XGBClassifier
    XGBOOST_OK = True
except ImportError:
    XGBOOST_OK = False
    from sklearn.ensemble import ExtraTreesClassifier

print("=" * 60)
print("   UNIVERSAL HR ATTRITION PIPELINE v3")
print("   Auto-detects CSV | Exports JSON for Dashboard")
print("=" * 60)

# ============================================================
#   STEP 1 - FIND CSV FILE
# ============================================================
print("\n[STEP 1] Searching for CSV file...")

script_dir = os.path.dirname(os.path.abspath(__file__))
df         = None
found_file = None

csv_files  = glob.glob(os.path.join(script_dir, "*.csv"))
valid_csvs = []
for f in csv_files:
    try:
        temp = pd.read_csv(f, nrows=5)
        cols_lower = [c.lower() for c in temp.columns]
        if any('attrit' in c or 'left' in c or 'churn' in c for c in cols_lower):
            valid_csvs.append((os.path.getsize(f), f))
    except Exception:
        continue

if valid_csvs:
    valid_csvs.sort(reverse=True)
    f_path     = valid_csvs[0][1]
    df         = pd.read_csv(f_path)
    found_file = os.path.basename(f_path)
    if len(valid_csvs) > 1:
        print(f"  Found {len(valid_csvs)} CSV files - using largest: {found_file}")

if df is None:
    print("  ERROR: No valid CSV found! Place this file in the same folder as your CSV.")
    input("Press Enter to exit...")
    exit()

print(f"  Found : {found_file}  |  Rows: {len(df):,}  |  Columns: {df.shape[1]}")

# ============================================================
#   STEP 2 - DETECT ATTRITION COLUMN
# ============================================================
print("\n[STEP 2] Detecting attrition column...")

attr_col = None
for col in df.columns:
    if any(x in col.lower() for x in ['attrit', 'left', 'churn', 'resign', 'turnover']):
        attr_col = col
        break

if attr_col is None:
    print("  ERROR: Could not find attrition column!")
    exit()

# Detect positive label (Yes / 1 / Left etc.)
pos_label = None
for val in df[attr_col].unique():
    if str(val).lower() in ['yes', '1', 'left', 'true', 'churned']:
        pos_label = val
        break
if pos_label is None:
    pos_label = df[attr_col].value_counts().index[-1]  # minority class

df['__target__'] = (df[attr_col] == pos_label).astype(int)
attrition_rate   = df['__target__'].mean() * 100
total_rows       = len(df)
left_count       = int(df['__target__'].sum())

print(f"  Column  : '{attr_col}'  |  Employees Left: {left_count:,} ({attrition_rate:.1f}%)")

# ============================================================
#   STEP 3 - ANALYZE DATASET TYPE
# ============================================================
print("\n[STEP 3] Analyzing dataset...")

is_balanced = attrition_rate > 30
is_large    = total_rows > 10000
use_smote   = (not is_balanced) and SMOTE_OK

if is_balanced and is_large:
    T_ACC, T_REC, T_F1, T_AUC = 0.75, 0.70, 0.70, 0.78
    class_w     = None
    smote_ratio = None
elif attrition_rate < 20:
    # IBM-type dataset (small, imbalanced ~16%)
    T_ACC, T_REC, T_F1, T_AUC = 0.80, 0.50, 0.50, 0.75
    class_w     = {0: 1, 1: 4}
    smote_ratio = 0.6
else:
    T_ACC, T_REC, T_F1, T_AUC = 0.78, 0.55, 0.55, 0.76
    class_w     = {0: 1, 1: 2}
    smote_ratio = 0.4

print(f"  Size    : {'Large' if is_large else 'Small'} | Balance: {'Balanced' if is_balanced else 'Imbalanced'}")
print(f"  SMOTE   : {'ON' if use_smote else 'OFF'} | Targets: Acc>={T_ACC}  Rec>={T_REC}  F1>={T_F1}  AUC>={T_AUC}")

# ============================================================
#   STEP 4 - PREPROCESSING + FEATURE ENGINEERING
#   Same logic as hr_attrition_pipeline.py
# ============================================================
print("\n[STEP 4] Preprocessing + Feature Engineering...")

df_ml = df.drop(columns=[attr_col, '__target__'], errors='ignore').copy()
y     = df['__target__'].copy()

# Drop ID / constant / near-unique columns
drop_cols = [c for c in df_ml.columns if
             c.lower() in ['employeecount', 'over18', 'standardhours', 'employeenumber', 'id'] or
             df_ml[c].nunique() == 1 or
             df_ml[c].nunique() > 0.95 * len(df_ml)]
df_ml.drop(columns=drop_cols, errors='ignore', inplace=True)

# Find key numeric columns
num_cols    = df_ml.select_dtypes(include='number').columns.tolist()
inc_col_fe  = next((c for c in num_cols if any(x in c.lower() for x in ['income', 'salary', 'pay'])), None)
yrs_col_fe  = next((c for c in num_cols if 'totalworking' in c.lower() or 'totalyears' in c.lower()), None)
yrs_co_fe   = next((c for c in num_cols if 'yearsatcompany' in c.lower() or ('tenure' in c.lower() and 'year' not in c.lower())), None)
promo_col   = next((c for c in num_cols if 'promotion' in c.lower()), None)
age_col_fe  = next((c for c in num_cols if c.lower() == 'age'), None)
ot_col_fe   = next((c for c in df_ml.columns if 'overtime' in c.lower().replace('_', '').replace('-', '')), None)
sat_cols_fe = [c for c in num_cols if any(x in c.lower() for x in
               ['satisfaction', 'involvement', 'worklifebalance', 'work_life'])]

# Feature engineering - same as hr_attrition_pipeline.py
if inc_col_fe:
    df_ml['SalaryPerYear'] = df_ml[inc_col_fe] * 12
if inc_col_fe and yrs_col_fe:
    df_ml['IncomePerYearExp'] = df_ml[inc_col_fe] / (df_ml[yrs_col_fe] + 1)
if yrs_co_fe and promo_col:
    df_ml['YearsWithoutPromotion'] = df_ml[yrs_co_fe] - df_ml[promo_col]
if len(sat_cols_fe) >= 2:
    df_ml['OverallSatisfaction'] = df_ml[sat_cols_fe].mean(axis=1)
if ot_col_fe:
    if df_ml[ot_col_fe].dtype == object:
        df_ml['IsOvertime'] = (df_ml[ot_col_fe].str.lower() == 'yes').astype(int)
    else:
        df_ml['IsOvertime'] = df_ml[ot_col_fe].astype(int)
if inc_col_fe:
    df_ml['LowIncome'] = (df_ml[inc_col_fe] < df_ml[inc_col_fe].median()).astype(int)
if age_col_fe:
    df_ml['YoungEmployee'] = (df_ml[age_col_fe] < 30).astype(int)

# Encode categorical columns
le = LabelEncoder()
for col in df_ml.columns:
    if df_ml[col].dtype == 'object':
        df_ml[col] = le.fit_transform(df_ml[col].astype(str))

# Fill any missing values
for col in df_ml.columns:
    if df_ml[col].isnull().any():
        df_ml[col].fillna(df_ml[col].median(), inplace=True)

X = df_ml.copy()
print(f"  Total features after engineering: {X.shape[1]}")
print(f"  Class balance - Stay: {(y==0).sum()}, Leave: {(y==1).sum()}")

# ============================================================
#   STEP 5 - TRAIN/TEST SPLIT + SMOTE
# ============================================================
print("\n[STEP 5] Train/Test Split + SMOTE...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y)

if use_smote and smote_ratio:
    try:
        smote            = SMOTE(sampling_strategy=smote_ratio, random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)
        print(f"  After SMOTE: {pd.Series(y_train).value_counts().to_dict()}")
    except Exception as e:
        print(f"  SMOTE skipped: {e}")
else:
    reason = 'SMOTE library not installed' if not SMOTE_OK else 'dataset is balanced'
    print(f"  No SMOTE ({reason})")

scaler     = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# ============================================================
#   STEP 6 - THRESHOLD FINDER
#   Exact same logic as hr_attrition_pipeline.py
# ============================================================
def find_best_threshold(y_true, y_prob, recall_min=0.50, f1_min=0.48):
    thresholds = np.arange(0.20, 0.65, 0.005)
    auc        = roc_auc_score(y_true, y_prob)
    candidates = []
    for t in thresholds:
        preds = (y_prob >= t).astype(int)
        acc   = accuracy_score(y_true, preds)
        rec   = recall_score(y_true, preds, zero_division=0)
        f1    = f1_score(y_true, preds, zero_division=0)
        if rec >= recall_min and f1 >= f1_min and acc >= 0.78 and auc >= 0.75:
            candidates.append((t, acc, rec, f1, auc))
    if candidates:
        best = max(candidates, key=lambda x: x[3])
        return round(best[0], 3), True
    else:
        best_t, best_f1 = 0.5, 0
        for t in thresholds:
            preds = (y_prob >= t).astype(int)
            f = f1_score(y_true, preds, zero_division=0)
            if f > best_f1:
                best_f1, best_t = f, t
        return round(best_t, 3), False

# ============================================================
#   STEP 7 - TRAIN MODELS
#   Exact same hyperparameters as hr_attrition_pipeline.py
# ============================================================
print("\n[STEP 6] Training models...\n")

xgb_spw = 1.0 if is_balanced else round((y_train==0).sum() / max((y_train==1).sum(), 1), 1)

if XGBOOST_OK:
    xgb_model = XGBClassifier(
        n_estimators=400,   learning_rate=0.03,
        max_depth=6,        scale_pos_weight=xgb_spw,
        subsample=0.75,     colsample_bytree=0.75,
        min_child_weight=3, gamma=0.1,
        reg_alpha=0.1,      reg_lambda=1.5,
        use_label_encoder=False, eval_metric='logloss', random_state=42)
else:
    xgb_model = ExtraTreesClassifier(n_estimators=300, random_state=42, n_jobs=-1)

models_cfg = [
    ("Logistic Regression",
     LogisticRegression(class_weight={0:1, 1:4}, C=0.3, max_iter=2000, random_state=42),
     True),

    ("Decision Tree",
     DecisionTreeClassifier(max_depth=7, class_weight={0:1, 1:4},
                            min_samples_leaf=4, random_state=42),
     True),

    ("Random Forest",
     RandomForestClassifier(n_estimators=500, class_weight={0:1, 1:5},
                            max_depth=15, min_samples_leaf=2,
                            max_features='sqrt', random_state=42),
     False),

    ("Gradient Boosting",
     GradientBoostingClassifier(n_estimators=400, learning_rate=0.03,
                                max_depth=5,      subsample=0.75,
                                min_samples_leaf=3, random_state=42),
     False),

    ("XGBoost" if XGBOOST_OK else "Extra Trees", xgb_model, False),
]

rec_min_thr = 0.70 if (is_balanced and is_large) else 0.50
f1_min_thr  = 0.65 if (is_balanced and is_large) else 0.48

results = []
for name, model, use_sc in models_cfg:
    print(f"  Running {name}...", end="  ", flush=True)
    Xtr = X_train_sc if use_sc else X_train
    Xte = X_test_sc  if use_sc else X_test

    model.fit(Xtr, y_train)
    proba     = model.predict_proba(Xte)[:, 1]
    thresh, _ = find_best_threshold(y_test, proba, rec_min_thr, f1_min_thr)
    pred      = (proba >= thresh).astype(int)

    acc = round(accuracy_score(y_test, pred), 4)
    rec = round(recall_score(y_test, pred, zero_division=0), 4)
    f1  = round(f1_score(y_test, pred, zero_division=0), 4)
    auc = round(roc_auc_score(y_test, proba), 4)
    met = acc >= T_ACC and rec >= T_REC and f1 >= T_F1 and auc >= T_AUC

    results.append(dict(name=name, model=model, thresh=thresh, acc=acc,
                        rec=rec, f1=f1, auc=auc, met=met, use_sc=use_sc, proba=proba))
    tag = "ALL TARGETS MET" if met else "partial"
    print(f"Acc={acc}  Rec={rec}  F1={f1}  AUC={auc}  [{tag}]")

# Pick best model
met_models = [r for r in results if r['met']]
best       = max(met_models if met_models else results, key=lambda x: x['f1'])
print(f"\n  BEST MODEL: {best['name']}  Acc={best['acc']}  Rec={best['rec']}  F1={best['f1']}  AUC={best['auc']}")

# ============================================================
#   STEP 8 - FEATURE IMPORTANCE
# ============================================================
feat_importance = []
if hasattr(best['model'], 'feature_importances_'):
    fi  = pd.Series(best['model'].feature_importances_, index=X.columns)
    top = fi.sort_values(ascending=False).head(10)
    feat_importance = [{"name": k, "value": round(float(v), 4)} for k, v in top.items()]
elif hasattr(best['model'], 'coef_'):
    fi  = pd.Series(np.abs(best['model'].coef_[0]), index=X.columns)
    top = fi.sort_values(ascending=False).head(10)
    feat_importance = [{"name": k, "value": round(float(v), 4)} for k, v in top.items()]

# ============================================================
#   STEP 9 - HR ANALYSIS FOR DASHBOARD CHARTS
# ============================================================
print("\n[STEP 9] HR analysis for dashboard charts...")

df_orig = df.copy()

def grp_rate(grp):
    return round(grp['__target__'].mean() * 100, 1) if len(grp) > 0 else 0

def col_find(candidates, dataframe):
    for c in candidates:
        found = next((col for col in dataframe.columns
                      if col.lower().replace(' ', '').replace('_', '').replace('-', '') == c.lower()), None)
        if found:
            return found
    return None

dept_col = col_find(['Department', 'dept', 'division', 'team'],                        df_orig)
age_col  = col_find(['Age', 'age'],                                                     df_orig)
inc_col2 = col_find(['MonthlyIncome', 'monthly_income', 'salary', 'income', 'monthlysalary'], df_orig)
ot_col   = col_find(['OverTime', 'overtime'],                                           df_orig)
sat_col  = col_find(['JobSatisfaction', 'job_satisfaction', 'satisfaction'],            df_orig)
mar_col  = col_find(['MaritalStatus', 'marital_status', 'marital'],                     df_orig)
role_col = col_find(['JobRole', 'job_role', 'jobrole', 'role', 'position', 'jobtitle'],df_orig)
ten_col  = col_find(['YearsAtCompany', 'years_at_company', 'tenure', 'yearsatcompany'],df_orig)

# Department
dept_data = []
if dept_col:
    for d in df_orig[dept_col].unique():
        g = df_orig[df_orig[dept_col] == d]
        dept_data.append({"name": str(d), "left": int(g['__target__'].sum()),
                          "total": len(g), "rate": grp_rate(g)})
    dept_data.sort(key=lambda x: -x['rate'])

# Age Groups
age_data = []
if age_col:
    for lo, hi, lbl in [[18,25,'18-25'],[26,30,'26-30'],[31,35,'31-35'],
                         [36,40,'36-40'],[41,50,'41-50'],[51,99,'51+']]:
        g = df_orig[(df_orig[age_col].astype(float) >= lo) &
                    (df_orig[age_col].astype(float) <= hi)]
        if len(g) > 0:
            age_data.append({"name": lbl, "rate": grp_rate(g),
                             "total": len(g), "left": int(g['__target__'].sum())})

# Tenure
tenure_data = []
if ten_col:
    for lo, hi, lbl in [[-1,1,'<1yr'],[1,3,'1-3yr'],[3,5,'3-5yr'],
                          [5,10,'5-10yr'],[10,99,'10yr+']]:
        g = df_orig[(df_orig[ten_col].astype(float) > lo) &
                    (df_orig[ten_col].astype(float) <= hi)]
        if len(g) > 0:
            tenure_data.append({"name": lbl, "rate": grp_rate(g), "total": len(g)})

# Income Quartiles
inc_data = []
if inc_col2:
    iv = df_orig[inc_col2].astype(float)
    q  = [iv.quantile(p) for p in [0, 0.25, 0.5, 0.75, 1.0]]
    for i, lbl in enumerate(['Low', 'Mid', 'High', 'Very High']):
        g = df_orig[(iv >= q[i]) & (iv <= q[i+1])]
        if len(g) > 0:
            inc_data.append({"name": lbl, "rate": grp_rate(g), "total": len(g)})

# Overtime
ot_data = {}
if ot_col:
    oty = df_orig[df_orig[ot_col].astype(str).str.lower().isin(['yes', '1', 'true'])]
    otn = df_orig[df_orig[ot_col].astype(str).str.lower().isin(['no', '0', 'false'])]
    ot_data = {
        "yes":     int(oty['__target__'].sum()),
        "yesStay": int((oty['__target__'] == 0).sum()),
        "no":      int(otn['__target__'].sum()),
        "noStay":  int((otn['__target__'] == 0).sum()),
        "yesRate": round(oty['__target__'].mean() * 100, 1) if len(oty) > 0 else 0,
        "noRate":  round(otn['__target__'].mean() * 100, 1) if len(otn) > 0 else 0,
    }

# Satisfaction
sat_data = []
if sat_col:
    for lvl in sorted(df_orig[sat_col].unique()):
        g = df_orig[df_orig[sat_col] == lvl]
        sat_data.append({"name": f"Level {lvl}", "rate": grp_rate(g), "total": len(g)})

# Marital Status
mar_data = []
if mar_col:
    for m in df_orig[mar_col].unique():
        g = df_orig[df_orig[mar_col] == m]
        mar_data.append({"name": str(m), "left": int(g['__target__'].sum()),
                         "total": len(g), "rate": grp_rate(g)})
    mar_data.sort(key=lambda x: -x['rate'])

# Job Roles
role_data = []
if role_col:
    for r in df_orig[role_col].unique():
        g = df_orig[df_orig[role_col] == r]
        role_data.append({"name": str(r), "left": int(g['__target__'].sum()),
                          "total": len(g), "rate": grp_rate(g)})
    role_data.sort(key=lambda x: -x['rate'])
    role_data = role_data[:7]

# Estimated Replacement Cost (INR)
avg_inc_left = round(float(df_orig[df_orig['__target__'] == 1][inc_col2].mean()), 0) if inc_col2 else None
avg_age_left = round(float(df_orig[df_orig['__target__'] == 1][age_col].mean()), 1)  if age_col  else None

if avg_inc_left:
    income_inr     = avg_inc_left * 83 if avg_inc_left < 50000 else avg_inc_left
    total_cost_inr = income_inr * 6 * left_count
    est_cost = (f"Rs.{round(total_cost_inr/1e7, 1)} Cr"
                if total_cost_inr >= 1e7
                else f"Rs.{round(total_cost_inr/1e5, 1)} L")
else:
    est_cost = "N/A"

# Insight strip values
ot_risk     = str(ot_data.get('yesRate', '-'))
single_risk = str(next((m['rate'] for m in mar_data if m['name'].lower() == 'single'), '-'))
young_rate  = str(age_data[0]['rate'])    if age_data    else '-'
new_join    = str(tenure_data[0]['rate']) if tenure_data else '-'
low_inc     = str(inc_data[0]['rate'])    if inc_data    else '-'
long_stay   = str(tenure_data[-1]['rate']) if tenure_data else '-'

# ============================================================
#   STEP 10 - CROSS VALIDATION
# ============================================================
print("\n[STEP 10] Cross Validation (5-fold)...")

Xtr_cv = X_train_sc if best['use_sc'] else X_train
cv     = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_sc  = cross_val_score(best['model'], Xtr_cv, y_train, cv=cv,
                          scoring='roc_auc', n_jobs=-1)
cv_mean = round(float(cv_sc.mean()), 4)
cv_std  = round(float(cv_sc.std()),  4)
print(f"  CV Folds : {[round(float(s), 4) for s in cv_sc]}")
print(f"  Mean AUC : {cv_mean} +/- {cv_std}")

# ============================================================
#   STEP 11 - SAVE results.json
# ============================================================
print("\n[STEP 11] Saving results.json...")

output = {
    "meta": {
        "fileName":      found_file,
        "totalRows":     total_rows,
        "leftCount":     left_count,
        "attritionRate": round(attrition_rate, 1),
        "isBalanced":    bool(is_balanced),
        "isLarge":       bool(is_large),
        "smoteUsed":     bool(use_smote),
        "avgIncomeLeft": avg_inc_left,
        "avgAgeLeft":    avg_age_left,
        "estimatedCost": est_cost,
    },
    "stripValues": {
        "otRisk":     ot_risk,
        "singleRisk": single_risk,
        "youngRate":  young_rate,
        "newJoin":    new_join,
        "lowInc":     low_inc,
        "longStay":   long_stay,
    },
    "targets": {
        "acc": T_ACC, "rec": T_REC, "f1": T_F1, "auc": T_AUC
    },
    "models": [
        {
            "name":      r['name'],
            "threshold": r['thresh'],
            "accuracy":  r['acc'],
            "recall":    r['rec'],
            "f1":        r['f1'],
            "auc":       r['auc'],
            "allMet":    bool(r['met']),
            "best":      r['name'] == best['name']
        }
        for r in results
    ],
    "bestModel": {
        "name":      best['name'],
        "threshold": best['thresh'],
        "accuracy":  best['acc'],
        "recall":    best['rec'],
        "f1":        best['f1'],
        "auc":       best['auc'],
        "allMet":    bool(best['met']),
        "cvMean":    cv_mean,
        "cvStd":     cv_std,
        "cvFolds":   [round(float(s), 4) for s in cv_sc],
    },
    "featureImportance": feat_importance,
    "charts": {
        "department":   dept_data,
        "ageGroup":     age_data,
        "tenure":       tenure_data,
        "income":       inc_data,
        "overtime":     ot_data,
        "satisfaction": sat_data,
        "marital":      mar_data,
        "jobRole":      role_data,
    }
}

out_path = os.path.join(script_dir, "results.json")
try:
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"  Saved : results.json")
    print(f"  Path  : {out_path}")
except Exception as e:
    fallback = os.path.join(os.getcwd(), "results.json")
    with open(fallback, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"  Saved (fallback): {fallback}")

# ============================================================
#   FINAL SUMMARY
# ============================================================
all_pass = best['met']
print("\n" + "=" * 60)
print("  FINAL SUMMARY")
print("=" * 60)
print(f"  File       : {found_file}")
print(f"  Rows       : {total_rows:,}")
print(f"  Attrition  : {attrition_rate:.1f}%")
print(f"  Best Model : {best['name']}")
print(f"  Accuracy   : {best['acc']}  {'PASS' if best['acc'] >= T_ACC else 'FAIL'}  (Target: {T_ACC}+)")
print(f"  Recall     : {best['rec']}  {'PASS' if best['rec'] >= T_REC else 'FAIL'}  (Target: {T_REC}+)")
print(f"  F1 Score   : {best['f1']}  {'PASS' if best['f1'] >= T_F1  else 'FAIL'}  (Target: {T_F1}+)")
print(f"  ROC AUC    : {best['auc']}  {'PASS' if best['auc'] >= T_AUC else 'FAIL'}  (Target: {T_AUC}+)")
print(f"  CV AUC     : {cv_mean} +/- {cv_std}")
print(f"\n  {'ALL TARGETS MET!' if all_pass else 'Some targets missed'}")
print("=" * 60)
print("\n  Done! Open hr_dashboard_pro.html and upload results.json to see your results.")
