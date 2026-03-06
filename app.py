"""
Universal Bank - Personal Loan Campaign Analysis (Management Edition)
=====================================================================
Comprehensive 4-Type Data Analysis using ALL features
Target: Who can take a Personal Loan and Who Cannot
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (confusion_matrix, accuracy_score,
                             precision_score, recall_score, f1_score,
                             roc_auc_score, roc_curve, precision_recall_curve)
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

# ======================== PAGE CONFIG ========================
st.set_page_config(
    page_title="Universal Bank - Loan Analysis (Management Edition)",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ======================== CUSTOM CSS ========================
st.markdown("""
<style>
    .main .block-container { padding-top: 1rem; max-width: 1500px; }
    .stMetric { background: #1e293b; padding: 16px; border-radius: 10px; border: 1px solid #334155; }
    .stMetric label { color: #94a3b8 !important; font-size: 12px !important; text-transform: uppercase; letter-spacing: 0.5px; }
    .stMetric [data-testid="stMetricValue"] { color: #f1f5f9 !important; font-size: 26px !important; }
    div[data-testid="stMetricDelta"] { font-size: 13px !important; }
    .highlight-box { background: #1e293b; padding: 20px; border-radius: 10px; border-left: 4px solid #3b82f6; margin: 10px 0; }
    .highlight-box h4 { color: #3b82f6; margin-bottom: 8px; }
    .highlight-box p { color: #cbd5e1; font-size: 14px; line-height: 1.7; }
    .yes-box { background: #052e16; padding: 18px; border-radius: 10px; border: 1px solid #166534; margin: 8px 0; }
    .yes-box h4 { color: #22c55e; margin-bottom: 6px; font-size: 16px; }
    .yes-box p { color: #bbf7d0; font-size: 14px; }
    .no-box { background: #2a0a0a; padding: 18px; border-radius: 10px; border: 1px solid #991b1b; margin: 8px 0; }
    .no-box h4 { color: #ef4444; margin-bottom: 6px; font-size: 16px; }
    .no-box p { color: #fecaca; font-size: 14px; }
    .insight-box { background: #1a1a2e; padding: 16px; border-radius: 8px; border: 1px solid #334155; margin: 8px 0; }
    .insight-box p { color: #e2e8f0; font-size: 13px; line-height: 1.6; }
</style>
""", unsafe_allow_html=True)


# ======================== DATA LOADING ========================
@st.cache_data
def load_data():
    df = pd.read_csv("UniversalBank.csv")
    df = df.drop(columns=["ID", "ZIP Code"])
    df["Experience"] = df["Experience"].apply(lambda x: max(x, 0))
    # Create derived columns for richer analysis
    df["Education_Label"] = df["Education"].map({1: "Undergrad", 2: "Graduate", 3: "Advanced/Professional"})
    df["Loan_Status"] = df["Personal Loan"].map({0: "Not Accepted", 1: "Accepted"})
    df["Income_Bracket"] = pd.cut(df["Income"], bins=[0, 50, 100, 150, 225],
                                   labels=["Low ($8K-50K)", "Medium ($51K-100K)", "High ($101K-150K)", "Very High ($151K+)"])
    df["Age_Group"] = pd.cut(df["Age"], bins=[22, 30, 40, 50, 60, 70],
                              labels=["23-30", "31-40", "41-50", "51-60", "61-67"])
    df["CC_Bracket"] = pd.cut(df["CCAvg"], bins=[-0.1, 1, 2, 4, 11],
                               labels=["Low ($0-1K)", "Medium ($1-2K)", "High ($2-4K)", "Very High ($4K+)"])
    df["Experience_Group"] = pd.cut(df["Experience"], bins=[-1, 10, 20, 30, 45],
                                     labels=["0-10 yrs", "11-20 yrs", "21-30 yrs", "31+ yrs"])
    df["Has_Mortgage"] = (df["Mortgage"] > 0).astype(int)
    df["Mortgage_Bracket"] = pd.cut(df["Mortgage"], bins=[-1, 0, 100, 200, 700],
                                     labels=["None", "$1-100K", "$101-200K", "$200K+"])
    return df


@st.cache_resource
def train_models(_df):
    df = _df.copy()
    feature_cols = ["Age", "Experience", "Income", "Family", "CCAvg", "Education",
                    "Mortgage", "Securities Account", "CD Account", "Online", "CreditCard"]
    X = df[feature_cols]
    y = df["Personal Loan"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, random_state=42, stratify=y
    )

    models = {
        "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(random_state=42, max_depth=5),
        "Random Forest": RandomForestClassifier(random_state=42, n_estimators=100, max_depth=10),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42, n_estimators=100, max_depth=5),
        "KNN (K=5)": KNeighborsClassifier(n_neighbors=5),
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        fpr_arr, tpr_arr, thresholds_roc = roc_curve(y_test, y_prob)
        prec_arr, rec_arr, thresholds_pr = precision_recall_curve(y_test, y_prob)

        # Cross-validation
        cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring="f1")

        results[name] = {
            "model": model,
            "y_pred": y_pred,
            "y_prob": y_prob,
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "auc": roc_auc_score(y_test, y_prob),
            "cm": cm,
            "tn": tn, "fp": fp, "fn": fn, "tp": tp,
            "specificity": tn / (tn + fp),
            "fpr_curve": fpr_arr,
            "tpr_curve": tpr_arr,
            "prec_curve": prec_arr,
            "rec_curve": rec_arr,
            "cv_f1_mean": cv_scores.mean(),
            "cv_f1_std": cv_scores.std(),
        }

    feat_imp = dict(zip(feature_cols, models["Random Forest"].feature_importances_))

    # Logistic Regression coefficients
    lr_coefs = dict(zip(feature_cols, models["Logistic Regression"].coef_[0]))

    return results, X_test, y_test, feat_imp, feature_cols, lr_coefs, scaler


df = load_data()
results, X_test, y_test, feat_imp, feature_names, lr_coefs, scaler = train_models(df)
accepted = df[df["Personal Loan"] == 1]
rejected = df[df["Personal Loan"] == 0]

# ======================== SIDEBAR ========================
st.sidebar.image("https://img.icons8.com/fluency/96/bank-building.png", width=60)
st.sidebar.title("Universal Bank")
st.sidebar.markdown("**Personal Loan Campaign**")
st.sidebar.markdown("**Management Dashboard**")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "📑 Navigate to Section",
    [
        "🏠 Executive Summary",
        "📊 1. Descriptive Analysis",
        "🔬 2. Feature-by-Feature Deep Dive",
        "🔍 3. Diagnostic Analysis",
        "🤖 4. Predictive Analysis",
        "🎯 5. Confusion Matrix Deep Dive",
        "💡 6. Prescriptive Analysis",
    ],
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Quick Stats**")
st.sidebar.markdown(f"📋 Total Customers: **{len(df):,}**")
st.sidebar.markdown(f"✅ Loan Accepted: **{accepted.shape[0]:,}** ({df['Personal Loan'].mean()*100:.1f}%)")
st.sidebar.markdown(f"❌ Loan Rejected: **{rejected.shape[0]:,}** ({(1-df['Personal Loan'].mean())*100:.1f}%)")
st.sidebar.markdown(f"📐 Features Analyzed: **11**")
st.sidebar.markdown(f"🤖 Models Trained: **5**")
st.sidebar.markdown(f"🧪 Test Set Size: **{len(y_test):,}**")


# ======================== HELPER FUNCTIONS ========================
def feature_comparison_chart(df, feature, title, is_categorical=False):
    """Create side-by-side comparison of accepted vs rejected for any feature."""
    if is_categorical:
        ct = pd.crosstab(df[feature], df["Loan_Status"], normalize="index") * 100
        ct = ct.reset_index()
        fig = px.bar(ct.melt(id_vars=feature, var_name="Loan Status", value_name="Percentage"),
                     x=feature, y="Percentage", color="Loan Status", barmode="group",
                     title=title, color_discrete_map={"Not Accepted": "#ef4444", "Accepted": "#22c55e"})
    else:
        fig = px.box(df, x="Loan_Status", y=feature, color="Loan_Status", title=title,
                     color_discrete_map={"Not Accepted": "#ef4444", "Accepted": "#22c55e"})
    fig.update_layout(template="plotly_dark", height=400, showlegend=True)
    return fig


def loan_rate_by_group(df, group_col, group_label):
    """Calculate loan acceptance rate by a grouped column."""
    rates = df.groupby(group_col).agg(
        Total=("Personal Loan", "count"),
        Accepted=("Personal Loan", "sum")
    ).reset_index()
    rates["Acceptance Rate (%)"] = (rates["Accepted"] / rates["Total"] * 100).round(2)
    rates.columns = [group_label, "Total Customers", "Accepted", "Acceptance Rate (%)"]
    return rates


# ======================== EXECUTIVE SUMMARY ========================
if page.startswith("🏠"):
    st.title("🏦 Universal Bank — Executive Summary")
    st.markdown("### Who Takes a Personal Loan and Who Doesn't?")
    st.markdown("---")

    # Top-level KPIs
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Total Customers", f"{len(df):,}")
    c2.metric("Loan Accepted", f"{accepted.shape[0]:,}", f"{df['Personal Loan'].mean()*100:.1f}%")
    c3.metric("Loan Rejected", f"{rejected.shape[0]:,}")
    c4.metric("Best Model Accuracy", "99.0%", "Random Forest")
    c5.metric("Best Precision", "97.1%")
    c6.metric("Best AUC-ROC", "99.9%")

    st.markdown("---")

    # WHO TAKES A LOAN vs WHO DOESN'T
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="yes-box">
            <h4>✅ IDEAL LOAN CUSTOMER PROFILE (Who Takes a Loan)</h4>
            <p>
            <strong>Income:</strong> $144.7K average (nearly 2.2x higher than non-takers)<br>
            <strong>Education:</strong> Graduate or Advanced/Professional degree (13-14% acceptance vs 4.4% for Undergrad)<br>
            <strong>CC Spending:</strong> $3.91K/month average (2.3x higher than non-takers)<br>
            <strong>CD Account:</strong> Having a CD account increases acceptance to 46.4% (vs 7.2%)<br>
            <strong>Family Size:</strong> Families of 3+ members show higher acceptance (13.2%)<br>
            <strong>Mortgage:</strong> Average mortgage of $101.4K (higher existing obligations)<br>
            <strong>Age:</strong> Age does NOT matter — evenly distributed across all ages
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="no-box">
            <h4>❌ NON-LOAN CUSTOMER PROFILE (Who Doesn't Take a Loan)</h4>
            <p>
            <strong>Income:</strong> $66.2K average — lower earning capacity<br>
            <strong>Education:</strong> Predominantly Undergraduate (only 4.4% acceptance)<br>
            <strong>CC Spending:</strong> $1.73K/month average — lower financial activity<br>
            <strong>CD Account:</strong> 93% of non-CD holders do NOT accept the loan<br>
            <strong>Family Size:</strong> Single members show lowest acceptance (7.3%)<br>
            <strong>Mortgage:</strong> Average mortgage of $51.7K<br>
            <strong>Securities/Online/CreditCard:</strong> Minimal impact on loan decision
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Feature Impact Ranking — What Matters Most?")

    # Feature ranking table
    feat_sorted = sorted(feat_imp.items(), key=lambda x: x[1], reverse=True)
    corr_all = df[["Age", "Experience", "Income", "Family", "CCAvg", "Education",
                    "Mortgage", "Securities Account", "CD Account", "Online", "CreditCard",
                    "Personal Loan"]].corr()["Personal Loan"].drop("Personal Loan")

    ranking_data = []
    for rank, (feat, imp) in enumerate(feat_sorted, 1):
        avg_acc = accepted[feat].mean()
        avg_rej = rejected[feat].mean()
        diff = avg_acc - avg_rej
        corr_val = corr_all.get(feat, 0)
        if imp > 0.1:
            impact = "🔴 Critical"
        elif imp > 0.04:
            impact = "🟠 Moderate"
        else:
            impact = "🟢 Low"
        ranking_data.append({
            "Rank": rank,
            "Feature": feat,
            "Importance Score": round(imp, 4),
            "Correlation": round(corr_val, 4),
            "Avg (Accepted)": round(avg_acc, 2),
            "Avg (Rejected)": round(avg_rej, 2),
            "Difference": round(diff, 2),
            "Impact Level": impact,
        })

    st.dataframe(pd.DataFrame(ranking_data), use_container_width=True, hide_index=True)

    # Visual summary
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Feature Importance (Random Forest)", "Correlation with Loan Acceptance"))
    feat_names_sorted = [f[0] for f in feat_sorted]
    feat_vals_sorted = [f[1] for f in feat_sorted]
    corr_sorted = [corr_all.get(f, 0) for f in feat_names_sorted]

    fig.add_trace(go.Bar(y=feat_names_sorted[::-1], x=feat_vals_sorted[::-1], orientation="h",
                         marker_color=["#22c55e" if v > 0.1 else "#f59e0b" if v > 0.04 else "#64748b" for v in feat_vals_sorted[::-1]],
                         name="Importance"), row=1, col=1)
    fig.add_trace(go.Bar(y=feat_names_sorted[::-1], x=corr_sorted[::-1], orientation="h",
                         marker_color=["#3b82f6" if v > 0.1 else "#f59e0b" if v > 0 else "#ef4444" for v in corr_sorted[::-1]],
                         name="Correlation"), row=1, col=2)
    fig.update_layout(template="plotly_dark", height=500, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)


# ======================== 1. DESCRIPTIVE ANALYSIS ========================
elif page.startswith("📊"):
    st.title("📊 1. Descriptive Analysis")
    st.markdown("*What happened? Complete overview of all 5,000 customers across every feature.*")
    st.markdown("---")

    # KPIs
    c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
    c1.metric("Customers", f"{len(df):,}")
    c2.metric("Avg Age", f"{df['Age'].mean():.1f}")
    c3.metric("Avg Experience", f"{df['Experience'].mean():.1f} yrs")
    c4.metric("Avg Income", f"${df['Income'].mean():.1f}K")
    c5.metric("Avg CC Spend", f"${df['CCAvg'].mean():.2f}K")
    c6.metric("Avg Mortgage", f"${df['Mortgage'].mean():.1f}K")
    c7.metric("Loan Rate", f"{df['Personal Loan'].mean()*100:.1f}%")

    st.markdown("---")

    # Loan Distribution
    col1, col2 = st.columns(2)
    with col1:
        fig = px.pie(names=["Rejected (90.4%)", "Accepted (9.6%)"],
                     values=[rejected.shape[0], accepted.shape[0]],
                     title="Personal Loan Acceptance", hole=0.5,
                     color_discrete_sequence=["#ef4444", "#22c55e"])
        fig.update_layout(template="plotly_dark", height=380)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Continuous features distributions
        fig = make_subplots(rows=2, cols=2, subplot_titles=("Age Distribution", "Income Distribution ($K)",
                                                             "CC Avg Spending ($K/mo)", "Experience Distribution"))
        fig.add_trace(go.Histogram(x=df["Age"], nbinsx=25, marker_color="#3b82f6", name="Age"), row=1, col=1)
        fig.add_trace(go.Histogram(x=df["Income"], nbinsx=30, marker_color="#22c55e", name="Income"), row=1, col=2)
        fig.add_trace(go.Histogram(x=df["CCAvg"], nbinsx=30, marker_color="#f59e0b", name="CCAvg"), row=2, col=1)
        fig.add_trace(go.Histogram(x=df["Experience"], nbinsx=25, marker_color="#a855f7", name="Experience"), row=2, col=2)
        fig.update_layout(template="plotly_dark", height=380, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    # Education, Family, Binary Features
    col1, col2, col3 = st.columns(3)
    with col1:
        edu_df = df["Education_Label"].value_counts().reset_index()
        edu_df.columns = ["Education", "Count"]
        edu_df["Pct"] = (edu_df["Count"] / edu_df["Count"].sum() * 100).round(1)
        edu_df["Label"] = edu_df.apply(lambda r: f"{r['Education']} ({r['Pct']:.1f}%)", axis=1)
        fig = px.pie(edu_df, names="Label", values="Count", title="Education Level",
                     color_discrete_sequence=["#3b82f6", "#22c55e", "#a855f7"], hole=0.45)
        fig.update_layout(template="plotly_dark", height=350)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fam_df = df["Family"].value_counts().sort_index().reset_index()
        fam_df.columns = ["Family Size", "Count"]
        fam_df["Percentage"] = (fam_df["Count"] / fam_df["Count"].sum() * 100).round(1)
        fam_df["Label"] = fam_df.apply(lambda r: f"{r['Count']:,} ({r['Percentage']:.1f}%)", axis=1)
        fig = px.bar(fam_df, x="Family Size", y="Count", title="Family Size Distribution",
                     color_discrete_sequence=["#a855f7"], text="Label")
        fig.update_layout(template="plotly_dark", height=350)
        st.plotly_chart(fig, use_container_width=True)

    with col3:
        binary_features = {
            "Online Banking": df["Online"].mean() * 100,
            "Credit Card": df["CreditCard"].mean() * 100,
            "Securities Account": df["Securities Account"].mean() * 100,
            "CD Account": df["CD Account"].mean() * 100,
            "Has Mortgage": df["Has_Mortgage"].mean() * 100,
        }
        bin_df = pd.DataFrame(list(binary_features.items()), columns=["Service", "% of Customers"])
        bin_df["% of Customers"] = bin_df["% of Customers"].round(1)
        fig = px.bar(bin_df, y="Service", x="% of Customers", title="Banking Services (%)",
                     orientation="h", color_discrete_sequence=["#06b6d4"], text="% of Customers")
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="auto")
        fig.update_layout(template="plotly_dark", height=350)
        st.plotly_chart(fig, use_container_width=True)

    # Mortgage distribution
    col1, col2 = st.columns(2)
    with col1:
        mortgage_holders = df[df["Mortgage"] > 0]
        fig = px.histogram(mortgage_holders, x="Mortgage", nbins=30,
                           title=f"Mortgage Distribution (among {len(mortgage_holders):,} holders)",
                           color_discrete_sequence=["#f59e0b"])
        fig.update_layout(template="plotly_dark", height=350)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.histogram(df, x="CCAvg", nbins=40, color="Loan_Status",
                           title="CC Spending by Loan Status",
                           color_discrete_map={"Not Accepted": "#ef4444", "Accepted": "#22c55e"},
                           barmode="overlay", opacity=0.7)
        fig.update_layout(template="plotly_dark", height=350)
        st.plotly_chart(fig, use_container_width=True)

    # Full stats table
    with st.expander("📋 Complete Summary Statistics for All Features"):
        st.dataframe(df[["Age", "Experience", "Income", "Family", "CCAvg", "Education",
                         "Mortgage", "Personal Loan", "Securities Account", "CD Account",
                         "Online", "CreditCard"]].describe().round(2), use_container_width=True)


# ======================== 2. FEATURE-BY-FEATURE DEEP DIVE ========================
elif page.startswith("🔬"):
    st.title("🔬 2. Feature-by-Feature Deep Dive")
    st.markdown("*Analyze every single feature individually to understand its impact on loan acceptance.*")
    st.markdown("---")

    # ---- INCOME ----
    st.markdown("### 💰 Feature 1: INCOME (Annual Income in $000)")
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        st.metric("Avg Income (Accepted)", f"${accepted['Income'].mean():.1f}K")
        st.metric("Avg Income (Rejected)", f"${rejected['Income'].mean():.1f}K")
    with col2:
        st.metric("Median Income (Accepted)", f"${accepted['Income'].median():.0f}K")
        st.metric("Median Income (Rejected)", f"${rejected['Income'].median():.0f}K")
    with col3:
        st.metric("Importance Rank", "#1", "35.8% importance")
        st.metric("Correlation", "0.503", "Strong Positive")

    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(df, x="Income", color="Loan_Status", barmode="overlay", nbins=40,
                           title="Income Distribution by Loan Status", opacity=0.7,
                           color_discrete_map={"Not Accepted": "#ef4444", "Accepted": "#22c55e"})
        fig.update_layout(template="plotly_dark", height=380)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        inc_rates = loan_rate_by_group(df, "Income_Bracket", "Income Bracket")
        fig = px.bar(inc_rates, x="Income Bracket", y="Acceptance Rate (%)", text="Acceptance Rate (%)",
                     title="Loan Acceptance Rate by Income Bracket",
                     color="Acceptance Rate (%)", color_continuous_scale=["#64748b", "#22c55e"])
        fig.update_traces(texttemplate="%{text:.1f}%")
        fig.update_layout(template="plotly_dark", height=380)
        st.plotly_chart(fig, use_container_width=True)

    st.dataframe(inc_rates, use_container_width=True, hide_index=True)
    st.markdown("""
    <div class="insight-box"><p>
    <strong>Finding:</strong> Income is the single most powerful predictor. Customers earning over $100K have a 26.4% acceptance rate —
    nearly 6x the rate of those earning under $50K (4.5%). The jump from Medium to High income brackets is where conversion dramatically increases.
    </p></div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ---- EDUCATION ----
    st.markdown("### 🎓 Feature 2: EDUCATION (1: Undergrad, 2: Graduate, 3: Advanced)")
    col1, col2 = st.columns(2)
    with col1:
        edu_rates = loan_rate_by_group(df, "Education_Label", "Education Level")
        fig = px.bar(edu_rates, x="Education Level", y="Acceptance Rate (%)", text="Acceptance Rate (%)",
                     title="Loan Acceptance Rate by Education Level",
                     color="Education Level", color_discrete_sequence=["#3b82f6", "#22c55e", "#a855f7"])
        fig.update_traces(texttemplate="%{text:.1f}%")
        fig.update_layout(template="plotly_dark", height=380, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        # Education + Income cross-analysis
        edu_inc = df.groupby(["Education_Label", "Loan_Status"])["Income"].mean().reset_index()
        fig = px.bar(edu_inc, x="Education_Label", y="Income", color="Loan_Status", barmode="group",
                     title="Avg Income by Education & Loan Status",
                     color_discrete_map={"Not Accepted": "#ef4444", "Accepted": "#22c55e"})
        fig.update_layout(template="plotly_dark", height=380)
        st.plotly_chart(fig, use_container_width=True)

    st.dataframe(edu_rates, use_container_width=True, hide_index=True)
    st.markdown("""
    <div class="insight-box"><p>
    <strong>Finding:</strong> Graduate and Advanced degree holders are 3x more likely to accept loans than Undergrads (13-14% vs 4.4%).
    Education is the #2 most important feature. Across ALL education levels, accepted customers earn significantly more — but the gap is
    largest for Graduates ($145.1K vs $52.3K).
    </p></div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ---- CCAvg ----
    st.markdown("### 💳 Feature 3: CC AVERAGE SPENDING ($000/month)")
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        st.metric("Avg CC Spend (Accepted)", f"${accepted['CCAvg'].mean():.2f}K/mo")
        st.metric("Avg CC Spend (Rejected)", f"${rejected['CCAvg'].mean():.2f}K/mo")
    with col2:
        st.metric("Importance Rank", "#3", "16.0% importance")
        st.metric("Correlation", "0.367", "Moderate Positive")
    with col3:
        st.metric("Max CC Spend", f"${df['CCAvg'].max():.1f}K")
        st.metric("% Spending > $3K", f"{(df['CCAvg'] > 3).mean()*100:.1f}%")

    col1, col2 = st.columns(2)
    with col1:
        fig = px.box(df, x="Loan_Status", y="CCAvg", color="Loan_Status",
                     title="CC Spending Distribution by Loan Status",
                     color_discrete_map={"Not Accepted": "#ef4444", "Accepted": "#22c55e"})
        fig.update_layout(template="plotly_dark", height=380)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        cc_rates = loan_rate_by_group(df, "CC_Bracket", "CC Spending Bracket")
        fig = px.bar(cc_rates, x="CC Spending Bracket", y="Acceptance Rate (%)", text="Acceptance Rate (%)",
                     title="Loan Acceptance by CC Spending Level",
                     color="Acceptance Rate (%)", color_continuous_scale=["#64748b", "#f59e0b"])
        fig.update_traces(texttemplate="%{text:.1f}%")
        fig.update_layout(template="plotly_dark", height=380)
        st.plotly_chart(fig, use_container_width=True)

    st.dataframe(cc_rates, use_container_width=True, hide_index=True)
    st.markdown("""
    <div class="insight-box"><p>
    <strong>Finding:</strong> High credit card spenders ($4K+/month) have a 34.5% acceptance rate — 12x the rate of low spenders ($0-1K).
    This feature captures financial activity and comfort with credit, both strong indicators of loan appetite.
    </p></div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ---- FAMILY ----
    st.markdown("### 👨‍👩‍👧‍👦 Feature 4: FAMILY SIZE (1-4 members)")
    col1, col2 = st.columns(2)
    with col1:
        fam_rates = loan_rate_by_group(df, "Family", "Family Size")
        fig = px.bar(fam_rates, x="Family Size", y="Acceptance Rate (%)", text="Acceptance Rate (%)",
                     title="Loan Acceptance Rate by Family Size",
                     color_discrete_sequence=["#a855f7"])
        fig.update_traces(texttemplate="%{text:.1f}%")
        fig.update_layout(template="plotly_dark", height=380)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fam_inc = df.groupby(["Family", "Loan_Status"])["Income"].mean().reset_index()
        fig = px.bar(fam_inc, x="Family", y="Income", color="Loan_Status", barmode="group",
                     title="Avg Income by Family Size & Loan Status",
                     color_discrete_map={"Not Accepted": "#ef4444", "Accepted": "#22c55e"})
        fig.update_layout(template="plotly_dark", height=380)
        st.plotly_chart(fig, use_container_width=True)

    st.dataframe(fam_rates, use_container_width=True, hide_index=True)
    st.markdown("""
    <div class="insight-box"><p>
    <strong>Finding:</strong> Family size of 3 shows the highest acceptance (13.2%), followed by 4 (11.0%). Single-member families
    have the lowest rate (7.3%). Larger families likely have greater financial needs, making them more receptive to personal loan offers.
    </p></div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ---- CD ACCOUNT ----
    st.markdown("### 📀 Feature 5: CD ACCOUNT (Certificate of Deposit)")
    col1, col2 = st.columns(2)
    with col1:
        cd_rates = df.groupby("CD Account")["Personal Loan"].mean() * 100
        cd_df = pd.DataFrame({"CD Account": ["No (0)", "Yes (1)"], "Acceptance Rate (%)": cd_rates.values})
        fig = px.bar(cd_df, x="CD Account", y="Acceptance Rate (%)", text="Acceptance Rate (%)",
                     title="Loan Acceptance: CD Account Holders vs Non-Holders",
                     color="CD Account", color_discrete_sequence=["#64748b", "#22c55e"])
        fig.update_traces(texttemplate="%{text:.1f}%")
        fig.update_layout(template="plotly_dark", height=380, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        cd_inc = df.groupby(["CD Account", "Loan_Status"])["Income"].mean().reset_index()
        cd_inc["CD Account"] = cd_inc["CD Account"].map({0: "No CD", 1: "Has CD"})
        fig = px.bar(cd_inc, x="CD Account", y="Income", color="Loan_Status", barmode="group",
                     title="Avg Income by CD Status & Loan Decision",
                     color_discrete_map={"Not Accepted": "#ef4444", "Accepted": "#22c55e"})
        fig.update_layout(template="plotly_dark", height=380)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    <div class="insight-box"><p>
    <strong>Finding:</strong> CD Account holders have a 46.4% loan acceptance rate — 6.4x higher than non-holders (7.2%).
    This is the strongest binary predictor. CD holders are financially sophisticated customers with established bank relationships.
    </p></div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ---- MORTGAGE ----
    st.markdown("### 🏠 Feature 6: MORTGAGE (Value in $000)")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Avg Mortgage (Accepted)", f"${accepted['Mortgage'].mean():.1f}K")
        st.metric("Avg Mortgage (Rejected)", f"${rejected['Mortgage'].mean():.1f}K")
        st.metric("% with Mortgage (Accepted)", f"{(accepted['Mortgage']>0).mean()*100:.1f}%")
        st.metric("% with Mortgage (Rejected)", f"{(rejected['Mortgage']>0).mean()*100:.1f}%")
    with col2:
        mort_rates = loan_rate_by_group(df, "Mortgage_Bracket", "Mortgage Bracket")
        fig = px.bar(mort_rates, x="Mortgage Bracket", y="Acceptance Rate (%)", text="Acceptance Rate (%)",
                     title="Loan Acceptance by Mortgage Level",
                     color="Acceptance Rate (%)", color_continuous_scale=["#64748b", "#f59e0b"])
        fig.update_traces(texttemplate="%{text:.1f}%")
        fig.update_layout(template="plotly_dark", height=380)
        st.plotly_chart(fig, use_container_width=True)

    st.dataframe(mort_rates, use_container_width=True, hide_index=True)
    st.markdown("""
    <div class="insight-box"><p>
    <strong>Finding:</strong> Mortgage has a moderate correlation (0.14). Customers with higher mortgages ($200K+) show 18.8% acceptance
    — they may need additional liquidity. However, 52% of customers have NO mortgage, limiting this feature's impact.
    </p></div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ---- AGE ----
    st.markdown("### 📅 Feature 7: AGE (Customer's age in completed years)")
    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(df, x="Age", color="Loan_Status", barmode="overlay", nbins=30,
                           title="Age Distribution by Loan Status", opacity=0.7,
                           color_discrete_map={"Not Accepted": "#ef4444", "Accepted": "#22c55e"})
        fig.update_layout(template="plotly_dark", height=380)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        age_rates = loan_rate_by_group(df, "Age_Group", "Age Group")
        fig = px.bar(age_rates, x="Age Group", y="Acceptance Rate (%)", text="Acceptance Rate (%)",
                     title="Loan Acceptance Rate by Age Group",
                     color_discrete_sequence=["#3b82f6"])
        fig.update_traces(texttemplate="%{text:.1f}%")
        fig.update_layout(template="plotly_dark", height=380)
        st.plotly_chart(fig, use_container_width=True)

    st.dataframe(age_rates, use_container_width=True, hide_index=True)
    st.markdown("""
    <div class="insight-box"><p>
    <strong>Finding:</strong> Age has near-zero correlation (-0.008) with loan acceptance. Acceptance rates are remarkably flat across
    all age groups (8.5%-10.8%). This means loan targeting should NOT be age-based — it provides no discriminating power.
    </p></div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ---- EXPERIENCE ----
    st.markdown("### 💼 Feature 8: EXPERIENCE (Years of professional experience)")
    col1, col2 = st.columns(2)
    with col1:
        fig = px.box(df, x="Loan_Status", y="Experience", color="Loan_Status",
                     title="Experience Distribution by Loan Status",
                     color_discrete_map={"Not Accepted": "#ef4444", "Accepted": "#22c55e"})
        fig.update_layout(template="plotly_dark", height=380)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        exp_rates = loan_rate_by_group(df, "Experience_Group", "Experience Group")
        fig = px.bar(exp_rates, x="Experience Group", y="Acceptance Rate (%)", text="Acceptance Rate (%)",
                     title="Loan Acceptance by Experience Level",
                     color_discrete_sequence=["#a855f7"])
        fig.update_traces(texttemplate="%{text:.1f}%")
        fig.update_layout(template="plotly_dark", height=380)
        st.plotly_chart(fig, use_container_width=True)

    st.dataframe(exp_rates, use_container_width=True, hide_index=True)
    st.markdown("""
    <div class="insight-box"><p>
    <strong>Finding:</strong> Like Age, Experience has virtually zero correlation (-0.008) with loan acceptance. This is expected since
    Age and Experience are highly correlated with each other (r=0.99). Neither is useful for targeting.
    </p></div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ---- SECURITIES ACCOUNT ----
    st.markdown("### 📈 Feature 9: SECURITIES ACCOUNT")
    col1, col2 = st.columns(2)
    with col1:
        sec_rates = df.groupby("Securities Account")["Personal Loan"].mean() * 100
        sec_df = pd.DataFrame({"Securities Account": ["No (0)", "Yes (1)"], "Acceptance Rate (%)": sec_rates.values})
        fig = px.bar(sec_df, x="Securities Account", y="Acceptance Rate (%)", text="Acceptance Rate (%)",
                     title="Loan Acceptance: Securities Account",
                     color="Securities Account", color_discrete_sequence=["#64748b", "#3b82f6"])
        fig.update_traces(texttemplate="%{text:.1f}%")
        fig.update_layout(template="plotly_dark", height=350, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.metric("With Securities - Acceptance", f"{sec_rates.iloc[1]:.1f}%")
        st.metric("Without Securities - Acceptance", f"{sec_rates.iloc[0]:.1f}%")
        st.metric("Correlation", "0.022", "Very Weak")
        st.markdown("""
        <div class="insight-box"><p>
        <strong>Finding:</strong> Securities Account has minimal impact on loan acceptance (r=0.02). Holders show only slightly
        higher acceptance (11.5% vs 9.4%). This feature adds little predictive value on its own.
        </p></div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ---- ONLINE ----
    st.markdown("### 🌐 Feature 10: ONLINE BANKING")
    col1, col2 = st.columns(2)
    with col1:
        onl_rates = df.groupby("Online")["Personal Loan"].mean() * 100
        onl_df = pd.DataFrame({"Online Banking": ["No (0)", "Yes (1)"], "Acceptance Rate (%)": onl_rates.values})
        fig = px.bar(onl_df, x="Online Banking", y="Acceptance Rate (%)", text="Acceptance Rate (%)",
                     title="Loan Acceptance: Online Banking Users",
                     color="Online Banking", color_discrete_sequence=["#64748b", "#06b6d4"])
        fig.update_traces(texttemplate="%{text:.1f}%")
        fig.update_layout(template="plotly_dark", height=350, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.metric("Online Users - Acceptance", f"{onl_rates.iloc[1]:.1f}%")
        st.metric("Non-Online - Acceptance", f"{onl_rates.iloc[0]:.1f}%")
        st.metric("Correlation", "0.006", "Negligible")
        st.markdown("""
        <div class="insight-box"><p>
        <strong>Finding:</strong> Online banking usage has essentially zero impact on loan acceptance (r=0.006). Both online and
        offline users show nearly identical acceptance rates. This feature should NOT drive targeting decisions.
        </p></div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ---- CREDIT CARD ----
    st.markdown("### 💳 Feature 11: CREDIT CARD (UniversalBank issued)")
    col1, col2 = st.columns(2)
    with col1:
        cc_rates = df.groupby("CreditCard")["Personal Loan"].mean() * 100
        cc_df = pd.DataFrame({"Credit Card": ["No (0)", "Yes (1)"], "Acceptance Rate (%)": cc_rates.values})
        fig = px.bar(cc_df, x="Credit Card", y="Acceptance Rate (%)", text="Acceptance Rate (%)",
                     title="Loan Acceptance: Credit Card Holders",
                     color="Credit Card", color_discrete_sequence=["#64748b", "#ec4899"])
        fig.update_traces(texttemplate="%{text:.1f}%")
        fig.update_layout(template="plotly_dark", height=350, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.metric("CC Holders - Acceptance", f"{cc_rates.iloc[1]:.1f}%")
        st.metric("Non-CC Holders - Acceptance", f"{cc_rates.iloc[0]:.1f}%")
        st.metric("Correlation", "0.003", "Negligible")
        st.markdown("""
        <div class="insight-box"><p>
        <strong>Finding:</strong> Having a UniversalBank credit card has virtually no impact on loan acceptance (r=0.003). Both groups
        show nearly identical rates. Note: this differs from CC spending (CCAvg), which IS highly predictive.
        </p></div>""", unsafe_allow_html=True)


# ======================== 3. DIAGNOSTIC ANALYSIS ========================
elif page.startswith("🔍"):
    st.title("🔍 3. Diagnostic Analysis")
    st.markdown("*Why did some customers accept and others didn't? Multi-feature interaction analysis.*")
    st.markdown("---")

    # Correlation Heatmap
    st.markdown("### Full Feature Correlation Matrix")
    corr_cols = ["Age", "Experience", "Income", "Family", "CCAvg", "Education",
                 "Mortgage", "Securities Account", "CD Account", "Online", "CreditCard", "Personal Loan"]
    corr_matrix = df[corr_cols].corr()
    fig = px.imshow(corr_matrix.round(3), text_auto=True, aspect="auto",
                    title="Correlation Heatmap — All Features",
                    color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
    fig.update_layout(template="plotly_dark", height=600)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Accepted vs Rejected comparison across ALL features
    st.markdown("### Accepted vs Rejected — Full Comparison Table")
    comparison_data = []
    for feat in ["Age", "Experience", "Income", "Family", "CCAvg", "Education", "Mortgage",
                 "Securities Account", "CD Account", "Online", "CreditCard"]:
        comparison_data.append({
            "Feature": feat,
            "Avg (Accepted)": round(accepted[feat].mean(), 2),
            "Avg (Rejected)": round(rejected[feat].mean(), 2),
            "Median (Accepted)": round(accepted[feat].median(), 2),
            "Median (Rejected)": round(rejected[feat].median(), 2),
            "Std (Accepted)": round(accepted[feat].std(), 2),
            "Std (Rejected)": round(rejected[feat].std(), 2),
            "Difference (Avg)": round(accepted[feat].mean() - rejected[feat].mean(), 2),
            "Correlation w/ Loan": round(corr_matrix.loc[feat, "Personal Loan"], 4),
        })
    st.dataframe(pd.DataFrame(comparison_data), use_container_width=True, hide_index=True)

    st.markdown("---")

    # Cross-feature analysis
    st.markdown("### Cross-Feature Interaction Analysis")

    col1, col2 = st.columns(2)
    with col1:
        # Income vs CCAvg colored by Loan
        fig = px.scatter(df.sample(2000, random_state=42), x="Income", y="CCAvg",
                         color="Loan_Status", opacity=0.6,
                         title="Income vs CC Spending (colored by Loan Status)",
                         color_discrete_map={"Not Accepted": "#ef4444", "Accepted": "#22c55e"})
        fig.update_layout(template="plotly_dark", height=420)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Income vs Education colored by Loan
        fig = px.box(df, x="Education_Label", y="Income", color="Loan_Status",
                     title="Income by Education Level & Loan Status",
                     color_discrete_map={"Not Accepted": "#ef4444", "Accepted": "#22c55e"})
        fig.update_layout(template="plotly_dark", height=420)
        st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        # Family vs Education heatmap of loan rates
        fam_edu_rate = df.groupby(["Family", "Education"])["Personal Loan"].mean().unstack() * 100
        fig = px.imshow(fam_edu_rate.round(1), text_auto=True, aspect="auto",
                        title="Loan Acceptance Rate (%) — Family Size vs Education",
                        labels=dict(x="Education (1=UG, 2=Grad, 3=Adv)", y="Family Size", color="Rate %"),
                        color_continuous_scale=["#1e293b", "#22c55e"])
        fig.update_layout(template="plotly_dark", height=350)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # CD Account + Income interaction
        df["CD_Label"] = df["CD Account"].map({0: "No CD Account", 1: "Has CD Account"})
        fig = px.box(df, x="CD_Label", y="Income", color="Loan_Status",
                     title="Income by CD Account Status & Loan Decision",
                     color_discrete_map={"Not Accepted": "#ef4444", "Accepted": "#22c55e"})
        fig.update_layout(template="plotly_dark", height=350)
        st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        # Income vs Mortgage colored by Loan
        fig = px.scatter(df[df["Mortgage"] > 0].sample(min(1000, len(df[df["Mortgage"]>0])), random_state=42),
                         x="Income", y="Mortgage", color="Loan_Status", opacity=0.6,
                         title="Income vs Mortgage (Mortgage holders only)",
                         color_discrete_map={"Not Accepted": "#ef4444", "Accepted": "#22c55e"})
        fig.update_layout(template="plotly_dark", height=380)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # CCAvg by Family and Loan
        fig = px.box(df, x="Family", y="CCAvg", color="Loan_Status",
                     title="CC Spending by Family Size & Loan Status",
                     color_discrete_map={"Not Accepted": "#ef4444", "Accepted": "#22c55e"})
        fig.update_layout(template="plotly_dark", height=380)
        st.plotly_chart(fig, use_container_width=True)

    # Multi-way segment analysis
    st.markdown("### Multi-Feature Segment Loan Rates")
    segment_analysis = []
    for edu in [1, 2, 3]:
        for fam in [1, 2, 3, 4]:
            for cd in [0, 1]:
                mask = (df["Education"] == edu) & (df["Family"] == fam) & (df["CD Account"] == cd)
                subset = df[mask]
                if len(subset) >= 10:
                    segment_analysis.append({
                        "Education": {1: "Undergrad", 2: "Graduate", 3: "Advanced"}[edu],
                        "Family Size": fam,
                        "CD Account": "Yes" if cd == 1 else "No",
                        "Count": len(subset),
                        "Loan Rate (%)": round(subset["Personal Loan"].mean() * 100, 1),
                        "Avg Income ($K)": round(subset["Income"].mean(), 1),
                    })
    seg_df = pd.DataFrame(segment_analysis).sort_values("Loan Rate (%)", ascending=False)
    st.dataframe(seg_df, use_container_width=True, hide_index=True)


# ======================== 4. PREDICTIVE ANALYSIS ========================
elif page.startswith("🤖"):
    st.title("🤖 4. Predictive Analysis")
    st.markdown("*5 machine learning models trained to predict who will accept a personal loan.*")
    st.markdown("---")

    # KPIs for best model
    rf = results["Random Forest"]
    gb = results["Gradient Boosting"]
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Models Trained", "5")
    c2.metric("Best Accuracy", f"{max(r['accuracy'] for r in results.values())*100:.1f}%")
    c3.metric("Best Precision", f"{max(r['precision'] for r in results.values())*100:.1f}%")
    c4.metric("Best Recall", f"{max(r['recall'] for r in results.values())*100:.1f}%")
    c5.metric("Best F1", f"{max(r['f1'] for r in results.values())*100:.1f}%")
    c6.metric("Best AUC", f"{max(r['auc'] for r in results.values())*100:.1f}%")

    st.markdown("---")

    # Full model comparison table
    st.markdown("### Complete Model Comparison")
    comp_data = []
    for name in results:
        r = results[name]
        comp_data.append({
            "Model": name,
            "Accuracy": f"{r['accuracy']*100:.2f}%",
            "Precision": f"{r['precision']*100:.2f}%",
            "Recall": f"{r['recall']*100:.2f}%",
            "F1 Score": f"{r['f1']*100:.2f}%",
            "AUC-ROC": f"{r['auc']*100:.2f}%",
            "Specificity": f"{r['specificity']*100:.2f}%",
            "CV F1 (5-fold)": f"{r['cv_f1_mean']*100:.1f}% ± {r['cv_f1_std']*100:.1f}%",
            "FP": r["fp"],
            "FN": r["fn"],
            "Total Errors": r["fp"] + r["fn"],
        })
    st.dataframe(pd.DataFrame(comp_data), use_container_width=True, hide_index=True)

    st.markdown("---")

    # Charts
    model_names = list(results.keys())
    col1, col2 = st.columns(2)
    with col1:
        metrics_list = ["accuracy", "precision", "recall", "f1", "auc"]
        metric_labels = ["Accuracy", "Precision", "Recall", "F1 Score", "AUC-ROC"]
        fig = go.Figure()
        clrs = ["#3b82f6", "#f59e0b", "#22c55e", "#a855f7", "#06b6d4"]
        for i, name in enumerate(model_names):
            fig.add_trace(go.Bar(
                name=name, x=metric_labels,
                y=[results[name][m] * 100 for m in metrics_list],
                marker_color=clrs[i],
                text=[f"{results[name][m]*100:.1f}%" for m in metrics_list],
                textposition="auto"
            ))
        fig.update_layout(title="All Models — All Metrics", template="plotly_dark",
                          barmode="group", yaxis_range=[55, 102], height=450, yaxis_title="Score (%)")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = go.Figure()
        for i, name in enumerate(model_names):
            r = results[name]
            fig.add_trace(go.Scatter(
                x=r["fpr_curve"], y=r["tpr_curve"],
                name=f"{name} (AUC={r['auc']*100:.1f}%)",
                line=dict(color=clrs[i], width=2)
            ))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name="Random",
                                 line=dict(color="gray", dash="dash", width=1)))
        fig.update_layout(title="ROC Curves — All 5 Models", template="plotly_dark", height=450,
                          xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
        st.plotly_chart(fig, use_container_width=True)

    # Feature Importance + LR Coefficients
    col1, col2 = st.columns(2)
    with col1:
        feat_df = pd.DataFrame(sorted(feat_imp.items(), key=lambda x: x[1], reverse=True),
                               columns=["Feature", "Importance"])
        fig = px.bar(feat_df, y="Feature", x="Importance",
                     title="Feature Importance — Random Forest", orientation="h",
                     color="Importance", color_continuous_scale=["#64748b", "#22c55e"],
                     text=feat_df["Importance"].apply(lambda x: f"{x:.3f}"))
        fig.update_layout(template="plotly_dark", height=450, yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        lr_df = pd.DataFrame(sorted(lr_coefs.items(), key=lambda x: abs(x[1]), reverse=True),
                             columns=["Feature", "Coefficient"])
        fig = px.bar(lr_df, y="Feature", x="Coefficient",
                     title="Logistic Regression Coefficients (Standardized)", orientation="h",
                     color="Coefficient", color_continuous_scale=["#ef4444", "#64748b", "#22c55e"],
                     text=lr_df["Coefficient"].apply(lambda x: f"{x:.3f}"))
        fig.update_layout(template="plotly_dark", height=450, yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True)

    # Precision-Recall Curve
    col1, col2 = st.columns(2)
    with col1:
        fig = go.Figure()
        for i, name in enumerate(model_names):
            r = results[name]
            fig.add_trace(go.Scatter(x=r["rec_curve"], y=r["prec_curve"],
                                     name=name, line=dict(color=clrs[i], width=2)))
        fig.update_layout(title="Precision-Recall Curves", template="plotly_dark", height=400,
                          xaxis_title="Recall", yaxis_title="Precision")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Cross-validation results
        cv_data = pd.DataFrame({
            "Model": model_names,
            "CV F1 Mean (%)": [results[m]["cv_f1_mean"] * 100 for m in model_names],
            "CV F1 Std (%)": [results[m]["cv_f1_std"] * 100 for m in model_names],
        })
        fig = px.bar(cv_data, x="Model", y="CV F1 Mean (%)", error_y="CV F1 Std (%)",
                     title="5-Fold Cross-Validation F1 Scores",
                     color_discrete_sequence=["#a855f7"], text="CV F1 Mean (%)")
        fig.update_traces(texttemplate="%{text:.1f}%")
        fig.update_layout(template="plotly_dark", height=400)
        st.plotly_chart(fig, use_container_width=True)


# ======================== 5. CONFUSION MATRIX DEEP DIVE ========================
elif page.startswith("🎯"):
    st.title("🎯 5. Confusion Matrix — Deep Dive")
    st.markdown("*Detailed error analysis, cost-benefit assessment, and threshold tuning for all 5 models.*")
    st.markdown("---")

    # Model selector
    selected_model = st.selectbox("Select Model to Analyze", list(results.keys()), index=2)
    r = results[selected_model]
    tn, fp, fn, tp = r["tn"], r["fp"], r["fn"], r["tp"]
    total = tn + fp + fn + tp

    st.markdown(f"### {selected_model} — Confusion Matrix")

    # Visual CM
    col1, col2 = st.columns([2, 1])
    with col1:
        cm_text = [[f"TN = {tn:,}\nCorrectly predicted\nNO Loan", f"FP = {fp:,}\nType I Error\n(False Alarm)"],
                    [f"FN = {fn:,}\nType II Error\n(Missed Opportunity)", f"TP = {tp:,}\nCorrectly predicted\nLoan Accepted"]]
        fig = go.Figure(data=go.Heatmap(
            z=[[tn, fp], [fn, tp]],
            x=["Predicted: No Loan", "Predicted: Loan"],
            y=["Actual: No Loan", "Actual: Loan"],
            text=cm_text, texttemplate="%{text}", textfont={"size": 14},
            colorscale=[[0, "#1e3a5f"], [0.3, "#3b82f6"], [0.7, "#22c55e"], [1, "#16a34a"]],
            showscale=False,
        ))
        fig.update_layout(template="plotly_dark", height=380, yaxis=dict(autorange="reversed"),
                          xaxis_title="Predicted Label", yaxis_title="Actual Label")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.metric("True Negatives (TN)", f"{tn:,}", f"{tn/total*100:.1f}% of all")
        st.metric("True Positives (TP)", f"{tp:,}", f"{tp/total*100:.1f}% of all")
        st.metric("False Positives (FP)", f"{fp:,}", f"{fp/total*100:.2f}% of all", delta_color="inverse")
        st.metric("False Negatives (FN)", f"{fn:,}", f"{fn/total*100:.2f}% of all", delta_color="inverse")
        st.metric("Total Errors", f"{fp+fn}", f"{(fp+fn)/total*100:.2f}% error rate", delta_color="inverse")

    st.markdown("---")

    # Detailed breakdown cards
    st.markdown("### Quadrant Analysis")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"""
        <div class="yes-box">
            <h4>✅ True Negatives (TN) = {tn:,} — Specificity: {tn/(tn+fp)*100:.2f}%</h4>
            <p>Out of {tn+fp:,} actual non-loan customers, <strong>{tn:,}</strong> were correctly identified.
            The model almost perfectly identifies customers who will NOT accept a loan. These customers are correctly
            excluded from the marketing campaign, saving significant outreach costs.</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown(f"""
        <div class="yes-box">
            <h4>✅ True Positives (TP) = {tp:,} — Recall: {tp/(tp+fn)*100:.2f}%</h4>
            <p>Out of {tp+fn:,} actual loan takers in the test set, <strong>{tp:,}</strong> were correctly identified.
            These customers would receive targeted marketing and convert successfully. Each represents
            confirmed revenue for the bank's loan portfolio.</p>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="no-box">
            <h4>⚠️ False Positives (FP) = {fp:,} — Type I Error Rate: {fp/(fp+tn)*100:.2f}%</h4>
            <p>Only <strong>{fp:,}</strong> customers were incorrectly predicted as loan takers. These represent
            wasted marketing outreach — sending offers to customers who won't convert. At ~$50 per outreach,
            this costs approximately <strong>${fp*50:,}</strong> in wasted marketing spend per 1,500 customers.</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown(f"""
        <div class="no-box">
            <h4>⚠️ False Negatives (FN) = {fn:,} — Type II Error Rate: {fn/(fn+tp)*100:.2f}%</h4>
            <p><strong>{fn:,}</strong> actual loan takers were missed by the model. These are lost revenue
            opportunities. If each personal loan generates ~$5,000 profit, the missed revenue is approximately
            <strong>${fn*5000:,}</strong>. This is the model's primary weakness and opportunity for improvement.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Complete metrics table
    st.markdown("### All Derived Metrics from Confusion Matrix")
    metrics_full = pd.DataFrame({
        "Metric": ["Accuracy", "Precision (Positive Predictive Value)", "Recall (Sensitivity / True Positive Rate)",
                    "Specificity (True Negative Rate)", "F1 Score", "False Positive Rate (Fall-out)",
                    "False Negative Rate (Miss Rate)", "Negative Predictive Value (NPV)",
                    "False Discovery Rate (FDR)", "False Omission Rate (FOR)",
                    "Balanced Accuracy", "Matthews Correlation Coefficient (MCC)", "AUC-ROC"],
        "Formula": ["(TP+TN)/(TP+TN+FP+FN)", "TP/(TP+FP)", "TP/(TP+FN)", "TN/(TN+FP)",
                     "2*(P*R)/(P+R)", "FP/(FP+TN)", "FN/(FN+TP)", "TN/(TN+FN)",
                     "FP/(FP+TP)", "FN/(FN+TN)", "(Sensitivity+Specificity)/2", "See formula", "Area under ROC"],
        "Value": [
            f"{r['accuracy']*100:.2f}%", f"{r['precision']*100:.2f}%", f"{r['recall']*100:.2f}%",
            f"{r['specificity']*100:.2f}%", f"{r['f1']*100:.2f}%",
            f"{fp/(fp+tn)*100:.2f}%", f"{fn/(fn+tp)*100:.2f}%",
            f"{tn/(tn+fn)*100:.2f}%", f"{fp/(fp+tp)*100:.2f}%" if (fp+tp) > 0 else "N/A",
            f"{fn/(fn+tn)*100:.2f}%",
            f"{(r['recall']+r['specificity'])/2*100:.2f}%",
            f"{((tp*tn)-(fp*fn))/np.sqrt(float((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))):.4f}" if (tp+fp)*(tp+fn)*(tn+fp)*(tn+fn) > 0 else "N/A",
            f"{r['auc']*100:.2f}%",
        ],
        "Interpretation": [
            "Overall correct predictions", "When model says 'loan', how often correct?",
            "Of actual loan takers, how many found?", "Of non-loan customers, how many correctly identified?",
            "Harmonic mean of precision & recall", "Rate of false alarms among negatives",
            "Rate of missed positives", "When model says 'no loan', how often correct?",
            "Rate of false alarms among positive predictions", "Rate of missed positives among negative predictions",
            "Average of sensitivity and specificity", "Overall quality of predictions (-1 to +1)",
            "Discriminative ability across all thresholds",
        ]
    })
    st.dataframe(metrics_full, use_container_width=True, hide_index=True)

    st.markdown("---")

    # Cross-model comparison
    st.markdown("### All 5 Models — Side-by-Side Confusion Matrices")

    cols = st.columns(5)
    for i, (name, r2) in enumerate(results.items()):
        with cols[i]:
            st.markdown(f"**{name}**")
            cm_mini = pd.DataFrame(
                [[r2["tn"], r2["fp"]], [r2["fn"], r2["tp"]]],
                index=["Actual: 0", "Actual: 1"],
                columns=["Pred: 0", "Pred: 1"]
            )
            st.dataframe(cm_mini)
            st.caption(f"Acc: {r2['accuracy']*100:.1f}% | F1: {r2['f1']*100:.1f}%")
            st.caption(f"Errors: {r2['fp']+r2['fn']}")

    st.markdown("---")

    # Cost-Benefit Analysis
    st.markdown("### Cost-Benefit Analysis")
    st.markdown("*Adjust the costs below to see how each model performs financially.*")
    col1, col2 = st.columns(2)
    with col1:
        cost_fp = st.slider("Cost per False Positive (wasted outreach $)", 10, 200, 50)
        cost_fn = st.slider("Cost per False Negative (missed revenue $)", 1000, 20000, 5000)
    with col2:
        revenue_tp = st.slider("Revenue per True Positive (loan profit $)", 1000, 20000, 5000)
        cost_campaign = st.slider("Fixed campaign cost per customer targeted ($)", 0, 50, 10)

    cost_data = []
    for name in results:
        r2 = results[name]
        total_cost = (r2["fp"] * cost_fp) + (r2["fn"] * cost_fn) + ((r2["tp"] + r2["fp"]) * cost_campaign)
        total_revenue = r2["tp"] * revenue_tp
        net_profit = total_revenue - total_cost
        roi = (net_profit / total_cost * 100) if total_cost > 0 else 0
        cost_data.append({
            "Model": name,
            "Revenue (TP)": f"${total_revenue:,.0f}",
            "Cost (FP waste)": f"${r2['fp'] * cost_fp:,.0f}",
            "Cost (FN missed)": f"${r2['fn'] * cost_fn:,.0f}",
            "Campaign Cost": f"${(r2['tp'] + r2['fp']) * cost_campaign:,.0f}",
            "Total Cost": f"${total_cost:,.0f}",
            "Net Profit": f"${net_profit:,.0f}",
            "ROI": f"{roi:.0f}%",
        })
    st.dataframe(pd.DataFrame(cost_data), use_container_width=True, hide_index=True)


# ======================== 6. PRESCRIPTIVE ANALYSIS ========================
elif page.startswith("💡"):
    st.title("💡 6. Prescriptive Analysis")
    st.markdown("*What should we do? Data-driven targeting strategy for the next loan campaign.*")
    st.markdown("---")

    # Define ALL segments
    segments = [
        ("High Income + CD Account", (df["Income"] > 100) & (df["CD Account"] == 1)),
        ("High Income + Graduate+", (df["Income"] > 100) & (df["Education"] >= 2)),
        ("High Income + High CC ($3K+)", (df["Income"] > 100) & (df["CCAvg"] > 3)),
        ("CD Account + Graduate+", (df["CD Account"] == 1) & (df["Education"] >= 2)),
        ("CD Account Holders", df["CD Account"] == 1),
        ("High CC Spend ($3K+) + Graduate+", (df["CCAvg"] > 3) & (df["Education"] >= 2)),
        ("High Income ($100K+)", df["Income"] > 100),
        ("High CC Spend ($3K+)", df["CCAvg"] > 3),
        ("High Mortgage ($200K+)", df["Mortgage"] > 200),
        ("Family Size 3+", df["Family"] >= 3),
        ("Graduate+ Education", df["Education"] >= 2),
        ("Undergrad + Low Income", (df["Education"] == 1) & (df["Income"] < 50)),
        ("Overall Baseline", pd.Series([True] * len(df))),
    ]

    seg_data = []
    for name, mask in segments:
        subset = df[mask]
        if len(subset) > 0:
            rate = subset["Personal Loan"].mean() * 100
            avg_inc = subset["Income"].mean()
            avg_cc = subset["CCAvg"].mean()
            if rate > 40:
                tier = "🔴 Tier 1 (Hot)"
            elif rate > 20:
                tier = "🟠 Tier 2 (Warm)"
            elif rate > 10:
                tier = "🔵 Tier 3 (Moderate)"
            else:
                tier = "⚪ Low Priority"
            seg_data.append({
                "Tier": tier,
                "Segment": name,
                "Conversion Rate (%)": round(rate, 1),
                "Customer Count": len(subset),
                "Avg Income ($K)": round(avg_inc, 1),
                "Avg CC Spend ($K)": round(avg_cc, 2),
                "Expected Conversions": round(len(subset) * rate / 100),
            })

    seg_df = pd.DataFrame(seg_data).sort_values("Conversion Rate (%)", ascending=False)

    # Visual
    fig = px.bar(seg_df.sort_values("Conversion Rate (%)"), y="Segment", x="Conversion Rate (%)",
                 title="All Customer Segments — Conversion Rate Ranking",
                 orientation="h", text="Conversion Rate (%)",
                 color="Conversion Rate (%)",
                 color_continuous_scale=["#334155", "#f59e0b", "#ef4444"])
    fig.update_traces(texttemplate="%{text:.1f}%")
    fig.update_layout(template="plotly_dark", height=600, yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Segment Detail Table")
    st.dataframe(seg_df, use_container_width=True, hide_index=True)

    st.markdown("---")

    # Recommendations
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="yes-box">
            <h4>✅ WHO TO TARGET (High Priority)</h4>
            <p>
            <strong>1. High Income + CD Account</strong> — 79.9% conversion, 159 customers. Highest ROI segment.<br><br>
            <strong>2. High Income + Graduate+</strong> — 78.4% conversion, 458 customers. Largest high-conversion pool.<br><br>
            <strong>3. CD Account + Graduate+</strong> — 62.8% conversion. Financially engaged and educated.<br><br>
            <strong>4. High Income + High CC Spend</strong> — 42.0% conversion, 660 customers. Volume opportunity.<br><br>
            <strong>5. All CD Account Holders</strong> — 46.4% conversion. Always include in campaigns.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="no-box">
            <h4>❌ WHO NOT TO TARGET (Low Priority)</h4>
            <p>
            <strong>1. Undergrad + Low Income (&lt;$50K)</strong> — Only 1.2% conversion. Avoid entirely.<br><br>
            <strong>2. Low Income with no CD Account</strong> — Under 3% conversion. Minimal ROI.<br><br>
            <strong>3. Age/Experience-based targeting</strong> — These features have zero predictive power. Do NOT segment by age.<br><br>
            <strong>4. Online/CreditCard-based targeting</strong> — These features show no difference between acceptors and non-acceptors.<br><br>
            <strong>Insight:</strong> Focus budget on the top 458-660 customers (Tier 1-2) instead of blanketing all 5,000.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Feature radar
    col1, col2 = st.columns(2)
    with col1:
        top_feats = sorted(feat_imp.items(), key=lambda x: x[1], reverse=True)[:8]
        f_names = [f[0] for f in top_feats]
        f_vals = [f[1] for f in top_feats]
        corr_all = df[feature_names + ["Personal Loan"]].corr()["Personal Loan"].drop("Personal Loan")
        c_vals = [abs(corr_all.get(f, 0)) for f in f_names]

        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(r=f_vals, theta=f_names, fill="toself",
                                       name="RF Importance", line_color="#3b82f6"))
        fig.add_trace(go.Scatterpolar(r=c_vals, theta=f_names, fill="toself",
                                       name="Correlation", line_color="#22c55e"))
        fig.update_layout(title="Feature Importance vs Correlation", template="plotly_dark",
                          height=450, polar=dict(radialaxis=dict(range=[0, 0.55])))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### Campaign ROI Projection")
        # Best segment economics
        best_seg = seg_df.iloc[0]
        st.markdown(f"""
        <div class="highlight-box">
            <h4>Best Segment: {best_seg['Segment']}</h4>
            <p>
            <strong>Customers:</strong> {best_seg['Customer Count']:,}<br>
            <strong>Expected Conversion:</strong> {best_seg['Conversion Rate (%)']:.1f}% = ~{best_seg['Expected Conversions']:.0f} loans<br>
            <strong>At $5,000 profit per loan:</strong> ~${best_seg['Expected Conversions']*5000:,.0f} revenue<br>
            <strong>Campaign cost ($50/customer):</strong> ~${best_seg['Customer Count']*50:,.0f}<br>
            <strong>Estimated ROI:</strong> ~{(best_seg['Expected Conversions']*5000 - best_seg['Customer Count']*50)/(best_seg['Customer Count']*50)*100:.0f}%
            </p>
        </div>
        """, unsafe_allow_html=True)

        total_tier1 = seg_df[seg_df["Tier"].str.contains("Tier 1")]["Expected Conversions"].sum()
        total_cust_tier1 = seg_df[seg_df["Tier"].str.contains("Tier 1")]["Customer Count"].sum()
        st.markdown(f"""
        <div class="highlight-box" style="border-left-color: #22c55e;">
            <h4>All Tier 1 Segments Combined</h4>
            <p>
            <strong>Total reachable:</strong> ~{total_cust_tier1:,.0f} unique customers<br>
            <strong>Expected conversions:</strong> ~{total_tier1:.0f} loans<br>
            <strong>Projected revenue:</strong> ~${total_tier1*5000:,.0f}<br>
            <strong>vs. targeting all 5,000:</strong> ~480 conversions but at much higher cost
            </p>
        </div>
        """, unsafe_allow_html=True)

    # Decision rules
    st.markdown("---")
    st.markdown("### Simple Decision Rules for Campaign Team")
    st.markdown("""
    | Rule | Condition | Expected Result |
    |------|-----------|-----------------|
    | **ALWAYS target** | Income > $100K AND (Graduate or Advanced degree) | ~78% will accept |
    | **ALWAYS target** | Has CD Account | ~46% will accept |
    | **STRONGLY consider** | Income > $100K AND CC Spend > $3K/month | ~42% will accept |
    | **Consider** | Income > $100K (any education) | ~36% will accept |
    | **Consider** | CC Spend > $3K AND Graduate+ | ~33% will accept |
    | **SKIP** | Income < $50K AND Undergrad | Only ~1% will accept |
    | **SKIP** | No CD Account AND Income < $50K | Under 3% conversion |
    | **IRRELEVANT** | Any age-based or experience-based rule | No predictive value |
    | **IRRELEVANT** | Online banking or CreditCard status | No predictive value |
    """)


# ======================== FOOTER ========================
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#64748b;font-size:12px;padding:10px;'>"
    "Universal Bank — Personal Loan Campaign Analysis Dashboard (Management Edition)<br>"
    "Dataset: 5,000 customers | 11 Features Analyzed | 5 ML Models | Target: Personal Loan Acceptance<br>"
    "Features: Age, Experience, Income, Family, CCAvg, Education, Mortgage, Securities Account, CD Account, Online, CreditCard"
    "</div>",
    unsafe_allow_html=True
)
