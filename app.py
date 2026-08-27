import streamlit as st
import pandas as pd
import altair as alt
import streamlit.components.v1 as components
from catboost import CatBoostClassifier 
import json


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Sleep Disorder Risk Prediction",
    page_icon="🌙",
    layout="wide"
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_catboost_model():

    model = CatBoostClassifier()

    model.load_model(
        "catboost_web13.cbm"
    )

    return model


model = load_catboost_model()


# ============================================================
# SIDEBAR
# ============================================================



st.sidebar.markdown(
    """
    <div style="
        font-size: 20px;
        font-weight: 700;
        white-space: nowrap;
        margin-bottom: 14px;
    ">
        🌙 Sleep Disorder Project
    </div>
    """,
    unsafe_allow_html=True
)

page = st.sidebar.radio(
    "功能選單",
    [
        "1. Model 說明",
        "2. 輸入與預測"
    ]
)
   
# ============================================================
# PAGE 1
# MODEL 說明
# ============================================================

if page == "1. Model 說明":

    col1, col2 = st.columns([0.55, 8])

    with col1:
        st.image("ai_ml_icon.png", width=100)

    with col2:
        st.markdown("## Model 說明")
        st.caption(
            "本專題使用多種 Tree-based Machine Learning Models "
            "進行睡眠障礙風險分類。"
        )

    # ========================================================
    # 0. DATASET / PROBLEM DEFINITION
    # ========================================================

    st.subheader("0. Dataset / Problem Definition")

    col_a, col_b, col_c, col_d = st.columns(4)

    with col_a:
        st.metric("Samples", "100,000")

    with col_b:
        st.metric("Features", "30")

    with col_c:
        st.metric("Task", "4-class")

    with col_d:
        st.metric("Classes", "Healthy → Severe")

    st.caption("Risk Levels：Healthy → Mild → Moderate → Severe")

    class_distribution = pd.DataFrame({
        "Risk Level": ["Healthy", "Mild", "Moderate", "Severe"],
        "Percentage": [54.16, 33.48, 8.30, 4.07]
    })

    risk_color_scale = alt.Scale(
        domain=["Healthy", "Mild", "Moderate", "Severe"],
        range=["#2e7d32", "#1976d2", "#f57c00", "#c62828"]
    )

    class_dist_chart = alt.Chart(class_distribution).mark_bar().encode(
        x=alt.X("Percentage:Q", title="佔比（%）"),
        y=alt.Y(
            "Risk Level:N",
            sort=["Healthy", "Mild", "Moderate", "Severe"],
            title=None
        ),
        color=alt.Color("Risk Level:N", scale=risk_color_scale, legend=None),
        tooltip=["Risk Level", "Percentage"]
    ).properties(height=180)

    st.altair_chart(class_dist_chart, width="stretch")

    st.caption(
        "Class Distribution 為完整 100,000 筆資料集的類別比例。"
        "資料集屬於 Imbalanced Data，"
        "Moderate 與 Severe 為少數類別，也是模型辨識的重點難點。"
    )

    st.divider()

    # ========================================================
    # 0.1 DATA PREPARATION
    # ========================================================

    st.subheader("0.1 Data Preparation")

    st.caption(
        "在模型訓練前，先完成資料品質檢查、Outlier 偵測、特徵與目標欄位定義，"
        "並以 Stratified Train/Test Split 保留各風險類別比例。"
    )

    prep_col1, prep_col2, prep_col3, prep_col4 = st.columns(4)

    with prep_col1:
        st.markdown(
            """
            **① Data Quality Check**

            - 檢查欄位型態
            - 檢查 Missing Values
            - 確認類別與數值欄位
            """
        )

    with prep_col2:
        st.markdown(
            """
            **② Outlier Detection**

            - Numerical Features
            - 使用 IQR Rule
            - 異常值記錄供後續檢查
            """
        )

    with prep_col3:
        st.markdown(
            """
            **③ Feature / Target Setup**

            - 移除 `person_id`
            - X = Predictors
            - y = `sleep_disorder_risk`
            """
        )

    with prep_col4:
        st.markdown(
            """
            **④ Train / Test Split**

            - Train 80%
            - Test 20%
            - `stratify=y`
            - `random_state=42`
            """
        )

    st.markdown("#### Outlier Detection — IQR Rule")

    st.caption(
        "使用 IQR（Interquartile Range，四分位距）找出數值型欄位中可能的異常值。"
    )

    components.html(
        """
<!DOCTYPE html>
<html>
<head>
<style>
    body {
        margin: 0;
        padding: 0;
        font-family: -apple-system, "Segoe UI", "Microsoft JhengHei", sans-serif;
        background: transparent;
        color: #1f2937;
    }

    .iqr-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 14px;
        margin-top: 8px;
        margin-bottom: 14px;
    }

    .iqr-card {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 14px;
        padding: 16px 18px;
        box-sizing: border-box;
        min-height: 132px;
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.06);
    }

    .step {
        font-size: 11px;
        font-weight: 800;
        color: #64748b;
        letter-spacing: 0.6px;
        margin-bottom: 5px;
    }

    .title {
        font-size: 15px;
        font-weight: 750;
        color: #0f172a;
        margin-bottom: 10px;
    }

    .formula {
        display: inline-block;
        background: #eff6ff;
        border: 1px solid #bfdbfe;
        color: #1d4ed8;
        border-radius: 9px;
        padding: 8px 12px;
        font-size: 15px;
        font-weight: 800;
        white-space: nowrap;
    }

    .note {
        margin-top: 9px;
        font-size: 11.5px;
        line-height: 1.5;
        color: #64748b;
    }

    .judge {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-left: 5px solid #2563eb;
        border-radius: 12px;
        padding: 13px 17px;
        font-size: 12.5px;
        line-height: 1.8;
        box-sizing: border-box;
    }

    .bad {
        color: #dc2626;
        font-weight: 800;
    }

    .good {
        color: #15803d;
        font-weight: 800;
    }
</style>
</head>

<body>

<div class="iqr-grid">

    <div class="iqr-card">
        <div class="step">STEP 1</div>
        <div class="title">計算四分位距</div>
        <div class="formula">IQR = Q3 − Q1</div>
        <div class="note">
            Q1 = 第 25 百分位數<br>
            Q3 = 第 75 百分位數
        </div>
    </div>

    <div class="iqr-card">
        <div class="step">STEP 2</div>
        <div class="title">計算下界</div>
        <div class="formula">Q1 − 1.5 × IQR</div>
        <div class="note">
            小於下界的數值<br>
            視為可能的異常值
        </div>
    </div>

    <div class="iqr-card">
        <div class="step">STEP 3</div>
        <div class="title">計算上界</div>
        <div class="formula">Q3 + 1.5 × IQR</div>
        <div class="note">
            大於上界的數值<br>
            視為可能的異常值
        </div>
    </div>

</div>

<div class="judge">
    <b>判斷方式</b><br>
    數值 &lt; 下界 → <span class="bad">Possible Outlier</span><br>
    數值 &gt; 上界 → <span class="bad">Possible Outlier</span><br>
    下界 ≤ 數值 ≤ 上界 → <span class="good">Normal Range</span>
</div>

</body>
</html>
        """,
        height=285,
        scrolling=False
    )

    st.markdown(
        """
        <div style="
            background:#eef6ff;
            border-left:6px solid #2563eb;
            border-radius:12px;
            padding:16px 20px;
            margin-top:12px;
            margin-bottom:18px;
            font-size:16px;
            line-height:1.7;
            color:#1e293b;
        ">
            <b style="font-size:18px;">✅ Outlier Handling Conclusion</b><br>
            本專題採用 <b>Detect → Log → Inspect</b> 的方式檢查異常值。
            經檢視後，所有異常值皆屬合理資料範圍，
            因此 <b style="color:#1d4ed8;">所有 Outliers 最終皆予以保留，不進行刪除。</b>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("#### Data Preparation Flow")

    components.html(
        """
<!DOCTYPE html>
<html>
<head>
<style>
    body {
        margin: 0;
        padding: 4px;
        font-family: -apple-system, "Segoe UI", "Microsoft JhengHei", sans-serif;
        background: transparent;
    }

    .flow {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 12px;
        flex-wrap: nowrap;
        padding: 14px 6px;
    }

    .box {
        min-width: 150px;
        padding: 14px 12px;
        border-radius: 12px;
        text-align: center;
        font-weight: 700;
        border: 1px solid #dbe3ef;
        background: #f8fafc;
        color: #334155;
        box-shadow: 0 2px 7px rgba(15, 23, 42, 0.05);
    }

    .sub {
        display: block;
        margin-top: 5px;
        font-size: 11px;
        font-weight: 500;
        color: #64748b;
    }

    .arrow {
        font-size: 24px;
        color: #64748b;
        font-weight: 700;
    }

    .final {
        background: #eef2ff;
        border-color: #c7d2fe;
        color: #4338ca;
    }
</style>
</head>

<body>

<div class="flow">
    <div class="box">
        Raw Data
        <span class="sub">100,000 samples</span>
    </div>

    <div class="arrow">→</div>

    <div class="box">
        Quality Check
        <span class="sub">types / missing values</span>
    </div>

    <div class="arrow">→</div>

    <div class="box">
        IQR Outlier Check
        <span class="sub">detect / log / inspect</span>
    </div>

    <div class="arrow">→</div>

    <div class="box">
        Feature Setup
        <span class="sub">remove ID / define X & y</span>
    </div>

    <div class="arrow">→</div>

    <div class="box final">
        Train / Test Split
        <span class="sub">80 / 20 + stratify</span>
    </div>
</div>

</body>
</html>
        """,
        height=120,
        scrolling=False
    )

    st.caption(
        "註：CatBoost 最終模型保留原始類別欄位，使用原生 categorical feature handling；"
        "Tree-based benchmark models 則依模型需求進行相對應的 encoding。"
    )

    st.divider()


    # ========================================================
    # 1-1 MODEL ARCHITECTURE
    # ========================================================

    st.header("1. Tree-based Models")

    st.write(
        """
        本專題使用的 8 個模型都是從最基礎的 **CART 決策樹** 演化而來，
        主要分成兩大家族：

        - 🔵 **Bagging 家族**：平行訓練多棵樹並投票，主要降低 **Variance**
        - 🟠 **Boosting 家族**：序列訓練，後面的樹修正前面的錯誤，主要降低 **Bias**
        """
    )

    components.html(
        """
<!DOCTYPE html>
<html>
<head>
<style>
    body { margin:0; padding:0; font-family: -apple-system, "Segoe UI", "Microsoft JhengHei", sans-serif; }
</style>
</head>
<body>

<svg viewBox="0 0 1180 470" width="100%" xmlns="http://www.w3.org/2000/svg">

    <defs>
        <marker id="arrow-blue" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto" markerUnits="strokeWidth">
            <path d="M0,0 L0,6 L9,3 z" fill="#2563eb" />
        </marker>
        <marker id="arrow-orange" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto" markerUnits="strokeWidth">
            <path d="M0,0 L0,6 L9,3 z" fill="#ea580c" />
        </marker>
        <filter id="shadow" x="-20%" y="-20%" width="140%" height="140%">
            <feDropShadow dx="0" dy="2" stdDeviation="3" flood-opacity="0.18"/>
        </filter>
    </defs>

    <!-- ============ CONNECTING LINES ============ -->
    <line x1="600" y1="90"  x2="275" y2="140" stroke="#2563eb" stroke-width="2.5" marker-end="url(#arrow-blue)" opacity="0.7"/>
    <line x1="600" y1="90"  x2="875" y2="140" stroke="#ea580c" stroke-width="2.5" marker-end="url(#arrow-orange)" opacity="0.7"/>

    <line x1="275" y1="210" x2="110" y2="285" stroke="#2563eb" stroke-width="2.5" marker-end="url(#arrow-blue)" opacity="0.6"/>
    <line x1="275" y1="210" x2="350" y2="285" stroke="#2563eb" stroke-width="2.5" marker-end="url(#arrow-blue)" opacity="0.6"/>

    <line x1="875" y1="210" x2="650" y2="285" stroke="#ea580c" stroke-width="2.5" marker-end="url(#arrow-orange)" opacity="0.6"/>
    <line x1="875" y1="210" x2="850" y2="285" stroke="#ea580c" stroke-width="2.5" marker-end="url(#arrow-orange)" opacity="0.6"/>
    <line x1="875" y1="210" x2="1060" y2="285" stroke="#ea580c" stroke-width="2.5" marker-end="url(#arrow-orange)" opacity="0.6"/>

    <!-- ============ ROOT: CART ============ -->
    <g filter="url(#shadow)">
        <rect x="510" y="20" width="180" height="70" rx="12" fill="#374151"/>
    </g>
    <text x="600" y="50" text-anchor="middle" fill="#ffffff" font-size="17" font-weight="700">CART</text>
    <text x="600" y="72" text-anchor="middle" fill="#d1d5db" font-size="12">單一決策樹</text>

    <!-- ============ ROW 1: Bagging / Gradient Boosting ============ -->
    <g filter="url(#shadow)">
        <rect x="180" y="140" width="190" height="70" rx="12" fill="#2563eb"/>
    </g>
    <text x="275" y="170" text-anchor="middle" fill="#ffffff" font-size="16" font-weight="700">Bagging</text>
    <text x="275" y="192" text-anchor="middle" fill="#dbeafe" font-size="11">Bootstrap + Voting</text>

    <g filter="url(#shadow)">
        <rect x="765" y="140" width="220" height="70" rx="12" fill="#ea580c"/>
    </g>
    <text x="875" y="170" text-anchor="middle" fill="#ffffff" font-size="16" font-weight="700">Gradient Boosting</text>
    <text x="875" y="192" text-anchor="middle" fill="#ffedd5" font-size="11">序列修正殘差</text>

    <!-- ============ ROW 2: Bagging 家族 children ============ -->
    <g filter="url(#shadow)">
        <rect x="25" y="285" width="170" height="70" rx="12" fill="#60a5fa"/>
    </g>
    <text x="110" y="315" text-anchor="middle" fill="#1e3a8a" font-size="15" font-weight="700">Random Forest</text>
    <text x="110" y="337" text-anchor="middle" fill="#1e3a8a" font-size="11">+ Feature Sampling</text>

    <g filter="url(#shadow)">
        <rect x="265" y="285" width="170" height="70" rx="12" fill="#60a5fa"/>
    </g>
    <text x="350" y="315" text-anchor="middle" fill="#1e3a8a" font-size="15" font-weight="700">Extra Trees</text>
    <text x="350" y="337" text-anchor="middle" fill="#1e3a8a" font-size="11">+ 隨機切分點</text>

    <!-- ============ ROW 2: Boosting 家族 children ============ -->
    <g filter="url(#shadow)">
        <rect x="575" y="285" width="150" height="70" rx="12" fill="#fb923c"/>
    </g>
    <text x="650" y="315" text-anchor="middle" fill="#7c2d12" font-size="15" font-weight="700">XGBoost</text>
    <text x="650" y="337" text-anchor="middle" fill="#7c2d12" font-size="10.5">Gradient+Hessian+正規化</text>

    <g filter="url(#shadow)">
        <rect x="775" y="285" width="150" height="70" rx="12" fill="#fb923c"/>
    </g>
    <text x="850" y="315" text-anchor="middle" fill="#7c2d12" font-size="15" font-weight="700">LightGBM</text>
    <text x="850" y="337" text-anchor="middle" fill="#7c2d12" font-size="11">Leaf-wise 生長</text>

    <g filter="url(#shadow)">
        <rect x="985" y="285" width="150" height="70" rx="12" fill="#fb923c"/>
    </g>
    <text x="1060" y="311" text-anchor="middle" fill="#7c2d12" font-size="15" font-weight="700">CatBoost</text>
    <text x="1060" y="330" text-anchor="middle" fill="#7c2d12" font-size="10">Symmetric Tree</text>
    <text x="1060" y="344" text-anchor="middle" fill="#7c2d12" font-size="10">Ordered Boosting</text>

    <!-- ============ LEGEND ============ -->
    <rect x="330" y="410" width="16" height="16" rx="3" fill="#2563eb"/>
    <text x="352" y="423" fill="#374151" font-size="13">Bagging 家族：平行訓練，降低 Variance</text>

    <rect x="680" y="410" width="16" height="16" rx="3" fill="#ea580c"/>
    <text x="702" y="423" fill="#374151" font-size="13">Boosting 家族：序列訓練，降低 Bias</text>

</svg>

</body>
</html>
        """,
        height=480,
        scrolling=False
    )

    st.divider()

    # ========================================================
    # 1.2 TREE STRUCTURE AT A GLANCE
    # ========================================================

    st.subheader("1.2 Tree Structure at a Glance")

    st.caption(
        "以 Static Overview 先看 8 個模型的結構差異。"
        "除了 Tree depth，也比較 Split Strategy、Leaf 的角色與 Tree Growth Pattern。"
        "圖為概念示意、非實際訓練樹。"
    )

    st.info(
        "🌳 **讀圖重點：** CART / Bagging / Random Forest / Extra Trees 以較深的樹為主，"
        "Leaf 比較像最終分類區域；Boosting family 通常以較淺的樹逐步加總修正，"
        "Leaf 表示對最終預測的 contribution / score。"
        "LightGBM 另外強調 Leaf-wise growth；CatBoost 則強調 Symmetric（Oblivious）Tree。"
    )

    components.html(
        """
<!DOCTYPE html>
<html>
<head>
<style>
    body {
        margin: 0;
        padding: 0;
        font-family: -apple-system, "Segoe UI", "Microsoft JhengHei", sans-serif;
        background: transparent;
    }
    .legend {
        display:flex;
        justify-content:center;
        gap:24px;
        margin: 2px 0 14px 0;
        font-size: 12px;
        color:#475569;
    }
    .legend span { display:flex; align-items:center; gap:7px; }
    .dot { width:10px; height:10px; border-radius:50%; display:inline-block; }
    .grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 14px;
        padding: 4px;
        box-sizing: border-box;
    }
    .tree-card {
        background: #ffffff;
        border-radius: 14px;
        border: 1px solid #e5e7eb;
        box-shadow: 0 3px 10px rgba(15,23,42,0.08);
        overflow: hidden;
        box-sizing: border-box;
    }
    .tree-card-header {
        padding: 8px 11px;
        color: #ffffff;
        font-weight: 700;
        font-size: 13px;
        display:flex;
        justify-content:space-between;
        align-items:center;
    }
    .depth-badge {
        background: rgba(255,255,255,0.20);
        border: 1px solid rgba(255,255,255,0.35);
        border-radius: 999px;
        padding: 2px 7px;
        font-size: 9.5px;
        font-weight: 700;
    }
    .tree-card-body { padding: 5px 8px 1px 8px; }
    .tree-caption {
        min-height: 43px;
        font-size: 10.5px;
        color: #4b5563;
        line-height: 1.42;
        padding: 0 10px 10px 10px;
    }
</style>
</head>
<body>

<div class="legend">
    <span><i class="dot" style="background:#2563eb"></i>Deep trees：Leaf → Class / Vote</span>
    <span><i class="dot" style="background:#ea580c"></i>Boosting trees：Leaf → Score / Contribution</span>
    <span><i class="dot" style="background:#7c3aed"></i>Growth pattern：Leaf-wise / Symmetric</span>
</div>

<div class="grid">

    <!-- CART: one deep tree -->
    <div class="tree-card">
        <div class="tree-card-header" style="background:#374151;">
            <span>CART</span><span class="depth-badge">Deep · depth 16</span>
        </div>
        <div class="tree-card-body">
            <svg viewBox="0 0 220 145" width="100%">
                <g stroke="#9ca3af" stroke-width="1.7">
                    <line x1="110" y1="14" x2="65" y2="42"/><line x1="110" y1="14" x2="155" y2="42"/>
                    <line x1="65" y1="42" x2="40" y2="70"/><line x1="65" y1="42" x2="88" y2="70"/>
                    <line x1="155" y1="42" x2="132" y2="70"/><line x1="155" y1="42" x2="180" y2="70"/>
                    <line x1="40" y1="70" x2="25" y2="100"/><line x1="40" y1="70" x2="53" y2="100"/>
                    <line x1="88" y1="70" x2="75" y2="100"/><line x1="88" y1="70" x2="100" y2="100"/>
                    <line x1="132" y1="70" x2="120" y2="100"/><line x1="132" y1="70" x2="145" y2="100"/>
                    <line x1="180" y1="70" x2="167" y2="100"/><line x1="180" y1="70" x2="195" y2="100"/>
                </g>
                <circle cx="110" cy="14" r="7" fill="#374151"/>
                <g fill="#6b7280"><circle cx="65" cy="42" r="6"/><circle cx="155" cy="42" r="6"/></g>
                <g fill="#9ca3af"><circle cx="40" cy="70" r="5"/><circle cx="88" cy="70" r="5"/><circle cx="132" cy="70" r="5"/><circle cx="180" cy="70" r="5"/></g>
                <g fill="#e5e7eb" stroke="#9ca3af"><rect x="17" y="100" width="16" height="14" rx="3"/><rect x="45" y="100" width="16" height="14" rx="3"/><rect x="67" y="100" width="16" height="14" rx="3"/><rect x="92" y="100" width="16" height="14" rx="3"/><rect x="112" y="100" width="16" height="14" rx="3"/><rect x="137" y="100" width="16" height="14" rx="3"/><rect x="159" y="100" width="16" height="14" rx="3"/><rect x="187" y="100" width="16" height="14" rx="3"/></g>
                <text x="110" y="122" text-anchor="middle" font-size="8.5" fill="#374151" font-weight="700">Leaf → Class</text>
                <text x="110" y="138" text-anchor="middle" font-size="9.5" fill="#475569">Best split → 深樹 → 最終分類</text>
            </svg>
        </div>
        <div class="tree-caption">每個 Node 尋找最佳 Split；Leaf 直接代表分類結果。單棵深樹解釋性高，但 Variance 也較高。</div>
    </div>

    <!-- Bagging: several deeper trees -->
    <div class="tree-card">
        <div class="tree-card-header" style="background:#2563eb;">
            <span>Bagging</span><span class="depth-badge">Deep · base 17</span>
        </div>
        <div class="tree-card-body">
            <svg viewBox="0 0 220 145" width="100%">
                <text x="38" y="12" font-size="9" text-anchor="middle">Bootstrap</text><text x="110" y="12" font-size="9" text-anchor="middle">Bootstrap</text><text x="182" y="12" font-size="9" text-anchor="middle">Bootstrap</text>
                <g stroke="#93c5fd" stroke-width="1.4">
                    <path d="M38 25 L20 45 M38 25 L56 45 M20 45 L12 66 M20 45 L29 66 M56 45 L47 66 M56 45 L65 66"/>
                    <path d="M110 25 L92 45 M110 25 L128 45 M92 45 L84 66 M92 45 L101 66 M128 45 L119 66 M128 45 L137 66"/>
                    <path d="M182 25 L164 45 M182 25 L200 45 M164 45 L156 66 M164 45 L173 66 M200 45 L191 66 M200 45 L209 66"/>
                </g>
                <g fill="#2563eb"><circle cx="38" cy="25" r="6"/><circle cx="110" cy="25" r="6"/><circle cx="182" cy="25" r="6"/></g>
                <g fill="#60a5fa"><circle cx="20" cy="45" r="4.5"/><circle cx="56" cy="45" r="4.5"/><circle cx="92" cy="45" r="4.5"/><circle cx="128" cy="45" r="4.5"/><circle cx="164" cy="45" r="4.5"/><circle cx="200" cy="45" r="4.5"/></g>
                <line x1="38" y1="75" x2="82" y2="98" stroke="#60a5fa"/><line x1="110" y1="75" x2="110" y2="98" stroke="#60a5fa"/><line x1="182" y1="75" x2="138" y2="98" stroke="#60a5fa"/>
                <text x="110" y="86" text-anchor="middle" font-size="8.5" fill="#1e3a8a" font-weight="700">Leaf → Class</text>
                <rect x="57" y="98" width="106" height="25" rx="8" fill="#1e40af"/><text x="110" y="115" text-anchor="middle" font-size="11" font-weight="700" fill="white">Majority Vote</text>
                <text x="110" y="138" text-anchor="middle" font-size="9.5" fill="#475569">Bootstrap + 深樹 + Vote</text>
            </svg>
        </div>
        <div class="tree-caption">Bootstrap 產生多組資料；每棵深 CART 的 Leaf 先給 Class，再由多棵樹 Majority Vote。</div>
    </div>

    <!-- Random Forest -->
    <div class="tree-card">
        <div class="tree-card-header" style="background:#3b82f6;">
            <span>Random Forest</span><span class="depth-badge">Deep · depth 22</span>
        </div>
        <div class="tree-card-body">
            <svg viewBox="0 0 220 145" width="100%">
                <g stroke="#93c5fd" stroke-width="1.4">
                    <path d="M38 24 L20 44 M38 24 L56 44 M20 44 L12 66 M20 44 L29 66 M56 44 L47 66 M56 44 L65 66"/>
                    <path d="M110 24 L92 44 M110 24 L128 44 M92 44 L84 66 M92 44 L101 66 M128 44 L119 66 M128 44 L137 66"/>
                    <path d="M182 24 L164 44 M182 24 L200 44 M164 44 L156 66 M164 44 L173 66 M200 44 L191 66 M200 44 L209 66"/>
                </g>
                <g fill="#3b82f6"><circle cx="38" cy="24" r="6"/><circle cx="110" cy="24" r="6"/><circle cx="182" cy="24" r="6"/></g>
                <g fill="#22c55e"><rect x="20" y="73" width="8" height="8"/><rect x="104" y="73" width="8" height="8"/><rect x="190" y="73" width="8" height="8"/></g>
                <g fill="#ef4444"><rect x="31" y="73" width="8" height="8"/><rect x="115" y="73" width="8" height="8"/><rect x="168" y="73" width="8" height="8"/></g>
                <text x="110" y="92" text-anchor="middle" font-size="8.8" fill="#1e40af">Node：Random Feature Subset</text>
                <text x="110" y="102" text-anchor="middle" font-size="8.2" fill="#1e40af" font-weight="700">Leaf → Class</text>
                <rect x="57" y="108" width="106" height="22" rx="8" fill="#1d4ed8"/><text x="110" y="123" text-anchor="middle" font-size="10.5" font-weight="700" fill="white">Vote</text>
                <text x="110" y="141" text-anchor="middle" font-size="9.2" fill="#475569">深樹 + feature randomness</text>
            </svg>
        </div>
        <div class="tree-caption">每個 Node 只從隨機 Feature Subset 中找最佳 Split；Leaf 仍輸出 Class，再由 Forest Vote。</div>
    </div>

    <!-- Extra Trees -->
    <div class="tree-card">
        <div class="tree-card-header" style="background:#60a5fa;">
            <span>Extra Trees</span><span class="depth-badge">Deep · depth 20</span>
        </div>
        <div class="tree-card-body">
            <svg viewBox="0 0 220 145" width="100%">
                <g stroke="#93c5fd" stroke-width="1.4" stroke-dasharray="4 3">
                    <path d="M38 24 L18 45 M38 24 L58 45 M18 45 L10 68 M18 45 L29 68 M58 45 L47 68 M58 45 L67 68"/>
                    <path d="M110 24 L90 45 M110 24 L130 45 M90 45 L82 68 M90 45 L101 68 M130 45 L119 68 M130 45 L139 68"/>
                    <path d="M182 24 L162 45 M182 24 L202 45 M162 45 L154 68 M162 45 L173 68 M202 45 L191 68 M202 45 L211 68"/>
                </g>
                <g fill="#60a5fa"><circle cx="38" cy="24" r="6"/><circle cx="110" cy="24" r="6"/><circle cx="182" cy="24" r="6"/></g>
                <text x="110" y="89" text-anchor="middle" font-size="8.8" fill="#1d4ed8">Node：Random Threshold</text>
                <text x="110" y="101" text-anchor="middle" font-size="8.2" fill="#1d4ed8" font-weight="700">Leaf → Class</text>
                <rect x="57" y="108" width="106" height="22" rx="8" fill="#1d4ed8"/><text x="110" y="123" text-anchor="middle" font-size="10.5" font-weight="700" fill="white">Vote</text>
                <text x="110" y="141" text-anchor="middle" font-size="9.2" fill="#475569">深樹 + 更隨機切點</text>
            </svg>
        </div>
        <div class="tree-caption">多棵較深樹，但 Split Threshold 更隨機，以增加 Tree Diversity。</div>
    </div>

    <!-- Gradient Boosting: shallow stumps -->
    <div class="tree-card">
        <div class="tree-card-header" style="background:#ea580c;">
            <span>Gradient Boosting</span><span class="depth-badge">Shallow · depth 3</span>
        </div>
        <div class="tree-card-body">
            <svg viewBox="0 0 220 145" width="100%">
                <defs><marker id="a1" markerWidth="7" markerHeight="7" refX="6" refY="3" orient="auto"><path d="M0,0 L0,6 L7,3 z" fill="#9a3412"/></marker></defs>
                <g fill="#ea580c"><circle cx="35" cy="35" r="7"/><circle cx="110" cy="35" r="7"/><circle cx="185" cy="35" r="7"/></g>
                <g stroke="#fdba74" stroke-width="1.7"><line x1="35" y1="42" x2="23" y2="62"/><line x1="35" y1="42" x2="47" y2="62"/><line x1="110" y1="42" x2="98" y2="62"/><line x1="110" y1="42" x2="122" y2="62"/><line x1="185" y1="42" x2="173" y2="62"/><line x1="185" y1="42" x2="197" y2="62"/></g>
                <line x1="56" y1="35" x2="88" y2="35" stroke="#9a3412" marker-end="url(#a1)"/><line x1="131" y1="35" x2="163" y2="35" stroke="#9a3412" marker-end="url(#a1)"/>
                <text x="72" y="25" text-anchor="middle" font-size="8.5" fill="#9a3412">residual</text><text x="147" y="25" text-anchor="middle" font-size="8.5" fill="#9a3412">residual</text>
                <rect x="57" y="94" width="106" height="25" rx="8" fill="#9a3412"/><text x="110" y="111" text-anchor="middle" font-size="10.5" font-weight="700" fill="white">Σ Prediction</text>
                <text x="110" y="138" text-anchor="middle" font-size="9.5" fill="#475569">多棵淺樹 sequentially 修正</text>
            </svg>
        </div>
        <div class="tree-caption">典型 Boosting 意象：弱學習器較淺，後一棵樹逐步修正前面的 Residual。</div>
    </div>

    <!-- XGBoost -->
    <div class="tree-card">
        <div class="tree-card-header" style="background:#f97316;">
            <span>XGBoost</span><span class="depth-badge">Shallow · depth 5</span>
        </div>
        <div class="tree-card-body">
            <svg viewBox="0 0 220 145" width="100%">
                <defs><marker id="a2" markerWidth="7" markerHeight="7" refX="6" refY="3" orient="auto"><path d="M0,0 L0,6 L7,3 z" fill="#c2410c"/></marker></defs>
                <g fill="#f97316"><circle cx="35" cy="32" r="7"/><circle cx="110" cy="32" r="7"/><circle cx="185" cy="32" r="7"/></g>
                <g stroke="#fed7aa" stroke-width="1.7"><line x1="35" y1="39" x2="23" y2="59"/><line x1="35" y1="39" x2="47" y2="59"/><line x1="110" y1="39" x2="98" y2="59"/><line x1="110" y1="39" x2="122" y2="59"/><line x1="185" y1="39" x2="173" y2="59"/><line x1="185" y1="39" x2="197" y2="59"/></g>
                <line x1="56" y1="32" x2="88" y2="32" stroke="#c2410c" marker-end="url(#a2)"/><line x1="131" y1="32" x2="163" y2="32" stroke="#c2410c" marker-end="url(#a2)"/>
                <rect x="35" y="72" width="150" height="22" rx="7" fill="#fff7ed" stroke="#fb923c"/><text x="110" y="87" text-anchor="middle" font-size="8.7" font-weight="700" fill="#9a3412">Gradient + Hessian + Regularization</text>
                <rect x="57" y="103" width="106" height="23" rx="8" fill="#c2410c"/><text x="110" y="119" text-anchor="middle" font-size="10.5" font-weight="700" fill="white">Σ Prediction</text>
                <text x="110" y="140" text-anchor="middle" font-size="9.5" fill="#475569">shallow boosting trees</text>
            </svg>
        </div>
        <div class="tree-caption">仍是較淺 Tree 的序列 Ensemble，加入二階梯度與 Regularization。</div>
    </div>

    <!-- LightGBM -->
    <div class="tree-card">
        <div class="tree-card-header" style="background:#fb923c;">
            <span>LightGBM</span><span class="depth-badge">Shallow-ish · depth 6</span>
        </div>
        <div class="tree-card-body">
            <svg viewBox="0 0 220 145" width="100%">
                <text x="110" y="14" text-anchor="middle" font-size="8.7" fill="#9a3412">Leaf-wise: grow highest-gain leaf</text>
                <line x1="105" y1="28" x2="60" y2="54" stroke="#fdba74" stroke-width="1.8"/><line x1="105" y1="28" x2="150" y2="54" stroke="#fdba74" stroke-width="1.8"/>
                <circle cx="105" cy="28" r="7" fill="#fb923c"/><rect x="43" y="54" width="34" height="18" rx="4" fill="#fed7aa" stroke="#fb923c"/>
                <circle cx="150" cy="54" r="6" fill="#fb923c"/><line x1="150" y1="60" x2="125" y2="84" stroke="#fdba74"/><line x1="150" y1="60" x2="177" y2="84" stroke="#fdba74"/>
                <circle cx="125" cy="84" r="5" fill="#fb923c"/><line x1="125" y1="89" x2="112" y2="108" stroke="#fdba74"/><line x1="125" y1="89" x2="138" y2="108" stroke="#fdba74"/>
                <g fill="#fed7aa" stroke="#fb923c"><rect x="99" y="108" width="26" height="16" rx="4"/><rect x="127" y="108" width="26" height="16" rx="4"/><rect x="164" y="84" width="26" height="16" rx="4"/></g>
                <text x="110" y="140" text-anchor="middle" font-size="9.5" fill="#475569">較淺但不對稱的 leaf-wise tree</text>
            </svg>
        </div>
        <div class="tree-caption">Boosting family；優先擴展 Gain 最大的 Leaf，因此樹形不對稱。</div>
    </div>

    <!-- CatBoost -->
    <div class="tree-card">
        <div class="tree-card-header" style="background:#7c3aed;">
            <span>CatBoost ✅</span><span class="depth-badge">Shallow · depth 5</span>
        </div>
        <div class="tree-card-body">
            <svg viewBox="0 0 220 145" width="100%">
                <text x="110" y="14" text-anchor="middle" font-size="8.7" fill="#5b21b6">same split rule within each level</text>
                <line x1="110" y1="28" x2="65" y2="55" stroke="#c4b5fd" stroke-width="1.8"/><line x1="110" y1="28" x2="155" y2="55" stroke="#c4b5fd" stroke-width="1.8"/>
                <circle cx="110" cy="28" r="7" fill="#7c3aed"/><circle cx="65" cy="55" r="6" fill="#8b5cf6"/><circle cx="155" cy="55" r="6" fill="#8b5cf6"/>
                <line x1="65" y1="61" x2="40" y2="85" stroke="#c4b5fd"/><line x1="65" y1="61" x2="90" y2="85" stroke="#c4b5fd"/><line x1="155" y1="61" x2="130" y2="85" stroke="#c4b5fd"/><line x1="155" y1="61" x2="180" y2="85" stroke="#c4b5fd"/>
                <g fill="#ede9fe" stroke="#7c3aed"><rect x="28" y="85" width="24" height="17" rx="4"/><rect x="78" y="85" width="24" height="17" rx="4"/><rect x="118" y="85" width="24" height="17" rx="4"/><rect x="168" y="85" width="24" height="17" rx="4"/></g>
                <rect x="31" y="112" width="72" height="18" rx="6" fill="#7c3aed"/><text x="67" y="125" text-anchor="middle" font-size="8.4" font-weight="700" fill="white">Ordered Boosting</text>
                <rect x="117" y="112" width="72" height="18" rx="6" fill="#6d28d9"/><text x="153" y="125" text-anchor="middle" font-size="8.4" font-weight="700" fill="white">Categorical</text>
                <text x="110" y="141" text-anchor="middle" font-size="9.5" fill="#475569">shallow symmetric tree</text>
            </svg>
        </div>
        <div class="tree-caption">較淺的 Symmetric（Oblivious）Tree：同一層使用相同 split rule；Leaf 輸出 score，再由 Boosting 累加。</div>
    </div>

</div>
</body>
</html>
        """,
        height=720,
        scrolling=False
    )

    st.markdown("#### 🧭 What changes across the trees?")

    tree_diff_df = pd.DataFrame({
        "Model": [
            "CART", "Bagging", "Random Forest", "Extra Trees",
            "Gradient Boosting", "XGBoost", "LightGBM", "CatBoost"
        ],
        "Split / Growth": [
            "Best split",
            "Bootstrap + best split",
            "Random feature subset + best split",
            "More-random split threshold",
            "Sequential residual correction",
            "Gradient + Hessian + regularized gain",
            "Leaf-wise growth",
            "Symmetric split rule per level"
        ],
        "Leaf meaning": [
            "Class",
            "Class → Vote",
            "Class → Vote",
            "Class → Vote",
            "Correction / contribution",
            "Regularized weight / score",
            "Contribution score",
            "Contribution score"
        ],
        "Tree image": [
            "Single deep tree",
            "Many deep trees",
            "Many deep diverse trees",
            "Many highly-random deep trees",
            "Many shallow trees",
            "Many shallow trees",
            "Asymmetric leaf-wise tree",
            "Shallow symmetric tree"
        ]
    })

    st.dataframe(
        tree_diff_df,
        width="stretch",
        hide_index=True
    )

    st.markdown("#### 🔎 Interactive Detail View")
    st.caption(
        "Static overview 看完後，可在下方選擇單一模型，放大查看其建樹機制與核心差異。"
    )

    structure_model_options = [
        "CART",
        "Bagging",
        "Random Forest",
        "Extra Trees",
        "Gradient Boosting",
        "XGBoost",
        "LightGBM",
        "CatBoost"
    ]

    selected_structure_model = st.selectbox(
        "選擇模型",
        structure_model_options,
        index=7,
        key="structure_explorer_model"
    )

    structure_svg_inner = {

        "CART": """
<svg viewBox="0 0 400 240" width="100%">
    <line x1="200" y1="40" x2="110" y2="95" stroke="#9ca3af" stroke-width="2.5"/>
    <line x1="200" y1="40" x2="290" y2="95" stroke="#9ca3af" stroke-width="2.5"/>
    <line x1="110" y1="95" x2="60" y2="160" stroke="#9ca3af" stroke-width="2.5"/>
    <line x1="110" y1="95" x2="160" y2="160" stroke="#9ca3af" stroke-width="2.5"/>
    <line x1="290" y1="95" x2="240" y2="160" stroke="#9ca3af" stroke-width="2.5"/>
    <line x1="290" y1="95" x2="340" y2="160" stroke="#9ca3af" stroke-width="2.5"/>
    <circle cx="200" cy="40" r="16" fill="#374151"/>
    <circle cx="110" cy="95" r="13" fill="#6b7280"/>
    <circle cx="290" cy="95" r="13" fill="#6b7280"/>
    <rect x="35" y="160" width="50" height="34" rx="6" fill="#e5e7eb" stroke="#9ca3af" stroke-width="1.5"/>
    <rect x="135" y="160" width="50" height="34" rx="6" fill="#e5e7eb" stroke="#9ca3af" stroke-width="1.5"/>
    <rect x="215" y="160" width="50" height="34" rx="6" fill="#e5e7eb" stroke="#9ca3af" stroke-width="1.5"/>
    <rect x="315" y="160" width="50" height="34" rx="6" fill="#e5e7eb" stroke="#9ca3af" stroke-width="1.5"/>
    <text x="200" y="220" font-size="13" fill="#374151" text-anchor="middle" font-weight="600">單一決策樹，依序切分資料直到葉節點</text>
</svg>
""",

        "Bagging": """
<svg viewBox="0 0 400 240" width="100%">
    <text x="80" y="20" font-size="20" text-anchor="middle">🎲</text>
    <text x="200" y="20" font-size="20" text-anchor="middle">🎲</text>
    <text x="320" y="20" font-size="20" text-anchor="middle">🎲</text>

    <text x="80" y="42" font-size="10" text-anchor="middle" fill="#1e40af">Bootstrap Sample</text>
    <text x="200" y="42" font-size="10" text-anchor="middle" fill="#1e40af">Bootstrap Sample</text>
    <text x="320" y="42" font-size="10" text-anchor="middle" fill="#1e40af">Bootstrap Sample</text>

    <line x1="80" y1="52" x2="55" y2="90" stroke="#93c5fd" stroke-width="2.5"/>
    <line x1="80" y1="52" x2="105" y2="90" stroke="#93c5fd" stroke-width="2.5"/>
    <line x1="200" y1="52" x2="175" y2="90" stroke="#93c5fd" stroke-width="2.5"/>
    <line x1="200" y1="52" x2="225" y2="90" stroke="#93c5fd" stroke-width="2.5"/>
    <line x1="320" y1="52" x2="295" y2="90" stroke="#93c5fd" stroke-width="2.5"/>
    <line x1="320" y1="52" x2="345" y2="90" stroke="#93c5fd" stroke-width="2.5"/>

    <circle cx="80" cy="52" r="12" fill="#2563eb"/>
    <circle cx="200" cy="52" r="12" fill="#2563eb"/>
    <circle cx="320" cy="52" r="12" fill="#2563eb"/>
    <circle cx="55" cy="90" r="9" fill="#93c5fd"/>
    <circle cx="105" cy="90" r="9" fill="#93c5fd"/>
    <circle cx="175" cy="90" r="9" fill="#93c5fd"/>
    <circle cx="225" cy="90" r="9" fill="#93c5fd"/>
    <circle cx="295" cy="90" r="9" fill="#93c5fd"/>
    <circle cx="345" cy="90" r="9" fill="#93c5fd"/>

    <line x1="80" y1="100" x2="150" y2="150" stroke="#60a5fa" stroke-width="2"/>
    <line x1="200" y1="100" x2="200" y2="150" stroke="#60a5fa" stroke-width="2"/>
    <line x1="320" y1="100" x2="250" y2="150" stroke="#60a5fa" stroke-width="2"/>

    <rect x="70" y="150" width="260" height="46" rx="10" fill="#1e40af"/>
    <text x="200" y="179" font-size="16" fill="#ffffff" text-anchor="middle" font-weight="700">🗳️ Majority Vote</text>

    <text x="200" y="220" font-size="13" fill="#1e3a8a" text-anchor="middle" font-weight="600">Bootstrap 抽樣多組資料，平行訓練多棵樹，最後多數決</text>
</svg>
""",

        "Random Forest": """
<svg viewBox="0 0 400 240" width="100%">
    <line x1="80" y1="42" x2="55" y2="80" stroke="#93c5fd" stroke-width="2.5"/>
    <line x1="80" y1="42" x2="105" y2="80" stroke="#93c5fd" stroke-width="2.5"/>
    <line x1="200" y1="42" x2="175" y2="80" stroke="#93c5fd" stroke-width="2.5"/>
    <line x1="200" y1="42" x2="225" y2="80" stroke="#93c5fd" stroke-width="2.5"/>
    <line x1="320" y1="42" x2="295" y2="80" stroke="#93c5fd" stroke-width="2.5"/>
    <line x1="320" y1="42" x2="345" y2="80" stroke="#93c5fd" stroke-width="2.5"/>

    <circle cx="80" cy="42" r="12" fill="#3b82f6"/>
    <circle cx="200" cy="42" r="12" fill="#3b82f6"/>
    <circle cx="320" cy="42" r="12" fill="#3b82f6"/>
    <circle cx="55" cy="80" r="9" fill="#93c5fd"/>
    <circle cx="105" cy="80" r="9" fill="#93c5fd"/>
    <circle cx="175" cy="80" r="9" fill="#93c5fd"/>
    <circle cx="225" cy="80" r="9" fill="#93c5fd"/>
    <circle cx="295" cy="80" r="9" fill="#93c5fd"/>
    <circle cx="345" cy="80" r="9" fill="#93c5fd"/>

    <rect x="65" y="95" width="10" height="10" fill="#ef4444"/>
    <rect x="80" y="95" width="10" height="10" fill="#d1d5db"/>
    <rect x="95" y="95" width="10" height="10" fill="#22c55e"/>

    <rect x="185" y="95" width="10" height="10" fill="#d1d5db"/>
    <rect x="200" y="95" width="10" height="10" fill="#22c55e"/>
    <rect x="215" y="95" width="10" height="10" fill="#d1d5db"/>

    <rect x="305" y="95" width="10" height="10" fill="#22c55e"/>
    <rect x="320" y="95" width="10" height="10" fill="#ef4444"/>
    <rect x="335" y="95" width="10" height="10" fill="#d1d5db"/>

    <text x="200" y="120" font-size="10" fill="#1e40af" text-anchor="middle">每次分裂僅隨機考慮部分特徵（彩色方塊）</text>

    <line x1="80" y1="128" x2="150" y2="150" stroke="#60a5fa" stroke-width="2"/>
    <line x1="200" y1="128" x2="200" y2="150" stroke="#60a5fa" stroke-width="2"/>
    <line x1="320" y1="128" x2="250" y2="150" stroke="#60a5fa" stroke-width="2"/>

    <rect x="70" y="150" width="260" height="46" rx="10" fill="#1d4ed8"/>
    <text x="200" y="179" font-size="16" fill="#ffffff" text-anchor="middle" font-weight="700">🗳️ Majority Vote</text>

    <text x="200" y="220" font-size="13" fill="#1e3a8a" text-anchor="middle" font-weight="600">Bagging + Random Feature Subset，降低樹間相關性</text>
</svg>
""",

        "Extra Trees": """
<svg viewBox="0 0 400 240" width="100%">
    <line x1="80" y1="42" x2="55" y2="80" stroke="#bfdbfe" stroke-width="2.5" stroke-dasharray="4,3"/>
    <line x1="80" y1="42" x2="105" y2="80" stroke="#bfdbfe" stroke-width="2.5" stroke-dasharray="4,3"/>
    <line x1="200" y1="42" x2="175" y2="80" stroke="#bfdbfe" stroke-width="2.5" stroke-dasharray="4,3"/>
    <line x1="200" y1="42" x2="225" y2="80" stroke="#bfdbfe" stroke-width="2.5" stroke-dasharray="4,3"/>
    <line x1="320" y1="42" x2="295" y2="80" stroke="#bfdbfe" stroke-width="2.5" stroke-dasharray="4,3"/>
    <line x1="320" y1="42" x2="345" y2="80" stroke="#bfdbfe" stroke-width="2.5" stroke-dasharray="4,3"/>

    <circle cx="80" cy="42" r="12" fill="#60a5fa"/>
    <circle cx="200" cy="42" r="12" fill="#60a5fa"/>
    <circle cx="320" cy="42" r="12" fill="#60a5fa"/>
    <circle cx="55" cy="80" r="9" fill="#bfdbfe"/>
    <circle cx="105" cy="80" r="9" fill="#bfdbfe"/>
    <circle cx="175" cy="80" r="9" fill="#bfdbfe"/>
    <circle cx="225" cy="80" r="9" fill="#bfdbfe"/>
    <circle cx="295" cy="80" r="9" fill="#bfdbfe"/>
    <circle cx="345" cy="80" r="9" fill="#bfdbfe"/>

    <text x="200" y="115" font-size="10" fill="#1d4ed8" text-anchor="middle">分裂閾值也隨機決定（虛線），非尋找最佳切點</text>

    <line x1="80" y1="128" x2="150" y2="150" stroke="#93c5fd" stroke-width="2"/>
    <line x1="200" y1="128" x2="200" y2="150" stroke="#93c5fd" stroke-width="2"/>
    <line x1="320" y1="128" x2="250" y2="150" stroke="#93c5fd" stroke-width="2"/>

    <rect x="70" y="150" width="260" height="46" rx="10" fill="#1d4ed8"/>
    <text x="200" y="179" font-size="16" fill="#ffffff" text-anchor="middle" font-weight="700">🗳️ Majority Vote</text>

    <text x="200" y="220" font-size="13" fill="#1e3a8a" text-anchor="middle" font-weight="600">Random Split Threshold → 增加 Tree Diversity</text>
</svg>
""",

        "Gradient Boosting": """
<svg viewBox="0 0 400 240" width="100%">
    <defs>
        <marker id="gbarrow2" markerWidth="9" markerHeight="9" refX="7" refY="3.5" orient="auto">
            <path d="M0,0 L0,7 L8,3.5 z" fill="#9a3412"/>
        </marker>
    </defs>

    <text x="60" y="15" font-size="12" text-anchor="middle" fill="#9a3412" font-weight="700">殘差</text>
    <text x="200" y="15" font-size="12" text-anchor="middle" fill="#9a3412" font-weight="700">殘差</text>
    <text x="340" y="15" font-size="12" text-anchor="middle" fill="#9a3412" font-weight="700">殘差</text>

    <line x1="110" y1="42" x2="165" y2="42" stroke="#9a3412" stroke-width="2" marker-end="url(#gbarrow2)"/>
    <line x1="250" y1="42" x2="305" y2="42" stroke="#9a3412" stroke-width="2" marker-end="url(#gbarrow2)"/>

    <circle cx="60" cy="42" r="14" fill="#ea580c"/>
    <text x="60" y="47" font-size="11" fill="#ffffff" text-anchor="middle" font-weight="700">T1</text>
    <line x1="60" y1="56" x2="35" y2="90" stroke="#fdba74" stroke-width="2"/>
    <line x1="60" y1="56" x2="85" y2="90" stroke="#fdba74" stroke-width="2"/>
    <circle cx="35" cy="90" r="8" fill="#fdba74"/>
    <circle cx="85" cy="90" r="8" fill="#fdba74"/>

    <circle cx="200" cy="42" r="14" fill="#ea580c"/>
    <text x="200" y="47" font-size="11" fill="#ffffff" text-anchor="middle" font-weight="700">T2</text>
    <line x1="200" y1="56" x2="175" y2="90" stroke="#fdba74" stroke-width="2"/>
    <line x1="200" y1="56" x2="225" y2="90" stroke="#fdba74" stroke-width="2"/>
    <circle cx="175" cy="90" r="8" fill="#fdba74"/>
    <circle cx="225" cy="90" r="8" fill="#fdba74"/>

    <circle cx="340" cy="42" r="14" fill="#ea580c"/>
    <text x="340" y="47" font-size="11" fill="#ffffff" text-anchor="middle" font-weight="700">T3</text>
    <line x1="340" y1="56" x2="315" y2="90" stroke="#fdba74" stroke-width="2"/>
    <line x1="340" y1="56" x2="365" y2="90" stroke="#fdba74" stroke-width="2"/>
    <circle cx="315" cy="90" r="8" fill="#fdba74"/>
    <circle cx="365" cy="90" r="8" fill="#fdba74"/>

    <line x1="200" y1="100" x2="200" y2="150" stroke="#9a3412" stroke-width="2" marker-end="url(#gbarrow2)"/>

    <rect x="90" y="150" width="220" height="46" rx="10" fill="#9a3412"/>
    <text x="200" y="179" font-size="15" fill="#ffffff" text-anchor="middle" font-weight="700">Σ Final Prediction</text>

    <text x="200" y="220" font-size="13" fill="#7c2d12" text-anchor="middle" font-weight="600">序列訓練，每棵樹修正前一棵的殘差</text>
</svg>
""",

        "XGBoost": """
<svg viewBox="0 0 400 240" width="100%">
    <defs>
        <marker id="xgbarrow2" markerWidth="9" markerHeight="9" refX="7" refY="3.5" orient="auto">
            <path d="M0,0 L0,7 L8,3.5 z" fill="#c2410c"/>
        </marker>
    </defs>

    <line x1="110" y1="42" x2="165" y2="42" stroke="#c2410c" stroke-width="2" marker-end="url(#xgbarrow2)"/>
    <line x1="250" y1="42" x2="305" y2="42" stroke="#c2410c" stroke-width="2" marker-end="url(#xgbarrow2)"/>

    <circle cx="60" cy="42" r="14" fill="#f97316"/>
    <text x="60" y="47" font-size="11" fill="#ffffff" text-anchor="middle" font-weight="700">T1</text>
    <line x1="60" y1="56" x2="35" y2="88" stroke="#fed7aa" stroke-width="2"/>
    <line x1="60" y1="56" x2="85" y2="88" stroke="#fed7aa" stroke-width="2"/>
    <circle cx="35" cy="88" r="8" fill="#fed7aa"/>
    <circle cx="85" cy="88" r="8" fill="#fed7aa"/>

    <circle cx="200" cy="42" r="14" fill="#f97316"/>
    <text x="200" y="47" font-size="11" fill="#ffffff" text-anchor="middle" font-weight="700">T2</text>
    <line x1="200" y1="56" x2="175" y2="88" stroke="#fed7aa" stroke-width="2"/>
    <line x1="200" y1="56" x2="225" y2="88" stroke="#fed7aa" stroke-width="2"/>
    <circle cx="175" cy="88" r="8" fill="#fed7aa"/>
    <circle cx="225" cy="88" r="8" fill="#fed7aa"/>

    <circle cx="340" cy="42" r="14" fill="#f97316"/>
    <text x="340" y="47" font-size="11" fill="#ffffff" text-anchor="middle" font-weight="700">T3</text>
    <line x1="340" y1="56" x2="315" y2="88" stroke="#fed7aa" stroke-width="2"/>
    <line x1="340" y1="56" x2="365" y2="88" stroke="#fed7aa" stroke-width="2"/>
    <circle cx="315" cy="88" r="8" fill="#fed7aa"/>
    <circle cx="365" cy="88" r="8" fill="#fed7aa"/>

    <rect x="100" y="102" width="200" height="30" rx="8" fill="#fff7ed" stroke="#c2410c" stroke-width="1.5"/>
    <text x="200" y="122" font-size="10" fill="#c2410c" text-anchor="middle" font-weight="700">🛡️ Gradient + Hessian + Regularization</text>

    <line x1="200" y1="132" x2="200" y2="150" stroke="#c2410c" stroke-width="2" marker-end="url(#xgbarrow2)"/>

    <rect x="90" y="150" width="220" height="46" rx="10" fill="#c2410c"/>
    <text x="200" y="179" font-size="15" fill="#ffffff" text-anchor="middle" font-weight="700">Σ Final Prediction</text>

    <text x="200" y="220" font-size="13" fill="#9a3412" text-anchor="middle" font-weight="600">Gradient Boosting + 二階梯度 + 正則化，防止過擬合</text>
</svg>
""",

        "LightGBM": """
<svg viewBox="0 0 400 240" width="100%">
    <text x="200" y="15" font-size="12" fill="#c2410c" text-anchor="middle" font-weight="700">往增益最大的葉節點持續分裂（不對稱生長）</text>

    <line x1="200" y1="30" x2="110" y2="65" stroke="#fdba74" stroke-width="2.5"/>
    <line x1="200" y1="30" x2="290" y2="65" stroke="#fdba74" stroke-width="2.5"/>
    <circle cx="200" cy="30" r="13" fill="#fb923c"/>

    <rect x="80" y="65" width="60" height="30" rx="6" fill="#fed7aa" stroke="#fb923c" stroke-width="1.5"/>
    <text x="110" y="84" font-size="10" fill="#9a3412" text-anchor="middle">Leaf</text>

    <circle cx="290" cy="65" r="11" fill="#fb923c"/>
    <line x1="290" y1="76" x2="230" y2="112" stroke="#fdba74" stroke-width="2.5"/>
    <line x1="290" y1="76" x2="345" y2="112" stroke="#fdba74" stroke-width="2.5"/>

    <circle cx="230" cy="112" r="10" fill="#fb923c"/>
    <line x1="230" y1="122" x2="195" y2="155" stroke="#fdba74" stroke-width="2.5"/>
    <line x1="230" y1="122" x2="260" y2="155" stroke="#fdba74" stroke-width="2.5"/>
    <rect x="165" y="155" width="55" height="30" rx="6" fill="#fed7aa" stroke="#fb923c" stroke-width="1.5"/>
    <rect x="235" y="155" width="55" height="30" rx="6" fill="#fed7aa" stroke="#fb923c" stroke-width="1.5"/>
    <text x="192" y="174" font-size="10" fill="#9a3412" text-anchor="middle">Leaf</text>
    <text x="262" y="174" font-size="10" fill="#9a3412" text-anchor="middle">Leaf</text>

    <rect x="315" y="112" width="55" height="30" rx="6" fill="#fed7aa" stroke="#fb923c" stroke-width="1.5"/>
    <text x="342" y="131" font-size="10" fill="#9a3412" text-anchor="middle">Leaf</text>

    <text x="200" y="220" font-size="13" fill="#9a3412" text-anchor="middle" font-weight="600">Leaf-wise 生長，不對稱樹型，收斂速度快</text>
</svg>
""",

        "CatBoost": """
<svg viewBox="0 0 400 240" width="100%">
    <text x="200" y="15" font-size="12" fill="#5b21b6" text-anchor="middle" font-weight="700">每一層使用「相同」分裂條件（Symmetric）</text>

    <line x1="200" y1="30" x2="110" y2="70" stroke="#c4b5fd" stroke-width="2.5"/>
    <line x1="200" y1="30" x2="290" y2="70" stroke="#c4b5fd" stroke-width="2.5"/>
    <circle cx="200" cy="30" r="13" fill="#7c3aed"/>

    <circle cx="110" cy="70" r="11" fill="#8b5cf6"/>
    <circle cx="290" cy="70" r="11" fill="#8b5cf6"/>

    <line x1="110" y1="81" x2="65" y2="118" stroke="#c4b5fd" stroke-width="2.5"/>
    <line x1="110" y1="81" x2="155" y2="118" stroke="#c4b5fd" stroke-width="2.5"/>
    <line x1="290" y1="81" x2="245" y2="118" stroke="#c4b5fd" stroke-width="2.5"/>
    <line x1="290" y1="81" x2="335" y2="118" stroke="#c4b5fd" stroke-width="2.5"/>

    <rect x="40" y="118" width="50" height="32" rx="6" fill="#ede9fe" stroke="#7c3aed" stroke-width="1.5"/>
    <rect x="130" y="118" width="50" height="32" rx="6" fill="#ede9fe" stroke="#7c3aed" stroke-width="1.5"/>
    <rect x="220" y="118" width="50" height="32" rx="6" fill="#ede9fe" stroke="#7c3aed" stroke-width="1.5"/>
    <rect x="310" y="118" width="50" height="32" rx="6" fill="#ede9fe" stroke="#7c3aed" stroke-width="1.5"/>

    <rect x="60" y="168" width="115" height="26" rx="8" fill="#7c3aed"/>
    <text x="117" y="185" font-size="11" fill="#ffffff" text-anchor="middle" font-weight="700">🔗 Ordered Boosting</text>

    <rect x="225" y="168" width="115" height="26" rx="8" fill="#6d28d9"/>
    <text x="282" y="185" font-size="11" fill="#ffffff" text-anchor="middle" font-weight="700">🏷️ Native Categorical</text>

    <text x="200" y="222" font-size="13" fill="#5b21b6" text-anchor="middle" font-weight="600">Symmetric（Oblivious）Tree，結構規則、推論效率高</text>
</svg>
"""

    }

    structure_descriptions = {

        "CART":
            "單一決策樹，依照 Feature 與 Split Threshold 逐層切分資料，直到葉節點做出分類。"
            "結構簡單、容易解釋，但單棵樹容易 Overfitting，也是後面所有 Ensemble 模型的基礎單元。",

        "Bagging":
            "使用 Bootstrap Sampling 從原始資料中重複抽樣出多組訓練集，平行訓練多棵獨立的 CART，"
            "最後透過 Majority Vote 決定分類結果，藉此降低單棵樹的 Variance。",

        "Random Forest":
            "在 Bagging 的基礎上，每次分裂節點時「只隨機考慮部分特徵」（圖中彩色方塊），"
            "進一步降低不同樹之間的相關性，讓 Ensemble 的效果更穩定。",

        "Extra Trees":
            "與 Random Forest 類似，但分裂的「閾值」也是隨機決定，而非搜尋最佳切點（圖中虛線），"
            "透過更隨機的 Split Threshold 增加樹間多樣性；實際效果仍取決於資料與參數設定。",

        "Gradient Boosting":
            "採 Sequential Training：先訓練一棵樹，計算殘差（預測錯的部分），"
            "下一棵樹專門學習修正這個殘差，如此反覆多輪，最後將所有樹的預測加總。",

        "XGBoost":
            "Gradient Boosting 的強化版本，加入二階梯度（Hessian）資訊與 L1/L2 Regularization，"
            "在保有 Boosting 精準度的同時，有效降低 Overfitting 風險。",

        "LightGBM":
            "改用 Leaf-wise（而非 Level-wise）生長策略，每次優先分裂「增益最大」的葉節點，"
            "因此樹型通常較不對稱，能把更多分裂集中在目前 Gain 較高的葉節點。",

        "CatBoost":
            "使用 Symmetric（Oblivious）Tree，每一層強制使用相同的分裂條件，結構規則、推論效率高，"
            "搭配 Ordered Boosting 降低訓練過程中 target leakage / prediction shift 造成的偏差，"
            "並具備原生 Categorical Feature 處理能力，是本專題最終選定的部署模型。"

    }

    structure_html_template = """
<!DOCTYPE html>
<html>
<head>
<style>
    body {
        margin: 0;
        padding: 0;
        font-family: -apple-system, "Segoe UI", "Microsoft JhengHei", sans-serif;
    }
    .diagram-wrap {
        display: flex;
        justify-content: center;
        align-items: center;
        width: 100%;
    }
    .diagram-wrap svg {
        max-width: 480px;
    }
</style>
</head>
<body>
<div class="diagram-wrap">
___INNER_SVG___
</div>
</body>
</html>
    """

    components.html(
        structure_html_template.replace(
            "___INNER_SVG___",
            structure_svg_inner[selected_structure_model]
        ),
        height=300,
        scrolling=False
    )

    st.info(
        structure_descriptions[selected_structure_model]
    )


    st.caption(
        "註：Interactive Detail View 同樣為教學示意圖，並非直接匯出的實際訓練 Tree Plot。"
    )

    st.divider()


    # ========================================================
    # 1.3 HYPERPARAMETER TUNING
    # ========================================================

    st.subheader("1.3 Hyperparameter Tuning")

    st.caption(
        "先以手動測試合理範圍，再用 GridSearchCV 系統化搜尋，"
        "得到以下各模型的最佳參數組合。"
    )

    components.html(
        """
<!DOCTYPE html>
<html>
<head>
<style>
    body {
        margin: 0;
        padding: 0;
        font-family: -apple-system, "Segoe UI", "Microsoft JhengHei", sans-serif;
    }
    .grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 14px;
        padding: 4px;
        box-sizing: border-box;
    }
    .card {
        border-radius: 12px;
        padding: 14px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.12);
        box-sizing: border-box;
    }
    .card-title {
        color: #ffffff;
        font-size: 15px;
        font-weight: 700;
        margin-bottom: 10px;
    }
    .sub-label {
        color: rgba(255,255,255,0.85);
        font-size: 11px;
        font-weight: 600;
        margin: 6px 0 4px 0;
    }
    .chip-row {
        display: flex;
        flex-wrap: wrap;
        gap: 6px;
    }
    .chip {
        background: rgba(255,255,255,0.92);
        border-radius: 999px;
        padding: 3px 9px;
        font-size: 11px;
        font-weight: 600;
        white-space: nowrap;
    }
</style>
</head>
<body>

<div class="grid">

    <div class="card" style="background:#374151;">
        <div class="card-title">CART</div>
        <div class="chip-row">
            <span class="chip">max_depth=16</span>
            <span class="chip">min_samples_split=5</span>
            <span class="chip">min_samples_leaf=1</span>
            <span class="chip">ccp_alpha=0.00005</span>
        </div>
    </div>

    <div class="card" style="background:#2563eb;">
        <div class="card-title">Bagging</div>
        <div class="sub-label">Base Tree</div>
        <div class="chip-row">
            <span class="chip">max_depth=17</span>
            <span class="chip">min_samples_split=2</span>
            <span class="chip">min_samples_leaf=1</span>
        </div>
        <div class="sub-label">Bagging</div>
        <div class="chip-row">
            <span class="chip">n_estimators=100</span>
            <span class="chip">max_samples=1.0</span>
            <span class="chip">max_features=1.0</span>
            <span class="chip">bootstrap=True</span>
        </div>
    </div>

    <div class="card" style="background:#3b82f6;">
        <div class="card-title">Random Forest</div>
        <div class="chip-row">
            <span class="chip">n_estimators=200</span>
            <span class="chip">max_depth=22</span>
            <span class="chip">max_features=0.7</span>
            <span class="chip">min_samples_split=2</span>
            <span class="chip">min_samples_leaf=1</span>
        </div>
    </div>

    <div class="card" style="background:#60a5fa;">
        <div class="card-title">Extra Trees</div>
        <div class="chip-row">
            <span class="chip">n_estimators=200</span>
            <span class="chip">max_depth=20</span>
            <span class="chip">max_features=1.0</span>
            <span class="chip">min_samples_split=2</span>
            <span class="chip">min_samples_leaf=1</span>
            <span class="chip">bootstrap=False</span>
        </div>
    </div>

    <div class="card" style="background:#ea580c;">
        <div class="card-title">Gradient Boosting</div>
        <div class="chip-row">
            <span class="chip">learning_rate=0.5</span>
            <span class="chip">n_estimators=200</span>
            <span class="chip">max_depth=3</span>
            <span class="chip">min_samples_leaf=20</span>
            <span class="chip">subsample=0.9</span>
        </div>
    </div>

    <div class="card" style="background:#f97316;">
        <div class="card-title">XGBoost</div>
        <div class="chip-row">
            <span class="chip">learning_rate=0.2</span>
            <span class="chip">n_estimators=300</span>
            <span class="chip">max_depth=5</span>
            <span class="chip">min_child_weight=60</span>
            <span class="chip">gamma=2</span>
            <span class="chip">reg_lambda=2</span>
            <span class="chip">reg_alpha=0</span>
            <span class="chip">subsample=1.0</span>
            <span class="chip">colsample_bytree=1.0</span>
        </div>
    </div>

    <div class="card" style="background:#fb923c;">
        <div class="card-title">LightGBM</div>
        <div class="chip-row">
            <span class="chip">learning_rate=0.1</span>
            <span class="chip">n_estimators=300</span>
            <span class="chip">num_leaves=15</span>
            <span class="chip">max_depth=6</span>
            <span class="chip">min_child_samples=50</span>
            <span class="chip">min_split_gain=0</span>
            <span class="chip">reg_lambda=1</span>
            <span class="chip">reg_alpha=0.1</span>
            <span class="chip">subsample=0.9</span>
            <span class="chip">subsample_freq=1</span>
            <span class="chip">colsample_bytree=1.0</span>
        </div>
    </div>

    <div class="card" style="background:#7c3aed;">
        <div class="card-title">CatBoost ✅</div>
        <div class="chip-row">
            <span class="chip">learning_rate=0.2</span>
            <span class="chip">iterations=600</span>
            <span class="chip">depth=5</span>
            <span class="chip">l2_leaf_reg=3</span>
            <span class="chip">border_count=254</span>
            <span class="chip">bootstrap_type='MVS'</span>
            <span class="chip">subsample=0.8</span>
            <span class="chip">random_strength=1</span>
            <span class="chip">rsm=1.0</span>
        </div>
    </div>

</div>

</body>
</html>
        """,
        height=560,
        scrolling=False
    )

    st.divider()


    # ========================================================
    # 2 MODEL COMPARISON
    # ========================================================

    st.header("2. Model Comparison")

    st.write(
        """
        模型比較分成兩個面向：

        **Overall Performance**

        - Test Accuracy
        - Macro F1

        **High-Risk Detection**

        - Moderate Recall
        - Severe Recall
        """
    )


    # ========================================================
    # MODEL COMPARISON CHART 1
    # ========================================================

    st.subheader(
        "2.1 Overall Model Performance"
    )

    st.caption(
        "Test Accuracy + Macro F1"
    )

    components.html(
        """
<!DOCTYPE html>

<html>

<head>

<script src="https://cdn.amcharts.com/lib/5/index.js"></script>
<script src="https://cdn.amcharts.com/lib/5/xy.js"></script>
<script src="https://cdn.amcharts.com/lib/5/themes/Animated.js"></script>

</head>

<body>

<div
    id="overall_chart"
    style="width:100%; height:520px;">
</div>

<script>


var root = am5.Root.new(
    "overall_chart"
);


root.setThemes([
    am5themes_Animated.new(root)
]);


var chart = root.container.children.push(
    am5xy.XYChart.new(root, {

        panX: false,
        panY: false,

        paddingLeft: 0,

        wheelX: "panX",
        wheelY: "zoomX",

        layout:
            root.verticalLayout

    })
);


// Legend

var legend = chart.children.push(
    am5.Legend.new(root, {

        centerX:
            am5.p50,

        x:
            am5.p50

    })
);


// Data

var data = [

{
    model: "CART",
    accuracy: 91.41,
    macro_f1: 82.00
},

{
    model: "Bagging",
    accuracy: 92.53,
    macro_f1: 83.00
},

{
    model: "Random Forest",
    accuracy: 92.69,
    macro_f1: 84.00
},

{
    model: "Extra Trees",
    accuracy: 89.36,
    macro_f1: 80.00
},

{
    model: "Gradient Boosting",
    accuracy: 95.49,
    macro_f1: 88.00
},

{
    model: "XGBoost",
    accuracy: 95.26,
    macro_f1: 87.00
},

{
    model: "LightGBM",
    accuracy: 95.32,
    macro_f1: 88.00
},

{
    model: "CatBoost",
    accuracy: 95.65,
    macro_f1: 88.26
}

];


// X Axis

var xRenderer =
    am5xy.AxisRendererX.new(
        root,
        {
            cellStartLocation:
                0.15,

            cellEndLocation:
                0.85,

            minGridDistance:
                20
        }
    );


var xAxis = chart.xAxes.push(
    am5xy.CategoryAxis.new(
        root,
        {

            categoryField:
                "model",

            renderer:
                xRenderer

        }
    )
);


xRenderer.labels.template.setAll({

    rotation: -30,

    centerY:
        am5.p50,

    centerX:
        am5.p100,

    fontSize:
        11

});


xAxis.data.setAll(
    data
);


// Y Axis

var yAxis = chart.yAxes.push(
    am5xy.ValueAxis.new(
        root,
        {

            min: 0,

            max: 100,

            strictMinMax:
                true,

            renderer:
                am5xy.AxisRendererY.new(
                    root,
                    {
                        strokeOpacity:
                            0.1
                    }
                )

        }
    )
);


yAxis.set(
    "numberFormat",
    "#'%'"
);


// Series function

function makeSeries(
    name,
    field,
    color
) {

    var series =
        chart.series.push(

            am5xy.ColumnSeries.new(
                root,
                {

                    name:
                        name,

                    xAxis:
                        xAxis,

                    yAxis:
                        yAxis,

                    valueYField:
                        field,

                    categoryXField:
                        "model"

                }
            )

        );


    series.set("fill", am5.color(color));
    series.set("stroke", am5.color(color));


    series.columns.template.setAll({

        tooltipText:
            "{name}\\n{categoryX}: {valueY.formatNumber('0.00')}%",

        width:
            am5.percent(80),

        strokeOpacity:
            0,

        cornerRadiusTL:
            6,

        cornerRadiusTR:
            6

    });


    series.data.setAll(
        data
    );


    series.appear();


    legend.data.push(
        series
    );

}


makeSeries(
    "Test Accuracy",
    "accuracy",
    0x0d9488
);

makeSeries(
    "Macro F1",
    "macro_f1",
    0x6366f1
);


chart.appear(
    1000,
    100
);


</script>

</body>

</html>
        """,

        height=560,

        scrolling=False
    )


    # ========================================================
    # MODEL COMPARISON CHART 2
    # ========================================================

    st.subheader(
        "2.2 High-Risk Detection Performance"
    )

    st.caption(
        "Moderate Recall + Severe Recall"
    )

    components.html(
        """
<!DOCTYPE html>

<html>

<head>

<script src="https://cdn.amcharts.com/lib/5/index.js"></script>
<script src="https://cdn.amcharts.com/lib/5/xy.js"></script>
<script src="https://cdn.amcharts.com/lib/5/themes/Animated.js"></script>

</head>


<body>

<div
    id="recall_chart"
    style="width:100%; height:520px;">
</div>


<script>


var root =
    am5.Root.new(
        "recall_chart"
    );


root.setThemes([
    am5themes_Animated.new(root)
]);


var chart =
    root.container.children.push(

        am5xy.XYChart.new(
            root,
            {

                panX: false,
                panY: false,

                paddingLeft: 0,

                wheelX: "none",
                wheelY: "none",

                layout:
                    root.verticalLayout

            }
        )

    );


var legend =
    chart.children.push(

        am5.Legend.new(
            root,
            {

                centerX:
                    am5.p50,

                x:
                    am5.p50

            }
        )

    );


var data = [

{
    model: "CART",
    moderate: 64.00,
    severe: 72.00
},

{
    model: "Bagging",
    moderate: 65.00,
    severe: 73.00
},

{
    model: "Random Forest",
    moderate: 64.00,
    severe: 73.00
},

{
    model: "Extra Trees",
    moderate: 54.00,
    severe: 69.00
},

{
    model: "Gradient Boosting",
    moderate: 71.00,
    severe: 77.00
},

{
    model: "XGBoost",
    moderate: 71.00,
    severe: 79.00
},

{
    model: "LightGBM",
    moderate: 72.00,
    severe: 80.00
},

{
    model: "CatBoost",
    moderate: 71.93,
    severe: 82.29
}

];


// Y Axis (Category - Model)

var yRenderer =
    am5xy.AxisRendererY.new(
        root,
        {

            cellStartLocation:
                0.15,

            cellEndLocation:
                0.85,

            minGridDistance:
                20

        }
    );

yRenderer.set("inversed", true);


var yAxis =
    chart.yAxes.push(

        am5xy.CategoryAxis.new(
            root,
            {

                categoryField:
                    "model",

                renderer:
                    yRenderer

            }
        )

    );


yRenderer.labels.template.setAll({
    fontSize: 12
});


yAxis.data.setAll(
    data
);


// X Axis (Value)

var xAxis =
    chart.xAxes.push(

        am5xy.ValueAxis.new(
            root,
            {

                min: 0,

                max: 100,

                strictMinMax:
                    true,

                renderer:
                    am5xy.AxisRendererX.new(
                        root,
                        {
                            strokeOpacity:
                                0.1
                        }
                    )

            }
        )

    );


xAxis.set(
    "numberFormat",
    "#'%'"
);


// Series

function makeSeries(
    name,
    field,
    color
) {

    var series =
        chart.series.push(

            am5xy.ColumnSeries.new(
                root,
                {

                    name:
                        name,

                    xAxis:
                        xAxis,

                    yAxis:
                        yAxis,

                    valueXField:
                        field,

                    categoryYField:
                        "model"

                }
            )

        );


    series.set("fill", am5.color(color));
    series.set("stroke", am5.color(color));


    series.columns.template.setAll({

        tooltipText:
            "{name}\\n{categoryY}: {valueX.formatNumber('0.00')}%",

        height:
            am5.percent(80),

        strokeOpacity:
            0,

        cornerRadiusTR:
            6,

        cornerRadiusBR:
            6

    });


    series.data.setAll(
        data
    );


    series.appear();


    legend.data.push(
        series
    );

}


makeSeries(
    "Moderate Recall",
    "moderate",
    0xf59e0b
);

makeSeries(
    "Severe Recall",
    "severe",
    0xdc2626
);


chart.appear(
    1000,
    100
);


</script>

</body>

</html>
        """,

        height=560,

        scrolling=False
    )


    # ========================================================
    # 2.3 FINAL MODEL SELECTION
    # ========================================================

    st.subheader("2.3 Final Model Selection")

    col_m1, col_m2, col_m3, col_m4 = st.columns(4)

    with col_m1:
        st.metric("Test Accuracy", "95.65%")

    with col_m2:
        st.metric("Severe Recall", "82.29%")

    with col_m3:
        st.metric("Categorical Handling", "Native")

    with col_m4:
        st.metric("Final Model", "CatBoost ✅")

    st.success(
        """
        **綜合 Overall Performance 與 High-Risk Detection 兩個面向來看，
        CatBoost 在 Test Accuracy 與 Severe Recall 上皆為最佳，
        並具備原生 Categorical Feature 處理能力，
        因此選定 CatBoost 作為最終部署模型。**
        """
    )

    st.divider()

    # ========================================================
    # 2.4 CONFUSION MATRIX BY MODEL
    # ========================================================

    st.subheader("2.4 Confusion Matrix by Model")

    st.write(
        """
        可選擇兩個模型，左右對照比較 Confusion Matrix。

        - 橫軸：Predicted Class
        - 縱軸：Actual Class
        - 對角線：正確分類
        - 非對角線：誤分類
        """
    )

    # 類別順序固定為：
    # [Healthy, Mild, Moderate, Severe]
    confusion_matrices = {
        "CART": [
            [10607, 224, 0, 0],
            [270, 6026, 397, 3],
            [0, 431, 1063, 166],
            [0, 13, 214, 586]
        ],
        "Bagging": [
            [10637, 194, 0, 0],
            [218, 6194, 282, 2],
            [0, 452, 1082, 126],
            [0, 5, 215, 593]
        ],
        "Random Forest": [
            [10676, 155, 0, 0],
            [230, 6197, 267, 2],
            [0, 467, 1070, 123],
            [0, 4, 214, 595]
        ],
        "Extra Trees": [
            [10414, 417, 0, 0],
            [492, 5991, 209, 4],
            [0, 640, 904, 116],
            [0, 9, 241, 563]
        ],
        "Gradient Boosting": [
            [10829, 2, 0, 0],
            [1, 6469, 226, 0],
            [0, 405, 1172, 83],
            [0, 0, 184, 629]
        ],
        "XGBoost": [
            [10817, 14, 0, 0],
            [20, 6424, 252, 0],
            [0, 391, 1172, 97],
            [0, 0, 173, 640]
        ],
        "LightGBM": [
            [10813, 18, 0, 0],
            [18, 6417, 261, 0],
            [0, 373, 1187, 100],
            [0, 0, 166, 647]
        ],
        "CatBoost": [
            [10831, 0, 0, 0],
            [0, 6436, 260, 0],
            [0, 362, 1194, 104],
            [0, 0, 144, 669]
        ]
    }

    model_options = [
        "CART",
        "Bagging",
        "Random Forest",
        "Extra Trees",
        "Gradient Boosting",
        "XGBoost",
        "LightGBM",
        "CatBoost"
    ]

    labels = ["Healthy", "Mild", "Moderate", "Severe"]

    def render_confusion_matrix(container, model_name, div_id):
        """在指定的 container（st.columns 的其中一格）畫出單一模型的 Confusion Matrix"""

        cm = confusion_matrices[model_name]

        heatmap_data = []
        for i, actual in enumerate(labels):
            for j, predicted in enumerate(labels):
                heatmap_data.append({
                    "actual": actual,
                    "predicted": predicted,
                    "value": int(cm[i][j])
                })

        data_json = json.dumps(heatmap_data)

        total_count = sum(sum(row) for row in cm)
        correct_count = sum(cm[i][i] for i in range(4))
        cm_accuracy = correct_count / total_count if total_count > 0 else 0

        with container:
            st.caption(
                f"{model_name} Accuracy from Confusion Matrix: {cm_accuracy:.2%}"
            )

            components.html(
                f"""
<!DOCTYPE html>
<html>
<head>
    <script src="https://cdn.amcharts.com/lib/5/index.js"></script>
    <script src="https://cdn.amcharts.com/lib/5/xy.js"></script>
    <script src="https://cdn.amcharts.com/lib/5/themes/Animated.js"></script>
</head>
<body>

<div id="{div_id}" style="width:100%; height:520px;"></div>

<script>

var root = am5.Root.new("{div_id}");

root.setThemes([
    am5themes_Animated.new(root)
]);

var chart = root.container.children.push(
    am5xy.XYChart.new(root, {{
        panX: false,
        panY: false,
        wheelX: "none",
        wheelY: "none",
        paddingLeft: 10,
        paddingRight: 10,
        paddingTop: 20,
        paddingBottom: 10
    }})
);

var data = {data_json};

var classes = [
    {{ category: "Healthy" }},
    {{ category: "Mild" }},
    {{ category: "Moderate" }},
    {{ category: "Severe" }}
];

var xRenderer = am5xy.AxisRendererX.new(root, {{
    minGridDistance: 40
}});

var xAxis = chart.xAxes.push(
    am5xy.CategoryAxis.new(root, {{
        categoryField: "category",
        renderer: xRenderer
    }})
);

xAxis.data.setAll(classes);

xRenderer.labels.template.setAll({{
    fontSize: 12,
    fontWeight: "500"
}});

var yRenderer = am5xy.AxisRendererY.new(root, {{
    inversed: true
}});

var yAxis = chart.yAxes.push(
    am5xy.CategoryAxis.new(root, {{
        categoryField: "category",
        renderer: yRenderer
    }})
);

yAxis.data.setAll(classes);

yRenderer.labels.template.setAll({{
    fontSize: 12,
    fontWeight: "500"
}});

var series = chart.series.push(
    am5xy.ColumnSeries.new(root, {{
        calculateAggregates: true,
        xAxis: xAxis,
        yAxis: yAxis,
        categoryXField: "predicted",
        categoryYField: "actual",
        valueField: "value",
        clustered: false
    }})
);

series.columns.template.setAll({{
    width: am5.percent(95),
    height: am5.percent(95),
    strokeWidth: 2,
    strokeOpacity: 0.15,
    cornerRadiusTL: 8,
    cornerRadiusTR: 8,
    cornerRadiusBL: 8,
    cornerRadiusBR: 8,
    tooltipText: "Actual: {{actual}}\\nPredicted: {{predicted}}\\nCount: {{value}}"
}});

series.set("heatRules", [{{
    target: series.columns.template,
    dataField: "value",
    min: am5.color(0xf5f3ff),
    max: am5.color(0x7c3aed),
    key: "fill"
}}]);

series.bullets.push(function() {{
    return am5.Bullet.new(root, {{
        sprite: am5.Label.new(root, {{
            text: "{{value}}",
            populateText: true,
            centerX: am5.p50,
            centerY: am5.p50,
            fontSize: 14,
            fontWeight: "600",
            fill: am5.color(0x000000)
        }})
    }});
}});

series.data.setAll(data);

series.columns.template.states.create("hover", {{
    scale: 1.05,
    strokeWidth: 3,
    strokeOpacity: 1
}});

chart.children.unshift(
    am5.Label.new(root, {{
        text: "{model_name} Confusion Matrix",
        fontSize: 16,
        fontWeight: "600",
        centerX: am5.p50,
        x: am5.p50,
        paddingBottom: 12
    }})
);

series.appear(1000);
chart.appear(1000, 100);

</script>
</body>
</html>
                """,
                height=550,
                scrolling=False
            )

    col_left, col_right = st.columns(2)

    with col_left:
        model_left = st.selectbox(
            "左邊模型",
            model_options,
            index=6,
            key="cm_model_left"
        )

    with col_right:
        # 右邊預設選跟左邊不同的模型，方便一開始就能看出差異
        default_right_index = 7 if model_options[7] != model_left else 6
        model_right = st.selectbox(
            "右邊模型",
            model_options,
            index=default_right_index,
            key="cm_model_right"
        )

    col_left_chart, col_right_chart = st.columns(2)

    render_confusion_matrix(col_left_chart, model_left, "cm_chart_left")
    render_confusion_matrix(col_right_chart, model_right, "cm_chart_right")

    # ========================================================
    # 1-3 FEATURE SELECTION
    # ========================================================

    st.header(
        "3. Feature Selection"
    )

    st.write(
        """
        使用 CatBoost Feature Importance
        對原始 30 個 Features 進行排序。

        前 6個 Features 已占約 80% 的 Feature Importance，
        因此進一步比較：

        Top6、Top9、Top12、Top15 與 Full30。
        """
    )


    # ========================================================
    # FEATURE IMPORTANCE PARETO
    # ========================================================

    st.subheader(
        "3.1 Top 15 Feature Importance & Cumulative Contribution"
    )

    st.caption(
        "左側 Y 軸為各 Feature 的 Importance (%)；右側 Y 軸為相對全部 30 個 Features 的累積重要比例。"
        "累積線標示 Top 6、Top 9、Top 12 與 Top 15 的累積比例。"
    )

    components.html(
        """
<!DOCTYPE html>
<html>
<head>
<script src="https://cdn.amcharts.com/lib/5/index.js"></script>
<script src="https://cdn.amcharts.com/lib/5/xy.js"></script>
<script src="https://cdn.amcharts.com/lib/5/themes/Animated.js"></script>
</head>

<body>

<div id="feature_chart" style="width:100%; height:590px;"></div>

<script>

var root = am5.Root.new("feature_chart");

root.setThemes([
    am5themes_Animated.new(root)
]);

var chart = root.container.children.push(
    am5xy.XYChart.new(root, {
        panX: false,
        panY: false,
        wheelX: "none",
        wheelY: "none",
        paddingLeft: 0,
        paddingRight: 28,
        layout: root.verticalLayout
    })
);


// ============================================================
// Top 15 Feature Importance
// 保留原始 importance，不將 Top15 重新 normalize 成 100%
// ============================================================

var data = [
    { feature: "Mental Health",   importance: 18.601980 },
    { feature: "Sleep Hours",     importance: 18.435356 },
    { feature: "BMI",             importance: 12.313130 },
    { feature: "Wake Episodes",   importance: 11.156521 },
    { feature: "Stress",          importance: 10.040040 },
    { feature: "Sleep Latency",   importance: 9.803530 },
    { feature: "Shift Work",      importance: 3.018657 },
    { feature: "Cognitive Score", importance: 3.011662 },
    { feature: "Sleep Quality",   importance: 2.996818 },
    { feature: "Alcohol",         importance: 2.530269 },
    { feature: "REM %",           importance: 2.413257 },
    { feature: "Age",             importance: 1.961132 },
    { feature: "Caffeine",        importance: 1.061179 },
    { feature: "Weekend Diff",    importance: 0.321348 },
    { feature: "Screen Time",     importance: 0.283907 }
];


// ============================================================
// Cumulative Importance
// ============================================================

var sum = 0;

for (var i = 0; i < data.length; i++) {

    sum += data[i].importance;
    data[i].cumulative = sum;

    if (i === 5) {
        data[i].markerLabel = "Top 6\\n80.35%";
        data[i].labelDx = 0;
    }
    else if (i === 8) {
        data[i].markerLabel = "Top 9\\n89.38%";
        data[i].labelDx = 0;
    }
    else if (i === 11) {
        data[i].markerLabel = "Top 12\\n96.28%";
        data[i].labelDx = -5;
    }
    else if (i === 14) {
        data[i].markerLabel = "Top 15\\n97.95%";
        data[i].labelDx = -38;
    }
}


// ============================================================
// X Axis
// ============================================================

var xRenderer = am5xy.AxisRendererX.new(root, {
    minGridDistance: 10
});

xRenderer.labels.template.setAll({
    paddingTop: 12,
    rotation: -45,
    centerY: am5.p50,
    centerX: am5.p100,
    fontSize: 10
});

var xAxis = chart.xAxes.push(
    am5xy.CategoryAxis.new(root, {
        categoryField: "feature",
        renderer: xRenderer
    })
);

xAxis.data.setAll(data);


// ============================================================
// Left Y Axis: Feature Importance (%)
// ============================================================

var leftRenderer = am5xy.AxisRendererY.new(root, {
    strokeOpacity: 0.1
});

var yAxis = chart.yAxes.push(
    am5xy.ValueAxis.new(root, {
        min: 0,
        max: 20,
        strictMinMax: true,
        numberFormat: "#'%'",
        renderer: leftRenderer
    })
);


// ============================================================
// Right Y Axis: Cumulative Importance (%)
// ============================================================

var paretoRenderer = am5xy.AxisRendererY.new(root, {
    opposite: true,
    strokeOpacity: 0.1
});

var paretoAxis = chart.yAxes.push(
    am5xy.ValueAxis.new(root, {
        min: 0,
        max: 100,
        strictMinMax: true,
        numberFormat: "#'%'",
        renderer: paretoRenderer
    })
);

paretoRenderer.grid.template.setAll({
    forceHidden: true
});


// ============================================================
// Feature Importance Columns
// ============================================================

var series = chart.series.push(
    am5xy.ColumnSeries.new(root, {
        name: "Feature Importance",
        xAxis: xAxis,
        yAxis: yAxis,
        valueYField: "importance",
        categoryXField: "feature"
    })
);

series.set("fill", am5.color(0x67b4d8));
series.set("stroke", am5.color(0x67b4d8));

series.columns.template.setAll({
    width: am5.percent(72),
    tooltipText: "{categoryX}\\nImportance: {valueY.formatNumber('0.00')}%",
    strokeOpacity: 0,
    cornerRadiusTL: 6,
    cornerRadiusTR: 6
});

series.data.setAll(data);


// ============================================================
// Cumulative Importance Line
// ============================================================

var paretoSeries = chart.series.push(
    am5xy.LineSeries.new(root, {
        name: "Cumulative Importance",
        xAxis: xAxis,
        yAxis: paretoAxis,
        valueYField: "cumulative",
        categoryXField: "feature",
        stroke: am5.color(0x3568c0),
        fill: am5.color(0x3568c0),
        maskBullets: false,
        tooltip: am5.Tooltip.new(root, {
            labelText: "{categoryX}\\nCumulative: {valueY.formatNumber('0.00')}%"
        })
    })
);

paretoSeries.strokes.template.setAll({
    strokeWidth: 3
});


// 所有累積節點
paretoSeries.bullets.push(function() {

    return am5.Bullet.new(root, {
        sprite: am5.Circle.new(root, {
            radius: 5,
            fill: am5.color(0x3568c0),
            stroke: am5.color(0xffffff),
            strokeWidth: 2
        })
    });

});


// Top 6 / Top 9 / Top 12 / Top 15 標籤
paretoSeries.bullets.push(function(root, series, dataItem) {

    var ctx = dataItem.dataContext;

    if (!ctx.markerLabel) {
        return undefined;
    }

    return am5.Bullet.new(root, {
        sprite: am5.Label.new(root, {
            text: ctx.markerLabel,
            textAlign: "center",
            centerX: am5.p50,
            centerY: am5.p100,
            dx: ctx.labelDx || 0,
            dy: 10,
            fontSize: 11,
            fontWeight: "600",
            fill: am5.color(0x1e4f9a),

            background: am5.RoundedRectangle.new(root, {
                fill: am5.color(0xffffff),
                fillOpacity: 0.97,
                stroke: am5.color(0x9dbcea),
                strokeOpacity: 1,
                cornerRadiusTL: 7,
                cornerRadiusTR: 7,
                cornerRadiusBL: 7,
                cornerRadiusBR: 7
            }),

            paddingLeft: 7,
            paddingRight: 7,
            paddingTop: 4,
            paddingBottom: 4
        })
    });

});

paretoSeries.data.setAll(data);


// ============================================================
// Legend
// ============================================================

var legend = chart.children.push(
    am5.Legend.new(root, {
        centerX: am5.p50,
        x: am5.p50,
        marginTop: 12
    })
);

legend.data.setAll([
    series,
    paretoSeries
]);


series.appear(1000);
paretoSeries.appear(1000);
chart.appear(1000, 100);

</script>

</body>
</html>
        """,
        height=640,
        scrolling=False
    )


    # ========================================================
    # FEATURE SET COMPARISON TABLE
    # ========================================================

    st.subheader(
        "3.2 Feature Set Comparison"
    )

    feature_results = pd.DataFrame({

        "Feature Set": [
            "Top6",
            "Top9",
            "Top12",
            "Top15",
            "Full30"
        ],

        "Features": [
            6,
            9,
            12,
            15,
            30
        ],

        "Test Accuracy": [
            0.9097,
            0.9254,
            0.9526,
            0.9564,
            0.9565
        ],

        "Macro F1": [
            0.8305,
            0.8481,
            0.8772,
            0.8819,
            0.8826
        ],

        "Moderate Recall": [
            0.7024,
            0.7030,
            0.7229,
            0.7241,
            0.7193
        ],

        "Severe Recall": [
            0.7392,
            0.7835,
            0.8278,
            0.8192,
            0.8229
        ]

    })


    st.dataframe(

        feature_results.style.format({

            "Test Accuracy":
                "{:.2%}",

            "Macro F1":
                "{:.2%}",

            "Moderate Recall":
                "{:.2%}",

            "Severe Recall":
                "{:.2%}"

        }),

        width="stretch",

        hide_index=True
    )


    st.success(
        """
        Top15 的 Test Accuracy 為 95.64%，
        幾乎與 Full30 的 95.65% 相同，
        但 Features 數量由 30 降為 15。
        """
    )

    st.divider()

    # ========================================================
    # 3.3 DEPLOYMENT-READY FEATURE SET
    # ========================================================

    st.subheader("3.3 Deployment-Ready Feature Set (Web13)")

   

    st.caption(
        "由 Top15 模型層級特徵集出發，再考量使用者資料可取得性，"
        "移除兩個不易取得的特徵，形成適合 Web Deployment 的 13-feature subset。"
    )

    col_remove1, col_remove2 = st.columns(2)

    with col_remove1:
        st.warning(
            """
            **Cognitive Performance Score**

            一般使用者通常沒有標準化認知測驗結果，
            不適合作為日常 Web 輸入。
            """
        )

    with col_remove2:
        st.warning(
            """
            **REM Percentage**

            通常需要穿戴式裝置或睡眠監測設備取得，
            一般使用者不一定能提供。
            """
        )

    st.success(
        """
        **Top15 → Usability Filter → Web13**

        移除上述 2 個低可取得性 Features 後，
        最終形成適合實際部署的 **13 個 Features**。
        """
    )

    st.markdown("#### Web13 Model Performance")

    col_w1, col_w2, col_w3, col_w4 = st.columns(4)

    with col_w1:
        st.metric("Test Accuracy", "94.89%")

    with col_w2:
        st.metric("Macro F1", "87.28%")

    with col_w3:
        st.metric("Moderate Recall", "72.47%")

    with col_w4:
        st.metric("Severe Recall", "82.04%")

    st.info(
        """
        **Web13 vs Full30**

        Web13（13 個 Features）相較 Full30，
        **Test Accuracy 僅下降約 0.76 個百分點**，
        而 **Severe Recall 幾乎不變**。

        代表移除 2 個一般使用者不容易取得的 Features 後，
        模型表現仍維持在高水準，
        同時兼顧 **Usability** 與 **Performance**。
        """
    )



    st.divider()

    # ========================================================
    # 4. CONCLUSION / PROJECT SUMMARY
    # ========================================================

    st.header("4. Conclusion / Project Summary")

    st.success(
        """
        本專題以 **100,000 筆睡眠資料** 建立四分類睡眠障礙風險預測模型，
        並比較 **8 種 Tree-based Machine Learning Models**。

        最終選擇 **CatBoost** 作為核心模型，
        因為其在整體分類表現與 **Severe 高風險辨識能力** 之間取得較佳平衡。

        在 Feature Selection 階段，進一步由 **Full 30 Features → Top15**，
        再考量實際使用者資料可取得性，形成 **13 個特徵的
        Deployment-Ready Feature Set（Web13）**。

        最終將 **Web13 CatBoost** 模型部署為 **Streamlit Web Application**，
        讓使用者可直接輸入日常可取得資訊並進行風險預測。
        """
    )

# ============================================================
# PAGE 2
# INPUT + PREDICTION
# ============================================================

elif page == "2. 輸入與預測":

    st.title(
        "🌙 Sleep Disorder Risk Prediction"
    )

    st.write(
        "請輸入您的生活與睡眠資訊，系統將使用 CatBoost 預測睡眠障礙風險。"
    )

    components.html(
        """
<!DOCTYPE html>
<html>
<head>
<style>
    body { margin:0; padding:0; font-family: -apple-system, "Segoe UI", "Microsoft JhengHei", sans-serif; }
</style>
</head>
<body>

<svg viewBox="0 0 1000 110" width="100%" xmlns="http://www.w3.org/2000/svg">

    <defs>
        <marker id="flow-arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto" markerUnits="strokeWidth">
            <path d="M0,0 L0,6 L9,3 z" fill="#6b7280" />
        </marker>
        <filter id="flow-shadow" x="-20%" y="-20%" width="140%" height="140%">
            <feDropShadow dx="0" dy="2" stdDeviation="2.5" flood-opacity="0.15"/>
        </filter>
    </defs>

    <line x1="215" y1="55" x2="255" y2="55" stroke="#6b7280" stroke-width="2.5" marker-end="url(#flow-arrow)"/>
    <line x1="475" y1="55" x2="515" y2="55" stroke="#6b7280" stroke-width="2.5" marker-end="url(#flow-arrow)"/>
    <line x1="675" y1="55" x2="715" y2="55" stroke="#6b7280" stroke-width="2.5" marker-end="url(#flow-arrow)"/>

    <g filter="url(#flow-shadow)">
        <rect x="10" y="20" width="205" height="70" rx="12" fill="#0f766e"/>
    </g>
    <text x="112" y="50" text-anchor="middle" fill="#ffffff" font-size="15" font-weight="700">14 User Inputs</text>
    <text x="112" y="70" text-anchor="middle" fill="#ccfbf1" font-size="11">使用者填寫的欄位</text>

    <g filter="url(#flow-shadow)">
        <rect x="260" y="20" width="215" height="70" rx="12" fill="#2563eb"/>
    </g>
    <text x="367" y="50" text-anchor="middle" fill="#ffffff" font-size="15" font-weight="700">13 Web13 Features</text>
    <text x="367" y="70" text-anchor="middle" fill="#dbeafe" font-size="11">特徵工程後的模型輸入</text>

    <g filter="url(#flow-shadow)">
        <rect x="520" y="20" width="155" height="70" rx="12" fill="#7c3aed"/>
    </g>
    <text x="597" y="50" text-anchor="middle" fill="#ffffff" font-size="15" font-weight="700">CatBoost</text>
    <text x="597" y="70" text-anchor="middle" fill="#ede9fe" font-size="11">Final Model</text>

    <g filter="url(#flow-shadow)">
        <rect x="720" y="20" width="230" height="70" rx="12" fill="#c62828"/>
    </g>
    <text x="835" y="50" text-anchor="middle" fill="#ffffff" font-size="15" font-weight="700">Risk Prediction</text>
    <text x="835" y="70" text-anchor="middle" fill="#fde8e8" font-size="11">Healthy / Mild / Moderate / Severe</text>

</svg>

</body>
</html>
        """,
        height=110,
        scrolling=False
    )


    # ========================================================
    # BASIC INFORMATION
    # ========================================================

    st.header(
        "1. 基本資料"
    )


    col1, col2 = st.columns(2)


    with col1:

        age = st.number_input(
            "年齡",
            min_value=18,
            max_value=69,
            value=33,
            step=1
        )


    with col2:

        shift_work_text = st.radio(
            "是否從事輪班工作？",
            [
                "否",
                "是"
            ]
        )


    shift_work = (
        1
        if shift_work_text == "是"
        else 0
    )


    mental_health_text = st.selectbox(
        "心理健康狀況",
        [
            "無相關狀況",
            "焦慮",
            "憂鬱",
            "焦慮與憂鬱皆有"
        ]
    )


    mental_health_map = {

        "無相關狀況":
            "Healthy",

        "焦慮":
            "Anxiety",

        "憂鬱":
            "Depression",

        "焦慮與憂鬱皆有":
            "Both"

    }


    mental_health_condition = (
        mental_health_map[
            mental_health_text
        ]
    )


    # ========================================================
    # BODY DATA
    # ========================================================

    st.header(
        "2. 身體資料"
    )


    col1, col2 = st.columns(2)


    with col1:

        height_cm = st.number_input(
            "身高（cm）",
            min_value=120.0,
            max_value=220.0,
            value=160.0,
            step=0.5
        )


    with col2:

        weight_kg = st.number_input(
            "體重（kg）",
            min_value=30.0,
            max_value=200.0,
            value=60.0,
            step=0.5
        )


    height_m = (
        height_cm / 100
    )


    bmi = (
        weight_kg
        /
        (height_m ** 2)
    )


    st.metric(
        "BMI",
        f"{bmi:.1f}"
    )


    # ========================================================
    # SLEEP DATA
    # ========================================================

    st.header(
        "3. 睡眠與壓力"
    )


    sleep_duration_hrs = st.slider(

        "平均每晚睡眠時間（小時）",

        min_value=3.0,

        max_value=10.5,

        value=6.4,

        step=0.1
    )


    wake_episodes_per_night = st.number_input(

        "平均每晚醒來幾次？",

        min_value=0,

        max_value=8,

        value=3,

        step=1
    )


    sleep_latency_mins = st.number_input(

        "躺下後通常需要多久才能睡著？（分鐘）",

        min_value=1,

        max_value=58,

        value=19,

        step=1
    )


    stress_score = st.slider(

        "最近的壓力程度",

        min_value=1.0,

        max_value=10.0,

        value=5.8,

        step=0.1
    )


    sleep_quality_score = st.slider(

        "最近的睡眠品質",

        min_value=1.0,

        max_value=10.0,

        value=4.9,

        step=0.1
    )


    weekend_sleep_diff_hrs = st.slider(

        "假日通常比平日多睡或少睡幾小時？",

        min_value=-1.0,

        max_value=3.0,

        value=1.2,

        step=0.1
    )


    # ========================================================
    # LIFESTYLE
    # ========================================================

    st.header(
        "4. 生活習慣"
    )


    # --------------------------------------------------------
    # Alcohol
    # --------------------------------------------------------

    st.subheader(
        "🍷 睡前飲酒"
    )


    alcohol_type = st.selectbox(

        "飲酒種類",

        [
            "無",
            "啤酒",
            "紅酒",
            "烈酒",
            "高酒精啤酒",
            "調酒"
        ]
    )


    alcohol_amount = st.number_input(

        "飲酒數量（杯 / 罐 / shot）",

        min_value=0.0,

        max_value=6.0,

        value=0.0,

        step=0.5
    )


    alcohol_unit_map = {

        "無":
            0.0,

        "啤酒":
            1.0,

        "紅酒":
            1.0,

        "烈酒":
            1.0,

        "高酒精啤酒":
            1.5,

        "調酒":
            1.5

    }


    alcohol_units_before_bed = (

        alcohol_unit_map[
            alcohol_type
        ]

        *

        alcohol_amount

    )


    alcohol_units_before_bed = min(

        alcohol_units_before_bed,

        6.0

    )


    st.write(

        f"估算 Alcohol Unit："
        f"**{alcohol_units_before_bed:.1f}**"

    )


    # --------------------------------------------------------
    # Caffeine
    # --------------------------------------------------------

    st.subheader(
        "☕ 睡前咖啡因"
    )


    caffeine_type = st.selectbox(

        "含咖啡因飲品",

        [
            "無",
            "綠茶",
            "紅茶",
            "可樂",
            "美式咖啡",
            "拿鐵",
            "濃縮咖啡",
            "能量飲料"
        ]
    )


    caffeine_amount = st.number_input(

        "數量（杯 / 罐 / shot）",

        min_value=0.0,

        max_value=5.0,

        value=0.0,

        step=0.5
    )


    caffeine_map = {

        "無":
            0,

        "綠茶":
            30,

        "紅茶":
            45,

        "可樂":
            35,

        "美式咖啡":
            100,

        "拿鐵":
            80,

        "濃縮咖啡":
            65,

        "能量飲料":
            80

    }


    caffeine_mg_before_bed = (

        caffeine_map[
            caffeine_type
        ]

        *

        caffeine_amount

    )


    caffeine_mg_before_bed = min(

        caffeine_mg_before_bed,

        400

    )


    st.write(

        f"估算咖啡因攝取量："
        f"**{caffeine_mg_before_bed:.0f} mg**"

    )


    # --------------------------------------------------------
    # Screen Time
    # --------------------------------------------------------

    screen_time_before_bed_mins = st.slider(

        "睡前使用手機、平板、電腦或電視多久？（分鐘）",

        min_value=2,

        max_value=180,

        value=51,

        step=1
    )


    # ========================================================
    # CREATE INPUT DATA
    # ========================================================

    input_data = pd.DataFrame([{

        "mental_health_condition":
            mental_health_condition,

        "sleep_duration_hrs":
            sleep_duration_hrs,

        "bmi":
            bmi,

        "wake_episodes_per_night":
            wake_episodes_per_night,

        "stress_score":
            stress_score,

        "sleep_latency_mins":
            sleep_latency_mins,

        "shift_work":
            shift_work,

        "sleep_quality_score":
            sleep_quality_score,

        "alcohol_units_before_bed":
            alcohol_units_before_bed,

        "age":
            age,

        "caffeine_mg_before_bed":
            caffeine_mg_before_bed,

        "weekend_sleep_diff_hrs":
            weekend_sleep_diff_hrs,

        "screen_time_before_bed_mins":
            screen_time_before_bed_mins

    }])


    # ========================================================
    # MODEL INPUT PREVIEW
    # ========================================================

    with st.expander(
        "🔎 查看 13 個 Model Features"
    ):

        st.dataframe(

            input_data,

            width="stretch"

        )


    # ========================================================
    # PREDICTION
    # ========================================================

    st.divider()


    if st.button(

        "🌙 開始預測",

        type="primary",

        width="stretch"

    ):


        prediction = model.predict(
            input_data
        )


        probabilities = model.predict_proba(
            input_data
        )


        predicted_class = (
            prediction[0][0]
        )


        prob = (
            probabilities[0]
        )


        confidence = max(
            prob
        )


        # ----------------------------------------------------
        # Result
        # ----------------------------------------------------

        st.header(
            "預測結果"
        )


        if predicted_class == "Healthy":

            st.success(
                "🟢 預測風險等級：Healthy"
            )


        elif predicted_class == "Mild":

            st.info(
                "🟡 預測風險等級：Mild"
            )


        elif predicted_class == "Moderate":

            st.warning(
                "🟠 預測風險等級：Moderate"
            )


        elif predicted_class == "Severe":

            st.error(
                "🔴 預測風險等級：Severe"
            )


        # ----------------------------------------------------
        # Confidence
        # ----------------------------------------------------

        st.metric(

            "模型信心分數",

            f"{confidence:.2%}"

        )


        # ----------------------------------------------------
        # Probability
        # ----------------------------------------------------

        probability_df = pd.DataFrame({

            "Risk Level":
                model.classes_,

            "Probability":
                prob

        })


        st.subheader(
            "各風險類別預測分數"
        )


        st.dataframe(

            probability_df.style.format({

                "Probability":
                    "{:.2%}"

            }),

            width="stretch",

            hide_index=True

        )


        # ----------------------------------------------------
        # Disclaimer
        # ----------------------------------------------------

        st.caption(
            "⚠️ 本結果為機器學習模型之風險預測，"
            "僅供專題展示與風險參考，不代表醫療診斷。"
        )