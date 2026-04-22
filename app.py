import streamlit as st
import joblib
import numpy as np

st.set_page_config(page_title="妊娠期糖尿病风险预测", page_icon="🏥")

# ---------- 加载模型 ----------
@st.cache_resource
def load_model():
    model = joblib.load("./models/best_model.pkl")
    scaler = joblib.load("./models/scaler.pkl")

    if hasattr(scaler, 'feature_names_in_'):
        feature_names = list(scaler.feature_names_in_)
        print("✅ 从 scaler 读取到特征名:", feature_names)
    else:
        feature_names = [
            "Age", "No of Pregnancy", "Previous Pregnancy Outcome", "BMI", "HDL",
            "Family History", "unexplained prenetal loss", "Large Child or Birth Default",
            "PCOS", "Sys BP", "Dia BP", "Hemoglobin", "Sedentary Lifestyle", "Prediabetes",
            "Pulse_Pressure", "MAP", "Age_squared", "BMI_squared", "Age_BMI_interaction",
            "SBP_BMI_interaction", "BMI_FamilyHistory", "Age_FamilyHistory", "PCOS_BMI",
            "GDM_Risk_Score"
        ]
        print("⚠️ scaler 无特征名，使用预设列表（已移除标签列）")

    label_col = "Class Label(GDM /Non GDM)"
    if label_col in feature_names:
        feature_names.remove(label_col)

    print(f"最终特征数量: {len(feature_names)}")
    return model, scaler, feature_names

model, scaler, FEATURE_ORDER = load_model()

# ---------- 界面 ----------
st.title("🏥 妊娠期糖尿病风险预测系统")
st.markdown("请填写以下基本信息（带 * 为必填）")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("🔢 年龄 (岁)", min_value=18, max_value=50, value=28, step=1, help="建议范围：18~45岁")
    pregnancies = st.number_input("🔢 怀孕次数", min_value=0, max_value=20, value=1, step=1)
    bmi = st.number_input("🔢 BMI 指数 (kg/m²)", min_value=15.0, max_value=50.0, value=22.0, step=0.1, help="建议范围：18.5~24.9")
    hdl = st.number_input("🔢 高密度脂蛋白 HDL (mg/dL)", min_value=20.0, max_value=120.0, value=60.0, step=1.0, help="建议范围：40~90")
    sys_bp = st.number_input("🔢 收缩压/高压 (mmHg)", min_value=70, max_value=200, value=110, step=1, help="建议范围：90~140")
    dia_bp = st.number_input("🔢 舒张压/低压 (mmHg)", min_value=40, max_value=120, value=70, step=1, help="建议范围：60~90")
    hemoglobin = st.number_input("🔢 血红蛋白 (g/dL)", min_value=8.0, max_value=18.0, value=12.0, step=0.1, help="建议范围：11~15")

with col2:
    pregnancies_previous = st.number_input(
        "🔢 既往分娩次数（产次）",
        min_value=0, max_value=2, value=0, step=1,
        help="过去达到可存活孕周（>28周）并分娩的次数。通常范围为0~2次"
    )
    family_history = st.radio("📌 糖尿病家族史", options=["否", "是"], horizontal=True)
    unexplained_loss = st.radio("📌 不明原因体重下降", options=["否", "是"], horizontal=True)
    large_child = st.radio("📌 既往巨大儿史", options=["否", "是"], horizontal=True)
    pcos = st.radio("📌 多囊卵巢综合征 (PCOS)", options=["否", "是"], horizontal=True)
    sedentary = st.radio("📌 久坐生活方式", options=["否", "是"], horizontal=True)
    prediabetes = st.radio("📌 糖尿病前期", options=["否", "是"], horizontal=True)

# ---------- 构建输入字典（动态匹配 FEATURE_ORDER 中的键名）----------
input_dict = {}

# 1. 基础连续特征（键名与训练一致）
input_dict["Age"] = float(age)
input_dict["No of Pregnancy"] = float(pregnancies)
input_dict["BMI"] = float(bmi)
input_dict["HDL"] = float(hdl)
input_dict["Sys BP"] = float(sys_bp)
input_dict["Dia BP"] = float(dia_bp)
input_dict["Hemoglobin"] = float(hemoglobin)

# 2. 既往分娩次数（智能匹配中英文键名）
preg_key_candidates = ["Previous Pregnancy Outcome", "前次怀孕的妊娠情况", "Gestation in previous Pregnancy"]
preg_key = next((k for k in FEATURE_ORDER if k in preg_key_candidates), None)
if preg_key is None:
    st.error("❌ 无法在特征列表中找到既往妊娠情况对应的列名，请联系管理员。")
    st.stop()
input_dict[preg_key] = float(pregnancies_previous)

# 3. 二分类特征（智能匹配中英文键名）
binary_mappings = {
    "Family History": (["Family History", "家族历史"], 1.0 if family_history == "是" else 0.0),
    "unexplained prenetal loss": (["unexplained prenetal loss", "无法解释的非净量损失"], 1.0 if unexplained_loss == "是" else 0.0),
    "Large Child or Birth Default": (["Large Child or Birth Default", "大子女或出生默认"], 1.0 if large_child == "是" else 0.0),
    "PCOS": (["PCOS", "多囊卵巢综合征"], 1.0 if pcos == "是" else 0.0),
    "Sedentary Lifestyle": (["Sedentary Lifestyle", "久坐生活方式"], 1.0 if sedentary == "是" else 0.0),
    "Prediabetes": (["Prediabetes", "糖尿病前期"], 1.0 if prediabetes == "是" else 0.0)
}

for eng_key, (candidates, value) in binary_mappings.items():
    actual_key = next((k for k in FEATURE_ORDER if k in candidates), eng_key)
    input_dict[actual_key] = value

# 4. 衍生特征（固定英文键名，训练时构造的）
input_dict["Pulse_Pressure"] = input_dict["Sys BP"] - input_dict["Dia BP"]
input_dict["MAP"] = input_dict["Dia BP"] + (input_dict["Pulse_Pressure"] / 3.0)
input_dict["Age_squared"] = input_dict["Age"] ** 2
input_dict["BMI_squared"] = input_dict["BMI"] ** 2
input_dict["Age_BMI_interaction"] = input_dict["Age"] * input_dict["BMI"]
input_dict["SBP_BMI_interaction"] = input_dict["Sys BP"] * input_dict["BMI"]

# 获取家族史、PCOS、糖尿病前期的实际键名
fh_key = next((k for k in FEATURE_ORDER if k in ["Family History", "家族历史"]), "Family History")
pcos_key = next((k for k in FEATURE_ORDER if k in ["PCOS", "多囊卵巢综合征"]), "PCOS")
prediabetes_key = next((k for k in FEATURE_ORDER if k in ["Prediabetes", "糖尿病前期"]), "Prediabetes")

input_dict["BMI_FamilyHistory"] = input_dict["BMI"] * input_dict[fh_key]
input_dict["Age_FamilyHistory"] = input_dict["Age"] * input_dict[fh_key]
input_dict["PCOS_BMI"] = input_dict[pcos_key] * input_dict["BMI"]

# 5. GDM 风险综合评分
risk_score = 0
if age < 25: risk_score += 0
elif age < 30: risk_score += 1
elif age < 35: risk_score += 2
else: risk_score += 3

if bmi < 24: risk_score += 0
elif bmi < 28: risk_score += 1
else: risk_score += 2

risk_score += input_dict[fh_key] * 2
risk_score += input_dict[pcos_key] * 2
risk_score += input_dict[prediabetes_key] * 2
input_dict["GDM_Risk_Score"] = float(risk_score)

# ---------- 预测 ----------
if st.button("开始预测", type="primary", use_container_width=True):
    feature_values = [input_dict[feat] for feat in FEATURE_ORDER]
    X_input = np.array([feature_values])
    X_scaled = scaler.transform(X_input)
    prob = model.predict_proba(X_scaled)[0][1]

    st.divider()
    st.metric("患病概率", f"{prob:.1%}")
    if prob >= 0.5:
        st.error("⚠️ 高风险，建议进一步进行口服葡萄糖耐量试验（OGTT）")
    else:
        st.success("✅ 低风险，继续保持健康生活方式并定期产检")

# ---------- 调试信息 ----------
with st.expander("🔧 调试信息（查看输入特征向量）"):
    st.write("特征顺序:", FEATURE_ORDER)
    st.write("特征数量:", len(FEATURE_ORDER))
    if 'feature_values' in locals():
        st.write("输入值:", feature_values)
    else:
        st.write("尚未进行预测")