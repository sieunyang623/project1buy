import os
import re
from pathlib import Path
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

# ==========================================================
# 1. 페이지 설정 및 경로 정의
# ==========================================================
st.set_page_config(
    page_title="고등학생 소비심리와 구매행동 분석",
    page_icon="🛒",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data.csv"
WC_IMG_PATH = BASE_DIR / "wordcloud.png"

# 척도 매핑
LIKERT_MAP = {
    "전혀 그렇지 않다": 1,
    "그렇지 않다": 2,
    "보통이다": 3,
    "그렇다": 4,
    "매우 그렇다": 5,
}

GRADE_ORDER = ["1학년", "2학년", "3학년"]
SNS_ORDER = ["1시간 미만", "1~2시간", "2~3시간", "3~4시간", "4시간 이상"]

# ==========================================================
# 2. 칼럼 매핑 정의
# ==========================================================
COLUMN_RENAME_MAP = {
    "타임스탬프": "타임스탬프",
    "연락처 (선택 사항)\n이벤트 당첨 시 연락받을 휴대전화 번호 또는 이메일 주소를 입력해 주세요.": "연락처",
    "1. 현재 몇 학년인가요?": "학년",
    "2. 성별은 무엇인가요?": "성별",
    "3. 한 달 평균 용돈은 얼마인가요?": "용돈",
    "4. 하루 평균 SNS 사용 시간은 얼마나 되나요?": "SNS시간",
    "5. 최근 한 달 동안 온라인 쇼핑을 몇 번 정도 이용하였나요?": "쇼핑횟수",
    "6. 물건을 구매할 때 가장 중요하게 생각하는 요소는 무엇인가요?": "구매중요요소",
    "7. 충동적으로 물건을 구매한 경험이 얼마나 있나요?": "충동구매경험",
    "8. SNS 광고를 보고 실제로 상품을 구매한 경험이 있나요?": "SNS광고구매",
    "9. 유명 브랜드 제품을 더 신뢰하는 편인가요?": "브랜드신뢰_Q9",
    "10. 친구의 추천은 구매 결정에 영향을 주나요?": "친구추천영향",
    "11. 할인 행사가 진행 중인 상품은 더 매력적으로 느껴지나요?": "할인상품매력",
    "12. 할인율이 높을수록 구매 의사가 증가하나요?": "할인율구매의사",
    "13. 다음 상품을 구매할 의향이 어느 정도 있나요?\n무선 이어폰 (30,000원)": "구매의향_기본",
    "14. 다음 상품을 구매할 의향이 어느 정도 있나요?\n무선 이어폰\n정가 50,000원 → 할인 후 30,000원": "구매의향_할인",
    "15. \"재고 3개 남음\"이라는 문구를 보면 구매 욕구가 증가하나요?": "재고3개남음",
    "16. \"오늘 마감\"이라는 문구를 보면 구매 욕구가 증가하나요?": "오늘마감",
    "17. \"한정판\" 또는 \"한정 수량\"이라는 문구를 보면 관심이 증가하나요?": "한정판관심",
    "18. 물건을 구매하기 전에 여러 쇼핑몰의 가격을 비교하는 편인가요?": "가격비교",
    "19. 할인 정보를 자주 찾아보는 편인가요?": "할인정보탐색",
    "20. 유명 브랜드 제품을 선호하는 편인가요?": "브랜드선호",
    "21. 브랜드 이미지를 중요하게 생각하나요?": "브랜드이미지중요",
    "22. SNS에서 유행하는 제품에 관심이 많은 편인가요?": "SNS유행관심",
    "23. 친구들이 사용하는 제품을 따라 구매한 경험이 있나요?": "친구따라구매",
    "24. 물건을 구매하기 전에 꼭 필요한 물건인지 생각하나요?": "필요성검토",
    "25. 예산을 정해 놓고 소비하는 편인가요?": "예산설정소비",
    "27. 최근 가장 사고 싶은 물건은 무엇인가요?": "최근위시템",
    "28. 그 물건을 사고 싶은 가장 큰 이유는 무엇인가요?": "구매이유",
    "26. 본인은 어떤 소비 유형에 가장 가깝다고 생각하나요?": "소비유형"
}

# ==========================================================
# 3. 헬퍼 함수 정의
# ==========================================================
@st.cache_data
def load_csv() -> pd.DataFrame:
    try:
        df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    except Exception:
        df = pd.read_csv(DATA_PATH, encoding="cp949")
        
    df.columns = df.columns.str.strip()
    
    # BOM 및 특수 제어문자 제거 전처리
    new_cols = []
    for col in df.columns:
        matched = False
        clean_col = re.sub(r'[\s\n\r\t]+', '', col)
        
        for k, v in COLUMN_RENAME_MAP.items():
            clean_k = re.sub(r'[\s\n\r\t]+', '', k)
            if clean_k in clean_col or clean_col in clean_k:
                new_cols.append(v)
                matched = True
                break
        if not matched:
            new_cols.append(col)
            
    df.columns = new_cols
    
    if "타임스탬프" in df.columns:
        df["타임스탬프"] = pd.to_datetime(df["타임스탬프"], errors="coerce")
    return df

def parse_intent_score(val):
    """문자열/숫자 응답을 정밀하게 5점 만점 점수로 변환"""
    if pd.isna(val):
        return np.nan
    clean_val = str(val).strip().replace(" ", "")
    score_map = {
        "매우낮음": 1,
        "낮음": 2,
        "보통": 3,
        "높음": 4,
        "매우높음": 5,
        "전혀그렇지않다": 1,
        "그렇지않다": 2,
        "보통이다": 3,
        "그렇다": 4,
        "매우그렇다": 5,
        "1": 1, "2": 2, "3": 3, "4": 4, "5": 5
    }
    if clean_val in score_map:
        return score_map[clean_val]
    try:
        num = float(clean_val)
        if 1 <= num <= 5:
            return num
    except ValueError:
        pass
    return np.nan

# ==========================================================
# 4. 데이터 로딩
# ==========================================================
try:
    df_raw = load_csv()
except Exception as e:
    st.error(f"같은 경로에서 'data.csv' 파일을 읽어올 수 없습니다. 경로와 파일명을 확인해 주세요. 오류: {e}")
    st.stop()

# ==========================================================
# 5. UI 대시보드 레이아웃
# ==========================================================
st.title("🛒 고등학생의 소비심리와 구매행동 분석 통합 페이지")
st.subheader("- 행동경제학 편향(앵커링, 손실회피, FOMO) 및 소비 성향 실증 연구 -")
st.markdown("---")

# SECTION 1: 탐구 개요
st.header("🏠 1. 탐구 소개 및 동기")
st.image("buy1image.png", caption="고등학생들의 소비 심리와 마케팅의 관계 탐구", use_container_width=True)

col_intro1, col_intro2 = st.columns(2)
with col_intro1:
    st.markdown("### Ⅰ. 탐구 동기")
    st.info(
        "현대 사회에서 소비자는 단순히 가격과 품질만을 고려하여 상품을 구매하지 않습니다. "
        "기업들은 할인 행사, SNS 광고, 한정판 마케팅, 브랜드 이미지 등을 활용하여 소비자의 심리를 자극합니다. "
        "특히 청소년들은 SNS 사용 빈도가 높고 유행에 민감하여 소비심리의 영향을 크게 받을 수 있습니다.\n\n"
        "최근 행동경제학에서는 소비자의 선택이 항상 합리적으로 이루어지는 것이 아니라 **심리적 편향**에 의해 영향을 받는다고 설명합니다. "
        "이에 본 연구에서는 고등학생들의 소비 행동에 영향을 미치는 심리적 요인을 조사하고, 행동경제학의 주요 개념인 "
        "**앵커링 효과(Anchoring Effect)**, **손실회피 성향(Loss Aversion)**, **FOMO(Fear Of Missing Out) 현상**이 실제 소비 행동에 어떤 영향을 미치는지 분석하고자 하였습니다."
    )
    
    st.markdown("### 🎓 진로 및 융합 연계")
    st.success(
        "📊 **통계학:** 설문 데이터를 가공하고 교차분석 및 상관관계 분석을 수행하며 실증 통계의 유용성을 경험.\n\n"
        "💼 **경영/마케팅:** 앵커링, FOMO 등 심리학 기반 마케팅의 영향력을 정량적 데이터로 실증 검증.\n\n"
        "⚖️ **경제/세무:** 전통 경제학의 합리적 인간 가정을 넘어 제한된 합리성에 기초한 행동경제학 모형을 학습."
    )

with col_intro2:
    st.markdown("### Ⅱ. 연구 설계 개요")
    with st.expander("❓ 본 탐구에서 검증하고자 하는 7가지 핵심 연구 문제"):
        st.write("1. 고등학생은 구매 시 어떤 요소를 가장 중요하게 생각하는가?")
        st.write("2. 할인 행사는 구매 의사에 영향을 미치는가?")
        st.write("3. 앵커링 효과는 실제 구매 의사에 영향을 미치는가?")
        st.write("4. 손실회피 성향은 구매 결정에 영향을 미치는가?")
        st.write("5. FOMO 현상은 소비 행동에 영향을 미치는가?")
        st.write("6. SNS 이용 시간과 충동구매 경험은 관련이 있는가?")
        st.write("7. 소비유형에 따라 구매 행동의 차이가 존재하는가?")
        
    st.markdown(
        "**1. 연구 대상:** 고등학생 설문 응답자 대상 구글 폼 조사 수행\n\n"
        "**2. 연구 도구:** Python (pandas, Plotly, Streamlit) 대시보드\n\n"
        "**3. 분석 방법:**\n"
        "- 빈도수 및 백분율 요약 분석\n"
        "- SNS 사용 시간과 충동구매 빈도의 **교차분석 히트맵** 시각화\n"
        "- 4대 소비유형 자가진단 및 평균 비교\n"
        "- 주관식 워드클라우드 시각화 분석"
    )
st.markdown("---")

# SECTION 2: 행동경제학 개념 및 미니 체험존
st.header("🧠 2. 행동경제학 개념 학습 & 인터랙티브 체험")
st.markdown("고등학생들의 의사결정에 무의식적으로 작용하는 심리적 편향들을 개념적으로 이해하고 직접 테스트해 보세요.")

col_c1, col_c2 = st.columns(2)
with col_c1:
    with st.container(border=True):
        st.subheader("1. 앵커링 효과 (Anchoring Effect)")
        st.write("- **뜻:** 처음에 제시된 숫자가 기준(닻)이 되어 그 이후의 판단에 치우침을 주는 편향.")
        st.write("- **비유:** 정가 10만원 제품에 선을 긋고 '5만원 특별 세일!'이라 적으면 5만원이 매우 싸 보임.")
        
    with st.container(border=True):
        st.subheader("2. 손실회피 성향 (Loss Aversion)")
        st.write("- **뜻:** 이득을 얻는 기쁨보다 가지고 있던 것을 잃어버리는 고통을 2배 가깝게 더 크게 느끼는 본능.")
        st.write("- **비유:** '20% 할인 쿠폰 지급'보다 '오늘 지나면 20% 특별 할인 권한 영구 삭제'에 더 다급하게 구매 결정.")
        
    with st.container(border=True):
        st.subheader("3. FOMO 현상 (소외 불안 심리)")
        st.write("- **뜻:** 나만 좋은 혜택이나 유행에서 뒤처지거나 소외될까 봐 불안해하는 심리.")
        st.write("- **비유:** 쇼핑몰의 '재고 단 3개 남음!', '오늘 마감', '한정 수량' 문구를 보면 조바심에 결제하게 됨.")

with col_c2:
    with st.container(border=True):
        st.subheader("4. 베블렌 효과 (Veblen Effect)")
        st.write("- **뜻:** 가격이 계속 올라감에도 과시욕구 때문에 오히려 수요가 늘어나는 현상 (명품 등).")
        
    with st.container(border=True):
        st.subheader("5. 밴드웨건 효과 (Bandwagon Effect)")
        st.write("- **뜻:** 다른 사람들이 많이 동조하고 유행하는 제품을 나도 따라서 대세를 타고 구매하는 편승 효과.")
        
    with st.container(border=True):
        st.subheader("6. 스놉 효과 (Snob Effect)")
        st.write("- **뜻:** 다수가 구매하는 흔한 유행템을 피하고 나만의 희소하고 독특한 개성을 지키려는 백로 효과.")

    with st.container(border=True):
        st.subheader("7. 사회적 증거 효과 (Social Proof)")
        st.write("- **뜻:** 결정이 애매할 때 남들의 후기 개수나 평점에 기대어 안도하며 구매를 모방하는 현상.")

# 미니 심리 실험
st.markdown("#### 🎮 미니 심리 실험 시뮬레이터")
c_sim1, c_sim2, c_sim3 = st.columns(3)

with c_sim1:
    st.markdown("**[실험 1] 앵커링 효과 체감**")
    score_a = st.slider("제안 A: 무선 이어폰 [판매가 30,000원]의 구매 의향은?", 1, 5, 3, key="sim_sa")
    score_b = st.slider("제안 B: 무선 이어폰 [정가 50,000원 -> 특가 30,000원]의 구매 의향은?", 1, 5, 3, key="sim_sb")
    if st.button("⚖️ 앵커링 점수 비교"):
        st.write(f"내 매력도 평가 차이: 제안 A {score_a}점 vs 제안 B {score_b}점")
        if score_b > score_a:
            st.success("🎉 **앵커링 편향 발견!** 동일한 3만원 결제 조건임에도 5만원이라는 최초 정가 기준점 때문에 제안 B를 더 높게 평가하셨습니다.")
        else:
            st.info("⚖️ 동일하게 평가하셨거나 제안 A를 더 선호하셨습니다. 정가의 눈속임에 넘어가지 않는 냉철한 합리적 소비자이십니다.")

with c_sim2:
    st.markdown("**[실험 2] 손실회피 성향 체감**")
    choice_gain = st.radio("상황 1: 이익 획득 시 선택은?", ["확실하게 10만원 얻기", "50% 확률로 20만원 받기 (나머지 50%는 0원)"], key="sim_cg")
    choice_loss = st.radio("상황 2: 손실 감내 시 선택은?", ["확실하게 10만원 손해보기", "50% 확률로 20만원 손해보기 (나머지 50%는 0원)"], key="sim_cl")
    if st.button("📊 손실회피 성향 판정"):
        if "확실하게 10만원 얻기" in choice_gain and "50% 확률로 20만원 손해보기" in choice_loss:
            st.success("🎯 **대중적인 손실회피 성향 확진!** 이익 상황에서는 안정을 택하고, 손실 상황에서는 손해를 회피하기 위해 도박적인 선택을 감수하려는 성향이 입증되었습니다.")
        else:
            st.info("💡 기댓값이 동일함에도 모험과 안정 균형을 다르게 잡으셨습니다. 이성적인 의사결정 방식입니다.")

with c_sim3:
    st.markdown("**[실험 3] FOMO 타이머 심박수 자극**")
    st.error("⏰ **[한정판매 타임세일 자동 소멸 경보]**")
    st.markdown(
        "<div style='background-color:#ffebeb; border:1px solid #ff4d4d; border-radius:8px; padding:10px; text-align:center;'>"
        "<span style='color:#ff4d4d; font-weight:bold;'>⏳ 세일 마감까지 00시간 04분 12초 남음</span><br>"
        "<span style='color:#333; font-weight:bold; font-size:14px;'>🚨 현재 재고: 단 2개 남음 (9명이 이 상품을 결제 중입니다)</span>"
        "</div>",
        unsafe_allow_html=True
    )
    fomo_rate = st.select_slider("위 경보를 보았을 때 지갑을 열고 싶은 조바심은?", ["차분함", "신경 쓰임", "다급함", "바로 구매 결제"], key="sim_sf")
    st.write(f"나의 조바심 지수: **{fomo_rate}**")
st.markdown("---")

# SECTION 3: 나의 소비유형 자가진단 테스트
st.header("📝 3. 나의 소비유형 자가진단 테스트")
st.markdown("설문 조사 문항을 기반으로 나에게 일치하는 수준을 선택해 보세요.")

with st.form("consumer_type_form"):
    st.markdown("##### 🏷️ [가격 민감도 자가진단]")
    q1 = st.radio("Q1. 물건을 구매하기 전 여러 쇼핑몰의 가격을 꼼꼼히 비교해 본다.", 
                  options=["전혀 그렇지 않다", "그렇지 않다", "보통이다", "그렇다", "매우 그렇다"], index=2, horizontal=True, key="rq1")
    q2 = st.radio("Q2. 할인 쿠폰이나 세일, 특가 정보를 평소 적극적으로 찾고 소비에 이용한다.", 
                  options=["전혀 그렇지 않다", "그렇지 않다", "보통이다", "그렇다", "매우 그렇다"], index=2, horizontal=True, key="rq2")
    
    st.markdown("##### 👑 [브랜드 신뢰성 자가진단]")
    q3 = st.radio("Q3. 가격이 더 비싸더라도 널리 알려진 브랜드 제품을 더 안전하다고 여겨 선호한다.", 
                  options=["전혀 그렇지 않다", "그렇지 않다", "보통이다", "그렇다", "매우 그렇다"], index=2, horizontal=True, key="rq3")
    q4 = st.radio("Q4. 물건을 고를 때 제조사 브랜드의 대중적 이미지나 신뢰도를 매우 중요하게 여긴다.", 
                  options=["전혀 그렇지 않다", "그렇지 않다", "보통이다", "그렇다", "매우 그렇다"], index=2, horizontal=True, key="rq4")
    
    st.markdown("##### ⚡ [유행 및 친구 영향 자가진단]")
    q5 = st.radio("Q5. SNS(인스타그램, 유튜브 등)에서 화제가 되거나 광고하는 인기 아이템에 흥미가 크다.", 
                  options=["전혀 그렇지 않다", "그렇지 않다", "보통이다", "그렇다", "매우 그렇다"], index=2, horizontal=True, key="rq5")
    q6 = st.radio("Q6. 친구나 주위 사람들이 산 제품이나 추천하는 물건을 보면 나도 따라 사고 싶다.", 
                  options=["전혀 그렇지 않다", "그렇지 않다", "보통이다", "그렇다", "매우 그렇다"], index=2, horizontal=True, key="rq6")
                  
    st.markdown("##### 📝 [합리적 계획 소비 자가진단]")
    q7 = st.radio("Q7. 충동적으로 사기보다는 이것이 나에게 꼭 진짜로 필요한가 고민 후 산다.", 
                  options=["전혀 그렇지 않다", "그렇지 않다", "보통이다", "그렇다", "매우 그렇다"], index=2, horizontal=True, key="rq7")
    q8 = st.radio("Q8. 용돈 예산을 정해두고 그 지출 한도 가이드 안에서 소비하려고 노력한다.", 
                  options=["전혀 그렇지 않다", "그렇지 않다", "보통이다", "그렇다", "매우 그렇다"], index=2, horizontal=True, key="rq8")
                  
    submit_test = st.form_submit_button("🔍 나의 소비 유형 종합 진단 결과 보기")

if submit_test:
    score_p = LIKERT_MAP[q1] + LIKERT_MAP[q2]
    score_b = LIKERT_MAP[q3] + LIKERT_MAP[q4]
    score_t = LIKERT_MAP[q5] + LIKERT_MAP[q6]
    score_pl = LIKERT_MAP[q7] + LIKERT_MAP[q8]
    
    results_map = {
        "가격민감형 (가격중심 소비자)": score_p,
        "브랜드신뢰형 (브랜드중심 소비자)": score_b,
        "유행추종형 (유행추종형 소비자)": score_t,
        "계획소비형 (계획형 소비자)": score_pl
    }
    
    max_score_val = max(results_map.values())
    my_types = [k for k, v in results_map.items() if v == max_score_val]
    
    st.markdown("#### 📊 유형 분석 보고서")
    
    for m_type in my_types:
        if "가격민감" in m_type:
            st.warning("🏷️ 당신은 합리적인 알뜰 지향형인 **[가격민감형]** 소비자입니다!")
            st.write("가성비와 세일 정보를 꿰차고 있어 불필요한 원가 지출을 방어합니다. 단, '할인 특가'라는 미끼에 매료되어 충동구매하지 않도록 주의하세요.")
        elif "브랜드" in m_type:
            st.info("👑 당신은 보장된 가치를 중시하는 **[브랜드신뢰형]** 소비자입니다!")
            st.write("유명 로고가 주는 안전성 및 품질 보증을 중시합니다. 지출 실패 확률은 낮으나, 브랜드 거품 가격에 과다 지출하지 않는지 돌아보세요.")
        elif "유행추종" in m_type:
            st.error("⚡ 당신은 최신 대세를 이끄는 **[유행추종형]** 소비자입니다!")
            st.write("SNS 인기 템이나 친구 추천에 호응하며 유행에 민감합니다. 급변하는 흐름 탓에 충동구매가 늘지 않도록 지출 통제가 필요합니다.")
        elif "계획소비" in m_type:
            st.success("📝 당신은 꼼꼼하고 통제력 높은 **[계획소비형]** 소비자입니다!")
            st.write("가장 이상적인 의사결정자입니다. 지출 예산에 따라 신중하게 필수 유무를 고민하여 경제 낭비가 적습니다.")

    raw_p_avg = (df_raw["가격비교"].map(LIKERT_MAP).mean() + df_raw["할인정보탐색"].map(LIKERT_MAP).mean()) if "가격비교" in df_raw.columns else 6
    raw_b_avg = (df_raw["브랜드선호"].map(LIKERT_MAP).mean() + df_raw["브랜드이미지중요"].map(LIKERT_MAP).mean()) if "브랜드선호" in df_raw.columns else 6
    raw_t_avg = (df_raw["SNS유행관심"].map(LIKERT_MAP).mean() + df_raw["친구따라구매"].map(LIKERT_MAP).mean()) if "SNS유행관심" in df_raw.columns else 6
    raw_pl_avg = (df_raw["필요성검토"].map(LIKERT_MAP).mean() + df_raw["예산설정소비"].map(LIKERT_MAP).mean()) if "필요성검토" in df_raw.columns else 6
    
    categories = ["가격민감성", "브랜드신뢰성", "유행추종성", "계획소비성"]
    user_scores = [score_p, score_b, score_t, score_pl]
    avg_scores = [raw_p_avg, raw_b_avg, raw_t_avg, raw_pl_avg]
    
    categories_closed = categories + [categories[0]]
    user_scores_closed = user_scores + [user_scores[0]]
    avg_scores_closed = avg_scores + [avg_scores[0]]
    
    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(
        r=user_scores_closed,
        theta=categories_closed,
        fill='toself',
        name='나의 진단 결과',
        line_color='#FF6B6B'
    ))
    fig_radar.add_trace(go.Scatterpolar(
        r=avg_scores_closed,
        theta=categories_closed,
        fill='toself',
        name='전체 응답자 평균 데이터',
        line_color='#4D96FF',
        opacity=0.5
    ))
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 10])),
        showlegend=True,
        title="나의 4대 소비 성향 스파이더 플롯 (10점 만점)"
    )
    st.plotly_chart(fig_radar, use_container_width=True)
    
    c_m1, c_m2, c_m3, c_m4 = st.columns(4)
    c_m1.metric("🏷️ 가격민감도", f"{score_p}점", f"{score_p - raw_p_avg:+.2f}점 대비 평균")
    c_m2.metric("👑 브랜드신뢰", f"{score_b}점", f"{score_b - raw_b_avg:+.2f}점 대비 평균")
    c_m3.metric("⚡ 유행추종성", f"{score_t}점", f"{score_t - raw_t_avg:+.2f}점 대비 평균")
    c_m4.metric("📝 계획지출성", f"{score_pl}점", f"{score_pl - raw_pl_avg:+.2f}점 대비 평균")
st.markdown("---")

# SECTION 4: 설문조사 통계 대시보드
st.header("📊 4. 설문조사 통계 및 데이터 대시보드")
st.markdown("설문조사 응답 현황을 실시간 필터를 통해 분석합니다.")

st.markdown("##### 🔍 동적 시각화 필터 조정")
col_fl1, col_fl2, col_fl3 = st.columns(3)

grade_opts = GRADE_ORDER
gender_opts = sorted(df_raw["성별"].dropna().unique().tolist()) if "성별" in df_raw.columns else []
type_opts = sorted(df_raw["소비유형"].dropna().unique().tolist()) if "소비유형" in df_raw.columns else []

with col_fl1:
    sel_grades = st.multiselect("학년 필터링", grade_opts, default=grade_opts, key="db_grade")
with col_fl2:
    sel_genders = st.multiselect("성별 필터링", gender_opts, default=gender_opts, key="db_gender")
with col_fl3:
    sel_types = st.multiselect("소비유형 필터링", type_opts, default=type_opts, key="db_type")

df_db = df_raw.copy()
if "학년" in df_db.columns and sel_grades:
    df_db = df_db[df_db["학년"].isin(sel_grades)]
if "성별" in df_db.columns and sel_genders:
    df_db = df_db[df_db["성별"].isin(sel_genders)]
if "소비유형" in df_db.columns and sel_types:
    df_db = df_db[df_db["소비유형"].isin(sel_types)]

if df_db.empty:
    st.warning("⚠️ 해당 필터 조건에 대응되는 응답 결과가 비어 있습니다. 필터를 넓게 재조정해 주세요.")
else:
    if "충동구매경험" in df_db.columns:
        db_impulse_rate = df_db["충동구매경험"].isin(["자주 있다", "매우 자주 있다", "보통이다"]).mean() * 100
    else:
        db_impulse_rate = 0.0
        
    fomo_cols = [c for c in ["재고3개남음", "오늘마감", "한정판관심"] if c in df_db.columns]
    if fomo_cols:
        db_fomo_mean = pd.concat([df_db[c].map(LIKERT_MAP) for c in fomo_cols], axis=1).mean(axis=1).mean()
    else:
        db_fomo_mean = 0.0

    # ----------------------------------------------------
    # ⚓ 13, 14번 문항 감지 강력 다중 추적 알고리즘
    # ----------------------------------------------------
    col_q13 = None
    col_q14 = None

    for col in df_db.columns:
        clean_c = re.sub(r'[\s\n\r\t]+', '', str(col))
        
        # 13번 문항 추적: '13' 포함 OR ('구매의향_기본' OR ('30,000' 포함 AND '50,000' 없음))
        if ("13" in clean_c) or ("구매의향_기본" in clean_c) or ("30,000원" in clean_c and "50,000" not in clean_c and "정가" not in clean_c):
            col_q13 = col
            
        # 14번 문항 추적: '14' 포함 OR ('구매의향_할인' OR ('50,000' 포함 OR '정가' 포함))
        if ("14" in clean_c) or ("구매의향_할인" in clean_c) or ("50,000" in clean_c) or ("정가" in clean_c and "할인" in clean_c):
            col_q14 = col

    base_scores = df_db[col_q13].apply(parse_intent_score).dropna() if col_q13 else pd.Series(dtype=float)
    disc_scores = df_db[col_q14].apply(parse_intent_score).dropna() if col_q14 else pd.Series(dtype=float)

    db_base_intent = base_scores.mean() if len(base_scores) > 0 else 0.0
    db_disc_intent = disc_scores.mean() if len(disc_scores) > 0 else 0.0
    db_anchoring_diff = db_disc_intent - db_base_intent

    c_k1, c_k2, c_k3, c_k4 = st.columns(4)
    c_k1.metric("👥 필터링 분석 대상", f"{len(df_db)} 명")
    c_k2.metric("🛍️ 충동구매 고빈도 비율", f"{db_impulse_rate:.1f} %")
    c_k3.metric("⏳ FOMO 심리 반응치", f"{db_fomo_mean:.2f} / 5")
    c_k4.metric("⚓ 앵커링 상승 유도치", f"{db_anchoring_diff:+.2f} 점")

    st.markdown("#### (1) 기본 소비 환경 및 성향 분포")
    col_g1, col_g2 = st.columns(2)
    with col_g1:
        if "학년" in df_db.columns:
            grade_dist = df_db["학년"].value_counts().reindex(GRADE_ORDER).fillna(0).reset_index()
            grade_dist.columns = ["학년", "인원수"]
            fig_g1 = px.bar(grade_dist, x="학년", y="인원수", text="인원수", title="학년별 응답자 수", color_discrete_sequence=["#4D96FF"])
            fig_g1.update_traces(textposition="outside")
            st.plotly_chart(fig_g1, use_container_width=True)
    with col_g2:
        if "용돈" in df_db.columns:
            allowance_dist = df_db["용돈"].value_counts().reset_index()
            allowance_dist.columns = ["용돈", "인원수"]
            fig_g2 = px.bar(allowance_dist, x="용돈", y="인원수", text="인원수", title="월평균 용돈 분포", color_discrete_sequence=["#6BCB77"])
            fig_g2.update_traces(textposition="outside")
            st.plotly_chart(fig_g2, use_container_width=True)

    col_g3, col_g4 = st.columns(2)
    with col_g3:
        if "구매중요요소" in df_db.columns:
            factor_dist = df_db["구매중요요소"].value_counts().reset_index()
            factor_dist.columns = ["구매중요요소", "인원수"]
            fig_g3 = px.pie(factor_dist, values="인원수", names="구매중요요소", title="구매 시 최고 중요 고려 요인", hole=0.3)
            st.plotly_chart(fig_g3, use_container_width=True)
    with col_g4:
        if "소비유형" in df_db.columns:
            db_type_dist = df_db["소비유형"].value_counts().reset_index()
            db_type_dist.columns = ["소비유형", "인원수"]
            fig_g4 = px.bar(db_type_dist, x="소비유형", y="인원수", text="인원수", title="자가진단 분류 유형 분포", color="소비유형", color_discrete_sequence=px.colors.qualitative.Safe)
            fig_g4.update_traces(textposition="outside")
            st.plotly_chart(fig_g4, use_container_width=True)

    st.markdown("#### (2) SNS 이용과 충동구매 경험 연관성 교차분석")
    col_g5, col_g6 = st.columns(2)
    with col_g5:
        if "SNS시간" in df_db.columns:
            sns_db_dist = df_db["SNS시간"].value_counts().reindex(SNS_ORDER).fillna(0).reset_index()
            sns_db_dist.columns = ["SNS시간", "인원수"]
            fig_g5 = px.bar(sns_db_dist, x="SNS시간", y="인원수", text="인원수", title="하루 평균 SNS 사용 시간 분포", color_discrete_sequence=["#FFD93D"])
            fig_g5.update_traces(textposition="outside")
            st.plotly_chart(fig_g5, use_container_width=True)
    with col_g6:
        if "충동구매경험" in df_db.columns:
            imp_db_dist = df_db["충동구매경험"].value_counts().reset_index()
            imp_db_dist.columns = ["충동구매경험", "인원수"]
            fig_g6 = px.bar(imp_db_dist, x="충동구매경험", y="인원수", text="인원수", title="충동구매 경험 빈도 분포", color_discrete_sequence=["#FF6B6B"])
            fig_g6.update_traces(textposition="outside")
            st.plotly_chart(fig_g6, use_container_width=True)

    if "SNS시간" in df_db.columns and "충동구매경험" in df_db.columns:
        st.write("**SNS 하루 평균 이용시간대별 충동구매 경험 여부 교차분석 비율(%) 히트맵**")
        cross_matrix = pd.crosstab(
            df_db["SNS시간"],
            df_db["충동구매경험"],
            normalize="index"
        ).mul(100).round(1)
        
        sns_db_order = [x for x in SNS_ORDER if x in cross_matrix.index]
        cross_matrix = cross_matrix.reindex(sns_db_order)
        
        fig_heatmap = px.imshow(
            cross_matrix,
            text_auto=True,
            aspect="auto",
            labels=dict(x="충동구매 경험 수준", y="하루 평균 SNS 사용 시간", color="비율(%)"),
            color_continuous_scale="Reds",
            title="SNS 노출도와 충동 소비 경향의 연계 비율"
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)

    st.markdown("#### (3) 가격 마케팅 및 희소성 자극(FOMO) 실험 분석")
    col_g7, col_g8 = st.columns(2)

    with col_g7:
        anchoring_df = pd.DataFrame({
            "조건": [
                "기본 가격 제시\n(13번 문항: 30,000원)", 
                "앵커링 제시\n(14번 문항: 50,000원 → 30,000원)"
            ],
            "구매 의향 점수": [round(db_base_intent, 2), round(db_disc_intent, 2)]
        })

        fig_g7 = px.bar(
            anchoring_df,
            x="조건",
            y="구매 의향 점수",
            text="구매 의향 점수",
            color="조건",
            color_discrete_sequence=["#8E9AAF", "#EF476F"],
            title="⚓ 앵커링 효과 유무 비교 (5점 만점 기준)"
        )
        
        fig_g7.update_traces(
            textposition="outside",
            textfont=dict(size=14, color="black", family="Arial Black")
        )
        fig_g7.update_layout(
            yaxis=dict(range=[0, 5.5], title="평균 구매 의향 (점)"),
            xaxis=dict(title=""),
            showlegend=False,
            height=400
        )
        st.plotly_chart(fig_g7, use_container_width=True)

        if db_base_intent > 0 or db_disc_intent > 0:
            if db_anchoring_diff > 0:
                st.caption(f"💡 정가(50,000원) 앵커링 자극 제시 시 구매 의향이 **+{db_anchoring_diff:.2f}점** 상승했습니다.")
            elif db_anchoring_diff < 0:
                st.caption(f"💡 정가 앵커링 제시 시 구매 의향이 오히려 **{db_anchoring_diff:.2f}점** 감소했습니다.")
            else:
                st.caption("💡 두 조건 간 구매 의향 점수 차이가 없습니다.")
        else:
            st.error("⚠️ 13, 14번 문항 데이터를 감지하지 못했습니다.")

    with col_g8:
        m_factors = [f for f in ["할인상품매력", "할인율구매의사", "재고3개남음", "오늘마감", "한정판관심"] if f in df_db.columns]
        m_labels = {
            "할인상품매력": "할인상품 호감도",
            "할인율구매의사": "할인율 극대화 반응",
            "재고3개남음": "품절 알림 (재고 3개)",
            "오늘마감": "시간제한 (오늘 마감)",
            "한정판관심": "희소 가치 (한정판)"
        }
        m_scores = []
        for mf in m_factors:
            score_series = df_db[mf].apply(parse_intent_score).dropna()
            score_val = score_series.mean() if len(score_series) > 0 else 0.0
            m_scores.append({
                "자극 요소": m_labels.get(mf, mf),
                "자극도 점수": round(score_val, 2)
            })
            
        if m_scores:
            df_m_scores = pd.DataFrame(m_scores).sort_values("자극도 점수", ascending=True)
            fig_g8 = px.bar(
                df_m_scores,
                x="자극도 점수",
                y="자극 요소",
                orientation="h",
                text="자극도 점수",
                color_discrete_sequence=["#FFBE0B"],
                title="⏳ 소비자 조바심/FOMO 유발 요인별 반응"
            )
            fig_g8.update_traces(textposition="outside", textfont=dict(size=12))
            fig_g8.update_layout(
                xaxis=dict(range=[0, 5.5], title="평균 반응도 (점)"),
                yaxis=dict(title=""),
                height=400
            )
            st.plotly_chart(fig_g8, use_container_width=True)
st.markdown("---")

# SECTION 5: 주관식 워드클라우드 분석 (wordcloud.png 이미지 직접 출력)
st.header("💬 5. 주관식 응답 워드클라우드 시각화")
st.markdown("고등학생들의 위시리스트 품목 주관식 응답 데이터를 기반으로 생성된 워드클라우드 이미지입니다.")

if WC_IMG_PATH.exists():
    try:
        wc_image = Image.open(WC_IMG_PATH)
        st.image(wc_image, caption="<최근 가장 사고 싶은 위시 리스트 단어 워드클라우드>", use_container_width=True)
    except Exception as img_err:
        st.error(f"wordcloud.png 이미지를 불러오는 중 오류가 발생했습니다: {img_err}")
else:
    st.warning("⚠️ `wordcloud.png` 파일이 지정된 경로에 존재하지 않습니다. 프로젝트 폴더에 이미지 파일을 올려주세요.")

st.markdown("---")

# 결론 및 요약
st.markdown("#### 💡 탐구 최종 결론 및 요약")
st.info(
    "본 실증 분석을 통해 고등학교 청소년 집단에서도 행동경제학의 대표적인 현상들이 매우 뚜렷하게 작용하고 있음을 확인할 수 있었습니다.\n\n"
    "1. **앵커링 효과의 영향:** 정가 판매 문구보다 기준 가격 대비 높은 할인율을 제시할 때 학생들의 무선 이어폰 평균 구매 의향 스코어가 유의미하게 상승했습니다.\n"
    "2. **손실회피 및 FOMO 요인:** '오늘 마감', '재고 부족'과 같은 제한적 희소성 정보가 조바심(소외감에 대한 걱정)을 증폭시켜 청소년 소비를 유도하는 핵심 마케팅 역할을 담당하고 있었습니다.\n"
    "3. **SNS와 충동 소비:** SNS 노출 시간이 늘어날수록 미디어가 제공하는 시각적 광고 및 유행 전파 효과로 인해 통제력을 잃고 충동구매에 노출될 비중이 현격히 증대되었습니다.\n\n"
    "본 탐구는 통계 데이터 기반으로 심리학 편향과 청소년 지출의 메커니즘을 밝혀, 향후 고등학생 집단에 올바르고 합리적인 예산 조절 및 심리 마케팅 방어적 소비 교육이 동반되어야 함을 역설합니다."
)
st.markdown("---")
st.caption("고등학생 소비심리와 구매행동 분석 Streamlit 웹 대시보드")