
import argparse
import json
from typing import Dict, List, Tuple
import math
import pandas as pd

CATALOG_PATH = "hobby_catalog_full_100_service.xlsx"

USER_KEYS = [
    "E_introvert","O_openness","C_conscientiousness","A_autonomy","K_competence",
    "Energy_dynamic","Extrinsic_drive","Online_affinity","Breadth_preference","Process_focus"
]

SURVEY_TO_KEY = {
    "Q6":  "E_introvert",
    "Q7":  "O_openness",
    "Q8":  "C_conscientiousness",
    "Q9":  "A_autonomy",
    "Q10": "K_competence",
    "Q11": "Energy_dynamic",
    "Q12": "Extrinsic_drive",
    "Q13": "Online_affinity",
    "Q14": "Breadth_preference",
    "Q15": "Process_focus"
}

# Feature mapping between user vector and hobby attributes
HOBBY_KEYS = {
    "E_introvert": None,  # handled via social_mode bonus
    "O_openness": "openness_alignment",
    "C_conscientiousness": "conscientiousness_alignment",
    "A_autonomy": "autonomy_alignment",
    "K_competence": "competence_alignment",
    "Energy_dynamic": "activity_energy",
    "Extrinsic_drive": None,   # handled via monetizable bonus
    "Online_affinity": None,   # handled via location/online bonus
    "Breadth_preference": "commitment_depth",  # inverse later if needed
    "Process_focus": None      # handled via explanation / minor weight
}

# Weights for the similarity score
W = {
    "O_openness": 1.0,
    "C_conscientiousness": 1.2,
    "A_autonomy": 1.0,
    "K_competence": 1.4,
    "Energy_dynamic": 1.1,
    "Breadth_preference": 0.8  # note: this compares to commitment_depth
}

def normalize_likert(x: int) -> float:
    if not (1 <= int(x) <= 5):
        raise ValueError("Likert must be 1..5")
    return (int(x) - 1) / 4.0

def build_user_vector(survey: Dict[str, int]) -> Dict[str, float]:
    z = {}
    for q, key in SURVEY_TO_KEY.items():
        if q not in survey:
            raise KeyError(f"Missing {q}")
        z[key] = normalize_likert(survey[q])
    return z

def load_catalog(path: str = CATALOG_PATH) -> pd.DataFrame:
    return pd.read_excel(path, sheet_name="catalog_full")

def passes_filters(user_ctx: Dict, row: pd.Series) -> bool:
    # budget filter
    if "monthly_budget" in user_ctx and pd.notna(row["avg_cost_month_krw"]):
        if row["avg_cost_month_krw"] > user_ctx["monthly_budget"]:
            return False
    # time filter (rough: session time <= available hours per session)
    if "session_time_limit_hours" in user_ctx and pd.notna(row["avg_session_time_hours"]):
        if row["avg_session_time_hours"] > user_ctx["session_time_limit_hours"]:
            return False
    # offline constraint
    if "offline_ok" in user_ctx:
        if row["needs_offline"] and not user_ctx["offline_ok"]:
            return False
    return True

def weighted_cosine(user: Dict[str, float], row: pd.Series) -> float:
    num = 0.0
    du = 0.0
    dh = 0.0
    for ukey, hkey in HOBBY_KEYS.items():
        if hkey is None:
            continue
        w = W.get(ukey, 1.0)
        uval = user.get(ukey, 0.0)
        hval = float(row.get(hkey, 0.0))
        # If comparing Breadth_preference to commitment_depth, consider inverse alignment:
        if ukey == "Breadth_preference":
            # user prefers breadth (0..1), hobby is depth (0..1). Alignment = 1 - |u - (1 - depth)|
            # For scoring, treat as similarity between u and (1 - depth)
            hval = 1.0 - hval
        num += w * uval * hval
        du  += w * (uval ** 2)
        dh  += w * (hval ** 2)
    if du <= 0 or dh <= 0:
        return 0.0
    return num / (math.sqrt(du) * math.sqrt(dh) + 1e-8)

def bonuses(user: Dict[str, float], row: pd.Series) -> float:
    b = 0.0
    # Sociality bonus: introvert prefers solo/parallel small group; extrovert prefers community
    intro = user["E_introvert"]
    if intro >= 0.6 and str(row["social_mode"]).lower() == "solo":
        b += 0.08
    if intro < 0.4 and str(row["social_mode"]).lower() in {"parallel","community"}:
        b += 0.06

    # Monetization bonus for high external motivation
    if user["Extrinsic_drive"] >= 0.6 and bool(row["monetizable"]):
        b += 0.07

    # Online affinity
    if user["Online_affinity"] >= 0.6 and (str(row["location_type"]).lower() in {"online","any"} or bool(row["online_available"])):
        b += 0.05

    # Healthy dopamine bias: competence/conscientiousness high + dynamic activities
    if (user["K_competence"] >= 0.6 or user["C_conscientiousness"] >= 0.6) and float(row.get("activity_energy",0.0)) >= 0.6:
        b += 0.05

    return b

def explain_reason(user: Dict[str, float], row: pd.Series, base: float, total: float) -> str:
    parts = []
    # Pick top 2-3 drivers
    drivers = []
    drivers.append(("성장/숙련(유능성)", user["K_competence"], row.get("competence_alignment",0)))
    drivers.append(("계획/꾸준함(성실성)", user["C_conscientiousness"], row.get("conscientiousness_alignment",0)))
    drivers.append(("새로운 경험(개방성)", user["O_openness"], row.get("openness_alignment",0)))
    drivers.append(("자율성/창작", user["A_autonomy"], row.get("autonomy_alignment",0)))
    drivers.append(("활동 에너지", user["Energy_dynamic"], row.get("activity_energy",0)))
    # Convert to contribution proxy
    drivers = [(name, float(u)*float(h)) for name,u,h in drivers]
    drivers.sort(key=lambda x: x[1], reverse=True)
    for name,score in drivers[:3]:
        parts.append(name)
    social = str(row["social_mode"]).lower()
    if user["E_introvert"] >= 0.6 and social == "solo":
        parts.append("혼자 몰입")
    if user["E_introvert"] < 0.4 and social in {"parallel","community"}:
        parts.append("함께하는 활동")
    if user["Extrinsic_drive"] >= 0.6 and bool(row["monetizable"]):
        parts.append("수익화 가능")
    return " · ".join(parts)

def recommend(
    survey: Dict[str,int],
    user_ctx: Dict,
    top_k: int = 10,
    catalog_path: str = CATALOG_PATH
) -> pd.DataFrame:
    """
    사용자 설문과 컨텍스트를 기반으로 취미를 추천합니다.
    벡터 연산을 사용하여 성능을 개선하고 로직을 간소화했습니다.
    """
    user = build_user_vector(survey)
    df = load_catalog(catalog_path)

    # 1. 필터링 (벡터 연산)
    filtered_df = df.copy()
    if "monthly_budget" in user_ctx:
        filtered_df = filtered_df[~(filtered_df["avg_cost_month_krw"] > user_ctx["monthly_budget"])]
    if "session_time_limit_hours" in user_ctx:
        filtered_df = filtered_df[~(filtered_df["avg_session_time_hours"] > user_ctx["session_time_limit_hours"])]
    if "offline_ok" in user_ctx and not user_ctx["offline_ok"]:
        filtered_df = filtered_df[~filtered_df["needs_offline"]]

    if filtered_df.empty:
        return pd.DataFrame()

    # 2. 점수 계산 (apply 사용)
    scores = filtered_df.apply(
        lambda row: pd.Series({
            'score_base': weighted_cosine(user, row),
            'score_bonus': bonuses(user, row)
        }), axis=1
    )
    filtered_df['score_base'] = scores['score_base']
    filtered_df['score_total'] = scores['score_base'] + scores['score_bonus']

    # 3. 상위 K개 선택 및 순위 부여
    top = filtered_df.nlargest(top_k, 'score_total').copy()
    top['rank'] = range(len(top))
    
    # 4. 추천 이유 생성
    top['reason'] = top.apply(lambda row: explain_reason(user, row, row['score_base'], row['score_total']), axis=1)
    
    return top[["rank","hobby_id","name_ko","short_desc","avg_cost_month_krw","avg_session_time_hours",
                "social_mode","location_type","tags","score_total","score_base","reason"]]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--survey_json", type=str, required=False,
                        help='JSON like {"Q6":4,"Q7":5,...,"Q15":3}')
    parser.add_argument("--budget", type=int, default=100000)
    parser.add_argument("--time_limit", type=float, default=3.0)
    parser.add_argument("--offline_ok", action='store_true', help="Set this flag if offline activities are okay.")
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--catalog", type=str, default=CATALOG_PATH)
    parser.add_argument("--out_csv", type=str, default="sample_output_top10.csv")
    args = parser.parse_args()

    if args.survey_json:
        survey = json.loads(args.survey_json)
    else:
        # sample default
        survey = {"Q6":4,"Q7":5,"Q8":4,"Q9":4,"Q10":5,"Q11":3,"Q12":2,"Q13":4,"Q14":3,"Q15":3}

    user_ctx = {
        "monthly_budget": args.budget,
        "session_time_limit_hours": args.time_limit,
        "offline_ok": args.offline_ok
    }

    top = recommend(survey, user_ctx, top_k=args.topk, catalog_path=args.catalog)
    top.to_csv(args.out_csv, index=False, encoding="utf-8-sig")
    print(top.to_string(index=False))

if __name__ == "__main__":
    main()
