from pathlib import Path
import dash
from dash import dcc, html, Input, Output, dash_table
import plotly.graph_objects as go
import pandas as pd
import numpy as np

RNG = np.random.default_rng(42)

TOP_N = 10
N_DEBTORS = 15_000
TOTAL_PHONES_A = TOTAL_PHONES_B = 100_000
DEBTS_A, DEBTS_B = 20_000, 30_000
MAX_DEBTS_A, MAX_DEBTS_B = 20, 10

COMMISSION = {1: 0.05, 2: 0.04, 3: 0.03, 4: 0.02}
COST_SMS = 0.5
ROBOT_COST = {1: 0.01, 2: 0.03}
BASE_EFF = {"voice": 0.000011, "robot": 0.0000005, "sms": 0.00000025}
BONUS_2, BONUS_3 = 0.005, 0.01
VOICE_COST_BY_TYPE = {1: 1.0, 2: 2.0, 3: 3.0}
MAX_HORIZON_DAYS = 90
STRATEGY_STEP = 1


def scale_counts_to_total(counts: np.ndarray, target: int, lo: int, hi: int) -> np.ndarray:
    n = len(counts)
    counts = counts.astype(int)
    tot = max(int(counts.sum()), 1)
    counts = np.clip(np.round(counts * (target / tot)).astype(int), lo, hi)
    diff = target - int(counts.sum())
    guard = 0
    while diff != 0 and guard < 5_000_000:
        j = guard % n
        if diff > 0 and counts[j] < hi:
            counts[j] += 1
            diff -= 1
        elif diff < 0 and counts[j] > lo:
            counts[j] -= 1
            diff += 1
        guard += 1
    return counts


def generate_debts(name: str, total_debts: int, max_per_debtor: int, cat_mode: str, sum_dist: str, days_dist: str) -> pd.DataFrame:
    ids = [f"{name}_D_{i}" for i in range(N_DEBTORS)]
    debts_per = RNG.integers(1, max_per_debtor + 1, size=N_DEBTORS)
    debts_per = scale_counts_to_total(debts_per, total_debts, 1, max_per_debtor)
    rows = []
    for i, d_id in enumerate(ids):
        for _ in range(int(debts_per[i])):
            if cat_mode == "uniform":
                cat = int(RNG.integers(1, 5))
            else:
                cat = int(np.clip(np.rint(RNG.normal(2.5, 0.85)), 1, 4))
            amt = RNG.normal(50_000, 25_000) if sum_dist == "normal" else RNG.uniform(100, 100_000)
            if days_dist == "normal":
                dpd = int(RNG.normal(500, 200))
            else:
                dpd = int(RNG.integers(90, 1000))
            rows.append({"debtor_id": d_id, "category": cat, "amount": amt, "days_overdue": dpd})
    df = pd.DataFrame(rows)
    df["amount"] = df["amount"].clip(100, 100_000)
    df["days_overdue"] = df["days_overdue"].clip(90, 1000)
    return df


def generate_phones(debtor_ids: list, total_phones: int, min_ph: int, max_ph: int) -> pd.DataFrame:
    n = len(debtor_ids)
    counts = RNG.integers(min_ph, max_ph + 1, size=n)
    counts = scale_counts_to_total(counts, total_phones, min_ph, max_ph)
    rows = []
    for d_id, k in zip(debtor_ids, counts):
        for pidx in range(int(k)):
            t = int(RNG.integers(1, 4))
            rows.append({"debtor_id": d_id, "phone_idx": pidx + 1, "phone_type": t, "voice_cost": VOICE_COST_BY_TYPE[t]})
    return pd.DataFrame(rows)


def generate_robot_mins(debtor_ids: list) -> pd.Series:
    return pd.Series(ROBOT_COST[1], index=debtor_ids, name="robot_min")


def debtor_voice_min_cost(phones: pd.DataFrame) -> pd.Series:
    return phones.groupby("debtor_id")["voice_cost"].min()


def all_strategies_upto90(step: int = 1, max_days: int = 90):
    out = []
    for v in range(0, max_days + 1, step):
        for r in range(0, max_days - v + 1, step):
            for s in range(0, max_days - v - r + 1, step):
                total = v + r + s
                out.append(
                    {
                        "name": f"V{v}_R{r}_S{s}",
                        "voice": v,
                        "robot": r,
                        "sms": s,
                        "total_days": total,
                    }
                )
    return out


STRATEGIES = all_strategies_upto90(STRATEGY_STEP, MAX_HORIZON_DAYS)

def create_portfolio_data(df_debts, phones, robot_min, label=None):
    d = df_debts["days_overdue"].to_numpy()
    bp = np.select([d <= 180, d <= 720], [0.05, 0.03], default=0.01)
    
    comm = df_debts["category"].map(COMMISSION).to_numpy()
    ac = df_debts["amount"].to_numpy() * comm
    
    voice_min = phones.groupby("debtor_id")["voice_cost"].min()
    
    data = {
        "n_debtors": df_debts["debtor_id"].nunique(),
        "bp": bp,
        "ac": ac,
        "sum_voice": voice_min.sum(),
        "sum_robot": robot_min.sum(),
    }
    
    if label:
        data["label"] = label
    
    return data


def calculate_profit(data: dict, voice_days: int, robot_days: int, sms_days: int):
    total_days = voice_days + robot_days + sms_days
    
    if total_days > 90:
        raise ValueError(f"Слишком много дней: {total_days} > 90")
    
    channels_used = (voice_days > 0) + (robot_days > 0) + (sms_days > 0)
    
    if channels_used == 3:
        bonus = BONUS_3 
    elif channels_used == 2:
        bonus = BONUS_2 
    else:
        bonus = 0.000 
    
    voice_effect = voice_days * BASE_EFF["voice"]    
    robot_effect = robot_days * BASE_EFF["robot"]  
    sms_effect   = sms_days   * BASE_EFF["sms"]  
    
    total_boost = voice_effect + robot_effect + sms_effect + bonus
    
    final_probability = data["bp"] + total_boost
    final_probability = np.clip(final_probability, 0, 1)
    
    expected_income = np.dot(data["ac"], final_probability)
    
    voice_cost = voice_days * data["sum_voice"]
    robot_cost = robot_days * data["sum_robot"]
    sms_cost   = sms_days   * data["n_debtors"] * 0.5
    
    total_cost = voice_cost + robot_cost + sms_cost
    
    profit = expected_income - total_cost
    
    if total_cost > 0:
        roi_percent = (profit / total_cost) * 100
    else:
        roi_percent = 0.0
    
    avg_probability = final_probability.mean()
    
    return {
        "profit": profit,
        "cost": total_cost,
        "revenue": expected_income,
        "roi": roi_percent,
        "avg_probability": avg_probability
    }


def evaluate_portfolio(data: dict) -> pd.DataFrame:
    rows = []
    for st in STRATEGIES:
        result = calculate_profit(data, st["voice"], st["robot"], st["sms"])  
        
        rows.append({
            "strategy": st["name"],
            "voice_days": st["voice"],
            "robot_days": st["robot"],
            "sms_days": st["sms"],
            "total_days": st["total_days"],
            "profit": result["profit"],           
            "cost": result["cost"],               
            "revenue": result["revenue"],         
            "roi": result["roi"],                 
            "avg_final_prob": result["avg_probability"],  
        })
    return pd.DataFrame(rows).sort_values(["profit", "roi"], ascending=[False, False]).reset_index(drop=True)

DEBTOR_IDS_A = [f"A_D_{i}" for i in range(N_DEBTORS)]
DEBTOR_IDS_B = [f"B_D_{i}" for i in range(N_DEBTORS)]

df_a_debts = generate_debts("A", DEBTS_A, MAX_DEBTS_A, "uniform", "normal", "uniform")
df_b_debts = generate_debts("B", DEBTS_B, MAX_DEBTS_B, "normal", "uniform", "normal")

phones_a = generate_phones(DEBTOR_IDS_A, TOTAL_PHONES_A, 1, 20)
phones_b = generate_phones(DEBTOR_IDS_B, TOTAL_PHONES_B, 1, 10)

robot_min_a = generate_robot_mins(DEBTOR_IDS_A)
robot_min_b = generate_robot_mins(DEBTOR_IDS_B)

data_a = create_portfolio_data(df_a_debts, phones_a, robot_min_a)
data_b = create_portfolio_data(df_b_debts, phones_b, robot_min_b)

df_results_a = evaluate_portfolio(data_a)
df_results_b = evaluate_portfolio(data_b)


GENERATED_DATA_DIR = Path(__file__).resolve().parent / "generated_data"
EXPORT_NOTE = ""

def export_generated_csv() -> str:
    global EXPORT_NOTE
    GENERATED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    robot_a = pd.DataFrame({"debtor_id": robot_min_a.index.astype(str), "robot_min_cost": robot_min_a.values})
    robot_b = pd.DataFrame({"debtor_id": robot_min_b.index.astype(str), "robot_min_cost": robot_min_b.values})
    files = {
        "debts_A.csv": df_a_debts,
        "debts_B.csv": df_b_debts,
        "phones_A.csv": phones_a,
        "phones_B.csv": phones_b,
        "robot_min_A.csv": robot_a,
        "robot_min_B.csv": robot_b,
        "strategies_results_A.csv": df_results_a,
        "strategies_results_B.csv": df_results_b,
    }
    err = []
    for name, obj in files.items():
        path = GENERATED_DATA_DIR / name
        try:
            obj.to_csv(path, index=False)
        except OSError as e:
            err.append(f"{name}: {e}")
    EXPORT_NOTE = str(GENERATED_DATA_DIR)
    if err:
        EXPORT_NOTE += " | Ошибки записи: " + "; ".join(err)
    return EXPORT_NOTE

export_generated_csv()


CARD_STYLE = {"borderRadius": "10px", "padding": "6px"}

app = dash.Dash(__name__)

app.layout = html.Div(
    style={"minHeight": "100vh", "backgroundColor": "#ffffff", "padding": "24px 16px 40px", "fontFamily": "Segoe UI, system-ui, sans-serif", "color": "#1e293b"},
    children=[
        html.Div(
            style={"maxWidth": "1180px", "margin": "0 auto"},
            children=[
                html.Div(
                    style={"display": "flex", "flexWrap": "wrap", "justifyContent": "space-between", "alignItems": "flex-end", "gap": "16px", "marginBottom": "20px", "paddingBottom": "16px", "borderBottom": "1px solid #e2e8f0"},
                    children=[
                        html.Div([html.H1("Стратегии взыскания", style={"margin": "0 0 4px 0", "fontSize": "26px", "color": "#1e3a5f"})]),
                        html.Div(
                            [
                                html.Label("Клиент", style={"fontSize": "11px", "color": "#64748b", "display": "block", "marginBottom": "4px"}),
                                dcc.Dropdown(id="client-selector", options=[{"label": "Клиент A", "value": "A"}, {"label": "Клиент B", "value": "B"}], value="A", clearable=False, style={"width": "280px"}),
                            ]
                        ),
                    ],
                ),
                html.H2(f"Топ-{TOP_N} по прибыли", style={"fontSize": "17px", "color": "#1e3a5f", "margin": "8px 0 8px 0"}),
                html.Div(id="profit-table"),
                html.H2("Графики", style={"fontSize": "17px", "color": "#1e3a5f", "margin": "32px 0 12px 0"}),
                html.Div(
                    style={"display": "grid", "gridTemplateColumns": "repeat(auto-fit, minmax(480px, 1fr))", "gap": "16px"},
                    children=[
                        html.Div(dcc.Graph(id="profit-chart", config={"responsive": True}), style=CARD_STYLE),
                        html.Div(dcc.Graph(id="roi-chart", config={"responsive": True}), style=CARD_STYLE),
                    ],
                ),
            ],
        ),
    ],
)



def fmt_rub(x):
    return f"{x:,.0f} ₽".replace(",", " ")


def describe_strategy(v, r, s):
    v, r, s = int(v), int(r), int(s)
    t = v + r + s
    bits = []
    if v:
        bits.append(f"голос {v} дн.")
    if r:
        bits.append(f"робот {r} дн.")
    if s:
        bits.append(f"СМС {s} дн.")
    if not bits:
        bits.append("без контактов")
    channels = (v > 0) + (r > 0) + (s > 0)
    bon = " Бонус: 3 канала." if channels >= 3 else " Бонус: 2 канала." if channels == 2 else ""
    return ", ".join(bits) + f" Всего {t} дн. из {MAX_HORIZON_DAYS}.{bon}"


def fig_layout(title, xaxis_title):
    return {
        "template": "plotly_white",
        "title": {"text": title, "font": {"size": 15, "color": "#1e3a5f"}},
        "xaxis_title": xaxis_title,
        "height": max(300, 44 * TOP_N + 88),
        "margin": dict(l=92, r=16, t=48, b=36),
        "showlegend": False,
    }



@app.callback(
    [Output("profit-table", "children"), Output("profit-chart", "figure"), Output("roi-chart", "figure")],
    [Input("client-selector", "value")],
)
def update(client):
    df = df_results_a if client == "A" else df_results_b
    name = "Клиент A" if client == "A" else "Клиент B"

    top = df.nlargest(TOP_N, "profit").copy()
    top["rank"] = range(1, len(top) + 1)
    top["method_desc"] = top.apply(lambda r: describe_strategy(r["voice_days"], r["robot_days"], r["sms_days"]), axis=1)
    top["v_r_s"] = top["voice_days"].astype(int).astype(str) + "/" + top["robot_days"].astype(int).astype(str) + "/" + top["sms_days"].astype(int).astype(str)
    for col, key in [("revenue_disp", "revenue"), ("cost_disp", "cost"), ("profit_disp", "profit")]:
        top[col] = top[key].map(fmt_rub)
    top["roi_disp"] = top["roi"].map(lambda x: f"{x:.1f}%")
    top["profit_k"] = top["profit"] / 1e3
    recs = top[["rank", "strategy", "method_desc", "v_r_s", "revenue_disp", "cost_disp", "profit_disp", "profit", "roi_disp"]].to_dict("records")

    money_cols = ("revenue_disp", "cost_disp", "profit_disp", "roi_disp")
    tbl = dash_table.DataTable(
        data=recs,
        columns=[
            {"name": "№", "id": "rank", "type": "numeric"},
            {"name": "Код", "id": "strategy"},
            {"name": "Описание", "id": "method_desc"},
            {"name": "В/Р/С", "id": "v_r_s"},
            {"name": "Доход", "id": "revenue_disp"},
            {"name": "Затраты", "id": "cost_disp"},
            {"name": "Прибыль", "id": "profit_disp"},
            {"name": "ROI", "id": "roi_disp"},
        ],
        fill_width=True,
        page_size=TOP_N,
        style_cell={"padding": "8px", "whiteSpace": "normal"},
        style_cell_conditional=[{"if": {"column_id": c}, "textAlign": "right"} for c in money_cols],
        style_data_conditional=[
            {"if": {"filter_query": "{profit} > 0", "column_id": "profit_disp"}, "color": "#047857"},
            {"if": {"filter_query": "{profit} < 0", "column_id": "profit_disp"}, "color": "#be123c"},
        ],
    )
    wrap = html.Div(tbl)

    green, red = "#10b981", "#f43f5e"
    tpc = top.sort_values("profit_k", ascending=True)
    hov = tpc["method_desc"].str.replace("\n", "<br>", regex=False)
    bar_text = tpc["profit"].map(lambda v: f"{v:,.0f} ₽")
    fig_p = go.Figure(
        go.Bar(
            y=tpc["strategy"],
            x=tpc["profit_k"],
            orientation="h",
            marker_color=[green if x > 0 else red for x in tpc["profit"]],
            text=bar_text,
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>%{customdata[0]}<br>Прибыль: %{customdata[1]:,.0f} ₽ (%{x:,.2f} тыс. ₽)<br>ROI: %{customdata[2]:.1f}%<extra></extra>",
            customdata=np.stack([hov.values, tpc["profit"].values, tpc["roi"].values], axis=-1),
        )
    )
    fig_p.update_layout(**fig_layout(f"{name}: топ-{TOP_N} по прибыли", "Прибыль, тыс. ₽"))
    x_lo, x_hi = float(tpc["profit_k"].min()), float(tpc["profit_k"].max())
    pad = max(0.5, (x_hi - x_lo) * 0.2 + 1e-9)
    fig_p.update_xaxes(range=[x_lo - pad, x_hi + pad])

    troi = df.nlargest(TOP_N, "roi").copy()
    troi["md"] = troi.apply(lambda r: describe_strategy(r["voice_days"], r["robot_days"], r["sms_days"]), axis=1)
    troi = troi.sort_values("roi", ascending=True)
    h2 = troi["md"].str.replace("\n", "<br>", regex=False)
    fig_r = go.Figure(
        go.Bar(
            y=troi["strategy"],
            x=troi["roi"],
            orientation="h",
            marker_color=[green if x > 0 else red for x in troi["roi"]],
            text=troi["roi"].round(1),
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>%{customdata[0]}<br>ROI: %{x:.1f}%<br>Прибыль: %{customdata[1]:.2f} млн ₽<extra></extra>",
            customdata=np.stack([h2.values, (troi["profit"] / 1e6).values], axis=-1),
        )
    )
    fig_r.update_layout(**fig_layout(f"{name}: топ-{TOP_N} по ROI", "ROI, %"))
    r_lo, r_hi = float(troi["roi"].min()), float(troi["roi"].max())
    r_pad = max(0.5, (r_hi - r_lo) * 0.2 + 1e-9)
    fig_r.update_xaxes(range=[r_lo - r_pad, r_hi + r_pad])

    return wrap, fig_p, fig_r

if __name__ == "__main__":
    app.run(debug=False, port=8050)
