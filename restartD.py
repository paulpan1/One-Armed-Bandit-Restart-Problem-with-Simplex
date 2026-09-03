import math
import pandas as pd
from amplpy import AMPL

# ==============================================================
# 1. ΦΟΡΤΩΣΗ & ΕΠΙΛΥΣΗ ΜΟΝΤΕΛΟΥ AMPL
# ==============================================================
ampl = AMPL()
ampl.read("restartD.mod")  # Ανάγνωση του δυϊκού μοντέλου restart
ampl.read_data("RestartData.dat")

# ==============================================================
# 2. ΟΡΙΣΜΟΣ ΑΡΧΙΚΗΣ ΚΑΤΑΝΟΜΗΣ d(a,b)
# ==============================================================



ampl.eval(
    """
    let {(a,b) in States} d[a,b] := 0.0;
    let d[prior_alpha, prior_beta] := 1.0;
    
    # Απαγορεύουμε το RESTART (k=1) στην αρχική κατάσταση
    fix w[prior_alpha, prior_beta, 1] := 0.0;
"""
)


ampl.option["solver"] = "highs"
ampl.option["highs_options"] = "presolve=off"  # <--- ΠΡΟΣΘΗΚΗ
ampl.solve()

# ==============================================================
# 3. ΑΝΑΚΤΗΣΗ ΠΑΡΑΜΕΤΡΩΝ & ΔΙΑΧΩΡΙΣΜΟΣ W1 / W2 / W_TOTAL
# ==============================================================
gamma = float(ampl.get_parameter("c").value())
prior_alpha = int(ampl.get_parameter("prior_alpha").value())
prior_beta = int(ampl.get_parameter("prior_beta").value())

df_w = ampl.get_variable("w").get_values().to_pandas().reset_index()
df_w.columns = ["a", "b", "k", "w_value"]

df_w1 = (
    df_w[df_w["k"] == 1][["a", "b", "w_value"]].rename(
        columns={"w_value": "w_a_b_1"}
    )
)
df_w2 = (
    df_w[df_w["k"] == 2][["a", "b", "w_value"]].rename(
        columns={"w_value": "w_a_b_2"}
    )
)
df = pd.merge(df_w1, df_w2, on=["a", "b"])

df["W_total"] = df["w_a_b_1"] + df["w_a_b_2"]


# ΔΙΟΡΘΩΣΗ 2: Ρητή επιλογή CONTINUE στην αρχική κατάσταση (prior_alpha, prior_beta)
def derive_policy(row):
    if row["a"] == prior_alpha and row["b"] == prior_beta:
        return "CONTINUE (k=2)"
    elif row["w_a_b_2"] > 1e-7:
        return "CONTINUE (k=2)"
    elif row["w_a_b_1"] > 1e-7:
        return "RESTART (k=1)"
    else:
        return "UNREACHABLE"


df["Optimal_Action"] = df.apply(derive_policy, axis=1)

df_shadow = (
    ampl.get_constraint("Flow_Balance").get_values().to_pandas().reset_index()
)
df_shadow.columns = ["a", "b", "V_a_b"]
df = pd.merge(df, df_shadow, on=["a", "b"])

W1_dict = df.set_index(["a", "b"])["w_a_b_1"].to_dict()
W2_dict = df.set_index(["a", "b"])["w_a_b_2"].to_dict()

stopping_set = set(
    tuple(x) for x in df[df["w_a_b_1"] > 1e-7][["a", "b"]].to_numpy()
)

# ==============================================================
# 4. ΑΡΧΙΚΑ PRINTS & ΠΙΝΑΚΕΣ W (W1, W2, W_TOTAL)
# ==============================================================
matrix_w1 = df.pivot(index="a", columns="b", values="w_a_b_1")
matrix_w2 = df.pivot(index="a", columns="b", values="w_a_b_2")
matrix_w_total = df.pivot(index="a", columns="b", values="W_total")
matrix_policy = df.pivot(index="a", columns="b", values="Optimal_Action")
matrix_v = df.pivot(index="a", columns="b", values="V_a_b")

print("\n=======================================================")
print(
    f"ΑΝΤΙΚΕΙΜΕΝΙΚΗ ΤΙΜΗ ΔΥΪΚΟΥ (RESTART):"
    f" {ampl.get_objective('Total_Dual_Reward').value():.6f}"
)
print("=======================================================")

print("\n--- 1. ΠΙΝΑΚΑΣ w[a,b,1] (Συχνότητα Επιλογής RESTART) ---")
print(matrix_w1)

print("\n--- 2. ΠΙΝΑΚΑΣ w[a,b,2] (Συχνότητα Επιλογής CONTINUE) ---")
print(matrix_w2)

print("\n--- 3. ΠΙΝΑΚΑΣ W_total [a,b] (w1 + w2) ---")
print(matrix_w_total)

print("\n--- 4. ΒΕΛΤΙΣΤΗ ΠΟΛΙΤΙΚΗ (Action per State) ---")
print(matrix_policy)

print("\n--- 5. ΤΙΜΕΣ V(a,b) (Shadow Prices) ---")
print(matrix_v)

# ==============================================================
# 5. ΟΡΙΑΚΕΣ ΚΑΤΑΣΤΑΣΕΙΣ (THRESHOLDS)
# ==============================================================
print(
    "\n--- 6. ΟΡΙΑΚΕΣ ΚΑΤΑΣΤΑΣΕΙΣ ΚΑΙ ΚΑΤΩΦΛΙΑ ΜΕΤΑΒΑΣΗΣ ΣΤΟ RESTART ---"
)

df["sum_ab"] = df["a"] + df["b"]
df["p_success"] = df["a"] / df["sum_ab"]

threshold_results = []
grouped = df.groupby("sum_ab")

for sum_val, group in grouped:
    n_stage = sum_val - prior_alpha - prior_beta
    sorted_group = group.sort_values(by="a", ascending=False)
    restart_states = sorted_group[sorted_group["w_a_b_1"] > 1e-7]

    if not restart_states.empty:
        boundary_row = restart_states.iloc[0]
        a_thresh = int(boundary_row["a"])
        b_thresh = int(boundary_row["b"])
        p_thresh = boundary_row["p_success"]
        switch_info = (
            f"Switch to RESTART at p <= {p_thresh:.4f} (State:"
            f" ({a_thresh},{b_thresh}))"
        )
        threshold_results.append({
            "Stage n": n_stage,
            "Sum (a+b)": sum_val,
            "Stopping State (a,b)": f"({a_thresh}, {b_thresh})",
            "Threshold Prob (p*)": f"{p_thresh:.4f}",
            "Transition Condition": switch_info,
        })
    else:
        threshold_results.append({
            "Stage n": n_stage,
            "Sum (a+b)": sum_val,
            "Stopping State (a,b)": "None",
            "Threshold Prob (p*)": "N/A",
            "Transition Condition": "Always CONTINUE in this stage",
        })

print(pd.DataFrame(threshold_results).to_string(index=False))

# ==============================================================
# 7. ΥΠΟΛΟΓΙΣΜΟΣ TOTAL COST / TOTAL REWARD
# ==============================================================
try:
    df_cost = ampl.get_parameter("cost").get_values().to_pandas().reset_index()
    df_cost.columns = ["a", "b", "k", "cost_value"]

    c_k1 = df_cost[df_cost["k"] == 1].set_index(["a", "b"])["cost_value"]
    c_k2 = df_cost[df_cost["k"] == 2].set_index(["a", "b"])["cost_value"]

    df = df.join(c_k1.rename("c_a_b_1"), on=["a", "b"])
    df = df.join(c_k2.rename("c_a_b_2"), on=["a", "b"])

    df["Cost_k1"] = df["w_a_b_1"] * df["c_a_b_1"]
    df["Cost_k2"] = df["w_a_b_2"] * df["c_a_b_2"]
    df["Cost_total_state"] = df["Cost_k1"] + df["Cost_k2"]
    total_cost = df["Cost_total_state"].sum()

    print("\n=======================================================")
    print(f"TOTAL REWARD [sum w(a,b)*c(a,b)]: {total_cost:.6f}")
    print("=======================================================")

    matrix_cost = df.pivot(index="a", columns="b", values="Cost_total_state")
    print("\n--- ΠΙΝΑΚΑΣ COST/REWARD PER STATE w1*c1 + w2*c2 ---")
    print(matrix_cost)
except Exception:
    print(
        "\n[Σημείωση: Δεν βρέθηκε παράμετρος 'cost', χρησιμοποιείται η"
        " αντικειμενική τιμή AMPL.]"
    )

# ==============================================================
# 10. ΣΤΑΤΙΣΤΙΚΑ ΑΠΟΦΑΣΗΣ CONTINUE (k=2)
# ==============================================================
total_w2 = df["w_a_b_2"].sum()
total_continue_reward = ((df["a"] / (df["a"] + df["b"])) * df["w_a_b_2"]).sum()
avg_continue_reward = total_continue_reward / total_w2 if total_w2 > 0 else 0.0

df_continue_stats = pd.DataFrame([
    {
        "Μετρική": "1. Συνολικό W2 (Σ w_a_b_2)",
        "Τιμή": round(total_w2, 6)
    },
    {
        "Μετρική": "2. Συνολική Αμοιβή Continue (Σ [x/(x+y) * w_a_b_2])",
        "Τιμή": round(total_continue_reward, 6)
    },
    {
        "Μετρική": "3. Μέση Αμοιβή Continue (Σ / W2)",
        "Τιμή": round(avg_continue_reward, 6)
    }
])

print("\n==========================================================================================")
print("--- 10. ΣΤΑΤΙΣΤΙΚΑ ΑΠΟΦΑΣΗΣ CONTINUE (k=2) ---")
print("==========================================================================================")
print(df_continue_stats.to_string(index=False))

print("\n--- ΑΝΑΛΥΤΙΚΑ PRINTS ---")
print(f"1. W2 (Συνολικό αθροισμα w_a_b_2)                 : {total_w2:.6f}")
print(f"2. Σ [x/(x+y) * w(x,y,2)] (Συνολική αμοιβή Continue): {total_continue_reward:.6f}")
print(f"3. Διαιρέση (Σ / W2) (Μέση αμοιβή ανά μονάδα W2)   : {avg_continue_reward:.6f}")


