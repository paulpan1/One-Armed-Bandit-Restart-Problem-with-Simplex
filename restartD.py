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
# 2. ΟΡΙΣΜΟΣ ΑΡΧΙΚΗΣ ΚΑΤΑΝΟΜΗΣ d(a,b) = 1 στο (prior_alpha, prior_beta)
# KAI ΕΠΙΒΟΛΗ CONTINUE (k=2) ΣΤΗΝ ΑΡΧΙΚΗ ΚΑΤΑΣΤΑΣΗ
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
ampl.solve()

# ==============================================================
# 3. ΑΝΑΚΤΗΣΗ ΠΑΡΑΜΕΤΡΩΝ & ΔΙΑΧΩΡΙΣΜΟΣ W1 / W2 / W_TOTAL
# ==============================================================
gamma = float(ampl.get_parameter("c").value())  # c = gamma
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


def derive_policy(row):
  if row["w_a_b_2"] > 1e-7:
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
# 6. ΒΟΗΘΗΤΙΚΕΣ ΣΥΝΑΡΤΗΣΕΙΣ & ΔΥΝΑΜΙΚΟΣ ΠΡΟΓΡΑΜΜΑΤΙΣΜΟΣ (DP)
# ==============================================================
def R(x, y):
  return x / (x + y)


def compute_expected_discounted_time(a, b, gamma_val, p_alpha, p_beta):
  n_steps = a + b - p_alpha - p_beta
  if gamma_val == 1.0:
    return float(n_steps)
  return (1.0 - (gamma_val**n_steps)) / (1.0 - gamma_val)


def compute_valid_transition_probs(target_a, target_b, stopping_set):
  Q = {(target_a, target_b): 1.0}

  for u in range(target_a, 0, -1):
    for v in range(target_b, 0, -1):
      if u == target_a and v == target_b:
        continue

      if (u, v) in stopping_set:
        Q[(u, v)] = 0.0
        continue

      p_win = u / (u + v)
      p_loss = v / (u + v)

      q_win = Q.get((u + 1, v), 0.0)
      q_loss = Q.get((u, v + 1), 0.0)

      Q[(u, v)] = p_win * q_win + p_loss * q_loss

  return Q


def compute_expected_reward_for_state(
    target_a,
    target_b,
    gamma_val,
    W1_dict,
    W2_dict,
    stopping_set,
    p_alpha,
    p_beta,
):
  w_target_1 = W1_dict.get((target_a, target_b), 0.0)
  if w_target_1 <= 1e-9:
    return 0.0

  Q_matrix = compute_valid_transition_probs(target_a, target_b, stopping_set)

  expected_sum = 0.0
  for x in range(1, target_a + 1):
    for y in range(1, target_b + 1):
      if x == target_a and y == target_b:
        continue

      w_xy_2 = W2_dict.get((x, y), 0.0)
      if w_xy_2 > 1e-9:
        p_trans = Q_matrix.get((x, y), 0.0)
        expected_sum += R(x, y) * w_xy_2 * p_trans

  multiplier = (gamma_val ** (target_a + target_b - p_alpha - p_beta)) / w_target_1
  return multiplier * expected_sum


def compute_expected_reward_including_target(
    target_a,
    target_b,
    gamma_val,
    W1_dict,
    W2_dict,
    stopping_set,
    p_alpha,
    p_beta,
):
  w_target_1 = W1_dict.get((target_a, target_b), 0.0)
  if w_target_1 <= 1e-9:
    return 0.0

  Q_matrix = compute_valid_transition_probs(target_a, target_b, stopping_set)

  expected_sum = 0.0
  for x in range(1, target_a + 1):
    for y in range(1, target_b + 1):
      p_trans = Q_matrix.get((x, y), 0.0)
      if p_trans <= 1e-9:
        continue

      if x == target_a and y == target_b:
        expected_sum += R(x, y) * w_target_1 * p_trans
      else:
        w_xy_2 = W2_dict.get((x, y), 0.0)
        if w_xy_2 > 1e-9:
          expected_sum += R(x, y) * w_xy_2 * p_trans

  multiplier = (gamma_val ** (target_a + target_b - p_alpha - p_beta)) / w_target_1
  return multiplier * expected_sum


# ==============================================================
# 7. ΥΠΟΛΟΓΙΣΜΟΣ TOTAL COST / TOTAL REWARD: sum_{(x,a)} w(x,a) * c(x,a)
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
# 8. ΔΕΙΚΤΕΣ ΚΑΙ ΤΙΜΕΣ W ΓΙΑ ΚΑΤΑΣΤΑΣΕΙΣ RESTART (w1 > 0)
# ==============================================================
df_stopping = df[df["w_a_b_1"] > 1e-7].copy()
scale_factor = 1.0 / (1.0 - gamma) if gamma < 1.0 else 1.0

results = []
for _, row in df_stopping.iterrows():
  st_a, st_b = int(row["a"]), int(row["b"])

  exp_time = compute_expected_discounted_time(
      st_a, st_b, gamma, prior_alpha, prior_beta
  )
  exp_reward = compute_expected_reward_for_state(
      st_a,
      st_b,
      gamma,
      W1_dict,
      W2_dict,
      stopping_set,
      prior_alpha,
      prior_beta,
  )
  custom_metric = (
      scale_factor * (exp_reward / exp_time) if exp_time > 0 else 0.0
  )

  exp_reward_inc = compute_expected_reward_including_target(
      st_a,
      st_b,
      gamma,
      W1_dict,
      W2_dict,
      stopping_set,
      prior_alpha,
      prior_beta,
  )

  c_val_1 = row["c_a_b_1"] if "c_a_b_1" in row else 0.0

  results.append({
      "State (a,b)": f"({st_a},{st_b})",
      "W_total": round(row["W_total"], 6),
      "Cost_k1 (Restart Value)": round(c_val_1, 6),
      "E[Discounted Time]": round(exp_time, 4),
      "E[Discounted Reward]": round(exp_reward, 6),
      "disc rew": round(exp_reward_inc, 6),
      "Custom Metric": round(custom_metric, 6),
  })

df_results = (
    pd.DataFrame(results)
    .sort_values(by=["E[Discounted Time]"])
    .reset_index(drop=True)
)

print(
    "\n=========================================================================================="
)
print("--- 9. ΑΠΟΤΕΛΕΣΜΑΤΑ ΓΙΑ ΚΑΤΑΣΤΑΣΕΙΣ RESTART ---")
print(
    "=========================================================================================="
)
print(
    df_results[[
        "State (a,b)",
        "W_total",
        "Cost_k1 (Restart Value)",
        "E[Discounted Time]",
        "E[Discounted Reward]",
        "disc rew",
        "Custom Metric",
    ]].to_string(index=False)
)
# ==============================================================
# 10. ΣΤΑΤΙΣΤΙΚΑ ΑΠΟΦΑΣΗΣ CONTINUE (k=2)
# ==============================================================
# 1. Άθροισμα όλων των ποσοστών χρόνου Continue: W2_total = Σ w(a,b,2)
total_w2 = df["w_a_b_2"].sum()

# 2. Άθροισμα: Σ [x / (x + y) * w(x,y,2)]
total_continue_reward = ((df["a"] / (df["a"] + df["b"])) * df["w_a_b_2"]).sum()

# 3. Πηλίκο: Σ [x / (x + y) * w(x,y,2)] / W2
avg_continue_reward = total_continue_reward / total_w2 if total_w2 > 0 else 0.0

# Δημιουργία Συγκεντρωτικού Πίνακα
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

# 3 Μεμονωμένα Prints
print("\n--- ΑΝΑΛΥΤΙΚΑ PRINTS ---")
print(f"1. W2 (Συνολικό αθροισμα w_a_b_2)                 : {total_w2:.6f}")
print(f"2. Σ [x/(x+y) * w(x,y,2)] (Συνολική αμοιβή Continue): {total_continue_reward:.6f}")
print(f"3. Διαιρέση (Σ / W2) (Μέση αμοιβή ανά μονάδα W2)   : {avg_continue_reward:.6f}")
