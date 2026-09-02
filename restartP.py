from amplpy import AMPL
import pandas as pd

# 1. Αρχικοποίηση AMPL & Φόρτωση του restartP.mod
ampl = AMPL()
ampl.read("restartP.mod")  # <-- Αλλαγή σε restartP.mod
ampl.read_data("RestartData.dat")

# 2. Δυναμικός υπολογισμός d[a,b]
ampl.eval(
    """
    let {(a,b) in States} d[a,b] := 1.0 / card(States);
    printf "Sum of d[a,b]: %f\\n", sum{(a,b) in States} d[a,b];
"""
)

# Επίλυση με Simplex / HiGHS
ampl.option["solver"] = "highs"
ampl.solve()

# 3. Ανάκτηση παραμέτρων
N = int(ampl.get_parameter("N").value())
c = ampl.get_parameter("c").value()
prior_alpha = int(ampl.get_parameter("prior_alpha").value())
prior_beta = int(ampl.get_parameter("prior_beta").value())

# 4. Λήψη αποτελεσμάτων V(a,b)
df = ampl.get_variable("V").get_values().to_pandas().reset_index()
df.columns = ["a", "b", "V_val"]
v_dict = {(r["a"], r["b"]): r["V_val"] for _, r in df.iterrows()}

# 5. Υπολογισμός της αξίας RESTART
# Αξία Restart = η αξία της αρχικής κατάστασης V(prior_alpha, prior_beta)
val_restart_base = v_dict.get((prior_alpha, prior_beta), 0.0)

# 6. Υπολογισμός πολιτικής (CONTINUE vs RESTART) για κάθε κατάσταση
actions = []
val_continues = []

for _, row in df.iterrows():
  a = int(row["a"])
  b = int(row["b"])

  # Υπολογισμός αξίας Continue από τη δεξιά πλευρά της Bellman
  p_success = a / (a + b)
  p_failure = b / (a + b)
  v_succ = v_dict.get((a + 1, b), 0.0)
  v_fail = v_dict.get((a, b + 1), 0.0)

  val_continue = p_success + c * p_success * v_succ + c * p_failure * v_fail
  val_continues.append(val_continue)

  # Σύγκριση Continue vs Restart (με μικρή ανοχή 1e-6 για στρογγυλοποιήσεις float)
  if val_continue >= val_restart_base - 1e-6:
    actions.append("CONTINUE")
  else:
    actions.append("RESTART")

df["Val_Continue"] = val_continues
df["Action"] = actions

# Δημιουργία δισδιάστατων πινάκων (pivot)
matrix_v = df.pivot(index="a", columns="b", values="V_val")
matrix_action = df.pivot(index="a", columns="b", values="Action")

# --- OUTPUT ---
print("\n--- 1. ΔΙΣΔΙΑΣΤΑΤΟΣ ΠΙΝΑΚΑΣ ΤΙΜΩΝ V(a,b) ---")
print(matrix_v)

print(
    f"\n--- 2. ΑΞΙΑ RESTART V({prior_alpha},{prior_beta}):"
    f" {val_restart_base:.6f} ---"
)

print("\n--- 3. ΔΙΣΔΙΑΣΤΑΤΟΣ ΠΙΝΑΚΑΣ ΠΟΛΙΤΙΚΗΣ (CONTINUE vs RESTART) ---")
print(matrix_action)

print("\n--- 4. ΤΙΜΕΣ ΠΑΡΑΜΕΤΡΟΥ d[a,b] ---")
df_d = ampl.get_parameter("d").get_values().to_pandas().reset_index()
df_d.columns = ["a", "b", "d_val"]
matrix_d = df_d.pivot(index="a", columns="b", values="d_val")
print(matrix_d)
