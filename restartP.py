from amplpy import AMPL
import pandas as pd

# 1. Αρχικοποίηση AMPL & Φόρτωση του restartP.mod
ampl = AMPL()
ampl.read("restartP.mod")  
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

# 5. Υπολογισμός πολιτικής (CONTINUE vs RESTART) ακριβώς όπως το AMPL
actions = []
val_continues = []
val_restarts = []

for _, row in df.iterrows():
    a = int(row["a"])
    b = int(row["b"])

    # --- Α. Υπολογισμός αξίας CONTINUE (Με τη νέα συνοριακή λογική που ταιριάζει στο .mod) ---
    p_success = a / (a + b)
    p_failure = b / (a + b)
    
    # Αν η επόμενη κατάσταση Success είναι εντός States παίρνει τη V, αλλιώς του Restart
    if (a + 1, b) in v_dict:
        v_succ_term = v_dict[(a + 1, b)]
    else:
        v_succ_term = v_dict.get((prior_alpha + 1, prior_beta), 0.0)

    # Αν η επόμενη κατάσταση Failure είναι εντός States παίρνει τη V, αλλιώς του Restart
    if (a, b + 1) in v_dict:
        v_fail_term = v_dict[(a, b + 1)]
    else:
        v_fail_term = v_dict.get((prior_alpha, prior_beta + 1), 0.0)

    val_continue = p_success + c * p_success * v_succ_term + c * p_failure * v_fail_term
    val_continues.append(val_continue)

    # --- Β. Υπολογισμός αξίας RESTART (Χωρίς περιττούς ελέγχους) ---
    p_restart_succ = prior_alpha / (prior_alpha + prior_beta)
    p_restart_fail = prior_beta / (prior_alpha + prior_beta)
    
    term_succ = v_dict.get((prior_alpha + 1, prior_beta), 0.0)
    term_fail = v_dict.get((prior_alpha, prior_beta + 1), 0.0)

    val_restart = p_restart_succ + c * p_restart_succ * term_succ + c * p_restart_fail * term_fail
    val_restarts.append(val_restart)

    # --- Γ. Σύγκριση Continue vs Restart (με μικρή ανοχή 1e-6 για στρογγυλοποιήσεις float) ---
    if val_continue >= val_restart - 1e-6:
        actions.append("CONTINUE")
    else:
        actions.append("RESTART")

df["Val_Continue"] = val_continues
df["Val_Restart"] = val_restarts
df["Action"] = actions

# Δημιουργία δισδιάστατων πινάκων (pivot)
matrix_v = df.pivot(index="a", columns="b", values="V_val")
matrix_action = df.pivot(index="a", columns="b", values="Action")

# --- OUTPUT ---
print("\n--- 1. ΔΙΣΔΙΑΣΤΑΤΟΣ ΠΙΝΑΚΑΣ ΤΙΜΩΝ V(a,b) ---")
print(matrix_v)

print(
    f"\n--- 2. ΑΞΙΑ RESTART ΣΤΗΝ ΑΡΧΙΚΗ ΚΑΤΑΣΤΑΣΗ V({prior_alpha},{prior_beta}):"
    f" {v_dict.get((prior_alpha, prior_beta), 0.0):.6f} ---"
)

print("\n--- 3. ΔΙΣΔΙΑΣΤΑΤΟΣ ΠΙΝΑΚΑΣ ΠΟΛΙΤΙΚΗΣ (CONTINUE vs RESTART) ---")
print(matrix_action)

print("\n--- 4. ΤΙΜΕΣ ΠΑΡΑΜΕΤΡΟΥ d[a,b] ---")
df_d = ampl.get_parameter("d").get_values().to_pandas().reset_index()
df_d.columns = ["a", "b", "d_val"]
matrix_d = df_d.pivot(index="a", columns="b", values="d_val")
print(matrix_d)
