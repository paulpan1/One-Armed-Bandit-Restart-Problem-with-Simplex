# ==============================================================
# ΔΥΪΚΟ ΜΟΝΤΕΛΟ ΓΡΑΜΜΙΚΟΥ ΠΡΟΓΡΑΜΜΑΤΙΣΜΟΥ - RESTART (dual_restart.mod)
# ==============================================================

# -- 1. ΠΑΡΑΜΕΤΡΟΙ --
param N integer > 0;               # Ορίζοντας
param c >= 0, < 1;                 # Προεξοφλητικός παράγοντας
param prior_alpha integer >= 0;    # Prior α (α0)
param prior_beta integer >= 0;     # Prior β (β0)

# -- 2. ΣΥΝΟΛΟ ΚΑΤΑΣΤΑΣΕΩΝ (States) --
set States := {a in prior_alpha..(N + prior_alpha), 
               b in prior_beta..(N + prior_beta) 
               : (a - prior_alpha) + (b - prior_beta) < N};

# -- 3. ΣΥΝΟΛΟ ΕΝΕΡΓΕΙΩΝ (Actions) --
# k = 1 -> Επιλογή RESTART
# k = 2 -> Επιλογή CONTINUE
set Actions := {1, 2};

# -- 4. ΑΡΧΙΚΗ ΚΑΤΑΝΟΜΗ d(a,b) --
param d {States} default 1.0;

# -- 5. ΣΥΝΤΕΛΕΣΤΕΣ ΑΜΟΙΒΩΝ c_{a,b,k} --
# c_{a,b,1} = prior_alpha / (prior_alpha + prior_beta)  [Restart]
# c_{a,b,2} = a / (a + b)                              [Continue]
param cost {(a,b) in States, k in Actions} :=
    if k = 1 then
        prior_alpha / (prior_alpha + prior_beta)
    else
        a / (a + b);

# -- 6. ΜΕΤΑΒΛΗΤΕΣ ΔΥΪΚΟΥ (w_{a,b,k}) --
var w {States, Actions} >= 0;

# -- 7. ΑΝΤΙΚΕΙΜΕΝΙΚΗ ΣΥΝΑΡΤΗΣΗ --
maximize Total_Dual_Reward:
    sum {(a,b) in States, k in Actions} cost[a,b,k] * w[a,b,k];

# -- 8. ΠΕΡΙΟΡΙΣΜΟΙ ΔΙΑΤΗΡΗΣΗΣ ΡΟΗΣ (Flow Balance) --
# W1 = sum {(x,y) in States} w[x,y,1] : Το συνολικό άθροισμα ροών Restart
subject to Flow_Balance {(a,b) in States}:
    w[a,b,1] + w[a,b,2] 
    = d[a,b]
      # Εισερχόμενη ροή από μετάβαση Success (a-1, b)
      + (if (a-1, b) in States then 
            c * ((a-1)/(a+b-1)) * (
                w[a-1,b,2] + (if (a-1 = prior_alpha and b = prior_beta) then sum {(x,y) in States} w[x,y,1] else 0)
            ) 
         else 0)
      # Εισερχόμενη ροή από μετάβαση Failure (a, b-1)
      + (if (a, b-1) in States then 
            c * ((b-1)/(a+b-1)) * (
                w[a,b-1,2] + (if (a = prior_alpha and b-1 = prior_beta) then sum {(x,y) in States} w[x,y,1] else 0)
            ) 
         else 0);