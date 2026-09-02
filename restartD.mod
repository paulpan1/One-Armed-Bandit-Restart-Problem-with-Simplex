# ==============================================================
# ΔΥΪΚΟ ΜΟΝΤΕΛΟ ΓΡΑΜΜΙΚΟΥ ΠΡΟΓΡΑΜΜΑΤΙΣΜΟΥ - RESTART (restartD.mod)
# ==============================================================

# -- 1. ΠΑΡΑΜΕΤΡΟΙ --
param N integer > 0;               # Ορίζοντας
param c >= 0, < 1;                 # Προεξοφλητικός παράγοντας
param prior_alpha integer >= 0;    # Prior α (a0)
param prior_beta integer >= 0;     # Prior β (b0)

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
# ΔΙΟΡΘΩΣΗ 1: Αναλυτική διάσπαση στις 4 περιπτώσεις καταστάσεων
subject to Flow_Balance {(a,b) in States}:
    w[a,b,1] + w[a,b,2] 
    = d[a,b]
      + (
        # 1. Αρχική κατάσταση (prior_alpha, prior_beta)
        if (a = prior_alpha and b = prior_beta) then 
            0

        # 2. Ειδική κατάσταση 1: (prior_alpha + 1, prior_beta)
        else if (a = prior_alpha + 1 and b = prior_beta) then 
            c * (prior_alpha / (prior_alpha + prior_beta)) * w[prior_alpha, prior_beta, 2]
            + c * (prior_alpha / (prior_alpha + prior_beta)) * sum {(x,y) in States} w[x,y,1]
            + c * (prior_alpha / (prior_alpha + prior_beta)) * sum {(x,y) in States : (x - prior_alpha) + (y - prior_beta) = N - 1} w[x,y,2]

        # 3. Ειδική κατάσταση 2: (prior_alpha, prior_beta + 1)
        else if (a = prior_alpha and b = prior_beta + 1) then 
            c * (prior_beta / (prior_alpha + prior_beta)) * w[prior_alpha, prior_beta, 2]
            + c * (prior_beta / (prior_alpha + prior_beta)) * sum {(x,y) in States} w[x,y,1]
            + c * (prior_beta / (prior_alpha + prior_beta)) * sum {(x,y) in States : (x - prior_alpha) + (y - prior_beta) = N - 1} w[x,y,2]

        # 4. Υπόλοιπες καταστάσεις (s,f) στο States
        else 
            (if (a-1, b) in States then c * ((a-1)/(a+b-1)) * w[a-1,b,2] else 0)
            + (if (a, b-1) in States then c * ((b-1)/(a+b-1)) * w[a,b-1,2] else 0)
      );

# ΔΙΟΡΘΩΣΗ 2: Επιβολή της ενέργειας CONTINUE (k=2) στην αρχική κατάσταση (prior_alpha, prior_beta)
subject to Force_Continue_Initial:
    w[prior_alpha, prior_beta, 1] = 0;

    
