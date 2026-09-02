# -- ΠΑΡΑΜΕΤΡΟΙ --
param N > 0, integer;              
param c >= 0, < 1;                 
param mu_known >= 0;               

param prior_alpha >= 0, integer;   
param prior_beta >= 0, integer;    

# -- ΕΓΚΥΡΕΣ ΚΑΤΑΣΤΑΣΕΙΣ (Τριγωνικό πλέγμα) --
# Ο συνολικός αριθμός βημάτων δεν μπορεί να ξεπεράσει το N
set States := {a in prior_alpha..(N + prior_alpha), 
               b in prior_beta..(N + prior_beta) 
               : (a - prior_alpha) + (b - prior_beta) < N};

# -- ΣΥΝΤΕΛΕΣΤΕΣ & ΜΕΤΑΒΛΗΤΕΣ --
param d {States} default 1.0;
var V {States} >= 0;

# -- ΑΝΤΙΚΕΙΜΕΝΙΚΗ ΣΥΝΑΡΤΗΣΗ --
minimize Total_Value:
    sum {(a,b) in States} d[a,b] * V[a,b];

# -- ΠΕΡΙΟΡΙΣΜΟΙ --

# 1. Περιορισμός για την ενέργεια Restart
subject to Restart {(a,b) in States}:
    V[a,b] >= prior_alpha/(prior_alpha+prior_beta) 
              + c * (prior_alpha/(prior_alpha+prior_beta)) * (V[prior_alpha+1,prior_beta])
              + c * (prior_beta/(prior_alpha+prior_beta)) * (V[prior_alpha,prior_beta+1]);  

# 2. Περιορισμός για την ενέργεια Continue (με έλεγχο αν οι επόμενες καταστάσεις ανήκουν στο States)
subject to Continue {(a,b) in States}:
    V[a,b] >= a/(a+b) 
              + c * (a/(a+b)) * (if (a+1, b) in States then V[a+1,b] else 0)
              + c * (b/(a+b)) * (if (a, b+1) in States then V[a,b+1] else 0);  #synoriakes 
