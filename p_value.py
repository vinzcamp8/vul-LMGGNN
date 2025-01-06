'''
Test di McNemar
p-value < 0.05: Differenza statisticamente significativa tra i due modelli.
p-value >= 0.05: Nessuna evidenza che uno dei due modelli sia significativamente migliore.
'''

import pandas as pd
from scipy.stats import chi2

# Carica i dati raw
txt1 = "data/model/no_embsize_scheduler_and_weights/vul_lmgnn_5e-06_32_10_1e-06_0.5/BertGGCN_test_probabilities.txt"
txt2 = "data/model/vul_lmgnn_1e-05_64_100_1.3e-06_0.5/BertGGCN_test_probabilities.txt"
txt1_data = pd.read_csv(txt1, sep="\t", comment="#", header=None, names=["True", "Predicted", "Probability"])
txt2_data = pd.read_csv(txt2, sep="\t", comment="#", header=None, names=["True", "Predicted", "Probability"])

# Calcola le discordanti
b = ((txt1_data["Predicted"] == txt1_data["True"]) & (txt2_data["Predicted"] != txt2_data["True"])).sum()
c = ((txt2_data["Predicted"] == txt2_data["True"]) & (txt1_data["Predicted"] != txt1_data["True"])).sum()

# Calcola chi-quadro
chi_squared = (b - c)**2 / (b + c) if (b + c) > 0 else 0

# Calcola il p-value
p_value = chi2.sf(chi_squared, df=1)

txt1_name = txt1.split("/")[-2]
txt2_name = txt2.split("/")[-2]

print(f"b ({txt1_name} corretto, {txt2_name} sbagliato): {b}")
print(f"c ({txt2_name} corretto, {txt1_name} sbagliato): {c}")
print(f"Chi-squared: {chi_squared}")
print(f"P-value: {p_value}")