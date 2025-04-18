import matplotlib.pyplot as plt

# v   = [parameter, MDS]
Unet = [17264147, 0.62949]
Unet2 = [2033043, 0.56192]
v1 = [1138051, 0.44815]
v1_2 = [1653155, 0.45835]
v1_3 = [1322691, 0.44493] #--
v2 = [1146563, 0.45482]
v3 = [1417907, 0.48999]
v3_2 = [1933011, 0.49807]
v3_2_2_red4 = [1965711, 0.50043]
v3_2_2_red8 = [1958441, 0.50026]
v3_2_2_red16 = [1954806,0.49861]
v4 = [5477043, 0.48426]
v4_2 = [2544307, 0.50191]
v4_2_2_red4 = [2597931, 0.50761]
v4_2_2_red8 = [2580223, 0.50576]
v4_2_2_red16 = [2571369, 0.51225]
v4_2_2_red32 = [2566942, 0.51101]
v4_3 = [3241331, 0.51602]

models = {
    "Unet": Unet,
    "Small Unet": Unet2,
    "v1": v1,
    "v1.2": v1_2,
    "v1.3": v1_3,
    "v2": v2,
    "v3": v3,
    "v3.2": v3_2,
    "v3.2.2 red=4": v3_2_2_red4,
    "v3.2.2 red=8": v3_2_2_red8,
    "v3.2.2 red=16": v3_2_2_red16,
    #"v3.3": v3_3,
    "v4": v4,
    "v4.2": v4_2,
    "v4.2.2 red=4": v4_2_2_red4,
    "v4.2.2 red=8": v4_2_2_red8,
    "v4.2.2 red=16": v4_2_2_red16,
    "v4.2.2 red=32": v4_2_2_red32,
    "v4.3": v4_3,
    #"v4.4": v4_4,
}

# Compute efficiency = accuracy / parameters
efficiency_data = [(name, val[1] / (val[0]*1e-9)) for name, val in models.items() if val[0] != 0]

# Sort by efficiency
efficiency_data.sort(key=lambda x: x[1], reverse=True)

# Print sorted efficiencies
for name, eff in efficiency_data:
    print(f"{name}: efficiency = {eff:.8f}")

colors = plt.cm.get_cmap('tab20', 20)
i=0
# Plotting param vs accuracy
plt.figure(figsize=(10, 6))
for name, (params, acc) in models.items():
    if params != 0:
        plt.scatter(params, acc, label=name, color=colors(i), edgecolors='k')
        i = i + 1

# Add 80% of Unet's MDS as a horizontal line
threshold = 0.8 * Unet[1]
plt.axhline(y=threshold, color='r', linestyle='--', label=f'80% of Unet MDS ({threshold:.4f})')

plt.xlabel("Parameters")
plt.ylabel("Accuracy (MDS)")
plt.title("Model Parameter Count vs. Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Print max efficiency
max_eff = max(eff for _, eff in efficiency_data)
print(f"max_efficiency: {max_eff:.8f}")
