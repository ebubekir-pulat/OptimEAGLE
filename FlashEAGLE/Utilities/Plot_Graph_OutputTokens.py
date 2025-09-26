# Plot Output Tokens, Tokens Generated Per Second Data

from matplotlib import pyplot as plt

output_tokens = []
tokens_gen_per_s = []

plt.title("Output Tokens vs Tokens Generated Per Second")
plt.xlabel("Number of Output Tokens")
plt.ylabel("Tokens Generated Per Second")
plt.bar(output_tokens, tokens_gen_per_s)
plt.savefig("OutputTokens_vs_TokensGenPerS.png")