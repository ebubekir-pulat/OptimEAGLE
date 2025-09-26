# Plot Input Tokens, Tokens Generated Per Second Data

from matplotlib import pyplot as plt

input_tokens = []
tokens_gen_per_s = []

plt.title("Input Tokens vs Tokens Generated Per Second")
plt.xlabel("Number of Input Tokens")
plt.ylabel("Tokens Generated Per Second")
plt.bar(input_tokens, tokens_gen_per_s)
plt.savefig("InputTokens_vs_TokensGenPerS.png")