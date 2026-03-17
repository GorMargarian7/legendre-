import numpy as np
from scipy.special import legendre
import matplotlib.pyplot as plt
import seaborn as sns

# =========================
# 1. Կառուցում ենք մատրիցը
# =========================
def build_legendre_matrix(N):
    x = np.linspace(-1, 1, N)
    L = np.zeros((N, N), dtype=float)

    for j in range(N):
        Pj = legendre(j)
        for i in range(N):
            L[i, j] = Pj(x[i])

    return x, L


# =========================
# 2. Heatmap
# =========================
def plot_matrix(N):
    x, L = build_legendre_matrix(N)

    plt.figure(figsize=(12, 10))
    annot = True if N <= 8 else False

    sns.heatmap(L, annot=annot, fmt=".2f", cmap='viridis',
                xticklabels=[f'P{j}' for j in range(N)],
                yticklabels=[f'x{i}' for i in range(N)])

    plt.title(f'Legendre Matrix L (N={N})')
    plt.xlabel('Polynomial degree j')
    plt.ylabel('Node x_i')

    filename = f'legendre_matrix_{N}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

    print(f'✓ Saved: {filename}')


# =========================
# 3. Պոլինոմների գրաֆիկ
# =========================
def plot_polynomials(N):
    x_fine = np.linspace(-1, 1, 200)

    plt.figure(figsize=(12, 7))
    colors = plt.cm.viridis(np.linspace(0, 1, min(N, 8)))

    for j in range(min(N, 8)):
        Pj = legendre(j)
        y = Pj(x_fine)
        plt.plot(x_fine, y, label=f'P_{j}(x)', color=colors[j], linewidth=2)

    plt.title(f'Legendre Polynomials (first {min(N,8)})')
    plt.xlabel('x ∈ [-1, 1]')
    plt.ylabel('P_n(x)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axhline(0, color='black', lw=0.5)

    filename = f'legendre_polynomials_{N}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

    print(f'✓ Saved: {filename}')


# =========================
# 4. Կոդավորում
# =========================
def encode_text(text):
    # Տեքստ → ASCII
    m = np.array([ord(ch) for ch in text], dtype=float)
    N = len(m)

    # Մատրից
    x, L = build_legendre_matrix(N)

    # Կոդավորում
    c = L @ m

    return m, c, L


# =========================
# 5. Վերծանում
# =========================
def decode_text(c, L):
    # Հակադարձ մատրից
    L_inv = np.linalg.inv(L)

    # Վերականգնում
    m_rec = L_inv @ c

    # Կլորացում → սիմվոլներ
    m_rec = np.round(m_rec).astype(int)
    text_rec = ''.join(chr(val) for val in m_rec)

    return text_rec


# =========================
# 6. Գործարկում
# =========================

# Գրաֆիկներ
for N in [5, 8, 12]:
    plot_matrix(N)
    plot_polynomials(N)

# Տեքստ
text = "STUDENT-DEMO"

print("\nՍկզբնական տեքստ:", text)

# Կոդավորում
m, c, L = encode_text(text)

print("\nASCII վեկտոր m:")
print(m)

print("\nԿոդավորված վեկտոր c = L·m:")
print(c)

# Վերծանում
decoded_text = decode_text(c, L)

print("\nՎերականգնված տեքստ:")
print(decoded_text)

# determinant
print(f"\ndet(L) ≈ {np.linalg.det(L):.2e}")