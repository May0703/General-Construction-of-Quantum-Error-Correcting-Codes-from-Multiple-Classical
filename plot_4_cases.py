# This code plots the code distance vs. code dimension for four different quantum error-correcting code constructions in paper:
# General Construction of Quantum Error-Correcting Codes from Multiple Classical Codes
# Author: Yue Wu
# Date: November 2025

from sympy import factorint
import math
from itertools import permutations
from matplotlib import pyplot as plt
from dimenison import calculate_code_dimension

def get_all_factors(prime_factors):
    factors = []
    primes = []
    exponents = []
    for prime, exp in prime_factors.items():
        primes.append(prime)
        exponents.append(exp)
    def generate_factors(idx, current):
        if idx == len(primes):
            if current != 1:
                factors.append(current)
            return
        for exp in range(exponents[idx] + 1):
            generate_factors(idx + 1, current * (primes[idx] ** exp))
    generate_factors(0, 1)
    return sorted(set(factors))

def find_factor_triplets_no_one(n):
    prime_factors = factorint(n)
    all_factors = get_all_factors(prime_factors)
    triplets = {}
    for i in range(len(all_factors)):
        a = all_factors[i]
        if a > n**(1/3):
            continue
        remaining = n // a
        for j in range(i, len(all_factors)):
            b = all_factors[j]
            if remaining % b == 0:
                c = remaining // b
                if c in all_factors and c >= b:
                    triplet = tuple(sorted([a, b, c]))
                    if a * b * c == n:
                        triplets[triplet] = triplet
    return list(triplets.values())

def calculate_data(number):
    toric3Dnumber = number // 3
    fljnumber = number // 4
    toric3D_gnumber = number // 4
    flj_gnumber = number // 4

    fljks, fljds = [], []
    triplets = find_factor_triplets_no_one(fljnumber)
    for triplet in triplets:
        for perm in permutations(triplet):
            a, b, c = perm
            gcd_ab = math.gcd(a, b)
            k = calculate_code_dimension(a, b, c,2)
            d = min(2 * a * b / gcd_ab, a * b, c)
            fljks.append(k)
            fljds.append(d)

    toric3Dks, toric3Dds = [], []
    triplets = find_factor_triplets_no_one(toric3Dnumber)
    for triplet in triplets:
        for perm in permutations(triplet):
            a, b, c = perm
            k = calculate_code_dimension(a, b, c,1)
            d = min(a, b, c)
            toric3Dks.append(k)
            toric3Dds.append(d)

    flj_gks, flj_gds = [], []
    triplets = find_factor_triplets_no_one(flj_gnumber)
    for triplet in triplets:
        for perm in permutations(triplet):
            a, b, c = perm
            gcd_ab = math.gcd(a, b)
            k = calculate_code_dimension(a, b, c,4)
            d = min(2 * a * b / gcd_ab, a * b, c)
            flj_gks.append(k)
            flj_gds.append(d)

    toric3D_gks, toric3D_gds = [], []
    triplets = find_factor_triplets_no_one(toric3D_gnumber)
    for triplet in triplets:
        for perm in permutations(triplet):
            a, b, c = perm
            k = calculate_code_dimension(a, b, c,3)
            d = min(a, b, c) #min(a,b,c) is smaller than 4 here
            toric3D_gks.append(k)
            toric3D_gds.append(d)

    return {
        'toric3D': (toric3Dks, toric3Dds),
        'flj': (fljks, fljds),
        'toric3D_g': (toric3D_gks, toric3D_gds),
        'flj_g': (flj_gks, flj_gds)
    }

# Plotting
numbers = [144, 432]
fig, axs = plt.subplots(2, 1, figsize=(12, 12), sharex=False)

markers = ['s', '^', 'D', 'v']
colors = ['red', 'gold', 'green', 'blue']
labels = ['CASE A', 'CASE B', 'CASE C', 'CASE D']
keys = ['toric3D', 'flj', 'toric3D_g', 'flj_g']

for ax, number in zip(axs, numbers):
    data = calculate_data(number)
    for i, key in enumerate(keys):
        ks, ds = data[key]
        ax.scatter(ks, ds,
                   label=labels[i],
                   alpha=0.7,
                   marker=markers[i],
                   s=200,
                   color=colors[i])
    ax.set_ylabel('Code Distance (d)', fontsize=20)
    ax.legend(fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')

x_ticks0 = [3, 4, 6, 8, 12, 20, 36]
x_tick_labels0 = ['3', '4', '6', '8',"12",'20', '36']
x_ticks1 = [3, 4, 6, 8, 12, 20, 40, 60, 100]
x_tick_labels1 = ['3', '4', '6', '8',"12",'20', '40', '60', '100']

axs[0].set_ylim(1, 7)
axs[0].set_xlim(2, 40)
axs[0].set_xticks(x_ticks0)
axs[0].set_xticklabels(x_tick_labels0)
axs[0].tick_params(axis='x', labelsize=20)
axs[0].tick_params(axis='y', labelsize=20)

axs[1].set_ylim(1, 10)
axs[1].set_xlim(2, 120)
axs[1].set_xticks(x_ticks1)
axs[1].set_xticklabels(x_tick_labels1)
axs[1].tick_params(axis='x', labelsize=20)
axs[1].tick_params(axis='y', labelsize=20)

axs[0].tick_params(axis='x', which='minor', length=0)
axs[0].xaxis.set_minor_locator(plt.NullLocator())
axs[1].tick_params(axis='x', which='minor', length=0)
axs[1].xaxis.set_minor_locator(plt.NullLocator())

axs[1].set_xlabel('Code Dimension (k)', fontsize=20)

# Add (a) and (b) outside the subplots
fig.text(0.04, 0.96, '(a)', fontsize=24, va='top')
fig.text(0.22, 0.92, '(2,3,6)/(1,6,6)', fontsize=24, va='top')
fig.text(0.855, 0.78, '(3,3,4)', fontsize=24, va='top')
fig.text(0.17, 0.49, '(3,4,9)', fontsize=24, va='top')
fig.text(0.32, 0.49, '(2,6,9)', fontsize=24, va='top')
fig.text(0.835, 0.36, '(3,3,12)', fontsize=24, va='top')
fig.text(0.04, 0.52, '(b)', fontsize=24, va='top')

plt.tight_layout(rect=[0.05, 0.03, 1, 0.97])
plt.savefig("figure_144_432.pdf", format='pdf', bbox_inches='tight')
plt.show()