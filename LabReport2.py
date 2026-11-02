import random
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

POP_SIZE = 300       
CHROM_LEN = 80       
TARGET_ONES = 50 
MAX_FITNESS = 80
N_GENERATIONS = 50

TOURNAMENT_K = 3
CROSSOVER_RATE = 0.9
MUTATION_RATE = 1.0 / CHROM_LEN 

def fitness(individual: np.ndarray) -> float:
    """
    Returns 80 when the number of 1s equals 50.
    Penalizes linearly based on distance from TARGET_ONES.
    """
    ones = int(individual.sum())
    return MAX_FITNESS - abs(ones - TARGET_ONES)

def init_population(pop_size: int, chrom_len: int) -> np.ndarray:
    return np.random.randint(0, 2, size=(pop_size, chrom_len), dtype=np.int8)

def tournament_selection(pop: np.ndarray, fits: np.ndarray, k: int) -> np.ndarray:
    idxs = np.random.randint(0, len(pop), size=k)
    best_idx = idxs[np.argmax(fits[idxs])]
    return pop[best_idx].copy()

def single_point_crossover(p1: np.ndarray, p2: np.ndarray):
    if np.random.rand() > CROSSOVER_RATE:
        return p1.copy(), p2.copy()
    point = np.random.randint(1, CHROM_LEN)
    c1 = np.concatenate([p1[:point], p2[point:]])
    c2 = np.concatenate([p2[:point], p1[point:]])
    return c1, c2

def mutate(individual: np.ndarray) -> np.ndarray:
    mask = np.random.rand(CHROM_LEN) < MUTATION_RATE
    individual[mask] = 1 - individual[mask]
    return individual

def evolve(pop: np.ndarray, generations: int):
    best_history = []
    avg_history = []

    best_individual = None
    best_f = -np.inf

    for _ in range(generations):
        fits = np.array([fitness(ind) for ind in pop])
        gen_best_idx = np.argmax(fits)
        gen_best_f = fits[gen_best_idx]
        gen_avg_f = np.mean(fits)

        best_history.append(float(gen_best_f))
        avg_history.append(float(gen_avg_f))

        if gen_best_f > best_f:
            best_f = float(gen_best_f)
            best_individual = pop[gen_best_idx].copy()
        new_pop = []
        while len(new_pop) < len(pop):
            p1 = tournament_selection(pop, fits, TOURNAMENT_K)
            p2 = tournament_selection(pop, fits, TOURNAMENT_K)
            c1, c2 = single_point_crossover(p1, p2)
            c1 = mutate(c1)
            c2 = mutate(c2)
            new_pop.extend([c1, c2])
        pop = np.array(new_pop[:len(pop)], dtype=np.int8)

    return best_individual, best_f, best_history, avg_history

st.set_page_config(page_title="Bit Pattern GA (80 bits)", page_icon="🧬", layout="centered")

st.title("🧬 Genetic Algorithm: 80-Bit Pattern Generation")
st.caption("BSD3513 Artificial Intelligence — Lab Report 2")

with st.expander("ℹ️ Problem Setup & Specifications", expanded=True):
    st.write(
        f"""
- **Population Size**: `{POP_SIZE}`  
- **Chromosome Length**: `{CHROM_LEN}` bits  
- **Target Number of Ones**: `{TARGET_ONES}`  
- **Maximum Fitness**: `{MAX_FITNESS}` (at ones = {TARGET_ONES})  
- **Number of Generations**: `{N_GENERATIONS}`  
- **Selection**: Tournament (k={TOURNAMENT_K})  
- **Crossover**: Single-Point (rate={CROSSOVER_RATE})  
- **Mutation**: Bit-Flip (rate={MUTATION_RATE:.4f})  
"""
    )

col1, col2 = st.columns(2)
with col1:
    seed = st.number_input("Random Seed (for reproducibility)", min_value=0, value=42, step=1)
with col2:
    run_btn = st.button("Run Genetic Algorithm", type="primary")

if run_btn:
    random.seed(seed)
    np.random.seed(seed)

    population = init_population(POP_SIZE, CHROM_LEN)
    best_ind, best_fit, best_curve, avg_curve = evolve(population, N_GENERATIONS)

    ones_count = int(best_ind.sum())
    zeros_count = CHROM_LEN - ones_count
    bitstring = "".join(map(str, best_ind.tolist()))

    st.subheader("🏁 Optimization Results")
    st.metric(label="Best Fitness Achieved", value=f"{best_fit:.0f} / {MAX_FITNESS}")
    st.write(f"**Ones**: {ones_count} | **Zeros**: {zeros_count} | **Length**: {CHROM_LEN}")
    st.code(bitstring, language="text")
    st.subheader("📉 Convergence (Fitness per Generation)")
    fig, ax = plt.subplots(figsize=(7, 4.5))

    generations_range = range(1, N_GENERATIONS + 1)
    ax.plot(generations_range, best_curve, label="Best", color="tab:blue", linewidth=2.5)
    ax.plot(generations_range, avg_curve, label="Average", color="tab:green", linewidth=2.0, linestyle="--")
    ax.axhline(y=MAX_FITNESS, color="grey", linestyle=":", linewidth=1.5, label=f"Max = {MAX_FITNESS}")

    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness")
    ax.set_title("GA Convergence")
    ax.legend(loc="lower right")
    ax.grid(True, linestyle="--", alpha=0.4)

    st.pyplot(fig)

    if best_fit == MAX_FITNESS and ones_count == TARGET_ONES:
        st.success(f"Optimal condition met: Number of ones = {TARGET_ONES} with maximum fitness = {MAX_FITNESS}! ✅")
    else:
        st.info("Near-optimal result reached; try another random seed.")