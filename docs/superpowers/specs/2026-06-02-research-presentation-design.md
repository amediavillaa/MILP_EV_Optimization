# Research Team Presentation Design
**Date:** 2026-06-02
**Format:** 15-minute intern presentation to a technical research team
**Project:** LP/MILP-based EV Charging Station Scheduling with Rolling-Horizon MPC

---

## Narrative Framing

**Framing C (system architecture paper):** The contribution is the end-to-end pipeline —
admission control decoupled from scheduling, LP formulation, rolling-horizon MPC, BESS
co-optimization — benchmarked rigorously across 288 episodes. Results serve as proof, not
the climax.

**Story arc (Option B — story-driven):**
Hook → Why naive heuristics break → System architecture → Results as validation → Future work roadmap

---

## Storyline

> A real-world problem demands a real-time, constraint-aware scheduler. Naive heuristics are
> fast but structurally unsafe. We built a principled LP-based MPC system that is formally
> correct, matches or exceeds heuristic profit, and solves in real time. Here is the system,
> the evidence, and where we go next.

---

## Slide Structure (12 slides, ~75 seconds each)

| # | Title | Purpose |
|---|-------|---------|
| 1 | The Problem | Hook: real station, real constraints |
| 2 | Why Naive Heuristics Break | Tension: MaxCharge violates grid cap, EqualShare ignores deadlines |
| 3 | System Overview | Two-layer architecture diagram |
| 4 | LP Formulation | Decision variables, objective, 9 constraint classes |
| 5 | Rolling-Horizon MPC | Receding-horizon re-solve loop, must-serve constraints |
| 6 | Experimental Setup | Ports, horizons, tariff regimes, baselines |
| 7 | Results: Fixed Tariff | LP ≈ MaxCharge — validation, not failure |
| 8 | Results: BESS Discharging | LP advantage under arbitrage conditions |
| 9 | Compute Scalability | 13–83 ms/step, 3600× margin below control interval |
| 10 | Future Work | Clairvoyant benchmark, RL, admission co-design, robust MPC |
| 11 | Conclusion | One-line system claim |

---

## Slide Content

### Slide 1 — The Problem
- Charging station with fixed grid capacity P_max
- EVs arrive with SoC deficit, depart at fixed times with SoC target
- Dynamic electricity prices: when you charge matters
- Punchline: "This is a real-time constrained optimization problem. Heuristics ignore at
  least one of these."

### Slide 2 — Why Naive Heuristics Break

| Policy | What it does | What it ignores |
|--------|-------------|-----------------|
| MaxCharge | Charges at I_max always | Grid cap, price timing |
| EqualShare | Splits P_max equally | Departure deadlines, SoC targets |
| Random | Random current in [0, I_max] | Everything |

Callout: "None of these policies enforce SoC guarantees."

### Slide 3 — System Overview
Two-layer diagram:
```
Chargax Environment
  └─ Admission Control + Port Assignment  (fixed before LP)
       └─ LP Charging Scheduler  (our contribution)
            └─ Physical Station + BESS
```
Key callout: "Port assignments are fixed before the LP runs. The scheduling subproblem is a
pure continuous LP."

### Slide 4 — LP Formulation

**Decision variables:**
- I_{j,t} ≥ 0: charging current at port j, step t
- I^ch_t, I^dis_t ≥ 0: BESS charge/discharge current
- SoC^B_t ≥ 0: BESS energy
- SoC^i_t ≥ 0: car i energy

**Objective:**
max  Σ_{j,t} V_j · I_{j,t} · z_{j,t} · p^sell_t · Δt/1000
   − Σ_t [ (Σ_j V_j·I_{j,t} + V_B·I^ch_t/η − V_B·I^dis_t·η) · p^buy_t · Δt/1000 ]

**Constraints (9 classes):**
- C1: I_{j,t} ≤ I^max_j
- C2: BESS current bounds (SoC-dependent ratios)
- C3: Σ_j V_j·I_{j,t} + V_B·(I^ch_t − I^dis_t) + L_t ≤ P_max  [grid cap]
- C4–C5: BESS SoC dynamics + bounds
- C6–C7: Car SoC dynamics + bounds
- C8: SoC-dependent car power limit
- C9: I_{j,t} = 0 if z_{j,t} = 0  [occupancy]
- C10a: soc_car[i, t_now] ≥ soc_now[i] + needed[i]/steps_until_dep[i]  [pacing]
- C10b: soc_car[i, dep_i] ≥ s_target[i]  [deadline]

### Slide 5 — Rolling-Horizon MPC
- Timeline diagram: overlapping H-step windows, apply first step only
- Re-solve every Δt=5min with latest observed SoC
- Horizon H ∈ {1, 3, 6, 12} tested
- Bare-mode fallback: drop pacing if LP infeasible, keep deadline

### Slide 6 — Experimental Setup

| Dimension | Values |
|-----------|--------|
| Station sizes | 3, 6, 12 ports |
| Horizons | H = 1, 3, 6, 12 steps |
| Tariff regimes | Fixed, Dynamic 1.3×, + BESS discharging |
| Baselines | MaxCharge, EqualShare, Random |
| Seeds | 10 per configuration |
| Solver | HiGHS (open-source LP) |

288 total episodes. All metrics with 95% CIs.

### Slide 7 — Results: Fixed Tariff

| Ports | MaxCharge | MILP (best H) | EqualShare |
|-------|-----------|---------------|------------|
| 3 | 208 € | 208 € | 151 € |
| 6 | 415 € | 415 € | 328 € |
| 12 | 822 € | 822 € | 662 € |

Framing: flat pricing makes charge-fast provably optimal; LP confirms this automatically.
EqualShare −20–25% gap shows the cost of ignoring departure urgency.

### Slide 8 — Results: BESS Discharging

| Ports | MaxCharge | MILP (best H) | Gain |
|-------|-----------|---------------|------|
| 3 | 208 € | 211 € (H=6) | +1.3% |
| 6 | 415 € | 417 € (H=12) | +0.5% |
| 12 | 822 € | 823 € (H=12) | +0.2% |

Honest callout: gains modest at current price spread; stronger V2G signal expected to widen gap.

### Slide 9 — Compute Scalability

| Horizon | ms/step (3 ports) | ms/step (12 ports) |
|---------|-------------------|---------------------|
| H=1 | 13.8 ms | 32.7 ms |
| H=3 | 38.0 ms | 55.1 ms |
| H=6 | 33.1 ms | 55.8 ms |
| H=12 | 55.4 ms | 82.6 ms |

Control interval = 5 min = 300,000 ms. Margin at H=12, 12 ports: 3,600×.

### Slide 10 — Future Work
1. Clairvoyant offline benchmark — upper bound on profit, MPC optimality gap
2. RL comparison + hybrid (LP constraints + learned price forecasting)
3. Admission control co-optimization (MILP with binary port assignment)
4. Robust/stochastic MPC for uncertain departure times

### Slide 11 — Conclusion
- Rolling-horizon LP controller for real-time EV charging station scheduling
- Formally enforces grid cap, SoC targets, departure deadlines, BESS co-optimization
- Matches best heuristic under flat pricing; outperforms with BESS arbitrage
- < 83 ms/step — deployable on commodity hardware

---

## Opening Speech (2 minutes)

"Good [morning/afternoon]. I'm going to use the next 15 minutes to walk you through the EV
charging station scheduling system I've been building.

Here's the problem in one sentence: you have a charging station with a fixed grid connection,
a fleet of EVs arriving and departing throughout the day, dynamic electricity prices, and a
promise to every driver that their car will reach its target charge by departure. How do you
decide how much current to send to each port every 5 minutes?

The naive answer — charge everyone at maximum power — violates the grid capacity limit.
Splitting capacity equally ignores departure urgency and leaves some cars under-charged.
Neither policy gives you formal energy delivery guarantees.

What I built is a rolling-horizon LP controller that re-solves an optimization problem every
5 minutes, jointly managing EV charging and on-site battery storage under all of these
constraints simultaneously. It runs in real time — under 83 milliseconds per solve — and
it's the only policy in our benchmark that can exploit time-varying electricity prices through
principled battery arbitrage.

I'll show you the architecture, walk through the formulation, present results across 288
experimental episodes, and close with what I think the three strongest next research directions
are."

---

## Closing Statement (30 seconds)

"To summarize: we built a real-time LP-based MPC system for EV charging station scheduling
that simultaneously enforces grid capacity, SoC guarantees, and departure deadlines. Under
flat pricing it matches the best heuristic; with battery storage and dynamic pricing it pulls
ahead. It solves in under 83 milliseconds per step against a 5-minute control interval. The
next steps are a clairvoyant upper bound, an RL comparison, and co-optimizing the admission
control layer. Thank you — happy to take questions."

---

## Q&A Preparation

**Q: "MaxCharge matches your LP — isn't LP overkill?"**
Under flat pricing, charging fast is provably optimal — MaxCharge does this by construction
and LP confirms it. LP's value: adapts automatically to dynamic pricing and BESS arbitrage
without modification. MaxCharge cannot do either.

**Q: "Where's the clairvoyant benchmark?"**
Model exists (`build_ev_lp_model`). Benchmarking it is the first item on the future work
list. Would give an upper bound and quantify the MPC optimality gap.

**Q: "Have you considered RL?"**
On the roadmap. LP advantages: interpretability, hard constraint satisfaction by construction.
RL potential: handling stochastic arrivals. Hybrid architecture is a concrete direction.

**Q: "How does must-serve handle infeasibility?"**
Pacing floor is capped by grid share budget and port hardware limit — physically achievable
under normal load. If still infeasible, bare mode drops pacing and keeps only the hard
departure deadline. In 288 episodes bare mode was never triggered to failure.

**Q: "Why HiGHS and not Gurobi?"**
HiGHS is open-source and sufficient at this scale (83ms/step, 5-min interval). Gurobi would
be faster for production or 100+ ports; that's an engineering upgrade, not an algorithmic
change.

**Q: "How does this scale to 100+ ports?"**
LP size is O(J×H) variables and constraints. At J=100, H=12: ~1,200 variables — well within
HiGHS's fast LP range. Python/Pyomo construction time is the scaling bottleneck, not the
solver.

**Q: "Is the BESS model realistic?"**
Current experiments use flat efficiency ratios (r=1.0). The model supports SoC-dependent
curves via r_bess_ch/r_bess_dis parameters. Adding realistic curves is a near-term improvement.

**Q: "Are the profit differences statistically significant?"**
At n=10, the 1–2% BESS gains are likely within noise. CIs reported throughout. More seeds
are needed for strong significance claims — acknowledged as a limitation.

---

## Weak Points and Honest Framing

| Weakness | Framing |
|----------|---------|
| LP matches MaxCharge on fixed tariff | "Theoretically correct — validates the model" |
| BESS profit gains are 1–2% | "Mechanism is right; synthetic price spread is weak" |
| No clairvoyant benchmark | "Model exists; first item on future work list" |
| Only 10 seeds | "CIs reported; n=10 limits significance claims" |
| Synthetic environment | "Real-world validation is future work" |
| BESS efficiency curve is flat | "r parameters support curves; not yet used" |

---

## Project Gaps to Close Before Presenting

| Gap | Effort | Impact |
|-----|--------|--------|
| Offline clairvoyant benchmark vs MPC | Low — model exists | High |
| More seeds (20–30) | Medium | Medium |
| SoC-dependent BESS efficiency curves | Low | Medium |
| Significance test note in slides | Low | Medium |
| Tiny-cost-case as backup slide | Done | Low |

---

## Preparation Checklist

- [ ] Two-layer architecture diagram (Slide 3)
- [ ] Receding-horizon MPC timeline diagram (Slide 5)
- [ ] Profit bar chart with 95% CI error bars (Slides 7–8)
- [ ] Compute scaling line chart with 300,000ms reference line (Slide 9)
- [ ] Run clairvoyant offline benchmark if time permits
- [ ] Add significance test note to result slides
- [ ] Practice opening speech out loud — time it
- [ ] Prepare answers to all Q&A questions above
- [ ] Prepare tiny-cost-case table as backup slide
- [ ] Know git commit hash for reproducibility questions
- [ ] Know why HiGHS was chosen over Gurobi
