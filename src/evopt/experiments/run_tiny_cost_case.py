def make_small_data():
    J = 3
    T = 8
    I = 3

    V = {1: 230, 2: 230, 3: 230}
    I_max = {1: 16, 2: 16, 3: 16}

    p_buy = {
        1: 0.30, 2: 0.28, 3: 0.25, 4: 0.20,
        5: 0.22, 6: 0.27, 7: 0.33, 8: 0.36
    }
    L = {t: 5.0 for t in range(1, T + 1)}

    E = {1: 4.0, 2: 3.0, 3: 2.0}
    arrival = {1: 1, 2: 2, 3: 4}
    departure = {1: 6, 2: 7, 3: 8}

    # Fixed assignment y[i,j]
    y = {}
    port_of_ev = {1: 1, 2: 2, 3: 3}
    for i in range(1, I + 1):
        for j in range(1, J + 1):
            y[(i, j)] = 1 if port_of_ev[i] == j else 0

    # Occupancy z[j,t]
    z = {}
    for j in range(1, J + 1):
        for t in range(1, T + 1):
            z[(j, t)] = 0

    for i in range(1, I + 1):
        j = port_of_ev[i]
        for t in range(arrival[i], departure[i] + 1):
            z[(j, t)] = 1

    return {
        "J": J,
        "T": T,
        "I": I,
        "delta_t": 0.25,
        "P_max": 25.0,
        "V": V,
        "I_max": I_max,
        "L": L,
        "p_buy": p_buy,
        "E": E,
        "arrival": arrival,
        "departure": departure,
        "y": y,
        "z": z,
    }