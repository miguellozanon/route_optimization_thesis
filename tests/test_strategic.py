from brlo.strategic.network import default_network, stochastic_metrics


def test_stochastic_bounds_order():
    nd = default_network(T=4)
    m = stochastic_metrics(nd, time_limit=60)
    # WS <= RP <= EEV  (Birge & Louveaux, Ch. 4)
    assert m["WS"] <= m["RP"] + 1e-3
    assert m["RP"] <= m["EEV"] + 1e-3
    assert m["VSS"] >= -1e-3 and m["EVPI"] >= -1e-3
