import numpy as np

from core.hedging import hedge_quantity, simulate_delta_hedge_pnl


def test_hedge_quantity_long_and_short():
    delta = 0.6
    position = 10

    long_hedge = hedge_quantity(delta, position, "long")
    short_hedge = hedge_quantity(delta, position, "short")

    assert long_hedge == -6.0
    assert short_hedge == 6.0


def test_delta_hedge_simulation_outputs_shapes():
    result = simulate_delta_hedge_pnl(
        option_type="call",
        side="long",
        position=5,
        spot=100,
        strike=100,
        rate=0.05,
        volatility=0.2,
        time_to_expiry=1.0,
        n_points=21,
    )

    assert len(result["moves_pct"]) == 21
    assert len(result["option_pnl"]) == 21
    assert len(result["hedge_pnl"]) == 21
    assert len(result["total_pnl"]) == 21
    assert np.isclose(result["moves_pct"][0], -10.0)
    assert np.isclose(result["moves_pct"][-1], 10.0)
