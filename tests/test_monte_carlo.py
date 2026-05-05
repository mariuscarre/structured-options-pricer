from core.black_scholes import call_price, put_price
from core.monte_carlo import european_option_monte_carlo


def test_monte_carlo_close_to_black_scholes_call():
    bs = call_price(100, 100, 0.05, 0.2, 1.0)
    mc = european_option_monte_carlo(100, 100, 0.05, 0.2, 1.0, n_simulations=120_000, option_type="call", seed=7)
    assert abs(bs - mc) < 0.5


def test_monte_carlo_close_to_black_scholes_put():
    bs = put_price(100, 100, 0.05, 0.2, 1.0)
    mc = european_option_monte_carlo(100, 100, 0.05, 0.2, 1.0, n_simulations=120_000, option_type="put", seed=7)
    assert abs(bs - mc) < 0.5
