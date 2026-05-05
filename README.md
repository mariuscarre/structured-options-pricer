# structured-options-pricer

An educational options pricing project built with Python and Streamlit.

It includes:
- Black-Scholes pricing for European call and put options
- Greeks (delta, gamma, vega, theta, rho)
- Monte Carlo pricing for European call and put options
- Volatility strategies: long straddle and long strangle
- Basic pytest tests

## Project Structure

```text
structured-options-pricer/
  app.py
  requirements.txt
  README.md
  core/
    black_scholes.py
    monte_carlo.py
  risk/
    greeks.py
  instruments/
    volatility_strategies.py
  tests/
    test_black_scholes.py
    test_monte_carlo.py
    test_strategies.py
```

## Quick Start

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the Streamlit app:

```bash
streamlit run app.py
```

4. Run tests:

```bash
pytest -q
```

## Notes

- Inputs assume annualized volatility and time in years.
- Greeks are Black-Scholes closed-form Greeks for European options.
- Monte Carlo uses a one-step terminal simulation under geometric Brownian motion.
