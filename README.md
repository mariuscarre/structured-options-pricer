# Structured Options Pricer

Application pédagogique de pricing d'options et de produits structures, developpee avec Python et Streamlit.

## Fonctionnalites principales

- Section **Black-Scholes**: pricing des options europeennes (call/put)
- Section **Greeks**: delta, gamma, vega, theta, rho
- Section **Monte Carlo**: pricing stochastique des options europeennes
- Section **Strategies de volatilite**: long straddle, long strangle
- Section **Structuring Summary**: budget optionnel, decomposition et resume
- Section **Market Vanilla Options**: mode `synthetic` par defaut, mode `live` disponible

## Structure du projet

```text
structured-options-pricer/
  app.py
  requirements.txt
  README.md
  core/
  data/
  instruments/
  risk/
  tests/
```

## Installation rapide

1. Creer et activer un environnement virtuel
2. Installer les dependances:

```bash
pip install -r requirements.txt
```

3. Lancer l'application Streamlit:

```bash
streamlit run app.py
```

4. Lancer les tests:

```bash
pytest -q
```

## Deploiement Streamlit Cloud

Lien direct de l'application en ligne:

- [https://structured-options-pricer.streamlit.app](https://structured-options-pricer.streamlit.app)

Code source GitHub:

- [https://github.com/mariuscarre/structured-options-pricer](https://github.com/mariuscarre/structured-options-pricer)

Apres un push:
- faire un **Reboot app** dans Streamlit Cloud
- puis un refresh navigateur (`Ctrl+F5`)

## Notes techniques

- Les volatilites sont annualisees.
- Le temps est exprime en annees dans les modeles.
- Les Greeks sont calcules en formule fermee Black-Scholes.
- Le module Monte Carlo repose sur une dynamique GBM (geometric Brownian motion).
