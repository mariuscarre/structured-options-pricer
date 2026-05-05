# Structured Options Pricer

Application pédagogique de pricing d'options et de produits structures, developpee avec Python et Streamlit.

## Fonctionnalites principales

- Pricing Black-Scholes des options europeennes (call/put)
- Greeks (delta, gamma, vega, theta, rho)
- Pricing Monte Carlo pour options europeennes
- Strategies de volatilite (long straddle, long strangle)
- Module de structuration (budget optionnel, decomposition et resume)
- Module "Market Vanilla Options" avec:
  - mode `live` (donnees de marche via Yahoo Finance)
  - mode `synthetic` (surface implicite realiste, plus stable en demo)
  - nettoyage robuste des chaines d'options
  - ATM IV robuste
  - controles de qualite (liquidite, monotonicite, convexite)

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

## Utilisation du module Market Vanilla Options

- Le `Data Mode` est **par defaut sur `synthetic`** pour eviter les erreurs de quota Yahoo en demonstration.
- Le mode `live` reste disponible pour recuperer des quotes de marche reelles.
- Si Yahoo limite les requetes (`Too Many Requests`), basculer temporairement en `synthetic` puis reessayer en `live`.

## Deploiement Streamlit Cloud

L'application deployee suit la branche `main` du repo GitHub:

- [https://github.com/mariuscarre/structured-options-pricer](https://github.com/mariuscarre/structured-options-pricer)

Apres un push:
- faire un **Reboot app** dans Streamlit Cloud
- puis un refresh navigateur (`Ctrl+F5`)

## Notes techniques

- Les volatilites sont annualisees.
- Le temps est exprime en annees dans les modeles.
- Les Greeks sont calcules en formule fermee Black-Scholes.
- Le module Monte Carlo repose sur une dynamique GBM (geometric Brownian motion).
