# Sistem Inteligent de Predicție a Prețurilor Imobiliare folosind Rețele Neuronale MLP
**Autor:** Antoneac Ramona-Florina  
**An:** 2025

---


Un sistem inteligent de estimare a prețurilor imobiliare construit cu un model de regresie Multilayer Perceptron (MLP). Sistemul prezice prețul de vânzare al locuințelor pe baza caracteristicilor structurale, funcționale și de localizare, utilizând dataset-ul Kaggle *House Prices – Advanced Regression Techniques*.

## Descriere

Evaluarea tradițională a proprietăților se bazează pe experți imobiliari, ceea ce introduce subiectivitate și inconsistență. Acest proiect implementează o soluție de învățare supervizată — **SmartEstate AI** — care automatizează estimarea prețurilor prin învățarea relațiilor complexe și neliniare dintre caracteristicile locuinței și prețul final.

## Set de date

- **Sursă:** [House Prices – Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques) (Kaggle)
- **Dimensiune:** 1.460 exemple × 81 atribute
- **Variabila țintă:** `SalePrice` — prețul final de vânzare al proprietății
- **Tipuri de atribute:** numerice (suprafețe, anul construcției, scoruri de calitate) și categorice (cartier, tipuri de materiale, stilul clădirii)

## Metodologie

### Preprocesare
- Transformare logaritmică a variabilei țintă (`np.log1p`) pentru reducerea skewness-ului și a influenței outlierilor
- Atribute numerice: imputare cu mediana + normalizare StandardScaler
- Atribute categorice: imputare cu valoarea cea mai frecventă + OneHotEncoder
- Întreaga preprocesare este integrată într-un `Pipeline` sklearn

### Selecția atributelor (Model optimizat)
- Analiza coeficientului de corelație între atributele numerice și `SalePrice`
- Sunt păstrate doar atributele cu `|corr| > 0.3`
- Toate variabilele categorice sunt păstrate (eliminarea lor a dus la degradarea semnificativă a performanței)
- Dimensionalitatea a fost redusă de la 79 la 61 de atribute

### Atribute relevante identificate:
`OverallQual`, `GrLivArea`, `GarageCars`, `GarageArea`, `TotalBsmtSF`, `FullBath`, `TotRmsAbvGrd`, `YearBuilt`

### Arhitectura modelului
Au fost comparate mai multe arhitecturi MLP. Performanțele obținute pe setul complet de date:

| Straturi ascunse | RMSE | MAE | R² |
|---|---|---|---|
| (16,) | 30.790 | 20.149 | 0,876 |
| (32,) | 28.659 | 19.488 | 0,893 |
| **(64,)** | **27.036** | **18.576** | **0,905** |
| (64, 32) | 29.926 | 20.449 | 0,883 |
| (128, 64) | 38.134 | 24.421 | 0,810 |
| (128, 64, 32) | 31.417 | 21.193 | 0,871 |

**Cea mai bună arhitectură: `(64,)` — un singur strat ascuns cu 64 de neuroni**

### Hiperparametri
```python
MLPRegressor(
    hidden_layer_sizes=(64,),
    activation="relu",
    solver="adam",
    alpha=0.001,
    learning_rate_init=0.001,
    max_iter=2000,
    random_state=42
)
```

## Rezultate

**Cel mai bun model (toate atributele, arhitectura 64,):**
```
MSE  = 730.946.071,45
RMSE = 27.036,01
MAE  = 18.576,44
R²   = 0,9047
```

Modelul explică peste **90% din variația prețurilor** locuințelor. Distribuția erorilor este aproximativ simetrică în jurul valorii 0, ceea ce indică absența unui bias sistematic de supraestimare sau subestimare. Deviații mai mari apar doar pentru proprietățile cu prețuri foarte ridicate, comportament normal datorită reprezentării lor reduse în setul de antrenare.

## Utilizare

1. Clonează repository-ul
2. Plasează `train.csv` în același folder cu notebook-ul (descarcă de pe [Kaggle](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data))
3. Deschide și rulează `house_price_prediction.ipynb` în Jupyter sau Google Colab

## Tehnologii utilizate

`Python` · `scikit-learn` · `NumPy` · `Pandas` · `Matplotlib` · `Jupyter Notebook`

---
