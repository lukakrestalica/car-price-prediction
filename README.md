# Predikcija cene polovnih automobila

**Luka Krestalica — Task 4**

Regresioni projekat predviđa `priceUSD` iz marke, modela, godišta, kilometraže, motora i drugih karakteristika vozila. Sadrži neizmenjeni izvorni CSV, izvršenu Jupyter svesku, odvojene module za ceo ML postupak i trenirani Pipeline.

## Pokretanje

Preuzeti projekat preko **Code → Download ZIP**, raspakovati i otvoriti terminal u njegovom korenom folderu. Koristiti **Python 3.12** (testirano: 3.12.14).

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python predict_price.py
```

Na Linuxu/macOS-u koristiti `.venv/bin/python` umesto `.venv\Scripts\python`.

Poslednja komanda učitava sačuvani model i procenjuje primer Volkswagen Golf vozila iz zadatka. Dobijena procena je **12.411,41 USD**. Vrednost 9.990 USD u tekstu zadatka je ilustracija, a ne poznata stvarna cena koju treba ručno nametnuti modelu.

Za sopstveni automobil napraviti JSON fajl sa poljima iz primera i pokrenuti:

```powershell
.venv\Scripts\python predict_price.py --input automobil.json
```

Primer sadržaja:

```json
{
  "make": "volkswagen", "model": "golf", "year": 2014,
  "condition": "with mileage", "mileage(kilometers)": 180000,
  "fuel_type": "diesel", "volume(cm3)": 1600, "color": "black",
  "transmission": "mechanics", "drive_unit": "front-wheel drive", "segment": "C"
}
```

Ponovno treniranje svih kandidata, evaluacija i otvaranje analize:

```powershell
.venv\Scripts\python -m src.model_comparison
.venv\Scripts\python -m src.model_evaluation
.venv\Scripts\python -m jupyterlab notebooks/01_eda.ipynb
```

Alternativno, `python -m src.model_training` pokreće isti postupak poređenja i čuvanja. `python -m src.data_cleaning` ispisuje pregled očišćenih podataka. Module pokretati sa `-m` iz korena projekta, kako bi importi ostali ispravni.

Sveska podrazumevano čita sačuvane rezultate. Za ponavljanje treninga u njoj postaviti `RETRAIN=True`; ako izveštaji ne postoje, trening se pokreće automatski. Zatim izabrati **Restart Kernel and Run All**. Ponovno treniranje zamenjuje model i numeričke izveštaje; ponovno izvršavanje sveske osvežava njene rezultate i grafikone.

## Struktura

```text
data/cars.csv
notebooks/01_eda.ipynb
src/__init__.py
src/data_cleaning.py
src/feature_engineering.py
src/data_preprocessing.py
src/model_training.py
src/model_evaluation.py
src/model_comparison.py
models/car_price_model.joblib
reports/
predict_price.py
requirements.txt
README.md
```

Izvorni fajl ima **56.244 reda i 12 kolona**. Posle standardizacije i uklanjanja 88 duplikata ostaje **56.156 redova**. Izvorni CSV nije menjan. Izvor je fajl `cars.csv` dostavljen uz zadatak; projekat mu ne dodeljuje novu licencu.

## Čišćenje i nedostajući podaci

- Nazivi kilometraže i zapremine menjaju se u `mileage_km` i `engine_cm3`.
- Kategorije prelaze u mala slova, bez spoljašnjih razmaka.
- Cena mora biti pozitivan, konačan broj; u dostavljenom skupu nije bilo redova odbačenih po ovom pravilu.
- Nečitljivi ili sumnjivi ulazi postaju NaN: godina van 1886–2026 ili necela godina, kilometraža van 0–2.000.000 km, motor van 100–10.000 cm³. Ovo su dokumentovane granice kvaliteta, ne tvrdnja da svaki ekstrem nužno predstavlja grešku.
- Po ovim pravilima 231 kilometraža postaje nepoznata, a 86 zapremina pridružuje se 47 već nedostajućih. Nedostaje i 1.905 vrednosti pogona i 5.291 segment.
- Numeričke vrednosti popunjava medijana treninga uz indikatore nedostajanja; kategorije se popunjavaju oznakom `unknown`.

Ekstremne cene od 48 do 235.235 USD nisu proizvoljno brisane: mogu predstavljati oštećena, retka ili luksuzna vozila. Njihov uticaj vidljiv je kroz RMSE i primere najvećih grešaka.

## Karakteristike i pretprocesiranje

Izvedene su starost prema fiksnoj 2026. godini, kilometraža po godini te starosti, motor u litrima, indikator najmanje 200.000 km i kombinacija marke i modela. Fiksna godina čuva ponovljivost; njena promena zahteva novo treniranje. Bez datuma oglasa `mileage_per_year` je samo približna izvedena karakteristika, ne izmerena godišnja upotreba.

`SimpleImputer`, `StandardScaler` i `OneHotEncoder` nalaze se unutar Pipeline-a. Imputacija, skaliranje i kategorije uče se samo na treningu. Retke kategorije grupišu se uz `min_frequency=10` i `max_categories=150`; nepoznate oznake pri upotrebi ne ruše model. Izvorna godina i cm³ ne ponavljaju se uz starost i litre kao numerički ulazi.

## Podela i izbor

Koristi se grupna podela sa semenom 42. Oglasi sa identičnim očišćenim karakteristikama, čak i kada imaju različite cene, ostaju u istoj grupi. To smanjuje curenje podataka; bez VIN-a ne možemo garantovati razdvajanje svih oglasa istog fizičkog vozila.

- Trening: **33.693** reda.
- Validacija: **11.231** red.
- Test: **11.232** reda.
- Sačuvani model treniran je na treningu i validaciji zajedno: **44.924** reda.

Najniži validacioni MAE bira model, a RMSE služi za eventualno izjednačenje. Izbor se zaključava pre testiranja. Zatim se svi kandidati treniraju na istom razvojnom skupu i ocenjuju na istom testu. Test ne učestvuje u učenju niti naknadnom izboru parametara.

| Model | Validacioni MAE (USD) | Test MAE (USD) | Test RMSE (USD) | Test R² |
|---|---:|---:|---:|---:|
| Medijanska referenca | 4.801,36 | 4.969,82 | 8.948,45 | -0,0575 |
| Ridge regresija | 2.285,49 | 2.316,52 | 5.321,83 | 0,6260 |
| Stablo odluke | 1.366,90 | 1.361,01 | 3.396,66 | 0,8476 |
| **Random Forest** | **1.167,04** | **1.160,83** | **2.885,22** | **0,8901** |

Ridge je regularizovana linearna regresija. Stablo i šuma hvataju nelinearne odnose. Random Forest koristi 50 stabala, maksimalnu dubinu 20 i najmanje četiri uzorka u listu; parametri su izabrani pre poređenja, bez naknadnog podešavanja prema test rezultatima.

## Rezultati i njihovo značenje

**Izabran je Random Forest.** Test MAE je **1.160,83 USD**, MSE približno **8.324.476 USD²**, RMSE **2.885,22 USD**, a R² **0,8901**.

MAE znači prosečnu apsolutnu razliku od oko 1.161 dolara na testu, ne garantovan interval za pojedinačni automobil. Znatno veći RMSE pokazuje uticaj velikih promašaja. R² nije procenat automobila kojima je cena tačno pogođena. Sveska prikazuje stvarne cene, predikcije, apsolutne greške, najveće promašaje i reziduale. Sve test predikcije nalaze se u `reports/test_predictions.csv`.

Sačuvan je upravo model evaluiran na testu, bez naknadnog učenja na test podacima. `python -m src.model_evaluation` ponovo učitava fajl i reprodukuje metrike pomoću sačuvanih test indeksa. Negativna predikcija se ograničava na nulu u evaluaciji i pri korišćenju.

## Ograničenja

Godine proizvodnje u izvoru dosežu 2019, a datum oglasa, lokacija, oprema i istorija održavanja nisu dostupni. Zato model procenjuje cenu prema obrascima dostavljenog skupa, a ne potvrđenu današnju tržišnu ili ostvarenu prodajnu cenu. Retki i kolekcionarski modeli mogu imati velike greške. Nepoznata marka ili model mogu se obraditi, ali procena za njih nije posebno potvrđena.

Sledeća verzija zahteva novije podatke, vremensku evaluaciju, proveru po familijama vozila i unakrsnu validaciju na treningu. Model ne sadrži kalibrisan interval pouzdanosti.

Sačuvani `joblib` sadrži kompletan Pipeline i koristi klase iz `src/`, pa ceo folder mora ostati uz model. Učitavati samo proverene modele, jer serijalizovani Python objekti mogu izvršavati kod. Koristiti verzije iz `requirements.txt` za reprodukciju.

## Dokumentacija

- [RandomForestRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)
- [Mean absolute error](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_absolute_error.html)

Za predaju instruktoru poslati link javnog repozitorijuma.
