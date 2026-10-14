# Instrukcja dla zespołu: model grafików, równania, prompty i benchmark

Instrukcja krok po kroku. Komendy kopiujesz i wklejasz do terminala. Wszystkie uruchamiasz
**z głównego katalogu repozytorium** (tam, gdzie leży `pyproject.toml`).

---

## 0. O co w tym chodzi (2 minuty czytania)

Mamy model CP-SAT (OR-Tools), który układa miesięczny grafik dyżurów lekarzy. Model składa się z
**33 równań**: ograniczeń (np. „11 h odpoczynku”), definicji (np. „godziny przepracowane”) i celów
(np. „minimalizuj braki obsady”).

**Pytanie badawcze:** ile informacji trzeba dać modelowi AI (np. Claude), żeby sam, bez
oglądania naszego kodu, napisał poprawną implementację każdego równania?

Jak to sprawdzamy:
1. Dla każdego równania automatycznie powstaje **prompt**: opis interfejsu plus opis równania.
2. Model AI odpowiada kodem funkcji w Pythonie.
3. Kod jest **automatycznie weryfikowany**. Funkcję podstawia się do modelu i testuje na
   30 grafikach. Wynik porównuje się z niezależną „wyrocznią”, czyli tym, co równanie powinno zrobić.
   Wynik to PASS albo FAIL z konkretnym kontrprzykładem.
4. **Benchmark** powtarza to dla kilku **zestawów promptów** o rosnącej szczegółowości i pokazuje,
   od którego poziomu model „trafia”.

Słowniczek:

| Pojęcie | Co to znaczy |
|---|---|
| równanie | jedna funkcja w `rostering/equations.py`, np. `H3` (odpoczynek), `O3` (cel fazy 3) |
| wyrocznia | niezależna implementacja w czystym Pythonie (`rostering/oracles.py`), która dla konkretnego grafiku mówi, jaki powinien być wynik |
| zestaw promptów | folder w `prompts/sets/` z opisami wszystkich 33 równań na danym poziomie szczegółowości |
| przebieg (run) | jeden zestaw × jeden model × wszystkie (lub wybrane) równania; wyniki w `runs/...` |
| benchmark | kilka zestawów × kilka powtórzeń; na końcu tabela porównawcza |

---

## 1. Przygotowanie (robisz raz)

### 1.1 Zainstaluj `uv` (menedżer Pythona)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Zamknij i otwórz terminal na nowo, potem sprawdź: `uv --version`.

### 1.2 Pobierz repozytorium i przełącz się na branch

```bash
git clone https://github.com/nataliadybczak/Rostering-Problem-Project.git
cd Rostering-Problem-Project
git checkout MP_review
```

### 1.3 Zainstaluj zależności

```bash
uv sync
```

### 1.4 Sprawdź, że wszystko działa

```bash
uv run pytest
```

Trwa około 2 minut. Na końcu musi być `passed` i **zero** `failed`.

### 1.5 Claude CLI (do generowania kodu przez AI)

Potrzebne tylko do kroków 4 i 5.
1. Zainstaluj Claude Code: https://docs.claude.com/claude-code.
2. Uruchom raz `claude`, zaloguj się i wyjdź (`/exit`).
3. Sprawdź:

```bash
echo "Odpowiedz tylko: OK" | claude -p
```

Ma wypisać `OK`.

---

## 2. Szybki start (kopiuj-wklej)

```bash
uv run python -m rostering.verify --list                    # lista 33 równań
uv run python -m rostering.verify --all                     # nasze równania vs wyrocznia (~30 s), wszystko PASS
uv run python -m rostering.prompting sets                   # lista zestawów promptów
uv run python -m rostering.prompting prompts --set s1_intent --out podglad   # podgląd promptów w folderze podglad/
uv run python -m rostering.prompting run H3 O3 --set s3_spec --model sonnet  # AI pisze H3 i O3, weryfikacja
```

Wynik ostatniej komendy jest w `runs/<data>_s3_spec_claude-cli_sonnet/report.md`.

---

## 3. Sam model grafików (gdy potrzebujesz)

Aplikacja w przeglądarce (pierwsze ładowanie około 5 minut):

```bash
uv run streamlit run app/streamlit_new.py
```

Z linii poleceń, szybciej, z krótszymi limitami czasu:

```bash
uv run python -m rostering --dataset dec2026_pandemic --pandemic --time-limits 30 30 60
```

Wyniki trafiają do `results/<zbiór>/`. Opis matematyczny modelu jest w `docs/MODEL.md`.

---

## 4. Prompty i pojedyncze przebiegi

### 4.1 Zestawy promptów (od najuboższego)

| Zestaw | Co dostaje model AI |
|---|---|
| `s0_name` | tylko nazwę równania, np. „Minimum rest of 11 h.” |
| `s1_intent` | 1-2 zdania „językiem planisty”, bez szczegółów |
| `s2_rules` | dokładne reguły (kogo dotyczy, co liczymy, liczby), bez kodowania CP-SAT i dziedzin zmiennych |
| `s3_spec` | pełna specyfikacja słowna: reguły, dziedziny zmiennych, gdzie zapisać wynik |
| `s4_formal` | `s3_spec` + zapis formalny (prawie wzór) |

**Każdy** zestaw dostaje też ten sam opis interfejsu (`prompts/contract.md`). Jest tam opis pól,
z których funkcja może korzystać.

### 4.2 Przebieg dla jednego zestawu

```bash
uv run python -m rostering.prompting run --set s2_rules --model sonnet
```

Przydatne opcje:

| Opcja | Co robi | Przykład |
|---|---|---|
| lista ID na początku | tylko wybrane równania | `run H3 H7 O3 --set s2_rules` |
| `--model` | który model Claude | `--model opus`, `--model sonnet`, `--model haiku` |
| `--attempts N` | po FAIL model dostaje kontrprzykłady i poprawia (do N prób) | `--attempts 3` |
| `--jobs N` | ile równań naraz | `--jobs 6` |
| `--provider manual` | tylko zapisuje prompty; odpowiedzi wklejasz ręcznie (np. z ChatGPT) | patrz 4.4 |
| `--provider command --command "..."` | dowolny inny model z linii poleceń | `--command "ollama run qwen2.5-coder"` |
| `--provider reference` | „odpowiada” naszą implementacją; test samego mechanizmu (ma być 33/33) | |

### 4.3 Gdzie są wyniki i jak je czytać

Każdy przebieg tworzy folder `runs/<data>_<zestaw>_<dostawca>_<model>/`:

```text
report.html          <- raport z wykresami i metrykami (zacznij od tego)
report.md            <- krótki raport tekstowy
summary.csv          <- wynik każdego równania w tabeli (do Excela / pandas)
metrics.csv          <- wszystkie metryki (zgodność, cosinusy, ilość informacji)
config.json          <- ustawienia przebiegu
H3/attempt_1/
    prompt.md        <- dokładnie to, co dostał model
    response.md      <- surowa odpowiedź modelu
    candidate.py     <- wyciągnięty kod
    verify.json      <- wynik weryfikacji
```

Jak czytać komunikaty FAIL (zawsze `[nazwa grafiku] co poszło nie tak`):

| Komunikat | Znaczenie |
|---|---|
| `CP-SAT infeasible, oracle: no violation expected` | funkcja zabrania czegoś, co jest dozwolone (za ostra) |
| `CP-SAT feasible, oracle: <naruszenie>` | funkcja przepuszcza coś, co powinno być zabronione (za luźna) |
| `H[5]: expected 120, got 112` | zła wartość wyliczanej zmiennej |
| `(maximised) spread_N: expected 10, got 100` | wartość nie jest jednoznaczna (np. ograniczenie zamiast równości) |
| `optimum expected X, got Y` | zła funkcja celu |
| `oracle: infeasible, CP-SAT found a solution` | brakuje ograniczenia dziedziny (np. górnej granicy zmiennej) |
| `no python block defining ...` | model nie oddał kodu w wymaganym formacie |
| `error while building or solving: ...` | kod się wywala (np. zła nazwa pola) |

Nazwy grafików: `reference` (poprawny grafik), `empty` (pusty), `add (13, 122)` (dodano lekarzowi 13
dyżur 122), `remove 1-8`, `move 1-4`, `candidate covers ...` (kandydat przejmuje dyżury) itd.

### 4.4 Tryb ręczny (inny model przez przeglądarkę)

```bash
uv run python -m rostering.prompting run --set s2_rules --provider manual
```

1. W `runs/<przebieg>/<ID>/attempt_1/prompt.md` są prompty. Na początku rozmowy wklej zawartość `runs/<przebieg>/system.md`.
2. Odpowiedź modelu wklej do pliku `response.md` w tym samym folderze.
3. Uruchom: `uv run python -m rostering.prompting check runs/<przebieg>`.

---

## 5. Benchmark (główny eksperyment)

```bash
uv run python -m rostering.prompting benchmark --model sonnet --repeats 3
```

- Domyślnie: wszystkie zestawy × wszystkie 33 równania × 3 powtórzenia. To 5 × 33 × 3 = 495 zapytań do modelu.
- Powtórzenia są potrzebne, bo **ten sam prompt daje różne odpowiedzi**.
- **Najpierw zrób pilotaż** na kilku równaniach i 1 powtórzeniu (kilka minut):

```bash
uv run python -m rostering.prompting benchmark H3 H8 C1 D2b F1 U2 O1 O3 --repeats 1 --model sonnet
```

- Tylko wybrane zestawy: `--sets s1_intent s2_rules`.
- Z poprawkami po błędach: `--attempts 3`. Raport pokazuje wtedy osobno „za pierwszym razem” i „łącznie”.

Wynik: `runs/benchmark_<data>_claude-cli_<model>/report.html` (wykresy, patrz 5.1) i `report.md`:
- **Tabela 1** - skuteczność każdego zestawu (łącznie, za pierwszym razem, osobno dla ograniczeń / definicji / celów).
- **Tabela 2** - macierz równanie × zestaw: ile powtórzeń przeszło (np. `2/3`).
- Surowe dane do analizy: `benchmark.csv` (jeden wiersz = zestaw, powtórzenie, równanie).
- Szczegóły każdego przebiegu: podfoldery `<zestaw>_r<powtórzenie>/`.

Przykład z pilotażu (Sonnet, 8 równań, 1 powtórzenie, przed doprecyzowaniem F1-F4 i H7 w `s3_spec`):

| Zestaw | Skuteczność |
|---|---|
| s0_name | 12% |
| s1_intent | 25% |
| s2_rules | 62% |
| s3_spec | 88% |

Po doprecyzowaniu opisów w `s3_spec` pełny przebieg 33 równań dał 32/33. Jedyny błąd (H7) po
doprecyzowaniu przechodzi 3/3. F1-F4 po doprecyzowaniu przechodzą 2/2.

### 5.1 Raport HTML z wykresami i metrykami

Po każdym `run` i `benchmark` automatycznie powstaje `report.html` (otwórz w przeglądarce) oraz
`metrics.csv` (wszystkie liczby, wiersz = jedna odpowiedź modelu). Raport dla dowolnego,
także starszego, przebiegu wygenerujesz tak:

```bash
uv run python -m rostering.prompting report runs/<przebieg lub benchmark>
uv run python -m rostering.prompting report runs/<...> --offline   # działa bez internetu (plik ~4 MB)
open runs/<...>/report.html                                         # macOS: otwórz w przeglądarce
```

Metryki w raporcie (poza samym PASS/FAIL):

| Metryka | Co mierzy | Jak czytać |
|---|---|---|
| **passed** | odsetek równań, które przeszły weryfikację | główny wynik |
| **1st attempt** | to samo, ale tylko pierwsza próba (bez podpowiedzi) | ważne przy `--attempts > 1` |
| **agreement** (zgodność behawioralna) | odsetek 30 grafików testowych, na których kod zachowuje się jak specyfikacja | 1,00 = zawsze dobrze; 0,90 = myli się w 10% przypadków. Pokazuje „prawie dobre” rozwiązania, które w PASS/FAIL liczą się jako 0 |
| **code cos. lexical** | podobieństwo cosinusowe (TF-IDF) tokenów kodu do naszej implementacji | jak bardzo kod „wygląda” jak nasz (nazwy, wywołania API) |
| **code cos. AST** | podobieństwo cosinusowe (TF-IDF) struktury drzewa składni (pary typów węzłów) | podobieństwo „kształtu” kodu, niezależne od nazw zmiennych i komentarzy |
| **words / description** | średnia liczba słów opisu równania w zestawie | ile informacji dostał model |
| **cos. to formal** | podobieństwo cosinusowe (TF-IDF, bez słów funkcyjnych) opisu do formalnej definicji równania | jak blisko opis jest „wzoru” |
| **std over repeats** | odchylenie standardowe skuteczności między powtórzeniami | stabilność wyniku (słupki błędów na wykresie) |

Co to jest cosinus: każdy kod (lub tekst) zamieniamy na wektor liczności „słów”. Cosinus kąta
między dwoma wektorami to 1 dla identycznego składu i 0 dla zupełnie innego. Ważenie TF-IDF zmniejsza
wagę elementów występujących w każdym kodzie (np. `m`, `model`, nawiasy), więc liczy się to, co
charakterystyczne dla danego równania.

Wykresy w raporcie:
- skuteczność i zgodność dla każdego zestawu (ze słupkami błędów z powtórzeń);
- krzywa „ilość informacji → skuteczność”;
- mapy cieplne równanie × zestaw (PASS i zgodność);
- podział na ograniczenia, definicje i cele;
- wykres „czy podobny kod = poprawny kod?” (cosinus AST vs zgodność);
- rozkłady podobieństwa cosinusowego;
- podobieństwo opisów do definicji formalnej;
- liczba prób (przy `--attempts`).

Na dole raportu jest lista nieudanych odpowiedzi z pierwszym kontrprzykładem.

**Uwaga interpretacyjna.** Podobieństwo cosinusowe kodu to miara pomocnicza, nie miara poprawności.
Poprawny kod może wyglądać zupełnie inaczej niż nasz (np. `AddBoolOr` zamiast nierówności), a kod
bardzo podobny może mieć jeden błędny znak. O poprawności decyduje weryfikacja i agreement.

---

## 6. Co wolno i co warto modyfikować

### 6.1 Opisy równań w zestawie (najczęstsza zmiana)

Plik: `prompts/sets/<zestaw>/equations.md`.
- Każde równanie to sekcja zaczynająca się od `## ID`, np. `## H3`. **Nie zmieniaj nagłówków** (ID muszą zostać) i nie usuwaj sekcji, bo test sprawdza, czy każdy zestaw opisuje wszystkie 33 równania.
- Tekst pod nagłówkiem zmieniasz dowolnie.
- Podgląd gotowych promptów: `uv run python -m rostering.prompting prompts --set <zestaw> --out podglad`.

### 6.2 Nowy zestaw (np. „reguły bez liczb”)

1. Skopiuj folder: `cp -r prompts/sets/s2_rules prompts/sets/s2b_bez_liczb`.
2. Edytuj `prompts/sets/s2b_bez_liczb/equations.md` (pierwsza linia `# ...` to opis zestawu widoczny w `sets`).
3. Sprawdź: `uv run python -m rostering.prompting sets` i `uv run pytest tests/test_prompting.py`.
4. Uruchom: `... benchmark --sets s2_rules s2b_bez_liczb --repeats 3`.

Opcje zestawu (plik `set.json` w folderze zestawu, nieobowiązkowy):

```json
{"description": "opis zestawu", "descriptions_from": "s3_spec", "formal": true}
```

- `descriptions_from`: weź opisy z innego zestawu, zamiast z własnego `equations.md`.
- `formal`: dołącz zapis formalny równania (jak `s4_formal`).
- W folderze zestawu możesz też położyć własne `template.md`, `contract.md`, `system.md` albo `feedback.md`. Nadpisują one wspólne pliki z `prompts/`. Tak testuje się np. „mniej szczegółowy opis interfejsu”.

### 6.3 Wspólne pliki promptu (`prompts/`)

| Plik | Co to jest | Uwaga |
|---|---|---|
| `template.md` | szkielet promptu | **nie usuwaj** pól w klamrach: `{contract} {id} {name} {description} {formal} {available} {output_rule} {function}` |
| `contract.md` | opis pól obiektu `m`, z których funkcja korzysta | zmiana dotyczy wszystkich zestawów |
| `system.md` | polecenie systemowe dla modelu | |
| `feedback.md` | prompt poprawki po FAIL (`--attempts`) | pola `{prompt} {code} {failures}` muszą zostać |

### 6.4 Czego NIE ruszać (bez uzgodnienia)

| Plik / folder | Dlaczego |
|---|---|
| `rostering/equations.py` | równania referencyjne = „prawda” |
| `rostering/oracles.py` | wyrocznia - na niej opiera się ocena |
| `rostering/verify.py` | procedura oceny - zmiana unieważnia porównania między benchmarkami |
| `tests/golden/` | zapis oryginalnego modelu |

Jeśli świadomie zmieniamy model (np. poprawiamy błąd z `docs/MODEL.md`):
1. Zmień równanie w `equations.py` **i** wyrocznię w `oracles.py`.
2. `uv run pytest`: test równoważności z oryginałem celowo zgłosi zmianę.
3. `uv run python tests/record_fingerprints.py --write`, żeby zapisać nowy wzorzec.
4. Opisz zmianę w `docs/MODEL.md` i zregeneruj `docs/EQUATIONS.md`: `uv run python -m rostering.verify --markdown docs/EQUATIONS.md`.

---

## 7. Typowe problemy

| Problem | Rozwiązanie |
|---|---|
| `claude: command not found` | zainstaluj Claude Code (1.5) albo użyj `--provider manual` |
| `model call failed ... exited with 1` | uruchom `claude` raz ręcznie i zaloguj się; sprawdź limit zapytań na koncie |
| wszystko FAIL w `s0_name` | to normalne - to najuboższy zestaw |
| wynik różni się między powtórzeniami | normalne - dlatego `--repeats` |
| `verification timed out` | kod modelu jest bardzo wolny; zwiększ `--verify-timeout` albo uznaj za FAIL |
| `ModuleNotFoundError: rostering` | uruchamiasz spoza głównego katalogu repozytorium |
| chcę zachować wyniki | `runs/` nie trafia do gita; skopiuj `report.md` i `benchmark.csv` tam, gdzie trzymamy wyniki |

---

## 8. Ograniczenia metody (ważne przy pisaniu pracy)

1. **Weryfikacja jest ścisła i w izolacji.**
   - Jeśli implementacja z AI różni się od oryginału tylko w sytuacjach, które w pełnym modelu i tak wyklucza inne ograniczenie, dostanie FAIL.
   - Przykład: H7 zakodowane parami zamiast sumą. W pełnym modelu różnicę „przykrywa” H2.
   - Pełny zestaw `s3_spec` opisuje dlatego takie szczegóły wprost.
2. **Ocena jest tak mocna, jak dane testowe.**
   - Różnice, które w danych `dec2026_pandemic` nigdy nie występują, są niewykrywalne.
   - Przykład: odpoczynek 10, 11 czy 12 h daje to samo, bo w danych nie ma przerw 10-11 h.
3. **Oryginalny model ma osobliwości** (`docs/MODEL.md`, sekcja „Known issues”).
   - Weryfikacja wymaga ich odtworzenia, np. lista `weekly_regular_lookup` i ograniczenie F4.
   - Modele AI w uboższych zestawach często „poprawiają” je na wersję sensowniejszą i dostają FAIL. Warto to analizować osobno.
4. **Kod z AI uruchamia się lokalnie** (osobny proces z limitem czasu, ale bez piaskownicy).
   - `claude-cli` działa bez narzędzi i w pustym katalogu, więc nie widzi repozytorium.
   - Przy innych dostawcach sprawdźcie, czy model nie ma dostępu do naszego kodu, bo to zafałszowałoby wyniki.
