# Generieren von prompt spezifischen und generischen Essays

## Generieren von TF-IDF Scores
tf_idf.py<br>
Aufruf: <br>
```tf_idf.py [input_folder] [n-gram length] [--min-count] [--output]```<br>
Sollten die ersten beiden Werte nicht als Programmparameter übergeben werden,
wird ein Prompt diese vom Nutzer abfragen.

- input_folder: spezifiziert den Pfad zu den input Datensatz.
- n-gram length: spezifiziert die Größe der n-grame im Output.
- --min-count: minimale Gesamtanzahl eines n-grams um im Output zu erscheinen.<br>default = 5
- --output: der volle Dateipfad und Name für die Outputdatei.<br> default = "tf_idf_[n-gram länge]_output.csv"

Ausgabe: <br>
Die Ausgabe ist eine CSV-Datei die folgende daten beinhaltet:

- term : ein String der das n-gram als Tupel beinhaltet
- total_count : die Gesamtanzahl wie oft das n-gram im Datensatz vorkommt.
- tf_idf(c) : die tf_idf Werte für das n-gram in Cluster c.

## Essay Texte modifizieren
text_modifier.py<br>
Aufruf: <br>
```text_modifier.py [input_file] [boarder position] [--output]```<br>
Sollten die ersten beiden Werte nicht als Programmparameter übergeben werden,
wird ein Prompt diese vom Nutzer abfragen.
- input_file: spezifiziert den Pfad zu der input Datei.
- boarder-position: spezifiziert die Grenze ab wie viele Token in den ersten Output kommen.
- --output: der Dateipfad für den Output<br>Output1 = [output]/; Output2 = [output]_inv/<br> default = "modified_clusters[boarder-position]"

# Analyse der predictions
analysis_kaggle_feedback.py<br>
Analysiert werden die Labelverteilung (Klassen der Substrings) als auch die Anzahl der Token.

Die absolute Labelverteilung und die Anzahl der Token lassen sich mithilfe `analyse_cluster` ermitteln.
Für eine csv-Datei `filename.csv` mit dem Inhalt
```
class,predictionstring
Lead,"0 1 2 3"
Lead,"4 5 6"
NotRelevant,"7 8"
```
erzeugt ein beispielhafter Aufruf von `analyse_cluster` folgend Daten:
```
label_distribution, token_counts = analyse_cluster("filename.csv", ['Lead', 'Position', 'Claim'])
# label_distribution: [2, 0, 0]
# token_counts: {'Lead': [4, 3], 'Position': [], 'Claim': []}
```
Die Methode `wordcount_prediction_string` wird dafür aufgerufen und sollte die Anzahl der tokens in einem predictionstring
zurückgeben. Sie muss ggf. bei unterschiedlichen Formatierungen des predictionstring angepasst werden.

Die Methode `plot_lf_wc` zeichnet ein Balkendiagramm und ein Boxplot nebeneinander.
Als Eingabe können die Ausgabedaten von `analyse_cluster` verwendet werden.