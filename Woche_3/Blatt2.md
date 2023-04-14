# Bildverarbeitung Praktikum
## Blatt 2, Abgabe 27.04.2023
## Dr. Daniel Wiegreffe
### April 20, 2023
<br>
<br>

## 1. Box Filter

Für diese Aufgabe benötigen Sie eine weitere Bibliothek in Python, die Sie durch Pip wie folgt laden können.

```bash
pip install opencv-python
```

Oder nutzen sie die `requirements.txt` von Moodle, und installieren sie alle notwendigen Pakete auf einmal.

```bash
pip install -r requirements.txt
```
OpenCV ist eine Bibliothek, die Ihnen sehr viele Möglichkeiten der Bildverarbeitung liefert. Unter anderem die Funktion

```python
result = cv2.filter2D(img, -1, kernel)
```

Hiermit können Sie einen beliebigen Kernel auf ein Bild anwenden. Hierzu müssen Sie sich einen Kernel erstellen. Dies können Sie durch ein neues Numpy Array erreichen, wie auch in Aufgabe 1.

<br>

a) Erstellen sie einen Box Kernel (Größe 5x5) und wenden Sie ihn auf ein beliebiges Bild an. Vergessen sie die Normalisierung nicht.

b) Manipulieren Sie den Wert des zentralen eintrags in Ihrem Kernel. Nutzen Sie dabei immer größere Werte. Achten Sie darauf, dass Sie auch hier entsprechend normalisieren müssen. Was fällt bei belibig großen Werten auf?

## 2. Implementierung des Gaußfilters
Wie bei allen anderen Programmiersprachen gibt es in Python Funktionen. Die können direkt in Ihren Programmcode eingefügt werden. Eine kurze Zusammenfassung finde sie hier:

[https://pythonbuch.com/funktion.html](https://pythonbuch.com/funktion.html)

Sie sollen eine Funktion schreiben, die Ihnen einen Gaußfilter mit beliebiger Größe (Input als Parameter der Funktion) erstellt.
Nutzen Sie die Funktion filter2D von openCV um verschiedene Filter auf ein Bild anzuwenden.
```python
result = cv2.filter2D(img, -1, kernel)
```
a) Schreiben Sie eine Funktion, die Ihnen beliebig große Gaußfilter mit erzeugt.

b) Nutzen Sie ihre Funktion um Gaußfilter der Größe 3, 7, und 15 auf ein beliebiges Bild anzuwenden. Was fällt Ihnen auf?