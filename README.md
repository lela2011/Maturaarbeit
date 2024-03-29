# Closed-Shell-Restricted-Hartree-Fock-Paket
## Hintergrund
Dieses Repository beinhaltet den selbst geschriebenen Code zu meiner Maturaarbeit im Themenbereich *Computergestützte Quantenchemie*. Genauer habe ich mich mit Closed-Shell-Restricted-Hartree-Fock-Berechnungen befasst. Dazu wurde zuerst die Theorie erarbeitet. Diese wurde dann in Python implementiert. Zur Hilfe wurde das bereits existierende Paket *PySCF* verwendet.
## Kompatibilität
Das Paket wurde auf einer Linux Ubuntu 20.04 Distribution mit Python 3.8 getestet. Zum Zeitpunkt des Erstellens funktionierte das Paket *PySCF* nicht auf Python 3.10. Zusätzlich funktioniert *PySCF* nicht auf Windows Betriebssystemen. Ein Test auf MacOS konnte nicht durchgeführt werden, da kein solches Gerät zur Verfügung stand.
## Installation
Zur Installation sind die Pakete *Numpy* und *PySCF* nötig. Diese können über die Befehle
```bash
$ pip install numpy
$ pip install pyscf
```
genutzt werden. Falls die Installation von *PySCF* nicht funktioniert, kann der Befehl
```bash
$ pip install git+https://github.com/pyscf/pyscf
```
versucht werden. Falls das auch nicht funktioniert, können mehr Details auf der [Installations-Seite von PySCF](https://pyscf.org/install.html) gefunden werden. Zusätzlich muss der Inhalt des Ordners `HartreeFock` heruntergeladen werden.
## Verwendung
### Vorbereitung
Zur Verwendung des Pakets muss eine Python-Datei in der gleichen Ordner-Ebene wie die `RHF.py`-Datei angelegt werden. In dieser wird der Code geschrieben. Zur Berechnung eins Atoms müssen folgende Import-Statements gemacht werden.
```python
from Objects.Atom import Atom
from RHF import RHF
```
Zur Berechnung eins Moleküls muss zusätzlich der folgende Import gemacht werden.
```python
from Objects.Molecule import Molecule
```
### Anwendung für Atome
Um eine Closed-Shell-Restricted-Hartree-Fock-Berechnung zu starten, wird zuerst das zu berechnende Atom definiert. Die Definition eines Atoms wird durch die Initialisierung der Klasse `Atom` erreicht. Dabei muss das Elementsymbol, die Position im Raum in Bohr-Radien und die Kernladung übergeben werden.
```python
helium = Atom("He", (0, 0, 0), 2)
```
Um für ein definiertes Atom eine Berechnung zu starten, wird ein Objekt der Klasse `RHF` mit dem zu berechnenden Atom initialisiert. Wahlweise kann eine maximale Anzahl Iterationen angegeben werden. Der Standard-Wert ist als 1000 Iterationen definiert. In diesem Beispiel wurde eine maximale Anzahl an Iterationen von 250 gewählt.
```python
rhf_helium = RHF(helium, 250)
```
Durch Aufruf der Methode `calculate` wird die Berechnung gestartet. Die Rückgabewerte sind die totale Energie als Dezimalzahl, die molekulare Energie als Liste, die Expansionskoeffizienten der Basisfunktionen als Matrix, die Anzahl nötiger Iterationen als Ganzzahl und eine Angabe, ob die Berechnung konvergiert.
```python
e_tot, e_mol, coefficients, no_iterations, converges = rhf_helium.calculate()
```
Ab einer Energie-Differenz von $10^{-14}$ Ha zwischen zwei Iterationsschritten sieht der Algorithmus die Berechnung als konvergierend an und gibt die finalen Werte zurück. Es ist wichtig anzumerken, dass der Algorithmus nicht mit Konvergenzoptimierungen arbeitet.
### Anwendung für Moleküle
Um eine Closed-Shell-Restricted-Hartree-Fock-Berechnung zu starten, werden zuerst die Atome definiert, die das Molekül bilden. Die Definition eines Atoms wird durch die Initialisierung der Klasse `Atom` erreicht. Dabei muss das Elementsymbol, die Position im Raum in Bohr-Radien und die Kernladung übergeben werden.
```python
h = Atom("H", (0, 0, 0), 1)
f = Atom("F", (0, 0, 1), 9)
```
Die definierten Atome werden anschliessend als Liste der Klasse 'Molecule' übergeben. Zusätzlich muss eine Orbital-Besetzung in Form einer Liste übergeben werden. Die Liste ist nach der Energie der Orbitale aufsteigend geordnet und zeigt an, wie viele Elektronen in dem jeweiligen Orbital zu finden sind. Um sicherzustellen, dass die Definition der Schalenbesetzung verständlich ist, wird diese kurz für das Fluor-Wasserstoff Molekül erklärt.
Wasserstoff besitzt ein s-Orbital. Fluor besitzt zwei s-Orbitale und und drei p-Orbitale. Damit ist klar, weshalb die Liste die Länge sechs hat. Es fehlen total zwei Elektronen, damit alle Orbitale vollständig besetzt sind. Daraus kann man schliessen, dass fünf Orbitale doppelt besetzt sind und eines nicht. Damit findet man die Orbitalbesetzung `[2,2,2,2,2,0]`.
```python
fh = Molecule([h, f], [2, 2, 2, 2, 2, 0])
```
Um für ein definiertes Molekül eine Berechnung zu starten, wird ein Objekt der Klasse `RHF` mit dem zu berechnenden Molekül initialisiert. Wahlweise kann eine maximale Anzahl Iterationen angegeben werden. Der Standard-Wert ist als 1000 Iterationen definiert. In diesem Beispiel wurde eine maximale Anzahl an Iterationen von 250 gewählt.
```python
rhf_fh = RHF(fh, 250)
```
Durch Aufruf der Methode `calculate` wird die Berechnung gestartet. Die Rückgabewerte sind die totale Energie als Dezimalzahl, die molekulare Energie als Liste, die Expansionskoeffizienten der Basisfunktionen als Matrix, die Anzahl nötiger Iterationen als Ganzzahl und eine Angabe, ob die Berechnung konvergiert.
```python
e_tot, e_mol, coefficients, no_iterations, converges = rhf_fh.calculate()
```
Ab einer Energie-Differenz von $10^{-14}$ Ha zwischen zwei Iterationsschritten sieht der Algorithmus die Berechnung als konvergierend an und gibt die finalen Werte zurück. Es ist wichtig anzumerken, dass der Algorithmus nicht mit Konvergenzoptimierungen arbeitet.
