import os
import pandas as pd
import matplotlib.pyplot as plt



def plot_from_excel(base_path, files, column_indices, yAchse, title):
    """
    Liest mehrere Excel-Dateien aus einem angegebenen Basisverzeichnis, 
    extrahiert Daten aus den entsprechenden Spalten und erstellt einen Graphen. 
    Die x-Achse repräsentiert die Anzahl der Einträge in den jeweiligen Spalten.
    
    :param base_path: Basisverzeichnis, in dem die Excel-Dateien liegen
    :param files: Liste von Excel-Dateinamen
    :param column_indices: Liste der Spaltennummern (beginnend bei 0), die geplottet werden sollen, 
                            wobei jede Datei ihren eigenen Index hat
    :param yAchse: Beschriftung der Y-Achse
    :param title: Titel des Plots
    """
    plt.figure(figsize=(10, 6))  # Größe des Plots

    # Jede Datei durchgehen und die entsprechenden Spalteninhalte plotten
    for i, file in enumerate(files):
        file_path = os.path.join(base_path, file)  # Datei-Pfad erstellen
        # Excel-Datei lesen
        df = pd.read_excel(file_path)

        # Hol den Spaltenindex für diese Datei
        column_index = column_indices[i]

        # Überprüfen, ob der Spaltenindex innerhalb der Spaltenanzahl liegt
        if column_index < len(df.columns):
            data = df.iloc[:, column_index].dropna()  # NaN-Werte ignorieren
            
            # Plotten der Werte und Speichern der Linienreferenz
            line, = plt.plot(range(1, len(data) + 1), data, label=file, linewidth=2)
            
            # Marker am Ende der Linie hinzufügen, gleiche Farbe wie die Linie
            plt.plot(len(data), data.iloc[-1], 'o', markersize=8, color=line.get_color())
        else:
            print(f"Warnung: Datei {file} hat weniger als {column_index + 1} Spalten.")

    # Graph-Details
    plt.xlabel('Generation')
    plt.ylabel(yAchse)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()



def smooth_values_global_max(original_values):
    """
    Glättet eine Liste von Werten, indem jeder Wert das vorherige globale Maximum übernimmt.
    
    :param original_values: Liste der Originalwerte
    :return: Liste der geglätteten Werte
    """
    if not original_values:
        return []

    smoothed_values = [original_values[0]]  # Den ersten Wert behalten
    current_max = original_values[0]  # Aktuelles Maximum initialisieren

    for value in original_values[1:]:
        # Aktualisiere das aktuelle Maximum, wenn der neue Wert höher ist
        current_max = max(current_max, value)
        smoothed_values.append(current_max)

    return smoothed_values


def plot_from_excel_global_max(base_path, files, column_indices, yAchse, title):
    """
    Liest mehrere Excel-Dateien aus einem angegebenen Basisverzeichnis, 
    extrahiert Daten aus den entsprechenden Spalten, glättet die Daten und 
    erstellt einen Graphen. 
    Die x-Achse repräsentiert die Anzahl der Einträge in den jeweiligen Spalten.
    
    :param base_path: Basisverzeichnis, in dem die Excel-Dateien liegen
    :param files: Liste von Excel-Dateinamen
    :param column_indices: Liste der Spaltennummern (beginnend bei 0), die geplottet werden sollen, 
                            wobei jede Datei ihren eigenen Index hat
    :param yAchse: Beschriftung der Y-Achse
    :param title: Titel des Plots
    """
    plt.figure(figsize=(10, 6))  # Größe des Plots

    # Jede Datei durchgehen und die entsprechenden Spalteninhalte plotten
    for i, file in enumerate(files):
        file_path = os.path.join(base_path, file)  # Datei-Pfad erstellen
        # Excel-Datei lesen
        df = pd.read_excel(file_path)

        # Hol den Spaltenindex für diese Datei
        column_index = column_indices[i]

        # Überprüfen, ob der Spaltenindex innerhalb der Spaltenanzahl liegt
        if column_index < len(df.columns):
            data = df.iloc[:, column_index].dropna().tolist()  # NaN-Werte ignorieren und in Liste umwandeln
            smoothed_data = smooth_values_global_max(data)  # Glättung der Werte
            
            # Plotten der geglätteten Werte
            line, = plt.plot(range(1, len(smoothed_data) + 1), smoothed_data, label=file, linewidth=2)
            
            # Marker am Ende hinzufügen, gleiche Farbe wie die Linie
            plt.plot(len(smoothed_data), smoothed_data[-1], 'o', markersize=8, color=line.get_color())
        else:
            print(f"Warnung: Datei {file} hat weniger als {column_index + 1} Spalten.")

    # Graph-Details
    plt.xlabel('Generation')
    plt.ylabel(yAchse)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


def create_average_excel(base_path, input_files, output_file, column_index):
    """
    Erstellt eine Excel-Datei mit dem Durchschnitt der Werte aus einer bestimmten Spalte 
    über mehrere Excel-Dateien für jede Zeile. Leere Einträge werden durch 400 ersetzt.
    
    :param base_path: Basisverzeichnis, in dem die Excel-Dateien liegen
    :param input_files: Liste von Excel-Dateinamen
    :param output_file: Name der Ausgabedatei, in die die Durchschnittswerte gespeichert werden
    :param column_index: Index der Spalte (beginnend bei 0), die für die Berechnung des Durchschnitts verwendet werden soll
    """
    # Liste zur Speicherung der Datenrahmen
    dfs = []

    # Jede Excel-Datei einlesen und den relevanten DataFrame speichern
    for file in input_files:
        file_path = os.path.join(base_path, file)  # Datei-Pfad erstellen
        df = pd.read_excel(file_path)  # Excel-Datei lesen
        # Überprüfen, ob der Spaltenindex innerhalb der Spaltenanzahl liegt
        if column_index < len(df.columns):
            # Leere Einträge durch 400 ersetzen und relevante Spalte hinzufügen
            data_column = df.iloc[:, column_index].fillna(400)  # NaN-Werte durch 400 ersetzen
            dfs.append(data_column)
        else:
            print(f"Warnung: Datei {file} hat weniger als {column_index + 1} Spalten.")

    # Überprüfen, ob wir Daten haben
    if dfs:
        # Datenrahmen zusammenführen und den Durchschnitt berechnen
        combined_df = pd.concat(dfs, axis=1)  # Zusammenführen der Datenrahmen
        averages = combined_df.mean(axis=1)  # Durchschnitt über die Zeilen (axis=1)

        # Durchschnittswerte in einen DataFrame umwandeln
        averages_df = pd.DataFrame(averages, columns=['Durchschnitt'])

        # Ausgabedatei speichern
        output_file_path = os.path.join(base_path, output_file)
        averages_df.to_excel(output_file_path, index=False)

        print(f"Die Durchschnittswerte wurden in {output_file_path} gespeichert.")
    else:
        print("Keine Daten zum Berechnen des Durchschnitts gefunden.")


def count_generations(file_path):
    """
    Zählt die Anzahl der Zeilen in allen Spalten einer Excel-Datei.
    Gibt die Zeilenanzahl zurück, wenn alle Spalten dieselbe Anzahl an Zeilen haben,
    andernfalls wird ein Fehler ausgelöst.
    
    :param file_path: Pfad zur Excel-Datei
    :return: Anzahl der Zeilen, wenn alle Spalten dieselbe Anzahl an Zeilen haben
    :raises ValueError: Wenn die Spalten unterschiedlich viele Zeilen haben
    """
    df = pd.read_excel(file_path)
    row_counts = [df.iloc[:, index].dropna().count() for index in range(len(df.columns))]

    # Überprüfen, ob alle Spalten dieselbe Zeilenanzahl haben
    if len(set(row_counts)) > 1:
        raise ValueError(f"Fehler: Die Spalten in der Datei '{file_path}' haben unterschiedliche Anzahlen von Zeilen.")

    # Alle Spalten haben dieselbe Zeilenanzahl, also einen Wert zurückgeben
    return row_counts[0]
