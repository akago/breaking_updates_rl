from pathlib import Path
import json
import csv
import argparse
from typing import Dict, Any, List, Iterable


def flatten_dict(d: Dict[str, Any], parent_key: str = "", sep: str = ".") -> Dict[str, Any]:
    items: Dict[str, Any] = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(flatten_dict(v, new_key, sep=sep))
        else:
            items[new_key] = v
    return items


def read_jsonl_breaking_commits(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                # Ignorar líneas no válidas
                continue

            # Extraer información básica
            record = {
                "absolute_path_to_file_in_container": obj.get("absolute_path_to_file_in_container", ""),
                "breakingCommit": obj.get("breakingCommit", ""),
                "model": obj.get("model", ""),
                "accepted": obj.get("accepted", ""),
            }

            # Extraer todos los 'kind' de errors > BCs (únicos)
            kinds_set = set()
            errors = obj.get("errors", [])
            if isinstance(errors, list):
                for error in errors:
                    if isinstance(error, dict):
                        bcs = error.get("BCs", [])
                        if isinstance(bcs, list):
                            for bc in bcs:
                                if isinstance(bc, dict):
                                    kind = bc.get("kind", "")
                                    if kind:
                                        kinds_set.add(kind)

            # Convertir set a lista ordenada para mantener consistencia
            kinds_unique = sorted(list(kinds_set))

            # Agregar los kinds únicos como columna separada por punto y coma
            record["BC_kinds"] = "; ".join(kinds_unique) if kinds_unique else ""
            record["BC_kinds_count"] = len(kinds_unique)

            records.append(record)
    return records


def write_csv_from_records(records: Iterable[Dict[str, Any]], out_path: Path) -> None:
    records = list(records)
    if not records:
        # Crear archivo vacío con solo cabecera vacía
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("")
        return

    # Determinar todas las claves (columnas)
    fieldnames = sorted({k for r in records for k in r.keys()})
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in records:
            # Asegurar que todos los valores sean serializables como str/primitive
            row = {k: ("" if r.get(k) is None else r.get(k)) for k in fieldnames}
            writer.writerow(row)


def convert_jsonl_to_breaking_csv(input_path: str, output_path: str) -> None:
    in_p = Path(input_path)
    out_p = Path(output_path)
    records = read_jsonl_breaking_commits(in_p)
    write_csv_from_records(records, out_p)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert .jsonl -> CSV extrayendo campos de `breaking_commit` por fila."
    )
    # parser.add_argument("input", help="Ruta al archivo .jsonl de entrada")
    # parser.add_argument("output", help="Ruta al archivo .csv de salida")
    # args = parser.parse_args()
    input_file = "/home/xchen6/breaking_updates_rl/data/sft/sft_data_updated.jsonl"
    data_output = "data2.csv"
    convert_jsonl_to_breaking_csv(input_file, data_output)
 