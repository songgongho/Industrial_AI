from __future__ import annotations

from pathlib import Path

import pandas as pd

TABLE_HINTS = (
    "press",
    "profile",
    "result",
    "inspect",
    "log",
    "data",
    "measure",
    "process",
    "defect",
    "ng",
)


def discover_mdb_files(input_dir: Path) -> list[Path]:
    """Return sorted MDB files under the input directory."""
    return sorted(input_dir.glob("*.mdb"))


def _import_pyodbc():
    try:
        import pyodbc  # type: ignore
    except ImportError as exc:
        raise RuntimeError("pyodbc is not installed. Install requirements first.") from exc
    return pyodbc


def resolve_access_driver(preferred: str | None = None) -> str:
    """Resolve a usable Access ODBC driver name."""
    pyodbc = _import_pyodbc()
    drivers = pyodbc.drivers()

    if preferred:
        if preferred in drivers:
            return preferred
        raise RuntimeError(f"Preferred ODBC driver not found: {preferred}")

    for driver in drivers:
        if "Access Driver" in driver and ("*.mdb" in driver or "*.accdb" in driver):
            return driver

    raise RuntimeError(
        "No Access ODBC driver found. Install Microsoft Access Database Engine "
        "and match Python/ODBC bitness (x64/x86)."
    )


def _connect_mdb(mdb_path: Path, driver_name: str):
    pyodbc = _import_pyodbc()
    conn_str = f"DRIVER={{{driver_name}}};DBQ={mdb_path};"
    return pyodbc.connect(conn_str)


def _list_user_tables(conn) -> list[str]:
    cur = conn.cursor()
    names = []
    for row in cur.tables(tableType="TABLE"):
        if row.table_name and not str(row.table_name).startswith("MSys"):
            names.append(str(row.table_name))
    return names


def _get_columns(conn, table_name: str) -> list[str]:
    cur = conn.cursor()
    return [str(row.column_name) for row in cur.columns(table=table_name)]


def _get_row_count(conn, table_name: str) -> int:
    try:
        cur = conn.cursor()
        cur.execute(f"SELECT COUNT(*) FROM [{table_name}]")
        return int(cur.fetchone()[0])
    except Exception:
        return 0


def _score_table(table_name: str, columns: list[str], row_count: int) -> float:
    score = 0.0
    table_low = table_name.lower()
    cols_low = [c.lower() for c in columns]

    for hint in TABLE_HINTS:
        if hint in table_low:
            score += 1.5
        score += sum(1.0 for col in cols_low if hint in col)

    if row_count > 0:
        score += min(5.0, row_count / 20000.0)

    return score


def _choose_best_table(conn) -> str:
    candidates: list[tuple[float, int, str]] = []

    for table_name in _list_user_tables(conn):
        columns = _get_columns(conn, table_name)
        row_count = _get_row_count(conn, table_name)
        score = _score_table(table_name, columns, row_count)
        candidates.append((score, row_count, table_name))

    if not candidates:
        raise RuntimeError("No user table found in MDB.")

    candidates.sort(reverse=True)
    return candidates[0][2]


def load_best_table(mdb_path: Path, driver_name: str) -> tuple[str, pd.DataFrame]:
    """Load the most likely production table from one MDB file."""
    conn = _connect_mdb(mdb_path, driver_name)
    try:
        table_name = _choose_best_table(conn)
        frame = pd.read_sql(f"SELECT * FROM [{table_name}]", conn)
        return table_name, frame
    finally:
        conn.close()

