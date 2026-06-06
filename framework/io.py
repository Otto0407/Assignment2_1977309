"""
framework/io.py
---------------
Custom CSV I/O and dataset-splitting utilities. Uses only numpy (no pandas).
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------

def load_csv(
    filepath: str,
    delimiter: str = ',',
    has_header: bool = True,
) -> tuple:
    """
    Load a CSV file into a numpy array.

    Missing values (empty fields) are stored as np.nan.

    Parameters
    ----------
    filepath  : str  – path to the CSV file
    delimiter : str  – field separator, default ','
    has_header : bool – if True, first row is treated as column names

    Returns
    -------
    data    : np.ndarray, shape (n_rows, n_cols)  – float64 array
    headers : list of str  – column names (empty list if has_header=False)
    """
    headers: list = []
    rows: list = []
    with open(filepath, 'r', newline='', encoding='utf-8') as fh:
        for line_no, raw in enumerate(fh):
            line = raw.rstrip('\n').rstrip('\r')
            if not line:
                continue
            fields = line.split(delimiter)
            if has_header and line_no == 0:
                headers = [f.strip() for f in fields]
                continue
            parsed = []
            for field in fields:
                field = field.strip()
                if field == '' or field.lower() in ('nan', 'na', 'null', 'none'):
                    parsed.append(np.nan)
                else:
                    try:
                        parsed.append(float(field))
                    except ValueError:
                        parsed.append(np.nan)
            rows.append(parsed)
    if rows:
        data = np.array(rows, dtype=float)
    else:
        data = np.empty((0, len(headers) if headers else 0), dtype=float)
    return data, headers


# ---------------------------------------------------------------------------
# CSV saving
# ---------------------------------------------------------------------------

def save_csv(
    filepath: str,
    data: np.ndarray,
    headers: list = None,
) -> None:
    """
    Write a numpy array to a CSV file.

    Parameters
    ----------
    filepath : str
        Destination path.
    data : np.ndarray, shape (n, d)
        Data to write.
    headers : list of str or None
        Column names for the header row.
    """
    data = np.asarray(data)
    with open(filepath, 'w', newline='', encoding='utf-8') as fh:
        if headers:
            fh.write(','.join(str(h) for h in headers) + '\n')
        for row in data:
            fh.write(','.join(
                '' if (isinstance(v, float) and np.isnan(v)) else str(v)
                for v in row
            ) + '\n')


# ---------------------------------------------------------------------------
# Streaming CSV generator
# ---------------------------------------------------------------------------

def stream_csv(
    filepath: str,
    chunk_size: int = 100,
    delimiter: str = ',',
    has_header: bool = True,
):
    """
    Generator that yields chunks of rows from a CSV file.

    Parameters
    ----------
    filepath   : str
    chunk_size : int  – number of rows per chunk
    delimiter  : str
    has_header : bool

    Yields
    ------
    (X_chunk, headers) : (np.ndarray shape (chunk_size, d), list of str)
        The last chunk may have fewer than chunk_size rows.
    """
    headers: list = []
    chunk: list = []
    with open(filepath, 'r', newline='', encoding='utf-8') as fh:
        for line_no, raw in enumerate(fh):
            line = raw.rstrip('\n').rstrip('\r')
            if not line:
                continue
            fields = line.split(delimiter)
            if has_header and line_no == 0:
                headers = [f.strip() for f in fields]
                continue
            parsed = []
            for field in fields:
                field = field.strip()
                if field == '' or field.lower() in ('nan', 'na', 'null', 'none'):
                    parsed.append(np.nan)
                else:
                    try:
                        parsed.append(float(field))
                    except ValueError:
                        parsed.append(np.nan)
            chunk.append(parsed)
            if len(chunk) == chunk_size:
                yield np.array(chunk, dtype=float), headers
                chunk = []
    if chunk:
        yield np.array(chunk, dtype=float), headers


# ---------------------------------------------------------------------------
# Chunk splitter
# ---------------------------------------------------------------------------

def split_into_chunks(
    X: np.ndarray,
    y: np.ndarray,
    n_chunks: int,
) -> list:
    """
    Split (X, y) into n_chunks consecutive, roughly equal-sized chunks.

    Parameters
    ----------
    X        : np.ndarray, shape (n, d)
    y        : np.ndarray, shape (n,)
    n_chunks : int

    Returns
    -------
    list of (X_chunk, y_chunk) tuples
        Length == n_chunks. The last chunk absorbs any remainder rows.
    """
    X = np.asarray(X)
    y = np.asarray(y)
    n = X.shape[0]
    chunk_size = n // n_chunks
    chunks = []
    for i in range(n_chunks):
        start = i * chunk_size
        end = start + chunk_size if i < n_chunks - 1 else n
        chunks.append((X[start:end], y[start:end]))
    return chunks
