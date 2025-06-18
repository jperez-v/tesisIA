#!/usr/bin/env python3
import argparse
import logging
from pathlib import Path

def find_and_delete(dir_path: Path, extension: str, dry_run: bool, verbose: bool) -> None:
    """
    Busca recursivamente archivos con la extensión dada y los elimina.

    :param dir_path: Directorio raíz donde iniciar la búsqueda.
    :param extension: Extensión de archivo a eliminar (e.g. '.keras').
    :param dry_run: Si True, solo muestra qué archivos se eliminarían.
    :param verbose: Si True, muestra información detallada en logs.
    """
    if not dir_path.is_dir():
        logging.error(f"'{dir_path}' no es un directorio válido.")
        return

    pattern = f"*{extension}"
    files = list(dir_path.rglob(pattern))
    logging.info(f"Encontrados {len(files)} archivos con extensión '{extension}'.")

    for file in files:
        if dry_run:
            logging.info(f"[DRY-RUN] {file}")
        else:
            try:
                file.unlink()
                logging.info(f"Eliminado: {file}")
            except Exception as e:
                logging.error(f"Error al eliminar '{file}': {e}")

    if dry_run:
        logging.warning(
            "Dry-run activado: no se eliminaron archivos.\n"
            "Para eliminar realmente, ejecuta con el flag --no-dry-run."
        )

def main():
    parser = argparse.ArgumentParser(
        description="Eliminar archivos .keras recursivamente en un directorio."
    )
    parser.add_argument(
        "directory",
        nargs="?",
        default=".",
        help="Directorio raíz donde buscar (por defecto: directorio actual)."
    )
    parser.add_argument(
        "--no-dry-run",
        action="store_true",
        dest="confirm_delete",
        help="Confirma la eliminación real de archivos (desactiva dry-run)."
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Activa salida detallada (logging INFO)."
    )

    args = parser.parse_args()

    # Configuración de logging
    level = logging.INFO if args.verbose else logging.WARNING
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")

    root = Path(args.directory).resolve()
    find_and_delete(
        dir_path=root,
        extension=".png",
        dry_run=not args.confirm_delete,
        verbose=args.verbose
    )

if __name__ == "__main__":
    main()

