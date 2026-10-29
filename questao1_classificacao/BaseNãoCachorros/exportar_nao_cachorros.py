"""Exporta 100 imagens por espécie (não cachorro) do dataset Rapidata/Animals-10.

Seleção reproduzível: ROW_NUMBER() ordenado por hash(image.path), mantendo
apenas linhas em que o prefixo do nome do arquivo e o label concordam.
"""
import sys
from pathlib import Path

import duckdb
from PIL import Image

OUT_DIR = Path(__file__).resolve().parent
PER_SPECIES = 100
SPECIES = {"cat": 1, "chicken": 2, "cow": 3, "horse": 6, "sheep": 7}
HF_GLOB = "hf://datasets/Rapidata/Animals-10/**/*.parquet"


def resolve_source(con):
    """Retorna a expressão de leitura do parquet (hf:// ou download local)."""
    try:
        files = [r[0] for r in con.execute(f"SELECT file FROM glob('{HF_GLOB}')").fetchall()]
        files = [f for f in files if "train" in f]
        if not files:
            raise RuntimeError("nenhum parquet de treino encontrado via hf://")
        print(f"[hf://] parquets: {files}")
        return files
    except Exception as e:
        print(f"AVISO: hf:// falhou ({e!r}); baixando com huggingface_hub...", file=sys.stderr)
        from huggingface_hub import snapshot_download
        local = snapshot_download(
            repo_id="Rapidata/Animals-10",
            repo_type="dataset",
            allow_patterns=["data/train-*.parquet"],
            local_dir=OUT_DIR / "_parquet_cache",
        )
        files = sorted(str(p) for p in Path(local).rglob("train-*.parquet"))
        print(f"[local] parquets: {files}")
        return files


def main():
    con = duckdb.connect()
    files = resolve_source(con)
    values = ", ".join(f"('{s}', {l})" for s, l in SPECIES.items())
    query = f"""
        WITH src AS (
            SELECT image.path AS path, image.bytes AS bytes, label,
                   regexp_extract(image.path, '^([^_]+)_', 1) AS species
            FROM read_parquet({files!r})
        ), ok AS (
            SELECT src.* FROM src
            JOIN (VALUES {values}) AS m(species, label)
              ON src.species = m.species AND src.label = m.label
        ), ranked AS (
            SELECT *, ROW_NUMBER() OVER (PARTITION BY species ORDER BY hash(path)) AS rn
            FROM ok
        )
        SELECT species, path, bytes FROM ranked WHERE rn <= {PER_SPECIES} ORDER BY species, rn
    """
    rows = con.execute(query).fetchall()
    print(f"Linhas selecionadas: {len(rows)}")

    for _, path, data in rows:
        (OUT_DIR / Path(path).name).write_bytes(data)

    # Validação
    errors = []
    counts = {s: 0 for s in SPECIES}
    imgs = [p for p in OUT_DIR.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}]
    for p in imgs:
        pre = p.name.split("_")[0]
        if pre in counts:
            counts[pre] += 1
        if p.stat().st_size == 0:
            errors.append(f"vazio: {p.name}")
            continue
        try:
            with Image.open(p) as im:
                im.verify()
        except Exception as e:
            errors.append(f"não abre como imagem: {p.name}: {e!r}")

    print(f"\nPasta: {OUT_DIR}")
    print(f"Total de imagens: {len(imgs)}")
    for s, n in counts.items():
        print(f"  {s}_: {n}")
    if len(imgs) != PER_SPECIES * len(SPECIES) or any(n != PER_SPECIES for n in counts.values()):
        errors.append("contagem diferente do esperado (500 total, 100 por espécie)")
    if errors:
        print("\nERROS:", *errors, sep="\n  ")
        sys.exit(1)
    print("\nValidação OK.")


if __name__ == "__main__":
    main()
