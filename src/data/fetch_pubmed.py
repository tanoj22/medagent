"""Fetch PubMed abstracts for the MedAgent corpus via NCBI Entrez."""
import json
import os
import time
from pathlib import Path

from Bio import Entrez
from dotenv import load_dotenv

load_dotenv()
Entrez.email = os.environ["NCBI_EMAIL"]
Entrez.api_key = os.environ["NCBI_API_KEY"]

QUERY = (
    '("protein language model" OR "deep learning" OR "machine learning") '
    'AND ("drug discovery" OR "protein structure" OR "computational biology" '
    'OR "bioinformatics" OR "molecular design")'
)
YEARS = range(2020, 2027)   # 2020 through 2026
TARGET = 30000
BATCH_SIZE = 200            # records per efetch
OUTPUT = Path("data/raw/pubmed.jsonl")


def get_year_pmids(year):
    """All PMIDs for one year. Each year is < 10k, so no history server needed."""
    handle = Entrez.esearch(db="pubmed", term=QUERY, mindate=str(year),
                            maxdate=str(year), datetype="pdat", retmax=0)
    count = int(Entrez.read(handle)["Count"]); handle.close()

    pmids = []
    for start in range(0, count, 9000):
        for attempt in range(4):
            try:
                handle = Entrez.esearch(db="pubmed", term=QUERY, mindate=str(year),
                                        maxdate=str(year), datetype="pdat",
                                        retstart=start, retmax=9000)
                pmids.extend(Entrez.read(handle)["IdList"]); handle.close()
                break
            except Exception as e:
                print(f"    {year} page {start} failed (try {attempt+1}): {e}")
                time.sleep(2 ** attempt)
    print(f"  {year}: {count} papers, {len(pmids)} ids")
    return pmids


def get_all_pmids():
    all_pmids = []
    for year in YEARS:
        all_pmids.extend(get_year_pmids(year))
    seen, unique = set(), []
    for p in all_pmids:
        if p not in seen:
            seen.add(p); unique.append(p)
    return unique[:TARGET]


def parse_article(article):
    medline = article["MedlineCitation"]
    pmid = str(medline["PMID"])
    art = medline["Article"]
    title = str(art.get("ArticleTitle", ""))

    abstract = ""
    if "Abstract" in art:
        parts = art["Abstract"].get("AbstractText", [])
        abstract = " ".join(str(p) for p in parts)

    year = ""
    pub_date = art.get("Journal", {}).get("JournalIssue", {}).get("PubDate", {})
    if "Year" in pub_date:
        year = str(pub_date["Year"])

    authors = []
    for a in art.get("AuthorList", []):
        last, initials = a.get("LastName", ""), a.get("Initials", "")
        if last:
            authors.append(f"{last} {initials}".strip())

    return {"pmid": pmid, "title": title, "abstract": abstract,
            "year": year, "authors": authors}


def main():
    print("Collecting PMIDs year by year...")
    pmids = get_all_pmids()
    print(f"\nCollected {len(pmids)} unique PMIDs. Fetching abstracts...")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    written, skipped = 0, 0
    with OUTPUT.open("w", encoding="utf-8") as f:
        for start in range(0, len(pmids), BATCH_SIZE):
            batch = pmids[start:start + BATCH_SIZE]
            records = None
            for attempt in range(4):
                try:
                    handle = Entrez.efetch(db="pubmed", id=",".join(batch),
                                           retmode="xml")
                    records = Entrez.read(handle); handle.close()
                    break
                except Exception as e:
                    print(f"  batch at {start} failed (try {attempt+1}): {e}")
                    time.sleep(2 ** attempt)
            if records is None:
                print(f"  giving up on batch at {start}")
                continue
            for article in records.get("PubmedArticle", []):
                rec = parse_article(article)
                if not rec["abstract"]:
                    skipped += 1
                    continue
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                written += 1
            done = min(start + BATCH_SIZE, len(pmids))
            print(f"  {done}/{len(pmids)} processed | {written} written | {skipped} skipped")
            time.sleep(0.15)

    print(f"\nDone. Wrote {written} abstracts to {OUTPUT} (skipped {skipped}).")


if __name__ == "__main__":
    main()