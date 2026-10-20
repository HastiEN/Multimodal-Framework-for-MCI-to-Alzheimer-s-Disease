# What to commit and what not to commit

**Commit (code only)**
- `code/*.py` analysis scripts
- `run_pipeline.py`, `requirements.txt`
- `README.md`, `COMMIT_GUIDE.md`, `.gitignore`
- `REMAINING_TO_UPLOAD.md` while uploads are pending (delete it after the last batch)

**Never commit**
- ADNI data of any kind (`data/`, raw tables, processed tables) - ADNI's data-use terms forbid redistribution
- Any `.csv`, `.xlsx`, `.parquet`, `.pkl`, `.npz` result or data file
- `runs/`, `outputs/`, `final_result/`, `audit/`, checkpoints and logs
- Manuscript files (`.docx`, `.pdf`, review notes, `*ByCodex.md`)
- `__pycache__/`, virtual environments, credentials or tokens

Before each commit run `git status` and check that nothing from the "never" list is staged; the `.gitignore` already blocks most of it.
