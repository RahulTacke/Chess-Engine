# DataAndParser

The data files are too large to add to the GitHub repo.
They are available at the following Drive link:

https://drive.google.com/drive/folders/1BU0X3DivbauQ74VzbcMqWppYRTJO4aMb?usp=drive_link

Make sure to download them exactly as they are named in the Drive link so that
`.gitignore` does not miss them and crash the repo, or add any renames to `.gitignore`.

A chunk size variable in the parser controls how large each output file is; it is
currently set to 5 million positions per chunk.

## Running the parser

The only files required are `lichessparse.py` and `lichessdata.pgn.zst`.
Run from the repo root:

```bash
python3 DataAndParser/lichessparse.py
```

The parser writes chunks of the configured size as it reads the source file.
You can kill the process at any time — it will stop cleanly after the current chunk.

Progress is printed every 10,000 games, showing the game count and total positions captured so far.
