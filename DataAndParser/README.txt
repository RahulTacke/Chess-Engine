The data files are too large to add to the github repo.
I have included them in the following drive link

https://drive.google.com/drive/folders/1BU0X3DivbauQ74VzbcMqWppYRTJO4aMb?usp=drive_link

Make sure to download them exactly as theyre named in the drive link so the
git ignore doesnt miss them and crash the repo, or add renames to gitignore.

Additionally I have included a chunk size variable into the parser that allows you to
break the datasets into smaller peices, it is at 5 million positions per set right now.

To run the parser the only files you need are lichessparse.py and lichessdata.pgn.zst
The parser will then run on the data and return chunks of whatever size as it goes.
You can kill this process at any time and it will stop reading in the source file.

There is a built in print statement to show when chunks are made as well as an
update of how many moves captured in the past 10000 games.