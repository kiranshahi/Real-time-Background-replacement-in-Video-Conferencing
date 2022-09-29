import os
import shutil

mydir = input("Enter directory name: ")
ptrn = input("Enter pattern name: ")

res = (path for path, _, _ in os.walk(mydir) if ptrn in path)
for path in res:
    try:
        shutil.rmtree(path)
        print(path + 'is removed.')
    except OSError as e:
        print("Error: %s - %s." % (e.filename, e.strerror))
