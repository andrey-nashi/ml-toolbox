import json

class FileOperationJSON:

    def __init__(self, path: str):
        f = open(path, "r")
        data = json.load(f)
        f.close()

