import json, sys
from jsonschema import validate, Draft7Validator

def load(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def main():
    if len(sys.argv) < 3:
        print("Usage: python validate.py <schema.json> <payload.json>")
        sys.exit(1)
    schema = load(sys.argv[1])
    payload = load(sys.argv[2])
    v = Draft7Validator(schema)
    errs = [f"{e.message} at {'/'.join([str(x) for x in e.path])}" for e in v.iter_errors(payload)]
    if errs:
        print("INVALID")
        for e in errs: print("-", e)
        sys.exit(2)
    print("VALID")

if __name__ == '__main__':
    main()
