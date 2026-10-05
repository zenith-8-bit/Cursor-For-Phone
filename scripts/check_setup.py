import shutil, subprocess, urllib.request

print("=== Mobile Cursor setup check ===")
adb=shutil.which("adb")
print("adb:",adb or "NOT FOUND")

if adb:
    try: print(subprocess.check_output(["adb","devices"],text=True))
    except Exception as e: print("adb error:",e)

try:
    with urllib.request.urlopen("http://127.0.0.1:11434/api/tags",timeout=3) as r:
        print("Ollama:",r.status)
        print(r.read().decode()[:500])
except Exception as e:
    print("Ollama: NOT REACHABLE:",e)
