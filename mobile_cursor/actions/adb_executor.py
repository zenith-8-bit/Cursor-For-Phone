import subprocess
import time

class ADBExecutor:
    def __init__(self, adb_path="adb", timeout=15):
        self.adb = adb_path
        self.timeout = timeout

    def _run(self, *args):
        result = subprocess.run(
            [self.adb, *map(str, args)],
            capture_output=True, text=True, timeout=self.timeout
        )
        if result.returncode:
            raise RuntimeError(result.stderr.strip() or "ADB command failed")
        return result.stdout.strip()

    def connected(self):
        output = self._run("devices")
        return any(line.strip().endswith("\tdevice") for line in output.splitlines())

    def shell(self, *args):
        return self._run("shell", *args)

    def home(self): self.shell("input","keyevent","KEYCODE_HOME")
    def back(self): self.shell("input","keyevent","KEYCODE_BACK")
    def recents(self): self.shell("input","keyevent","KEYCODE_APP_SWITCH")

    def tap(self, x, y):
        self.shell("input","tap",x,y)

    def long_press(self, x, y, duration_ms=700):
        self.shell("input","swipe",x,y,x,y,duration_ms)

    def swipe(self, x1,y1,x2,y2,duration_ms=500):
        self.shell("input","swipe",x1,y1,x2,y2,duration_ms)

    def type_text(self, text):
        escaped = (text.replace("%","%25").replace(" ","%s")
                   .replace("&","\\&").replace("<","\\<").replace(">","\\>"))
        self.shell("input","text",escaped)

    def key(self, key):
        self.shell("input","keyevent",key)

    def open_app(self, package):
        self.shell("monkey","-p",package,"1")

    def clear_text(self):
        self.key("KEYCODE_MOVE_END")
        self.shell("input","keyevent","--longpress","KEYCODE_DEL")

    def wait(self, seconds=1):
        time.sleep(seconds)
