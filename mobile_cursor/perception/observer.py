import datetime as dt
import re
import subprocess
from .ocr import OCRReader
from ..models import Observation

class Observer:
    def __init__(self, adb, screenshot_dir):
        self.adb = adb
        self.screenshot_dir = screenshot_dir
        self.ocr = OCRReader()

    def screenshot(self):
        filename = self.screenshot_dir / (
            dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".png"
        )
        with open(filename, "wb") as fh:
            subprocess.run(
                [self.adb.adb, "exec-out", "screencap", "-p"],
                stdout=fh, stderr=subprocess.PIPE, check=True,
                timeout=self.adb.timeout
            )
        return filename

    def current_package(self):
        output = self.adb.shell("dumpsys","window","windows")
        match = re.search(r"mCurrentFocus=Window\{.*?\s([^/\s]+)/", output)
        return match.group(1) if match else ""

    def observe(self, last_result=""):
        connected = self.adb.connected()
        if not connected:
            return Observation(
                timestamp=dt.datetime.now().isoformat(),
                connected=False, last_result=last_result
            )
        screen = self.screenshot()
        return Observation(
            timestamp=dt.datetime.now().isoformat(),
            connected=True,
            package=self.current_package(),
            screen_path=str(screen),
            ocr_text=self.ocr.read(screen),
            last_result=last_result
        )
