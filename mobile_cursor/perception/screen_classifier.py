class ScreenClassifier:
    """Adapter for the future dedicated UI/screen model."""

    def analyze(self, image_path):
        return {"elements": []}

# Expected future output:
# {
#   "elements": [
#     {
#       "label": "Search",
#       "type": "button",
#       "bbox": [100,200,180,260],
#       "confidence": 0.93
#     }
#   ]
# }
