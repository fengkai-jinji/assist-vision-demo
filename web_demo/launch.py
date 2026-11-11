"""Double-click launcher: reuse this demo or start it with the current interpreter."""
import json
import subprocess
import sys
import time
import webbrowser
from pathlib import Path
from urllib.request import build_opener, ProxyHandler

HERE=Path(__file__).resolve().parent
URL='http://127.0.0.1:8765'
opener=build_opener(ProxyHandler({}))

def running():
    try:
        with opener.open(URL+'/api/status',timeout=1) as response: data=json.load(response)
        return data.get('models',{}).get('project')==str(HERE.parent/'runs/detect/products_v2/weights/best.pt')
    except (OSError,ValueError): return False

if __name__=='__main__':
    if running():
        webbrowser.open(URL)
        print('Demo is already running:',URL)
    else:
        child=subprocess.Popen([sys.executable,'-u',str(HERE/'app.py')],cwd=HERE.parent)
        try:
            for _ in range(40):
                if child.poll() is not None: raise RuntimeError('Demo failed to start; check the error above.')
                if running(): break
                time.sleep(.25)
            else: raise RuntimeError('Startup timed out.')
            webbrowser.open(URL)
            print('Keep this window open. Press Ctrl+C to stop the local server.',flush=True)
            child.wait()
        finally:
            if child.poll() is None: child.terminate()
