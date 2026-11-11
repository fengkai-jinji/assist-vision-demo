import io
import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, build_opener, ProxyHandler

import cv2
import numpy as np

from app import make_server, validate_stream_url


class FakeDetector:
    def info(self):
        return {'device':'test','models':{'project':'test.pt','general':'test.pt'}}

    def predict(self, frame, conf, mode, track=False):
        return {'objects':[], 'inference_ms':1.0, 'device':'test'}


class DemoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent)
        cls.server=make_server('127.0.0.1',0,FakeDetector(),Path(cls.temp.name))
        cls.thread=threading.Thread(target=cls.server.serve_forever,daemon=True)
        cls.thread.start()
        cls.base=f'http://127.0.0.1:{cls.server.server_port}'

    @classmethod
    def tearDownClass(cls):
        cls.server.state.stop()
        cls.server.shutdown();cls.server.server_close()
        assert Path(cls.temp.name).resolve().is_relative_to(Path(__file__).resolve().parent)
        cls.temp.cleanup()

    def call(self,path,data=None):
        if isinstance(data,dict):data=json.dumps(data).encode()
        request=Request(self.base+path,data=data)
        try:
            with build_opener(ProxyHandler({})).open(request,timeout=10) as res:return res.status,json.load(res)
        except HTTPError as e:return e.code,json.load(e)

    def test_image_round_trip(self):
        raw=cv2.imencode('.jpg',np.zeros((80,120,3),np.uint8))[1].tobytes()
        status,result=self.call('/api/image?conf=0.35&mode=project',raw)
        self.assertEqual(status,200)
        self.assertEqual((result['width'],result['height']),(120,80))
        self.assertTrue(result['image'].startswith('data:image/jpeg;base64,'))

    def test_invalid_image_and_confidence(self):
        self.assertEqual(self.call('/api/image',b'not an image')[0],400)
        self.assertEqual(self.call('/api/image?conf=nan',b'bad')[0],400)
        self.assertEqual(self.call('/api/image?mode=missing',b'bad')[0],400)

    def test_stream_sources_reject_files(self):
        for url in ['file:///C:/secret','F:/video.mp4','https://example.org/video','http://169.254.169.254/']:
            with self.assertRaises(ValueError):validate_stream_url(url)
        self.assertEqual(validate_stream_url('http://192.168.1.20:41737/video.mjpeg'),
                         'http://192.168.1.20:41737/video.mjpeg')

    def test_video_upload_and_end(self):
        path=Path(self.temp.name)/'fixture.avi'
        writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'MJPG'),10,(64,48))
        for _ in range(3):writer.write(np.zeros((48,64,3),np.uint8))
        writer.release()
        code,upload=self.call('/api/video?name=test.avi',path.read_bytes())
        self.assertEqual(code,200)
        code,start=self.call('/api/start',{'kind':'video','source':upload['id']})
        self.assertEqual(code,200)
        for _ in range(50):
            _,live=self.call('/api/live?id='+start['id'])
            if live['status'] in ['completed','error']:break
            time.sleep(.05)
        self.assertEqual(live['status'],'completed')
        self.assertGreater(live['sequence'],0)
        self.assertEqual(self.call('/api/stop',{'id':start['id']})[0],200)
        self.assertEqual(self.call('/api/stop',{})[0],400)  # 空 id 拒绝无条件停止

    def test_missing_video_and_stale_session(self):
        self.assertEqual(self.call('/api/start',{'kind':'video','source':'missing'})[0],400)
        self.assertEqual(self.call('/api/live?id=missing')[0],404)

    def test_old_page_cannot_stop_new_capture(self):
        path=Path(self.temp.name)/'long.avi'
        writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'MJPG'),10,(64,48))
        for _ in range(40):writer.write(np.zeros((48,64,3),np.uint8))
        writer.release()
        _,upload=self.call('/api/video?name=long.avi',path.read_bytes())
        _,first=self.call('/api/start',{'kind':'video','source':upload['id']})
        _,second=self.call('/api/start',{'kind':'video','source':upload['id']})
        self.call('/api/stop',{'id':first['id']})
        _,live=self.call('/api/live?id='+second['id'])
        self.assertNotEqual(live['status'],'stopped')
        self.assertFalse(self.server.state.event.is_set())
        self.call('/api/stop',{'id':second['id']})


if __name__=='__main__':unittest.main()
