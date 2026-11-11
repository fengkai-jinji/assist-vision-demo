"""Local-only photo/video/camera demo using the existing YOLO environment."""
import argparse
import base64
import ipaddress
import json
import math
import os
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from guide_demo.warning import WarningEngine
MODELS = {'project': ROOT/'runs/detect/products_v2/weights/best.pt',
          'general': ROOT/'models/yolo26n.pt'}
LABELS = {'bottle':'瓶／杯','chair':'椅子','door':'门','person':'人','car':'汽车',
          'bicycle':'自行车','motorcycle':'摩托车','bus':'公交车','truck':'卡车',
          'cup':'杯子','wine glass':'高脚杯','cell phone':'手机','laptop':'笔记本电脑',
          'keyboard':'键盘','mouse':'鼠标','tv':'屏幕','book':'书','backpack':'背包',
          'handbag':'手提包','umbrella':'雨伞','dining table':'桌子','couch':'沙发',
          'bed':'床','cat':'猫','dog':'狗','potted plant':'盆栽','clock':'钟',
          'refrigerator':'冰箱','sink':'水槽','toilet':'马桶','bench':'长椅'}


def settings(values):
    conf = float(values.get('conf', .35))
    mode = values.get('mode', 'project')
    if not math.isfinite(conf) or not .05 <= conf <= .95:
        raise ValueError('置信度必须在 0.05～0.95 之间。')
    if mode not in MODELS: raise ValueError('未知模型。')
    return conf, mode


def validate_stream_url(value):
    url = urlsplit(value)
    if url.scheme not in ('http','https','rtsp') or not url.hostname:
        raise ValueError('请输入手机 App 提供的 HTTP/MJPEG 或 RTSP 视频流地址。')
    try: address = ipaddress.ip_address(url.hostname)
    except ValueError: raise ValueError('请使用手机的局域网 IP 地址，不使用域名。')
    allowed = (address.is_loopback or
               any(address in net for net in (ipaddress.ip_network('10.0.0.0/8'),
                   ipaddress.ip_network('172.16.0.0/12'),ipaddress.ip_network('192.168.0.0/16'))))
    if not allowed: raise ValueError('只支持本机或局域网 IP，不支持公网地址。')
    try: _ = url.port  # 提前校验畸形端口，再启动采集线程
    except ValueError: raise ValueError('视频流地址端口不正确。')
    return value


class Detector:
    def __init__(self):
        import torch
        torch.set_num_threads(4)
        self.device = 0 if torch.cuda.is_available() else 'cpu'
        self.device_name = torch.cuda.get_device_name(0) if self.device == 0 else 'CPU'
        self.models = {}
        self.lock = threading.Lock()

    def info(self):
        return {'device':self.device_name,'models':{k:str(v) for k,v in MODELS.items()}}

    def predict(self, frame, conf, mode, track=False):
        from ultralytics import YOLO
        with self.lock:
            if mode not in self.models:
                if not MODELS[mode].is_file(): raise ValueError('模型权重不存在：'+str(MODELS[mode]))
                self.models[mode] = YOLO(MODELS[mode])
            model = self.models[mode]
            began = time.perf_counter()
            # persist=True 的跟踪器状态挂在共享 model 对象上，跨会话延续：
            # 换源后 id 不清零属已知行为（引擎每会话新建，无计数串扰）
            if track:
                result = model.track(frame,conf=conf,imgsz=640,device=self.device,
                                     persist=True,tracker='bytetrack.yaml',verbose=False)[0]
            else:
                result = model.predict(frame,conf=conf,imgsz=640,device=self.device,verbose=False)[0]
            elapsed = (time.perf_counter()-began)*1000
            objects = []
            for i,box in enumerate(result.boxes):
                class_id = int(box.cls.item())
                name = model.names.get(class_id) or f'未知类别{class_id}'
                xy = [round(float(v),1) for v in box.xyxy[0].tolist()]
                center = (xy[0]+xy[2])/(2*frame.shape[1])
                label = LABELS.get(name,name)
                if mode == 'general' and name == 'bottle': label = '瓶子'
                ident = int(box.id.item()) if getattr(box,'id',None) is not None else None
                objects.append({'class_id':class_id,'name':name,'label':label,'id':ident,
                    'confidence':round(float(box.conf.item()),4),'xyxy':xy,
                    'side':'左侧' if center < 1/3 else '右侧' if center > 2/3 else '中间'})
        return {'objects':objects,'inference_ms':round(elapsed,1),'device':self.device_name}


def frame_result(detector, frame, conf, mode, engine=None, track=False):
    height,width = frame.shape[:2]
    if max(height,width)>1280:
        scale = 1280/max(height,width)
        frame = cv2.resize(frame,(round(width*scale),round(height*scale)))
    result = detector.predict(frame,conf,mode,track=track)
    if engine is not None:
        decorated,warning = engine.observe(result['objects'],frame.shape[1],frame.shape[0])
        result['objects'] = decorated
        result['warning'] = warning
    else:
        result['warning'] = None  # 契约恒定：前端可无判断读取该键
    ok,encoded = cv2.imencode('.jpg',frame,[cv2.IMWRITE_JPEG_QUALITY,85])
    if not ok: raise ValueError('图像编码失败。')
    return result | {'image':'data:image/jpeg;base64,'+base64.b64encode(encoded).decode(),
                     'width':frame.shape[1],'height':frame.shape[0]}


class LiveState:
    def __init__(self, detector, storage):
        self.detector=detector; self.storage=storage
        self.storage.mkdir(parents=True,exist_ok=True)
        # uploads/ 只存临时副本：启动时清掉上次运行残留
        for leftover in self.storage.glob('*'):
            if leftover.is_file():
                try: leftover.unlink()
                except OSError: pass
        self.uploads={}; self.thread=None; self.event=threading.Event()
        self.lock=threading.Lock(); self.control=threading.Lock()
        self.snapshot={'id':None,'status':'idle','sequence':0}
        self.active_upload=None     # 当前会话正在读的上传副本，剪除时跳过
        self.pending_cleanup=set()  # Windows 下被占用删不掉的副本，会话结束后补删

    def stop(self):
        self.event.set()
        if self.thread and self.thread.is_alive(): self.thread.join(timeout=6)
        if self.thread and self.thread.is_alive():
            raise ValueError('上一个视频源仍在关闭，请稍后再试。')
        # 会话已结束：补删之前因文件占用没删掉的副本（排除新会话在用的）
        for path in list(self.pending_cleanup):
            if str(path) != str(self.active_upload or ''):
                try: path.unlink(); self.pending_cleanup.discard(path)
                except OSError: pass

    def start(self, data):
        conf,mode=settings(data)
        kind=data.get('kind'); source=data.get('source')
        if kind == 'video':
            if source not in self.uploads: raise ValueError('请先上传视频。')
            source=str(self.uploads[source])
        elif kind == 'camera':
            if type(source) is not int or not 0<=source<=4: raise ValueError('摄像头编号应为 0～4。')
        elif kind == 'phone': source=validate_stream_url(str(source))
        else: raise ValueError('未知视频来源。')
        with self.control:
            self.stop()
            self.event=threading.Event(); session=uuid.uuid4().hex
            # 记录本会话在用的上传副本，避免并发上传的剪除误删在用文件
            self.active_upload = self.uploads.get(source) if kind == 'video' else None
            with self.lock: self.snapshot={'id':session,'status':'connecting','sequence':0,'kind':kind}
            self.thread=threading.Thread(target=self.run,args=(session,kind,source,conf,mode,self.event),daemon=True)
            self.thread.start()
        return session

    def update(self, **data):
        with self.lock: self.snapshot=self.snapshot | data

    def run(self, session, kind, source, conf, mode, stop):
        cap=None
        engine=WarningEngine(need_frames=2,labels=LABELS)  # 实时流：同一目标连续 2 帧确认
        try:
            if kind == 'camera':
                cap=cv2.VideoCapture(source,cv2.CAP_DSHOW if os.name=='nt' else cv2.CAP_ANY)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH,640);cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
            elif kind == 'phone':
                cap=cv2.VideoCapture(source,cv2.CAP_FFMPEG,[cv2.CAP_PROP_OPEN_TIMEOUT_MSEC,5000,
                                                         cv2.CAP_PROP_READ_TIMEOUT_MSEC,3000])
            else: cap=cv2.VideoCapture(source)
            if not cap.isOpened():
                raise ValueError('无法打开视频源。电脑摄像头请检查隐私权限、占用情况或编号；手机请检查同一 Wi-Fi 和视频流地址。')
            cap.set(cv2.CAP_PROP_BUFFERSIZE,1)
            fps=cap.get(cv2.CAP_PROP_FPS)
            if not math.isfinite(fps) or fps<=0: fps=25
            total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if kind=='video' else 0
            step=max(1,round(fps/10)) if kind=='video' else 1
            index=0;sequence=0; began=time.monotonic()
            while not stop.is_set():
                ok,frame=cap.read()
                if not ok:
                    if kind!='video': raise ValueError('摄像头画面已中断，请检查连接后重新开始。')
                    if sequence==0: raise ValueError('视频无法解码，请使用 MP4/H.264 或重新导出视频。')
                    break
                index+=1
                if kind=='video' and stop.wait(max(0,(index-1)/fps-(time.monotonic()-began))): break
                result=frame_result(self.detector,frame,conf,mode,engine=engine,track=True)
                sequence+=1
                self.update(status='running',sequence=sequence,frame=result,
                            video_time=round(index/fps,2) if kind=='video' else None,
                            duration=round(total/fps,2) if total else None)
                for _ in range(step-1):
                    if cap.grab(): index+=1
            self.update(status='stopped' if stop.is_set() else 'completed')
        except Exception as error:
            self.update(status='stopped' if stop.is_set() else 'error',error=str(error))
        finally:
            if cap is not None: cap.release()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        if args and str(args[0]).startswith(('GET /api/live','GET /api/status')): return
        super().log_message(fmt,*args)

    def json_response(self, value, status=200):
        raw=json.dumps(value,ensure_ascii=False).encode('utf-8')
        self.send_response(status);self.send_header('Content-Type','application/json; charset=utf-8')
        self.send_header('Cache-Control','no-store');self.send_header('Content-Length',str(len(raw)))
        self.end_headers();self.wfile.write(raw)

    def do_GET(self):
        url=urlsplit(self.path);query=parse_qs(url.query)
        try:
            if url.path=='/':
                raw=(HERE/'index.html').read_bytes()
                self.send_response(200);self.send_header('Content-Type','text/html; charset=utf-8')
                self.send_header('Content-Length',str(len(raw)));self.end_headers();self.wfile.write(raw)
            elif url.path=='/api/status': self.json_response(self.server.state.detector.info())
            elif url.path=='/api/live':
                with self.server.state.lock: snapshot=self.server.state.snapshot.copy()
                if query.get('id',[''])[0]!=snapshot['id']: self.json_response({'error':'会话已结束或不存在。'},404)
                else:self.json_response(snapshot)
            else:self.json_response({'error':'页面不存在。'},404)
        except (ConnectionError,BrokenPipeError):pass

    def do_POST(self):
        url=urlsplit(self.path);query={k:v[0] for k,v in parse_qs(url.query).items()}
        try:
            origin=self.headers.get('Origin')
            if origin and urlsplit(origin).netloc!=self.headers.get('Host'):
                self.json_response({'error':'只允许本地 Demo 页面发起请求。'},403);return
            size=int(self.headers.get('Content-Length','0'))
            limit=300*1024*1024 if url.path=='/api/video' else 12*1024*1024
            if size<0 or size>limit:
                self.json_response({'error':'文件过大：图片最多 12 MB，视频最多 300 MB。'},413);return
            state=self.server.state
            if url.path=='/api/video':
                suffix=Path(query.get('name','')).suffix.lower()
                if suffix not in {'.mp4','.mov','.avi','.mkv','.webm','.m4v'}: raise ValueError('不支持此视频格式。')
                if not size: raise ValueError('视频文件为空。')
                ident=uuid.uuid4().hex;target=state.storage/(ident+suffix)
                remaining=size
                with target.open('wb') as stream:
                    while remaining:
                        chunk=self.rfile.read(min(1024*1024,remaining))
                        if not chunk: raise ValueError('视频上传中断。')
                        stream.write(chunk);remaining-=len(chunk)
                # 注册与剪除同锁串行：并发上传不会互相误删；跳过当前会话在用文件
                with state.control:
                    state.uploads[ident]=target
                    for key,value in list(state.uploads.items()):
                        if key != ident and str(value) != str(state.active_upload or ''):
                            state.uploads.pop(key,None)
                            try: value.unlink()
                            except OSError: state.pending_cleanup.add(value)  # 占用中，停会话后补删
                self.json_response({'id':ident});return
            raw=self.rfile.read(size)
            if url.path=='/api/image':
                conf,mode=settings(query)
                frame=cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_COLOR) if raw else None
                if frame is None: raise ValueError('图片无法读取，请选择 JPG、PNG 或 WebP。')
                # 单张图没有跨帧确认，need_frames=1 立即给等级与播报建议
                self.json_response(frame_result(state.detector,frame,conf,mode,
                                                engine=WarningEngine(need_frames=1,labels=LABELS)))
            elif url.path=='/api/start':self.json_response({'id':state.start(json.loads(raw))})
            elif url.path=='/api/stop':
                expected=json.loads(raw or b'{}').get('id')
                if not expected: raise ValueError('缺少会话 id，拒绝无条件停止。')
                with state.control:
                    if expected and expected!=state.snapshot['id']:
                        self.json_response({'status':'already_replaced'});return
                    state.stop()
                self.json_response({'status':'stopped'})
            else:self.json_response({'error':'接口不存在。'},404)
        except (ValueError,TypeError,KeyError) as error:self.json_response({'error':str(error)},400)
        except (ConnectionError,BrokenPipeError):pass
        except Exception as error:
            print('Inference error:',repr(error),flush=True)
            self.json_response({'error':'处理失败：'+str(error)},500)


def make_server(host='127.0.0.1',port=8765,detector=None,storage=None):
    server=ThreadingHTTPServer((host,port),Handler)
    server.daemon_threads=True
    server.state=LiveState(detector or Detector(),storage or HERE/'uploads')
    return server


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--port',type=int,default=8765)
    args=parser.parse_args()
    os.environ.setdefault('YOLO_CONFIG_DIR',str(ROOT/'Ultralytics'))
    server=make_server(port=args.port)
    print(f'YOLO demo ready: http://127.0.0.1:{server.server_port}',flush=True)
    print('Model:',MODELS['project'],'Device:',server.state.detector.info()['device'],flush=True)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:server.state.stop();server.server_close()
