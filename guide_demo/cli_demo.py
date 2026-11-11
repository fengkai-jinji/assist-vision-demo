# -*- coding: utf-8 -*-
"""导盲预警 CLI Demo：与 web_demo 共用 guide_demo.warning 引擎。

功能对标便携推理包的 blind_demo.py（结构：检测 → 距离分级 → 方位 → 连续帧确认
→ 中文语音 + 冷却打断），差异：跟踪用 ByteTrack（同一目标连续 N 帧确认）、
类别角色语义（门=通行提示）、语音改为可选依赖。

用法（YOLOv8 目录下）：
  python -m guide_demo.cli_demo --source 0                    # 摄像头实时
  python -m guide_demo.cli_demo --source 视频.mp4             # 视频文件
  python -m guide_demo.cli_demo --source 图片.jpg             # 单张图片
  python -m guide_demo.cli_demo --source 目录 --no-show       # 批量图片（调试）
按键（视频/摄像头窗口）：空格暂停 · Q/Esc 退出。
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from .warning import WarningEngine

TIER_COLOR = {'danger': (77, 72, 229), 'near': (46, 161, 240), 'mid': (168, 160, 58),
              'far': (143, 160, 127), 'quiet': (95, 102, 85)}  # BGR
TIER_TEXT = {'danger': '危险', 'near': '近', 'mid': '中', 'far': '远', 'quiet': '仅显示'}
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / 'runs/detect/products_v2/weights/best.pt'


class Speaker:
    """中文语音：优先 pyttsx3（离线 SAPI），不可用则打印到控制台。"""

    def __init__(self, enabled=True):
        self.enabled = enabled
        self.engine = None
        if enabled:
            try:
                import pyttsx3
                self.engine = pyttsx3.init()
                for voice in self.engine.getProperty('voices'):
                    if 'chinese' in voice.name.lower() or 'huihui' in voice.name.lower():
                        self.engine.setProperty('voice', voice.id)
                        break
            except ImportError:
                print('提示：未安装 pyttsx3（pip install pyttsx3），语音降级为控制台输出。')

    def say(self, text, interrupt=False):
        if not self.enabled:
            return
        if self.engine is None:
            print(f'[播报{"·打断" if interrupt else ""}] {text}')
            return
        try:
            if interrupt:
                self.engine.stop()
            self.engine.say(text)
            self.engine.runAndWait()
        except Exception as error:  # SAPI 偶发异常不中断检测循环
            print(f'语音播报失败：{error}')


def short_path(path):
    """Windows 中文路径兼容：imread/imdecode 用字节流已可读，这里保留给 VideoCapture。"""
    return str(path)


def imread_u(path):
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None


def draw_overlay(frame, objects, warning):
    height, width = frame.shape[:2]
    for obj in objects:
        x1, y1, x2, y2 = [int(v) for v in obj['xyxy']]
        tier = obj.get('tier') or 'far'
        color = TIER_COLOR['quiet' if obj.get('role') == 'quiet' else tier]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        text = f"{obj['label']}·{TIER_TEXT.get(tier, tier)} {obj['confidence'] * 100:.1f}%"
        baseline = max(18, height // 30)
        # CJK 全角字符宽约等于字号：按 UTF-8 字节数估宽（中文 3 字节/字，ASCII 1）
        width_px = int(len(text.encode('utf-8')) * baseline * 0.55) + 8
        cv2.rectangle(frame, (x1, max(0, y1 - baseline)), (x1 + width_px, y1), color, -1)
        cv2.putText(frame, text, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX,
                    baseline / 34.0, (16, 15, 10), 2)
    if warning:
        bar = TIER_COLOR.get(warning['level'], (168, 160, 58))
        cv2.rectangle(frame, (0, height - 44), (width, height), bar, -1)
        cv2.putText(frame, warning['text'], (14, height - 14), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 255, 255), 2)
    return frame


class Detector:
    def __init__(self, weights):
        os_env_fix()
        from ultralytics import YOLO
        if not Path(weights).is_file():
            raise SystemExit(f'模型权重不存在：{weights}')
        self.model = YOLO(weights)

    def infer(self, frame, conf, track):
        if track:
            return self.model.track(frame, conf=conf, imgsz=640, persist=True,
                                    tracker='bytetrack.yaml', verbose=False)[0]
        return self.model.predict(frame, conf=conf, imgsz=640, verbose=False)[0]


def os_env_fix():
    import os
    os.environ.setdefault('YOLO_CONFIG_DIR', str(ROOT / 'Ultralytics'))


def run_source(detector, speaker, source, conf, show, need_frames, save_dir=None):
    cap = cv2.VideoCapture(source if isinstance(source, int) else short_path(source))
    if isinstance(source, int):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        raise SystemExit(f'无法打开视频源：{source}（摄像头检查占用与隐私权限）')
    engine = WarningEngine(need_frames=need_frames)
    paused = False
    frame = None
    while True:
        if not paused:
            ok, frame = cap.read()
            if not ok:
                print('视频源结束或中断。')
                break
            result = detector.infer(frame, conf, track=True)
            dets = []
            for i, box in enumerate(result.boxes):
                ident = int(box.id.item()) if getattr(box, 'id', None) is not None else None
                name = detector.model.names[int(box.cls.item())]
                dets.append({'name': name,
                             'xyxy': [round(float(v), 1) for v in box.xyxy[0].tolist()],
                             'confidence': float(box.conf.item()), 'id': ident})
            objects, warning = engine.observe(dets, frame.shape[1], frame.shape[0])
            frame = draw_overlay(frame, objects, warning)
            if warning:
                speaker.say(warning['text'], warning['interrupt'])
                if save_dir:
                    path = Path(save_dir) / f'warn_{int(time.time() * 1000)}.jpg'
                    path.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imencode('.jpg', frame)[1].tofile(str(path))
            if show:
                cv2.imshow('guide_demo', frame)
        if show:
            key = cv2.waitKey(0 if paused else 1) & 0xFF
            if key in (ord('q'), 27):
                break
            if key == ord(' '):
                paused = not paused
        elif isinstance(source, int):
            time.sleep(0.03)  # 无窗口模式摄像头限速，避免空转
    cap.release()
    if show:
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--source', required=True,
                        help='摄像头编号 / 图片 / 视频文件 / 图片目录')
    parser.add_argument('--weights', default=str(DEFAULT_MODEL))
    parser.add_argument('--conf', type=float, default=0.35)
    parser.add_argument('--no-sound', action='store_true')
    parser.add_argument('--no-show', action='store_true', help='不弹窗（批量调试）')
    parser.add_argument('--save-warn', default=None, help='触发预警时保存截图的目录')
    args = parser.parse_args()

    detector = Detector(args.weights)
    speaker = Speaker(enabled=not args.no_sound)
    source = args.source
    if source.isdigit():
        run_source(detector, speaker, int(source), args.conf,
                   show=not args.no_show, need_frames=2, save_dir=args.save_warn)
    else:
        path = Path(source)
        if path.suffix.lower() in {'.mp4', '.mov', '.avi', '.mkv', '.webm', '.m4v'}:
            run_source(detector, speaker, str(path), args.conf,
                       show=not args.no_show, need_frames=2, save_dir=args.save_warn)
            return
        if path.is_dir():
            images = sorted(p for p in path.iterdir()
                            if p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.webp'})
            if not images:
                raise SystemExit(f'目录里没有图片：{path}')
            for image in images:
                frame = imread_u(image)
                if frame is None:
                    print(f'无法读取，跳过：{image.name}')
                    continue
                result = detector.infer(frame, args.conf, track=False)
                dets = [{'name': detector.model.names[int(box.cls.item())],
                         'xyxy': [round(float(v), 1) for v in box.xyxy[0].tolist()],
                         'confidence': float(box.conf.item())} for box in result.boxes]
                objects, warning = WarningEngine(need_frames=1).observe(
                    dets, frame.shape[1], frame.shape[0])
                print(f'{image.name}: {len(objects)} 目标'
                      + (f'，预警 {warning["text"]}' if warning else '，无预警'))
                if not args.no_show:
                    cv2.imshow('guide_demo', draw_overlay(frame, objects, warning))
                    if cv2.waitKey(300) & 0xFF in (ord('q'), 27):
                        break
            cv2.destroyAllWindows()
        else:
            frame = imread_u(path)
            if frame is None:
                raise SystemExit(f'图片无法读取：{path}')
            result = detector.infer(frame, args.conf, track=False)
            dets = [{'name': detector.model.names[int(box.cls.item())],
                     'xyxy': [round(float(v), 1) for v in box.xyxy[0].tolist()],
                     'confidence': float(box.conf.item())} for box in result.boxes]
            objects, warning = WarningEngine(need_frames=1).observe(
                dets, frame.shape[1], frame.shape[0])
            out = draw_overlay(frame, objects, warning)
            out_path = path.with_name(path.stem + '_预警.jpg')
            cv2.imencode('.jpg', out)[1].tofile(short_path(out_path))
            print(f'{len(objects)} 目标，预警：' + (warning['text'] if warning else '无'))
            print(f'结果图：{out_path}')


if __name__ == '__main__':
    main()
