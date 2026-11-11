# -*- coding: utf-8 -*-
"""导盲预警管线：几何距离分级 + 方位 + 同目标连续帧确认 + 播报冷却与打断。

纯逻辑模块，不依赖 torch / ultralytics / numpy，方便单元测试，
也可直接复用到 web_demo、CLI 或便携推理包。

距离分级参照便携推理包 demo/blind_demo.py 的几何方案（非模型输出）：
检测框底边在画面中的纵向位置（归一化 y2）与框高占画面比例（h）越大，目标越近。
与参考实现的主要差异：
1. 播报确认是"同一个目标（track_id 或类别键）连续 N 帧处于可播报等级"：
   同帧同键只计一次（跟踪器未分配 id 时同类多目标不会单帧凑满），
   目标掉回中/远等级即清零（near→远→near 不算连续）；
2. 按类别语义区分角色：门是通行目标（提示不报警），瓶/椅/人是障碍物；
3. 冷却按（等级, 类别）键控并逐候选回退——一个持续在场的危险不会无限期
   压住其他类别已确认的近级播报；危险级播报绕过全局最小间隔（可打断语义），
   非危险级受全局最小间隔节流。
"""
import time

# 等级从低到高；tier 为 None 表示"远"，仅显示不播报
LEVELS = ('mid', 'near', 'danger')

# 每级阈值：底边位置 y2 或框高占比 h 任一达标即判入（与参考实现一致的 OR 口径）；
# h 为 None 表示该级不考察框高
TIERS = {
    'danger': {'y2': 0.88, 'h': 0.55},
    'near': {'y2': 0.66, 'h': 0.34},
    'mid': {'y2': 0.48, 'h': None},
}

# 类别角色：obstacle 障碍物（near/danger 播报，mid 仅显示）/ target 通行目标
#（确认后按 mid 级提示）/ quiet 仅显示不参与播报
DEFAULT_ROLES = {
    'bottle': 'obstacle', 'cup': 'obstacle', 'wine glass': 'obstacle',
    'chair': 'obstacle', 'person': 'obstacle',
    'door': 'target',
    'car': 'quiet', 'truck': 'quiet', 'bus': 'quiet',
}
SILENT_ROLES = {'quiet'}

# 类别提级钩子：水坑这类地面障碍进入画面中部（约 3~5 米）就值得按"近"提醒
DEFAULT_BOOST = {'puddle': 'near'}

# 内置中文显示名，可被调用方传入的 labels 覆盖
CN_LABELS = {
    'bottle': '瓶子', 'cup': '杯子', 'wine glass': '高脚杯', 'chair': '椅子',
    'door': '门', 'person': '行人', 'puddle': '水坑', 'car': '汽车',
}

DIRECTION_LEFT, DIRECTION_CENTER, DIRECTION_RIGHT = '左前方', '正前方', '右前方'


def tier_of(y2n, hn, boost=None):
    """按几何阈值返回 'danger'/'near'/'mid' 或 None（远）。

    y2n：框底边纵向位置 / 画面高；hn：框高 / 画面高，均归一化 0~1。
    boost：类别提级目标等级，如水坑 mid 提为 near。
    """
    if hn >= TIERS['danger']['h'] or y2n >= TIERS['danger']['y2']:
        tier = 'danger'
    elif hn >= TIERS['near']['h'] or y2n >= TIERS['near']['y2']:
        tier = 'near'
    elif y2n >= TIERS['mid']['y2']:
        tier = 'mid'
    else:
        return None
    if boost and LEVELS.index(tier) < LEVELS.index(boost):
        tier = boost
    return tier


def direction_of(x1n, x2n):
    """按框中心水平位置返回方位：左前方 / 正前方 / 右前方。"""
    center = (x1n + x2n) / 2
    if center < 1 / 3:
        return DIRECTION_LEFT
    if center > 2 / 3:
        return DIRECTION_RIGHT
    return DIRECTION_CENTER


def _role_of(name, roles):
    return roles.get(name, roles.get(name.lower(), 'obstacle'))


def voice_text(level, direction, label):
    if level == 'danger':
        return f'危险！{direction}有{label}，请停下'
    if level == 'near':
        return f'{direction}近处有{label}，注意避让'
    return f'{direction}有{label}'


class WarningEngine:
    """逐帧观察检测结果，输出带等级/方位标注的目标列表与当帧播报（至多一条）。

    dets 约定：每个元素为 dict，含 'name'（英文类别名）、'xyxy'（像素坐标），
    可选 'label'（中文显示名）、'id'（跟踪编号，图像单帧模式可省略）。
    """

    def __init__(self, need_frames=2, danger_cooldown=2.0, near_cooldown=4.0,
                 target_cooldown=8.0, min_gap=1.0, clock=time.monotonic,
                 roles=None, boost=None, labels=None):
        self.need_frames = max(1, need_frames)
        self.cooldowns = {'danger': danger_cooldown, 'near': near_cooldown, 'mid': target_cooldown}
        self.min_gap = min_gap
        self.clock = clock
        self.roles = dict(DEFAULT_ROLES if roles is None else roles)
        self.boost = dict(DEFAULT_BOOST if boost is None else boost)
        for name, level in self.boost.items():
            if level not in LEVELS:
                raise ValueError(f'提级配置非法：{name} -> {level}（可选 {LEVELS}）')
        self.labels = dict(CN_LABELS, **(labels or {}))
        self.streaks = {}      # 候选键 -> 连续处于可播报等级的帧数
        self.last_spoken = {}  # (等级, 类别) -> 上次播报时刻
        self.last_any = None   # 上次任意播报时刻（全局最小间隔用）

    def reset(self):
        """更换视频源时调用，避免上一源的连续帧计数与冷却串场。"""
        self.streaks.clear()
        self.last_spoken.clear()
        self.last_any = None

    def observe(self, dets, width, height):
        """处理一帧。返回 (标注后的目标列表, 播报或 None)。"""
        now = self.clock()
        decorated = []
        frame_best = {}  # key -> 本帧该键的最优候选（同帧同键聚合，取等级更高/更近者）
        for det in dets:
            x1, y1, x2, y2 = det['xyxy']
            y2n = min(1.0, max(0.0, y2 / max(1, height)))
            hn = min(1.0, max(0.0, (y2 - y1) / max(1, height)))
            name = det.get('name', '')
            role = _role_of(name, self.roles)
            # quiet 角色不参与分级；其余按几何阈值分级
            tier = None if role in SILENT_ROLES else tier_of(y2n, hn, self.boost.get(name))
            item = dict(det)
            item['role'] = role
            item['tier'] = tier
            item['direction'] = direction_of(x1 / max(1, width), x2 / max(1, width))
            item['label'] = det.get('label') or self.labels.get(name) or name
            decorated.append(item)
            if tier not in ('near', 'danger'):
                continue
            key = (item.get('id'), name)
            cand = {'name': name, 'label': item['label'], 'tier': tier,
                    'direction': item['direction'], 'id': item.get('id'),
                    '_y2n': y2n, '_role': role}
            best = frame_best.get(key)
            if best is None or LEVELS.index(tier) > LEVELS.index(best['tier']) or \
                    (tier == best['tier'] and y2n > best['_y2n']):
                frame_best[key] = cand
        # 本帧实际处于可播报等级的键各计一次；掉出等级的键立即清零，
        # 保证"连续 N 帧处于该等级"的字面语义（near→远→near 不算连续）
        for key, cand in frame_best.items():
            self.streaks[key] = self.streaks.get(key, 0) + 1
            cand['_streak'] = self.streaks[key]
        for key in list(self.streaks):
            if key not in frame_best:
                del self.streaks[key]
        return decorated, self._announce(frame_best, now)

    def _announce(self, frame_best, now):
        """按 危险→近→通行提示 排序逐候选尝试，取第一个通过自身冷却
        （与全局最小间隔，危险级豁免）的候选；全部被挡则本帧不播报。"""
        confirmed = []
        for cand in frame_best.values():
            if cand['_streak'] < self.need_frames:
                continue
            if cand['_role'] == 'obstacle':
                confirmed.append((0 if cand['tier'] == 'danger' else 1, cand))
            elif cand['_role'] == 'target':
                confirmed.append((2, cand))
        if not confirmed:
            return None
        confirmed.sort(key=lambda pair: (pair[0], -pair[1]['_y2n']))
        for _, cand in confirmed:
            level = 'mid' if cand['_role'] == 'target' else cand['tier']
            last = self.last_spoken.get((level, cand['name']))
            if last is not None and now - last < self.cooldowns[level]:
                continue  # 该候选自身冷却中，轮到次优候选
            if level != 'danger' and self.last_any is not None and \
                    now - self.last_any < self.min_gap:
                continue  # 全局最小间隔只节流非危险级（危险可打断语义）
            self.last_spoken[(level, cand['name'])] = now
            self.last_any = now
            return {'level': level, 'name': cand['name'], 'label': cand['label'],
                    'id': cand['id'], 'direction': cand['direction'],
                    'text': voice_text(level, cand['direction'], cand['label']),
                    'interrupt': level == 'danger'}
        return None
