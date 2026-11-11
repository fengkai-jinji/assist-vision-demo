# -*- coding: utf-8 -*-
"""第二轮训练数据准备：按场景（视频级）划分 train/val，绝不按帧随机划分。

与 prepare_team_pilot.py / build_team_pilot.py 的关系：第一轮试训已经按视频级
划分（build_manifest.json 的 train_video_ids/val_video_ids）；本脚本把该约束
固化成可复用的准备流程，供组员完成标注复核后直接跑：

1. 从一个或多个"帧目录"（图片 + 同名 .txt 标签，文件名形如 ABC_001_a.jpg）
   收集样本，按场景键（成员_视频号）分组——同一视频的所有帧只进同一个集合；
2. 按比例划分场景键到 train/val，保证两个集合之间：
   - 场景键零交集（同视频不跨集）
   - 文件 SHA-256 零交集（排除重复图）
3. 可选 --with-person：只有当确实存在含 person 标注的样本时才启用 4 类，
   否则保持 bottle/chair/door 三类并给出提示，避免出现空类；
4. 输出到 datasets/team_video_round2_<日期>，目录已存在即拒绝，不覆盖旧成果。

用法（YOLOv8 目录下）：
  python scripts/prepare_round2.py --sources datasets/team_video_pilot_20261025/train \
      datasets/team_video_pilot_20261025/val --val-scenes ABC_007 ABC_024
  python scripts/prepare_round2.py --sources 新帧目录 --with-person
"""
import argparse
import datetime
import hashlib
import json
import random
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# 场景键默认匹配 成员缩写_视频号 前缀（如 ABC_001_a.jpg -> ABC_001）；
# 可用 --scene-regex 覆盖以适配其他命名规范。旧数据集回放图（legacy_*）整体归为 legacy 组。
SCENE_RE = re.compile(r'^([A-Za-z]+_\d+)')
IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.webp'}
# 0 bottle / 1 chair / 2 door 沿用第一轮口径；person 只有真的有标注才加
BASE_NAMES = {0: 'bottle', 1: 'chair', 2: 'door'}
PERSON_NAMES = {3: 'person'}


def scene_key(name, image=None, pattern=None):
    """ABC_001_a.jpg -> ABC_001；旧数据集回放图（legacy_*）整体归为 legacy 组。

    legacy 组按第一轮口径只进训练侧（回放本就来自旧 train，且已排除与
    旧 valid/test 的哈希交集）。取不出键又不是 legacy 的文件会被拒绝并点名。
    """
    if image is not None and image.name.lower().startswith('legacy'):
        return 'legacy'
    match = (pattern or SCENE_RE).match(name)
    if not match:
        raise ValueError(f'文件名不符合 场景键_帧 规范（可用 --scene-regex 自定义）：{name}')
    return match.group(1)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def find_label(image):
    """兼容两种布局：标签与图片同目录（成品风格）或 images/labels 兄弟目录。"""
    candidates = [image.with_suffix('.txt'),
                  image.parent.parent / 'labels' / image.with_suffix('.txt').name]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise ValueError(f'找不到标签文件（同目录与 labels/ 都没有）：{image}')


def collect(sources, pattern=None):
    """返回 [{image,label,scene,hash,classes}]；缺标签、坏标签直接报错，不静默跳过。"""
    samples = []
    for source in sources:
        source = Path(source)
        if not source.is_absolute():
            source = ROOT / source
        images = sorted(p for p in source.rglob('*')
                        if p.suffix.lower() in IMAGE_EXTS and p.parent.name != 'labels')
        if not images:
            raise ValueError(f'目录里没有图片：{source}')
        for image in images:
            label = find_label(image)
            classes = set()
            for line in label.read_text(encoding='utf-8').splitlines():
                parts = line.split()
                if parts:
                    classes.add(int(float(parts[0])))
            if not classes:
                raise ValueError(f'空标签（既无框也未标背景）：{label}')
            samples.append({'image': str(image), 'label': str(label),
                            'scene': scene_key(image.stem, image, pattern),
                            'hash': sha256(image), 'classes': sorted(classes)})
    return samples


def split_scenes(samples, val_ratio, val_forced):
    """按场景键划分；val_forced 里的场景进 val，legacy（旧回放）固定在训练侧。"""
    scenes = {}
    for sample in samples:
        scenes.setdefault(sample['scene'], []).append(sample)
    keys = sorted(scenes)
    random.shuffle(keys)
    val = {key for key in val_forced if key in scenes}
    for key in keys:  # 不足额时从剩余成员场景补（legacy 除外）
        if len(val) >= max(1, round(len(keys) * val_ratio)):
            break
        if key not in val and key != 'legacy':
            val.add(key)
    val.discard('legacy')
    train = {key for key in keys if key not in val}
    return {key: scenes[key] for key in train}, {key: scenes[key] for key in val}


def check_disjoint(train, val):
    train_hashes = {s['hash'] for s in train}
    val_hashes = {s['hash'] for s in val}
    overlap = train_hashes & val_hashes
    if overlap:
        raise ValueError(f'训练/验证存在 {len(overlap)} 张完全相同的图片，拒绝输出')
    train_scenes = {s['scene'] for s in train}
    val_scenes = {s['scene'] for s in val}
    both = train_scenes & val_scenes
    if both:
        raise ValueError(f'场景同时出现在训练与验证：{sorted(both)}')


def class_counts(samples):
    counts = {}
    for sample in samples:
        for cls in sample['classes']:
            counts[cls] = counts.get(cls, 0) + 1
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--sources', nargs='+', required=True,
                        help='帧目录（图片+同名txt），可多个')
    parser.add_argument('--val-ratio', type=float, default=0.25)
    parser.add_argument('--val-scenes', nargs='*', default=[],
                        help='强制进验证集的场景键，如 ABC_007 ABC_024')
    parser.add_argument('--with-person', action='store_true',
                        help='检测到 person 标注时启用 4 类，否则拒绝并提示')
    parser.add_argument('--seed', type=int, default=20261112)
    parser.add_argument('--scene-regex', default=None,
                        help='场景键正则（默认 [A-Za-z]+_数字 前缀），须含一个分组')
    parser.add_argument('--name', default=None,
                        help='输出目录名，默认 team_video_round2_<日期>')
    args = parser.parse_args()
    pattern = re.compile(args.scene_regex) if args.scene_regex else SCENE_RE

    out_name = args.name or f'team_video_round2_{datetime.date.today():%Y%m%d}'
    out = ROOT / 'datasets' / out_name
    if out.exists():
        raise SystemExit(f'输出目录已存在，不覆盖旧成果：{out}。'
                         '下一轮请换名字（脚本沿用第一轮"不覆盖已训练轮次"的约定）。')

    samples = collect(args.sources, pattern)
    print(f'共收集 {len(samples)} 帧、'
          f'{len({s["scene"] for s in samples})} 个场景')

    counts_all = class_counts(samples)
    with_person = args.with_person
    if with_person and counts_all.get(3, 0) == 0:
        raise SystemExit('--with-person 已指定，但没有任何样本含 person(3) 标注。'
                         '请先补标 person，或去掉该开关保持 3 类。')
    if not with_person and counts_all.get(3, 0):
        print(f'提示：发现 {counts_all[3]} 帧含 person 标注，'
              '但未加 --with-person，本轮仍按 3 类输出。')

    names = dict(BASE_NAMES, **(PERSON_NAMES if with_person else {}))
    for sample in samples:  # 类别越界的标签直接拦下
        for cls in sample['classes']:
            if cls not in names:
                raise ValueError(f'样本含未定义类别 {cls}（当前类别表 {names}）：'
                                 f'{sample["image"]}')

    random.seed(args.seed)
    train_map, val_map = split_scenes(samples, args.val_ratio, args.val_scenes)
    train = [s for group in train_map.values() for s in group]
    val = [s for group in val_map.values() for s in group]
    check_disjoint(train, val)

    for split, group in (('train', train), ('val', val)):
        (out / split / 'images').mkdir(parents=True)
        (out / split / 'labels').mkdir(parents=True)
        for sample in group:
            shutil.copy2(sample['image'], out / split / 'images' / Path(sample['image']).name)
            shutil.copy2(sample['label'], out / split / 'labels' / Path(sample['label']).name)

    lines = [f'path: {out.as_posix()}', 'train: train/images', 'val: val/images',
             'names:']
    lines += [f'  {i}: {n}' for i, n in names.items()]
    (out / 'data.yaml').write_text('\n'.join(lines) + '\n', encoding='utf-8')

    manifest = {
        'created': f'{datetime.datetime.now():%Y-%m-%d %H:%M:%S}',
        'sources': [str(Path(s).resolve()) for s in args.sources],
        'split_unit': 'scene (video-level)，禁止按帧随机划分',
        'train_scenes': sorted(train_map), 'val_scenes': sorted(val_map),
        'train_images': len(train), 'val_images': len(val),
        'train_class_image_counts': class_counts(train),
        'val_class_image_counts': class_counts(val),
        'train_val_exact_hash_overlap': 0,
        'names': names,
        'annotation_status': 'inherited from sources; must be human-certified before training',
        'seed': args.seed,
    }
    (out / 'build_manifest.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'场景划分 train {len(train_map)} / val {len(val_map)}，'
          f'图片 {len(train)} / {len(val)}')
    print(f'训练集类别分布 {class_counts(train)}')
    print(f'验证集类别分布 {class_counts(val)}')
    print(f'已输出：{out}')
    print('下一步：组员人工复核签字后，用新版本名训练（不要覆盖旧 runs）。')


if __name__ == '__main__':
    try:
        main()
    except (ValueError, SystemExit) as error:
        print(f'错误：{error}', file=sys.stderr)
        sys.exit(1)
