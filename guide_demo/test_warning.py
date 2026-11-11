# -*- coding: utf-8 -*-
"""guide_demo/warning.py 的单元测试。运行：python -m unittest guide_demo.test_warning"""
import unittest

from guide_demo import warning as W


def det(name, xyxy, ident=None, label=None):
    d = {'name': name, 'xyxy': xyxy}
    if ident is not None:
        d['id'] = ident
    if label:
        d['label'] = label
    return d


class FakeClock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


class TierTests(unittest.TestCase):
    def test_y2_levels(self):
        self.assertEqual(W.tier_of(0.95, 0.1), 'danger')
        self.assertEqual(W.tier_of(0.70, 0.1), 'near')
        self.assertEqual(W.tier_of(0.50, 0.1), 'mid')
        self.assertIsNone(W.tier_of(0.30, 0.1))

    def test_height_alone_triggers(self):
        self.assertEqual(W.tier_of(0.50, 0.60), 'danger')   # 框高过半即危险
        self.assertEqual(W.tier_of(0.30, 0.40), 'near')     # 框高 0.34 以上即近
        self.assertIsNone(W.tier_of(0.30, 0.20))            # 又高又远才不算

    def test_boundary_inclusive(self):
        self.assertEqual(W.tier_of(0.88, 0.0), 'danger')
        self.assertEqual(W.tier_of(0.66, 0.0), 'near')
        self.assertEqual(W.tier_of(0.48, 0.0), 'mid')

    def test_boost_raises_mid_to_near(self):
        self.assertEqual(W.tier_of(0.50, 0.10, boost='near'), 'near')
        self.assertEqual(W.tier_of(0.50, 0.10), 'mid')      # 无提级不受影响
        self.assertEqual(W.tier_of(0.95, 0.10, boost='near'), 'danger')  # 高级不被降

    def test_direction(self):
        self.assertEqual(W.direction_of(0.0, 0.2), W.DIRECTION_LEFT)
        self.assertEqual(W.direction_of(0.45, 0.55), W.DIRECTION_CENTER)
        self.assertEqual(W.direction_of(0.9, 1.0), W.DIRECTION_RIGHT)


class EngineTests(unittest.TestCase):
    def make(self, **kwargs):
        clock = FakeClock()
        engine = W.WarningEngine(clock=clock, **kwargs)
        return clock, engine

    def test_single_frame_not_confirmed(self):
        _, engine = self.make(need_frames=2)
        dets, spoken = engine.observe([det('chair', [0, 380, 200, 420], ident=1)], 640, 480)
        self.assertIsNone(spoken)
        self.assertEqual(dets[0]['tier'], 'near')           # 等级立即显示
        self.assertEqual(dets[0]['direction'], W.DIRECTION_LEFT)

    def test_same_id_two_frames_confirmed(self):
        _, engine = self.make(need_frames=2)
        engine.observe([det('chair', [0, 380, 200, 420], ident=1)], 640, 480)
        dets, spoken = engine.observe([det('chair', [0, 380, 200, 420], ident=1)], 640, 480)
        self.assertIsNotNone(spoken)
        self.assertEqual(spoken['level'], 'near')
        self.assertIn('左前方', spoken['text'])
        self.assertFalse(spoken['interrupt'])

    def test_alternating_ids_never_confirm(self):
        _, engine = self.make(need_frames=2)
        for i in range(6):
            _, spoken = engine.observe(
                [det('chair', [0, 380, 200, 420], ident=i % 2)], 640, 480)
            self.assertIsNone(spoken, f'第 {i} 帧不应播报')

    def test_streak_resets_after_absence(self):
        _, engine = self.make(need_frames=2)
        engine.observe([det('chair', [0, 380, 200, 420], ident=1)], 640, 480)
        engine.observe([], 640, 480)                        # 消失一帧，计数清零
        _, spoken = engine.observe([det('chair', [0, 380, 200, 420], ident=1)], 640, 480)
        self.assertIsNone(spoken)

    def test_danger_interrupts(self):
        _, engine = self.make(need_frames=1)
        _, spoken = engine.observe([det('person', [0, 0, 640, 480], ident=1)], 640, 480)
        self.assertEqual(spoken['level'], 'danger')
        self.assertTrue(spoken['interrupt'])
        self.assertIn('危险', spoken['text'])

    def test_danger_cooldown_and_recovery(self):
        clock, engine = self.make(need_frames=1)
        _, first = engine.observe([det('person', [0, 0, 640, 480], ident=1)], 640, 480)
        _, second = engine.observe([det('person', [0, 0, 640, 480], ident=1)], 640, 480)
        self.assertIsNotNone(first)
        self.assertIsNone(second)                           # 冷却期内不重复播报
        clock.advance(2.5)                                  # 超过 danger 冷却 2 秒
        _, third = engine.observe([det('person', [0, 0, 640, 480], ident=1)], 640, 480)
        self.assertIsNotNone(third)

    def test_min_gap_blocks_other_class(self):
        clock, engine = self.make(need_frames=1, min_gap=1.0)
        _, first = engine.observe([det('person', [0, 0, 640, 480], ident=1)], 640, 480)
        _, second = engine.observe([det('chair', [0, 0, 640, 480], ident=2)], 640, 480)
        self.assertIsNotNone(first)
        self.assertIsNone(second)                           # 全局最小间隔挡住另一个类
        clock.advance(1.5)
        _, third = engine.observe([det('chair', [0, 0, 640, 480], ident=2)], 640, 480)
        self.assertIsNotNone(third)

    def test_door_is_target_not_danger(self):
        _, engine = self.make(need_frames=1)
        dets, spoken = engine.observe([det('door', [0, 0, 640, 480], ident=1)], 640, 480)
        self.assertEqual(dets[0]['role'], 'target')
        self.assertIsNotNone(spoken)
        self.assertEqual(spoken['level'], 'mid')            # 门只提示，不进危险级
        self.assertFalse(spoken['interrupt'])
        self.assertIn('门', spoken['text'])

    def test_obstacle_beats_target_same_frame(self):
        _, engine = self.make(need_frames=1)
        _, spoken = engine.observe(
            [det('door', [0, 0, 640, 480], ident=1),
             det('chair', [0, 0, 640, 480], ident=2)], 640, 480)
        self.assertEqual(spoken['name'], 'chair')           # 有障碍时门让位

    def test_quiet_class_display_only(self):
        _, engine = self.make(need_frames=1)
        dets, spoken = engine.observe([det('car', [0, 0, 640, 480], ident=1)], 640, 480)
        self.assertEqual(dets[0]['role'], 'quiet')
        self.assertIsNone(dets[0]['tier'])
        self.assertIsNone(spoken)

    def test_no_track_id_counts_by_class(self):
        _, engine = self.make(need_frames=2)                # 图像单帧模式 id 缺省
        engine.observe([det('chair', [0, 380, 200, 420])], 640, 480)
        _, spoken = engine.observe([det('chair', [0, 380, 200, 420])], 640, 480)
        self.assertIsNotNone(spoken)

    def test_reset_clears_streaks(self):
        _, engine = self.make(need_frames=2)
        engine.observe([det('chair', [0, 380, 200, 420], ident=1)], 640, 480)
        engine.reset()
        _, spoken = engine.observe([det('chair', [0, 380, 200, 420], ident=1)], 640, 480)
        self.assertIsNone(spoken)

    def test_far_object_no_tier(self):
        _, engine = self.make(need_frames=1)
        dets, spoken = engine.observe([det('chair', [300, 0, 340, 100], ident=1)], 640, 480)
        self.assertIsNone(dets[0]['tier'])
        self.assertIsNone(spoken)

    def test_label_falls_back_to_name(self):
        _, engine = self.make(need_frames=1, labels={'chair': '椅子'})
        _, spoken = engine.observe([det('chair', [0, 0, 640, 480], ident=1)], 640, 480)
        self.assertIn('椅子', spoken['text'])

    def test_puddle_boost_via_engine(self):
        _, engine = self.make(need_frames=1)
        dets, _ = engine.observe([det('puddle', [100, 200, 500, 260], ident=1)], 640, 480)
        self.assertEqual(dets[0]['tier'], 'near')           # mid 被提为 near


if __name__ == '__main__':
    unittest.main()
