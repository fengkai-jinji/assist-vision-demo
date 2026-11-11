# GitHub 开源调研 —— 距离预警 / 跟踪 / 语音 / 同类导盲项目

调研日期：2026-11-12。目的：为 YOLOv8 工程补齐"距离三级预警 + 语音播报"能力选型。

## 1. 单目距离估计（后续进阶方向）

| 项目 | 结论 |
| --- | --- |
| [DepthAnything/Depth-Anything-V2](https://github.com/DepthAnything/Depth-Anything-V2)（NeurIPS 2024, Apache-2.0） | 单目深度基座模型。注意：**基础版只输出相对深度**；要输出真实米数需用其室内（NYUv2）/室外（KITTI）微调的 metric 版本。ViT-S 小模型可实时。 |
| [Video-Depth-Anything](https://github.com/DepthAnything/Video-Depth-Anything)（CVPR 2025） | V2 的视频版，长视频时深度更稳定，适合替代逐帧估计的闪烁问题。 |

**决策**：本轮预警用几何分级（框底边 y2 + 框高比，参照便携推理包方案），不上深度模型。理由：零下载、零额外延迟、4060 上推理预算留给检测；等预警管线跑通后再把 Depth Anything V2 ViT-S 作为可选开关接入，届时用已知距离标定物（门高 2m）校准尺度。

## 2. 目标跟踪（防抖升级，本轮采用）

| 结论 | 依据 |
| --- | --- |
| ultralytics 8.4.33 **自带** `model.track()`，默认 BoT-SORT，可选 `tracker='bytetrack.yaml'`；`persist=True` 跨帧保状态，结果里直接有 `track_id` | [官方 tracking 文档](https://docs.ultralytics.com/modes/track) |

**决策**：预警确认从"连续 2 帧有检测"升级为"**同一个 track_id 连续 N 帧处于该等级**"，消除两个不同物体各闪一帧凑数造成的误报。ByteTrack 更快，BoT-SORT 更稳，默认用 bytetrack（640 实时帧率优先），配置留成参数。无需装新包。

## 3. 中文语音（TTS）

| 方案 | 结论 |
| --- | --- |
| **浏览器 speechSynthesis（本轮 web_demo 采用）** | Windows 自带中文语音（Huihui 等），零安装、零网络，danger 可 `cancel()` 打断。缺点：换浏览器/系统语音包可能缺失，需做回退提示。 |
| [k2-fsa/sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) | 完全离线 TTS 引擎，中文模型如 `vits-piper-zh_CN-huayan-medium`（普通话女声）、kokoro 系列。有预编译二进制。**后续给便携推理包升级音质时用**，替代 pyttsx3 的 SAPI 依赖。 |
| pyttsx3 | 便携推理包当前方案，依赖 Windows SAPI 语音包，换机器可能没中文声音。不新增依赖。 |

## 4. 同类导盲开源项目（对标与差异化）

| 项目 | 可借鉴 |
| --- | --- |
| [drisskhattabi6/Obstacle-Detection-with-Alert-System-for-Visually-Impaired](https://github.com/drisskhattabi6/Obstacle-Detection-with-Alert-System-for-Visually-Impaired) | 检测 + 声音报警的完整闭环，结构简单清晰，可对照其报警触发条件。 |
| [beingaryan/Blind-Assistance-Object-Detection-and-Navigation](https://github.com/beingaryan/Blind-Assistance-Object-Detection-and-Navigation) | 近似距离估计 + 播报的组合。 |
| GitHub 话题页 [blind-navigation](https://github.com/topics/blind-navigation) / [visually-impaired-people](https://github.com/topics/visually-impaired-people) | 大量 YOLOv8 + 超声波/移动端组合项目，找数据集与外设思路时再翻。 |
| 论文 YOLO-OD（MDPI Sensors 2024）、[Outdoor Obstacle Detection (arXiv 2312.07571)](https://arxiv.org/html/2312.07571v1) | 小障碍物检出改进、室外场景基线，第二轮训练扩类别时可参考其类别设计。 |

**差异化结论**：现有开源多数是"检测+蜂鸣"，本项目已有两个独特点——立体声方位引导（desktop_demo）和按类别语义区分预警策略（门=目标提示而非障碍）。本轮把它们和距离分级合成完整管线。

## 落地顺序（本轮）

1. `guide_demo/warning.py` 纯逻辑引擎（几何分级 + 方位 + track_id 连续帧确认 + 冷却/打断 + 类别语义策略）+ 单元测试。
2. web_demo 后端：live 模式切 `model.track(persist=True)`，API 增加 `tier` / `direction` / `track_id` / `warning` 字段，旧字段不动。
3. 前端：预警横幅 + 等级徽标 + 浏览器语音（开关、danger 打断），按 uiendless 流程重做。
