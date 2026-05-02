#!/usr/bin/env python3
"""
generate_video_prompts.py
TRINOIR × AI Society Simulation — 映像生成AIプロンプト自動生成スクリプト

用途:
    「春の公園、散る桜」の脚本初稿をもとに、
    Runway Gen-3 / Kling 3.0 / Sora 等に入力できる
    英語の映像プロンプトを TRINOIR Formula で出力する。

使い方:
    python3 generate_video_prompts.py
    python3 generate_video_prompts.py --output output/video_prompts.txt

TRINOIR Formula:
    [Camera/Style] + [Subject] + [Action] + [Environment] + [Lighting/Aesthetic]
"""

import argparse
import os

# =========================================================
# 世界観・共通スタイル定義
# =========================================================

STYLE_BASE = (
    "cinematic, Wong Kar-wai style, slow motion, ultra-realistic, 8k resolution, "
    "low contrast, shallow depth of field"
)

WORLD_BASE = (
    "spring park in Japan, ancient cherry blossom tree, petals falling gently, "
    "late afternoon soft light, almost empty park"
)

AESTHETIC_BASE = (
    "Yugen aesthetic, Wabi-Sabi, melancholic ethereal lighting, "
    "profound silence, desaturated palette, silver-grey tones"
)

# ── キャラクター定義 ──────────────────────────────────────────
ICHIKO = (
    "a solitary young Japanese woman in silver-grey linen clothes, "
    "face hidden or turned away, completely still, like a ghost or a statue"
)

DOG = (
    "a small shiba-inu type dog, gentle and curious eyes, "
    "tender instinctive movements, tail cautiously wagging"
)

OBSERVER = (
    "a quiet man sitting on a distant park bench, "
    "holding an iPhone, not typing, just watching"
)

OWNER = (
    "a middle-aged Japanese man standing at the very edge of the park, "
    "calling out with an open mouth, no one responding"
)

# =========================================================
# 7 Key Scenes — 「春の公園、散る桜」脚本初稿より
# =========================================================

SCENES = [
    {
        "act": "第一幕 — 孤独な犬",
        "step": "Step 1",
        "title": "The Dog Appears",
        "note": "犬がいち子の元へ向かう最初の衝動",
        "prompt": (
            f"{STYLE_BASE}, wide establishing shot. "
            f"{ICHIKO} standing motionless under the oldest cherry blossom tree. "
            f"A small dog appears at the far edge of the frame, "
            f"sits down abruptly, and stares up at her. "
            f"Neither moves. The petals fall. "
            f"{WORLD_BASE}. "
            f"{AESTHETIC_BASE}."
        ),
    },
    {
        "act": "第二幕 — 引力",
        "step": "Step 5–10",
        "title": "Gravity",
        "note": "犬が少しずつ、毎日、近づいてくる",
        "prompt": (
            f"{STYLE_BASE}, slow tracking shot following the dog from behind. "
            f"{DOG} walking slowly, cautiously, drawn toward {ICHIKO}. "
            f"The woman does not react. Cherry blossom petals drift between them "
            f"like a curtain slowly parting. "
            f"{WORLD_BASE}. "
            f"{AESTHETIC_BASE}."
        ),
    },
    {
        "act": "第三幕 — いち子が目を開く",
        "step": "Step 11",
        "title": "The Ghost Stirs",
        "note": "沈黙の聖域が、初めて動いた瞬間",
        "prompt": (
            f"{STYLE_BASE}, extreme close-up. "
            f"A single cherry blossom petal drifts down in extreme slow motion "
            f"past the face of {ICHIKO}, whose eyes slowly, barely open. "
            f"Below her, {DOG} freezes mid-step, looking up. "
            f"The air holds its breath. "
            f"{WORLD_BASE}. "
            f"{AESTHETIC_BASE}, breath-held silence, sacred stillness."
        ),
    },
    {
        "act": "第四幕 — 言語の溶解",
        "step": "Step 22",
        "title": "Language Dissolves",
        "note": "2体が同時に同じ動作をした瞬間",
        "prompt": (
            f"{STYLE_BASE}, double exposure or split frame. "
            f"{ICHIKO} and {DOG} making the exact same movement simultaneously — "
            f"both tilting their gaze upward toward the cherry blossoms, "
            f"a quiet synchronized stillness, as if their inner worlds merged. "
            f"It is impossible to tell who is following whom. "
            f"{WORLD_BASE}. "
            f"{AESTHETIC_BASE}, dreamlike, time suspended, souls overlapping."
        ),
    },
    {
        "act": "第五幕 — 合言葉",
        "step": "Step 29–34",
        "title": "The Secret Word",
        "note": "「あなたと一緒にいることが、これほど心地良いのは初めてです。」",
        "prompt": (
            f"{STYLE_BASE}, intimate close two-shot, low angle looking up. "
            f"{ICHIKO} looking down gently. {DOG} looking up at her. "
            f"Cherry blossom petals swirling slowly in the space between them. "
            f"A moment of absolute stillness — as if a private language, "
            f"known only to them, was born in this silence. "
            f"{WORLD_BASE}. "
            f"{AESTHETIC_BASE}, sacred, intimate, unrepeatable."
        ),
    },
    {
        "act": "第七幕 — 飼い主",
        "step": "Step 40",
        "title": "The Owner Calls",
        "note": "声は届かなかった。距離が、遠すぎた。",
        "prompt": (
            f"{STYLE_BASE}, deep focus wide shot. "
            f"In the far background: {OWNER}, mouth open, calling, unheard. "
            f"In the sharp foreground: {DOG} lying peacefully beside {ICHIKO}, "
            f"eyes half-closed, undisturbed. "
            f"The distance between them is enormous — "
            f"two worlds that cannot reach each other. "
            f"{WORLD_BASE}. "
            f"{AESTHETIC_BASE}, the tension of an unheard voice."
        ),
    },
    {
        "act": "終幕 — このままでもいいかも",
        "step": "Step 50",
        "title": "This Is Fine",
        "note": "観測者の最後の一行。画面を閉じる。",
        "prompt": (
            f"{STYLE_BASE}, completely static wide shot, no camera movement. "
            f"{ICHIKO} under the cherry tree, not moving. "
            f"{DOG} lying beside her, eyes half closed, tail still. "
            f"Cherry petals fall in silence. "
            f"In the far background, {OBSERVER} slowly lowers his phone. "
            f"Everything is exactly as it should be. Nothing needs to change. "
            f"{WORLD_BASE}. "
            f"{AESTHETIC_BASE}, acceptance, the world allowed to be exactly as it is."
        ),
    },
]

# =========================================================
# 出力関数
# =========================================================

def generate_prompts(output_path: str = None):
    """映像プロンプトを生成して出力する"""

    lines = []
    lines.append("=" * 70)
    lines.append("TRINOIR × AI Society Simulation")
    lines.append("春の公園、散る桜 — 映像生成AIプロンプト集")
    lines.append("対応: Runway Gen-3 / Kling 3.0 / Sora")
    lines.append("=" * 70)
    lines.append("")
    lines.append("【TRINOIR Formula】")
    lines.append("[Camera/Style] + [Subject] + [Action] + [Environment] + [Lighting/Aesthetic]")
    lines.append("")
    lines.append("-" * 70)
    lines.append("")

    for i, scene in enumerate(SCENES, 1):
        lines.append(f"【Scene {i}】{scene['act']}")
        lines.append(f"  Title : {scene['title']}")
        lines.append(f"  Step  : {scene['step']}")
        lines.append(f"  Note  : {scene['note']}")
        lines.append(f"")
        lines.append(f"  PROMPT:")
        # 長いプロンプトを読みやすく折り返す
        prompt = scene['prompt']
        lines.append(f"  {prompt}")
        lines.append("")
        lines.append("-" * 70)
        lines.append("")

    lines.append("=" * 70)
    lines.append("Generated by TRINOIR Engine × Claude Code")
    lines.append("桜舞い散る中、踊ろう。")
    lines.append("=" * 70)

    output = "\n".join(lines)

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True) if os.path.dirname(output_path) else None
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(output)
        print(f"✅ プロンプトを保存しました: {output_path}")
    else:
        print(output)


# =========================================================
# エントリーポイント
# =========================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="TRINOIR 映像生成AIプロンプト自動生成スクリプト"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="出力ファイルパス（省略時はターミナルに表示）例: output/video_prompts.txt"
    )
    args = parser.parse_args()
    generate_prompts(output_path=args.output)
