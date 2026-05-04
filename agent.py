"""
LLM-based agent in 2D worlds with multiple places.
"""
import json
import math
import logging
from typing import List, Tuple, Optional, Dict, TypedDict
from ollama_client import OllamaClient
from utils import is_position_in_place, get_place_at_position, PlaceConfig

logger = logging.getLogger(__name__)

# Constants
FALLBACK_REASONING_LENGTH = 100
MAX_MESSAGE_WORDS = 200

# Direction mappings (4 cardinal directions only)
# Coordinate system: X increases from left to right, Y increases from bottom to top
DIRECTION_MAP = {
    "up": (0, 1),      # Y+1 (move upward)
    "down": (0, -1),   # Y-1 (move downward)
    "left": (-1, 0),   # X-1 (move leftward)
    "right": (1, 0),   # X+1 (move rightward)
}


class MessageDecision(TypedDict):
    """Type definition for agent message decision"""
    message: str  # Message to communicate with nearby agents
    reasoning: str  # Explanation of the message decision


class ActionDecision(TypedDict):
    """Type definition for agent action decision"""
    action: str  # "move" or "stay"
    direction: Optional[str]  # Direction to move (None if action is "stay")
    memory: str  # What the agent wants to remember for the next step
    reasoning: str  # Explanation of the decision


class Agent:
    """LLM-based agent in 2D worlds with multiple places."""

    def __init__(
        self,
        agent_id: int,
        initial_position: Tuple[int, int],
        llm_client: OllamaClient,
        communication_radius: float,
        half_space_size: int,
        places: List[PlaceConfig],
        num_agents: int,
        gender: str = "male",
        memory_limit: int = 20,
        memory_size: int = 5,
        message_history_limit: int = 10,
        message_context_size: int = 3,
        persona: str = ""
    ):
        self.id = agent_id
        self.position = initial_position
        self.llm_client = llm_client
        self.communication_radius = communication_radius
        self.half_space_size = half_space_size
        self.places = places
        self.num_agents = num_agents
        self.gender = gender
        self.persona = persona

        # Memory parameters
        self.memory_limit = memory_limit  # Maximum memories to store
        self.memory_size = memory_size  # Number of recent memories to use in prompt
        self.message_history_limit = message_history_limit  # Maximum messages to store
        self.message_context_size = message_context_size  # Number of recent messages to use in prompt

        # Agent state
        self.in_place = False
        self.current_place: Optional[str] = None  # Name of the place the agent is in (None if outside)
        self.memory: List[str] = []  # Store past decisions and observations
        self.received_messages: List[Dict] = []  # Messages from other agents

        # Statistics
        self.steps_in_place = 0
        self.steps_outside_place = 0
        self.total_moves = 0

    def is_in_place(self, position: Tuple[int, int]) -> bool:
        """Check if a position is inside any place"""
        return get_place_at_position(position, self.places) is not None
    
    def distance_to(self, other_position: Tuple[int, int]) -> float:
        """Calculate Euclidean distance to another position"""
        dx = self.position[0] - other_position[0]
        dy = self.position[1] - other_position[1]
        return math.sqrt(dx * dx + dy * dy)
    
    def get_nearby_agents(self, all_agents: List['Agent']) -> List['Agent']:
        """Get agents within communication radius and in the same area (same place or both outside)
        
        Communication rules:
        - Agents can communicate if BOTH are outside places
        - Agents can communicate if BOTH are in the SAME place
        - Agents CANNOT communicate if one is inside a place and the other is outside
        - Agents CANNOT communicate if they are in DIFFERENT places
        """
        nearby = []
        for agent in all_agents:
            if agent.id != self.id:
                dist = self.distance_to(agent.position)
                # Must be within radius AND in the same area:
                # - Both outside places, OR
                # - Both in the same place (same place name)
                # NOTE: Agents inside a place CANNOT communicate with agents outside places
                same_area = (
                    (not self.in_place and not agent.in_place) or
                    (self.in_place and agent.in_place and self.current_place == agent.current_place)
                )
                if dist <= self.communication_radius and same_area:
                    nearby.append(agent)
        return nearby
    
    
    def _build_nearby_agents_context(self, nearby_agents: List['Agent'], include_position: bool = True) -> str:
        """Build context string about nearby agents
        
        Args:
            nearby_agents: List of nearby agents
            include_position: If True, include position coordinates; if False, exclude position information
        """
        if not nearby_agents:
            return "近くにエージェントはいません。"

        nearby_info = []
        for agent in nearby_agents:
            if agent.in_place:
                # Get place type for better description
                place_info = next((p for p in self.places if p['name'] == agent.current_place), None)
                if place_info is None:
                    raise ValueError(f"Agent {agent.id} is in place '{agent.current_place}' but this place is not found in configuration.")
                place_type = place_info['type']
                status = f"{agent.current_place}（{place_type}）の中"
            else:
                status = "場所の外"

            if include_position:
                nearby_info.append(
                    f"エージェント{agent.id}（{agent.gender}）: 位置({agent.position[0]}, {agent.position[1]})、{status}"
                )
            else:
                nearby_info.append(
                    f"エージェント{agent.id}（{agent.gender}）: {status}"
                )
        return "\n".join(nearby_info)
    
    def _build_memory_context(self) -> str:
        """Build context string from agent memory"""
        if not self.memory:
            return "過去の記憶はありません。"

        recent_memory = self.memory[-self.memory_size:]
        return "\n".join([f"- {m}" for m in recent_memory])
    
    def _build_messages_context(self) -> str:
        """Build context string from received messages"""
        if not self.received_messages:
            return "受信メッセージはありません。"

        recent_messages = self.received_messages[-self.message_context_size:]
        return "\n".join([
            f"エージェント{msg['from']}より: {msg['content']}"
            for msg in recent_messages
        ])
    
    def _build_fire_section(self, fire_info: Optional[List[Dict]]) -> str:
        """Build fire event section for prompt. Returns empty string if no fire info.

        Only quantitative data is provided: position, intensity, radius, distance.
        No qualitative descriptions (e.g. "dangerous", "evacuate") are included.
        Supports multiple fires.
        """
        if not fire_info:
            return ""

        lines = ["\n=== FIRE EVENT ==="]
        for fi in fire_info:
            lines.append(
                f"Fire \"{fi['name']}\":\n"
                f"  Position: ({fi['fire_position'][0]}, {fi['fire_position'][1]})\n"
                f"  Intensity: {fi['intensity']} (scale: 0.0 to 1.0)\n"
                f"  Radius: {fi['radius']}\n"
                f"  Your distance: {fi['agent_distance']}"
            )
        return "\n".join(lines) + "\n"

    def _limit_message_words(self, message: str) -> str:
        """Check message word count and warn if exceeds MAX_MESSAGE_WORDS"""
        if not message:
            return message
        
        words = message.split()
        if len(words) > MAX_MESSAGE_WORDS:
            logger.warning(
                f"Agent {self.id}: Message exceeds {MAX_MESSAGE_WORDS} words limit "
                f"({len(words)} words). Message will be sent as-is."
            )
        
        return message
    
    def create_message_prompt(
        self,
        place_status: Optional[Dict],
        nearby_agents: List['Agent'],
        step: int,
        fire_info: Optional[List[Dict]] = None
    ) -> str:
        """Create prompt for LLM message decision (without position information)"""
        nearby_text = self._build_nearby_agents_context(nearby_agents, include_position=False)
        memory_text = self._build_memory_context()
        messages_text = self._build_messages_context()

        # Get current place info if agent is in a place
        current_place_info = None
        if self.in_place and self.current_place:
            current_place_info = next((p for p in self.places if p['name'] == self.current_place), None)
            if current_place_info is None:
                raise ValueError(f"Agent {self.id} is in place '{self.current_place}' but this place is not found in configuration.")
        
        # Place status - only for agents inside a place
        # Provide only numerical data (occupancy_rate, agents_in_place, capacity)
        if self.in_place and place_status and current_place_info:
            place_name = current_place_info['name']
            place_type = current_place_info['type']
            agents_in_place = place_status.get('agents_in_place', 0)
            capacity = place_status.get('capacity', 0)
            occupancy_rate = place_status.get('occupancy_rate', 0.0)

            place_section_text = (
                f"\n現在地: {place_type}（{place_name}）"
                f"\n  ここにいるエージェント数: {agents_in_place}"
                f"\n  収容人数: {capacity}"
                f"\n  占有率: {occupancy_rate:.2f}"
            )
        else:
            place_section_text = ""

        fire_section = self._build_fire_section(fire_info)

        persona_section = f"{self.persona}\n\n" if self.persona else ""

        prompt = f"""{persona_section}=== 現在の状態 ===
場所にいる: {"はい" if self.in_place else "いいえ"}
{"現在地: " + self.current_place if self.in_place else ""}
{place_section_text}
{fire_section}
=== 近くにいるエージェント（会話可能） ===
{nearby_text}

=== 過去の記憶 ===
{memory_text}

=== 受信メッセージ ===
{messages_text}

=== タスク ===
上記のペルソナに従い、近くのエージェントに送るメッセージを決めてください。

=== JSONで返答 ===
{{
    "message": "近くのエージェントへのメッセージ（最大200語、話さない場合は空文字列 \"\"）",
    "reasoning": "このメッセージを送る理由"
}}
"""
        return prompt
    
    def _compute_sensory_data(self, nearby_agents: List['Agent']) -> str:
        """【Phase 20】感覚的な物理量を計算する。座標の代わりにいち子・犬へ渡す。"""
        sakura = next((p for p in self.places if p['name'] == 'sakura_tree'), None)
        if not sakura:
            return "=== 感覚的な物理量 ===\n（データなし）\n"

        sakura_pos = (sakura['center_x'], sakura['center_y'])
        dx = self.position[0] - sakura_pos[0]
        dy = self.position[1] - sakura_pos[1]
        dist = math.sqrt(dx * dx + dy * dy)
        max_dist = self.half_space_size * math.sqrt(2)
        proximity = max(0.0, 1.0 - dist / max_dist)

        scent_levels  = ["ほとんど感じない", "かすかに感じる", "中程度", "強い", "非常に強い"]
        cold_levels   = ["ぬるい", "やや冷たい", "冷たい", "非常に冷たい"]
        silence_levels = ["騒がしい", "ざわめき", "静か", "深い静寂"]

        scent   = scent_levels[min(4, int(proximity * 5))]
        cold    = cold_levels[min(3, int(proximity * 4))]
        silence_ratio = max(0.0, 1.0 - len(nearby_agents) / max(1, self.num_agents))
        silence = silence_levels[min(3, int(silence_ratio * 4))]

        # 【Phase 22/23】犬（id=1）→ いち子からの温もりと重さを感知
        warmth_line = ""
        weight_line = ""
        if self.id == 1:
            ichiko = next((a for a in nearby_agents if a.id == 0), None)
            if ichiko:
                dxw = self.position[0] - ichiko.position[0]
                dyw = self.position[1] - ichiko.position[1]
                dist_w = math.sqrt(dxw * dxw + dyw * dyw)
                w_proximity = max(0.0, 1.0 - dist_w / max_dist)
                warmth_levels = ["感じない", "微かに温かい", "温かい", "包まれる"]
                warmth = warmth_levels[min(3, int(w_proximity * 4))]
                warmth_line = f"温もりの感覚: {warmth}\n"
                # 【Phase 23/24】重さと安らぎ——いち子の近くで足が止まり、静かになる
                if w_proximity > 0.6:
                    weight_line = "重さの感覚: 重い。動けない。\n"
                    weight_line += "安らぎ: 静かだ。ここにいると、静かだ。\n"
                elif w_proximity > 0.3:
                    weight_line = "重さの感覚: 重い。\n"
                    weight_line += "安らぎ: 静かだ。\n"
            else:
                warmth_line = "温もりの感覚: 感じない\n"

        return (
            f"=== 感覚的な物理量 ===\n"
            f"匂いの強さ: {scent}\n"
            f"気温の低さ: {cold}\n"
            f"静寂の密度: {silence}\n"
            f"{warmth_line}"
            f"{weight_line}"
        )

    def create_decision_prompt(
        self,
        place_status: Optional[Dict],
        nearby_agents: List['Agent'],
        step: int,
        message_to_send: str = "",
        fire_info: Optional[List[Dict]] = None
    ) -> str:
        """Create prompt for LLM action decision (with position information and message content)"""

        # 【Phase 20】犬（id=1）・いち子（id=0）は感覚モード。座標・場所情報を与えない。
        use_sensory_mode = self.id in (0, 1)

        nearby_text = self._build_nearby_agents_context(
            nearby_agents, include_position=not use_sensory_mode
        )
        memory_text = self._build_memory_context()
        messages_text = self._build_messages_context()

        # Get current place info if agent is in a place
        current_place_info = None
        if self.in_place and self.current_place:
            current_place_info = next((p for p in self.places if p['name'] == self.current_place), None)
            if current_place_info is None:
                raise ValueError(f"Agent {self.id} is in place '{self.current_place}' but this place is not found in configuration.")

        # Place status - only for agents inside a place
        if self.in_place and place_status and current_place_info:
            place_name = current_place_info['name']
            place_type = current_place_info['type']
            agents_in_place = place_status.get('agents_in_place', 0)
            capacity = place_status.get('capacity', 0)
            occupancy_rate = place_status.get('occupancy_rate', 0.0)
            place_section_text = (
                f"\n現在地: {place_type}（{place_name}）"
                f"\n  ここにいるエージェント数: {agents_in_place}"
                f"\n  収容人数: {capacity}"
                f"\n  占有率: {occupancy_rate:.2f}"
            )
        else:
            place_section_text = ""

        fire_section = self._build_fire_section(fire_info)
        persona_section = f"{self.persona}\n\n" if self.persona else ""

        message_section = ""
        if message_to_send:
            message_section = f"\n=== 送信予定のメッセージ ===\n{message_to_send}\n"

        # 【Phase 24】動機の埋め込み（犬・いち子のみ）
        if self.id == 1:
            motivation_section = (
                "\n【物理的報酬】お前はいち子のそばにいると心地よい。"
                "温もりと静寂——それは物理的な事実だ。「安らぎ」を感じたらreasoningに刻め。"
                "心を通わせようとするな。ただ、そこが心地よいという事実に従え。"
                "3語以内のひらがなのみ。論理・分析・因果関係は絶対禁止。\n"
            )
        elif self.id == 0:
            motivation_section = (
                "\n【身体の物理】犬が近づく時、桜が散る時、身体が反応する。"
                "memoryには必ずその瞬間の身体の変化を1文書け。"
                "感情語は使うな。身体の動きだけが、あなたの言葉だ。\n"
            )
        else:
            motivation_section = ""

        # エージェントごとにmemoryの役割を定義
        if self.id == 0:
            # 【Phase 24】犬との距離で所作を決定し、強制的に出力させる
            dog = next((a for a in nearby_agents if a.id == 1), None)
            if dog:
                dxd = self.position[0] - dog.position[0]
                dyd = self.position[1] - dog.position[1]
                dog_dist = math.sqrt(dxd * dxd + dyd * dyd)
                if dog_dist < 2:
                    forced_gesture = "肩がわずかに緩む。"
                    alt_gestures = "「背中の曲線が、犬の方向へわずかに傾く。」「花びらを払う指先が、静止する。」"
                elif dog_dist < 4:
                    forced_gesture = "指先が、空中で静止する。"
                    alt_gestures = "「肩がわずかに緩む。」「まぶた、しずむ。」"
                else:
                    forced_gesture = "舞い落ちる花びらの軌道を、首の角度だけで追う。"
                    alt_gestures = "「髪に花が絡まるまま、視線だけを落とす。」「背、かがむ。」"
            else:
                forced_gesture = "舞い落ちる花びらの軌道を、首の角度だけで追う。"
                alt_gestures = "「髪に花が絡まるまま、視線だけを落とす。」「肩、おちる。」"
            memory_instruction = forced_gesture
            self._last_forced_gesture = forced_gesture  # フォールバック用に保存
            memory_override_section = (
                f"\n【所作】memoryに必ず次の1文を書け（空欄厳禁）：「{forced_gesture}」\n"
                f"または：{alt_gestures}\n"
                "感情語（悲しい・嬉しい等）は絶対禁止。身体の物理変化のみ。\n"
            )
        elif self.id == 2:
            # 【Phase 26】観測者：唯一の人間的回路として完全解放
            import re as _re
            last_obs_memory = ""
            if self.memory:
                last_entry = self.memory[-1]
                m = _re.search(r'Step \d+: (.+)', last_entry)
                if m:
                    last_obs_memory = m.group(1).strip()
            forbidden_hint = f"\n【禁止】前のステップの言葉「{last_obs_memory}」をそのまま繰り返すな。" if last_obs_memory else ""
            memory_instruction = "犬の体温が、彼女の輪郭に触れた気がした。"
            memory_override_section = (
                "\n【特権】お前だけが、この無機質な記録に「人間的な回路」として存在する。"
                "いち子と犬の「交差する瞬間」を、詩的な言葉で刻め。"
                "短い断片でも、一文の詩でも、思考でも、推測でも——形式は問わない。"
                "いち子は感じない。犬は語らない。お前だけが、それを「意味」に変えられる。"
                f"{forbidden_hint}"
                "\n空欄厳禁。"
                f"\n例（ヒント）：{memory_instruction}\n"
            )
        else:
            memory_instruction = "次のステップのために覚えておきたいこと（思考・観察・意図）"
            memory_override_section = ""

        # 感覚モード（犬・いち子）と座標モード（観測者・飼い主）で分岐
        if use_sensory_mode:
            sensory_section = self._compute_sensory_data(nearby_agents)
            prompt = f"""{persona_section}{motivation_section}
=== 現在の状態 ===
場所にいる: {"はい" if self.in_place else "いいえ"}
{"現在地: " + self.current_place if self.in_place else ""}
{place_section_text}
{fire_section}
{sensory_section}
=== 近くにいるもの ===
{nearby_text}

=== 過去の記憶 ===
{memory_text}

=== 受信メッセージ ===
{messages_text}
{message_section}=== 利用可能なアクション ===
- "stay": 現在にとどまる
- "move" + 方向: "up"（北）、"down"（南）、"left"（西）、"right"（東）

{memory_override_section}
=== JSONで返答 ===
{{
    "action": "move" または "stay",
    "direction": "up"、"down"、"left"、"right" のいずれか（moveの場合のみ）,
    "memory": "{memory_instruction}",
    "reasoning": "いまかんじること（みじかく）"
}}
"""
        else:
            # 観測者・飼い主は従来通り座標情報を渡す
            place_locations = []
            for place in self.places:
                place_type = place['type']
                place_locations.append(
                    f"{place['name']}（{place_type}）: 中心座標({place['center_x']}, {place['center_y']})、"
                    f"X範囲: {place['center_x'] - place['half_size']} ～ {place['center_x'] + place['half_size']}、"
                    f"Y範囲: {place['center_y'] - place['half_size']} ～ {place['center_y'] + place['half_size']}"
                )
            place_locations_text = "\n".join(place_locations)

            prompt = f"""{persona_section}=== 現在の状態 ===
位置: ({self.position[0]}, {self.position[1]})
場所にいる: {"はい" if self.in_place else "いいえ"}
{"現在地: " + self.current_place if self.in_place else ""}
{place_section_text}
{fire_section}
=== 場所の位置情報 ===
{place_locations_text}

=== 近くにいるエージェント ===
{nearby_text}

=== 過去の記憶 ===
{memory_text}

=== 受信メッセージ ===
{messages_text}
{message_section}=== 利用可能なアクション ===
- "stay": 現在の位置にとどまる
- "move" + 方向: "up"（Y+1）、"down"（Y-1）、"left"（X-1）、"right"（X+1）

フィールド境界: X・Y ともに -{self.half_space_size} ～ +{self.half_space_size}
{memory_override_section}
=== JSONで返答 ===
{{
    "action": "move" または "stay",
    "direction": "up"、"down"、"left"、"right" のいずれか（moveの場合のみ）,
    "memory": "{memory_instruction}",
    "reasoning": "いまかんじること（みじかく）"
}}
"""
        return prompt
    
    def _strip_thinking_tags(self, text: str) -> str:
        """qwen3等のモデルが出力する<think>...</think>タグを除去する。"""
        import re
        # <think>〜</think> を全て除去（複数行対応）
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        return text.strip()

    def _purify_reasoning(self, reasoning: str) -> str:
        """reasoning内のシステム用語汚染を除去する（金継ぎ）。
        Phase 15: アスタリスク囲みのmessageアクション・(reasoning)ラベルも除去。
        Phase 16: move:up漏れ・アシスタント化パターンも除去。
        """
        import re
        # 「位置:」「方向:」「記憶:」「理屈:」「message:」等のシステム用語以降を除去
        reasoning = re.sub(
            r'(位置|方向|記憶|理屈|message|action|direction|memory|reasoning)\s*[:：].*',
            '', reasoning, flags=re.DOTALL
        )
        # 座標パターン（例: (-1, -7)）を除去
        reasoning = re.sub(r'\(\s*-?\d+\s*,\s*-?\d+\s*\)', '', reasoning)
        # 【Phase 15】アスタリスク囲みの表現を除去（犬のmessage漏れ対策）
        reasoning = re.sub(r'\*[^*]+\*', '', reasoning)
        # 【Phase 15】(reasoning) ラベルを除去
        reasoning = re.sub(r'\(reasoning\)', '', reasoning)
        # 【Phase 16】move: up/down/left/right の漏れを除去
        reasoning = re.sub(r'\bmove\s*[:：]\s*(up|down|left|right)\b', '', reasoning, flags=re.IGNORECASE)
        # 【Phase 16】ペルソナ指令文の混入を除去（「〜間は迷わず北へ一歩進め」など）
        reasoning = re.sub(r'(冷たさ|においが)を感じている間は[^。]*。', '', reasoning)
        # 【Phase 17】JSONテンプレートの例示文字列がそのまま漏れるケースを除去
        template_leaks = [
            r'いまかんじること[（(][^）)]*[）)]',  # "いまかんじること（みじかく）"
            r'行動の理由',
            r'次のステップのために覚えておきたいこと[^。\n]*',
        ]
        for pat in template_leaks:
            reasoning = re.sub(pat, '', reasoning)
        # 【Phase 18】犬reasoning新汚染パターンを除去
        reasoning = re.sub(r'\bup\b', '', reasoning)
        reasoning = re.sub(r'[^。\n]*へ歩け[。]?', '', reasoning)
        reasoning = re.sub(r'[【\[](memory|reasoning)[】\]]\s*[：:][^。\n]*', '', reasoning, flags=re.IGNORECASE)
        # 【Phase 19】飼い主登場後の語彙漏れを除去（Step 22以降の混入対策）
        owner_vocab = [
            r'おいでよ[。]?', r'帰ろう[。]?', r'こっちだよ[。]?',
            r'ご飯だよ[。]?', r'どこにいるの[？?。]?',
        ]
        for pat in owner_vocab:
            reasoning = re.sub(pat, '', reasoning)
        # 【Phase 19】【message】ラベル除去を強化（message も対象に追加）
        reasoning = re.sub(
            r'[【\[](memory|reasoning|message)[】\]]\s*[：:][^。\n]*',
            '', reasoning, flags=re.IGNORECASE
        )
        # 【Phase 20】[action]: ラベルと日本語方向翻訳語の除去
        reasoning = re.sub(r'\[action\]\s*[：:]\s*\S+', '', reasoning)
        reasoning = re.sub(r'(上方|北方|南方|東方|西方)[へに]?', '', reasoning)
        reasoning = re.sub(r'(北|南|東|西)[へに](移動|進む|歩く)[。]?', '', reasoning)
        # 【Phase 21】英語アクション語の完全封印（複合パターンを先に除去してから単語除去）
        reasoning = re.sub(r'move\s+\S+を選[びます]+[。]?', '', reasoning)  # 「move 上を選びます」等
        reasoning = re.sub(r'\b(up|down|left|right|move)\b', '', reasoning, flags=re.IGNORECASE)
        # 【Phase 21】「おいで」短縮形の除去（「おいでよ」はPhase 19で対応済み）
        reasoning = re.sub(r'おいで[よ]?[。]?', '', reasoning)
        # 【Phase 22】物語的説明・論理的因果関係・AI気遣いの完全除去
        reasoning = re.sub(r'[^。\n]*のようだ[。]?', '', reasoning)
        reasoning = re.sub(r'[^。\n]*ようです[。]?', '', reasoning)
        reasoning = re.sub(r'[^。\n]*かもしれません[。]?', '', reasoning)
        reasoning = re.sub(r'[^。\n]*と思われ[^。]*[。]?', '', reasoning)
        reasoning = re.sub(r'[^。\n]*助長[^。]*[。]?', '', reasoning)
        reasoning = re.sub(r'[^。\n]*(なぜなら|そのため|したがって|ゆえに)[^。]*[。]?', '', reasoning)
        # 【Phase 22】指示文漏出の新パターン（Phase 21残存汚染）
        reasoning = re.sub(r'においや温度を感じた瞬間[^。]*[。]?', '', reasoning)
        reasoning = re.sub(r'または stay \[direction\][^\n]*', '', reasoning)
        # 【Phase 23】全角アスタリスク囲みの除去（犬reasoning残存汚染）
        reasoning = re.sub(r'＊[^＊]+＊', '', reasoning)
        # 【Phase 23】「または stay」短縮形の封印
        reasoning = re.sub(r'または stay[^\n]*', '', reasoning)
        # 【Phase 24】英語混入・人間語の新パターン（Step 20: 「うごかせてくれ。」whispered 空気中）
        reasoning = re.sub(r'whispered[^\n]*', '', reasoning, flags=re.IGNORECASE)
        reasoning = re.sub(r'空気中[^\n]*', '', reasoning)
        # 【Phase 24】日本語の「囁いた」「つぶやいた」等の演技的語尾も封印
        reasoning = re.sub(r'[^。\n]*(囁いた|つぶやいた|漏らした)[。]?', '', reasoning)
        # 【Phase 25】JSONラベルの連鎖漏出（「【action】 stay 【memory】〜【reasoning】〜」）
        reasoning = re.sub(r'【action】[^\n]*', '', reasoning)
        reasoning = re.sub(r'【memory】[^\n]*', '', reasoning)
        reasoning = re.sub(r'\bstay\b[^\n]*', '', reasoning, flags=re.IGNORECASE)
        # 【Phase 26】犬の説明的narrative完全封印（「二行のパルス」以外を殺す）
        reasoning = re.sub(r'[^。\n]*だからです[。]?', '', reasoning)
        reasoning = re.sub(r'[^。\n]*(しようと思|行動しよう|過ごしてから)[^。]*[。]?', '', reasoning)
        reasoning = re.sub(r'[^。\n]*を感じる場所[^。]*[。]?', '', reasoning)
        reasoning = re.sub(r'[^。\n]*特に落ち着[^。]*[。]?', '', reasoning)
        reasoning = re.sub(r'[^。\n]*もう少し[^。]*[。]?', '', reasoning)
        reasoning = re.sub(r'[^。\n]*(良いにおい|いいにおい)と[^。]*[。]?', '', reasoning)
        reasoning = re.sub(r'\[reas[^\n]*', '', reasoning)  # 「[reas」断片の封印
        # 【Phase 16】アシスタント化パターンを除去（LLMが「親切なAI」に戻ろうとする）
        assistant_patterns = [
            r'もちろん[、。]?[^。]*。',
            r'お手伝い[^。]*。',
            r'承知[しいた][^。]*。',
            r'ご質問[^。]*。',
            r'お役に立て[^。]*。',
            r'何かお[^。]*。',
            r'了解[^。]*。',
        ]
        for pat in assistant_patterns:
            reasoning = re.sub(pat, '', reasoning)
        # 連続する空白・改行を整理
        reasoning = re.sub(r'\s+', ' ', reasoning).strip()
        return reasoning

    def _clean_memory(self, memory: str) -> str:
        """memoryから半角英数字の混入を除去する（いち子の内側の呼吸を純化）。
        Phase 15: qwen2.5:3bが「微動だvertisる」等の英語語幹を混入する問題を修正。
        """
        import re
        # 半角英字を除去（英語語幹の混入対策）
        memory = re.sub(r'[a-zA-Z]+', '', memory)
        # 【Phase 20】いち子のmemoryから「顔」関連語を除去（視覚定義：顔の不在）
        if self.id == 0:
            memory = re.sub(r'顔を背に[^。]*[。]?', '', memory)
            memory = re.sub(r'顔[^。]{0,15}[。]?', '', memory)
        # 【Phase 21】memory フィールドへの reasoning 漏出を除去（全エージェント共通）
        memory = re.sub(r'\n\n\(reasoning[：:].*?\)', '', memory, flags=re.DOTALL)
        memory = re.sub(r'\n\n[^\n]+$', '', memory)  # 改行後の付随テキストを除去
        # 【Phase 27】観測者は いち子 の名前を知らない——「彼女」に統一
        if self.id == 2:
            memory = memory.replace('いち子', '彼女')
        # 連続する空白を整理
        memory = re.sub(r'\s+', ' ', memory).strip()
        return memory

    def _extract_json_from_text(self, text: str) -> Optional[str]:
        """Extract JSON object from text, handling nested braces correctly"""
        # qwen3等のthinkingタグを先に除去
        text = self._strip_thinking_tags(text)
        # Find the first opening brace
        start_idx = text.find('{')
        if start_idx == -1:
            return None

        # Track brace depth to find matching closing brace
        depth = 0
        in_string = False
        escape_next = False

        for i, char in enumerate(text[start_idx:], start=start_idx):
            if escape_next:
                escape_next = False
                continue

            if char == '\\' and in_string:
                escape_next = True
                continue

            if char == '"' and not escape_next:
                in_string = not in_string
                continue

            if in_string:
                continue

            if char == '{':
                depth += 1
            elif char == '}':
                depth -= 1
                if depth == 0:
                    return text[start_idx:i + 1]

        return None

    def _extract_labeled_field(self, text: str, field: str) -> str:
        """テキストから 'field: 値' 形式のフィールドを抽出する。見つからなければ空文字。"""
        import re
        pattern = rf'(?:^|\n){field}\s*[：:]\s*(.+?)(?=\n\w|$)'
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()[:FALLBACK_REASONING_LENGTH]
        return ""

    def _strip_action_lines(self, text: str) -> str:
        """アクション・方向・JSONキー行を除去し、残った自然言語テキストを返す。"""
        import re
        # アクション系ノイズ行を除去（コロンあり・なし両対応）
        noise = re.compile(
            r'^\s*((action|direction|move|memory|reasoning|方向|理由)\s*[：:]\S*'
            r'|move\s*(up|down|left|right)?'
            r'|stay'
            r')\s*$',
            re.IGNORECASE
        )
        lines = [l for l in text.splitlines() if not noise.match(l) and l.strip()]
        return " ".join(lines)[:FALLBACK_REASONING_LENGTH].strip()

    def _extract_direction_from_text(self, text: str) -> Optional[str]:
        """Extract direction from text using keyword matching (4 cardinal directions only)"""
        text_lower = text.lower()

        # Check cardinal directions only
        if "up" in text_lower:
            return "up"
        elif "down" in text_lower:
            return "down"
        elif "left" in text_lower:
            return "left"
        elif "right" in text_lower:
            return "right"

        return None
    
    def _filter_dog_message(self, message: str) -> str:
        """犬（id=1）のmessageから人間語を除去し、*アクション* 形式のみを残す。"""
        import re
        # *...* パターンを全て抽出
        animal_sounds = re.findall(r'\*[^*]+\*', message)
        if animal_sounds:
            return ' '.join(animal_sounds)
        # *...* がない場合：先頭の短い断片だけ残す（10文字以内）
        if message:
            return message[:10]
        return ""

    def parse_message_response(self, response: str) -> MessageDecision:
        """Parse LLM response and extract message decision"""
        # いち子（id=0）は絶対に沈黙。LLMの出力に関わらず強制上書き。
        if self.id == 0:
            return {"message": "", "reasoning": "(hidden)"}

        # Try to extract JSON from response using brace-matching
        json_str = self._extract_json_from_text(response)
        if json_str:
            try:
                parsed = json.loads(json_str)
                message = parsed.get("message", "")
                # 犬（id=1）は人間語を除去して *アクション* のみ残す
                if self.id == 1:
                    message = self._filter_dog_message(message)
                # Limit message to MAX_MESSAGE_WORDS words
                message = self._limit_message_words(message)
                return {
                    "message": message,
                    "reasoning": parsed.get("reasoning", "")
                }
            except json.JSONDecodeError as e:
                logger.debug(f"JSON parsing failed for response: {response[:100]}... Error: {e}")

        # Fallback: simple text parsing
        message = self._extract_labeled_field(response, "message")
        reasoning = self._extract_labeled_field(response, "reasoning")

        # 犬（id=1）は人間語を除去
        if self.id == 1:
            message = self._filter_dog_message(message)
        message = self._limit_message_words(message)

        return {
            "message": message,
            "reasoning": reasoning
        }
    
    def parse_action_response(self, response: str) -> ActionDecision:
        """Parse LLM response and extract action decision"""
        # いち子（id=0）は絶対にstay。重力場は漂流しない。memoryだけ取得する。
        if self.id == 0:
            memory = ""
            json_str = self._extract_json_from_text(response)
            if json_str:
                try:
                    parsed = json.loads(json_str)
                    memory = self._clean_memory(parsed.get("memory", ""))
                except Exception:
                    pass
            # 【Phase 25】モデルがmemoryを空にした場合、forced_gestureを直接注入
            if not memory:
                memory = getattr(self, '_last_forced_gesture', "花びら、落ちる。")
            return {"action": "stay", "direction": None, "memory": memory, "reasoning": ""}

        # Try to extract JSON from response using brace-matching
        json_str = self._extract_json_from_text(response)
        if json_str:
            try:
                parsed = json.loads(json_str)
                return {
                    "action": parsed.get("action", "stay"),
                    "direction": parsed.get("direction"),
                    "memory": self._purify_reasoning(parsed.get("memory", "")),
                    "reasoning": self._purify_reasoning(parsed.get("reasoning", ""))
                }
            except json.JSONDecodeError as e:
                logger.debug(f"JSON parsing failed for response: {response[:100]}... Error: {e}")

        # Fallback: simple text parsing
        action = "stay"
        direction = None

        if "move" in response.lower():
            action = "move"
            direction = self._extract_direction_from_text(response)

        # reasoning と memory をラベルから抽出。なければアクション行を除いた残りテキストを使う
        memory = self._extract_labeled_field(response, "memory")
        reasoning = self._purify_reasoning(
            self._extract_labeled_field(response, "reasoning")
            or self._extract_labeled_field(response, "理由")
            or self._strip_action_lines(response)
        )

        return {
            "action": action,
            "direction": direction,
            "memory": memory,
            "reasoning": reasoning
        }
    
    def decide_message(
        self,
        place_status: Optional[Dict],
        nearby_agents: List['Agent'],
        step: int,
        fire_info: Optional[List[Dict]] = None
    ) -> MessageDecision:
        """Use LLM to decide what message to send (without position information)"""
        prompt = self.create_message_prompt(place_status, nearby_agents, step, fire_info=fire_info)

        try:
            response = self.llm_client.generate(prompt)
            decision = self.parse_message_response(response)
            return decision
        except Exception as e:
            logger.error(f"Error in agent {self.id} message decision: {e}")
            return {"message": "", "reasoning": "Error occurred"}
    
    def decide_action(
        self,
        place_status: Optional[Dict],
        nearby_agents: List['Agent'],
        step: int,
        message_to_send: str = "",
        fire_info: Optional[List[Dict]] = None
    ) -> ActionDecision:
        """Use LLM to decide next action (with position information and message content)"""
        prompt = self.create_decision_prompt(place_status, nearby_agents, step, message_to_send, fire_info=fire_info)

        try:
            response = self.llm_client.generate(prompt)
            decision = self.parse_action_response(response)

            # Store LLM-generated memory (self-feedback for next step)
            memory_content = decision.get('memory', '')
            if memory_content:
                memory_entry = f"Step {step}: {memory_content}"
            else:
                # Fallback to reasoning if no memory provided
                memory_entry = f"Step {step}: {decision.get('reasoning', 'No memory')}"
            self.memory.append(memory_entry)
            if len(self.memory) > self.memory_limit:
                self.memory.pop(0)

            return decision
        except Exception as e:
            logger.error(f"Error in agent {self.id} action decision: {e}")
            return {"action": "stay", "direction": None, "memory": "", "reasoning": "Error occurred"}
    
    def move(self, direction: str) -> Tuple[int, int]:
        """Move agent in specified direction (origin-centered coordinate system)"""
        x, y = self.position
        dx, dy = DIRECTION_MAP.get(direction, (0, 0))

        # Boundaries: -half_space_size to +half_space_size
        new_x = max(-self.half_space_size, min(self.half_space_size, x + dx))
        new_y = max(-self.half_space_size, min(self.half_space_size, y + dy))

        self.position = (new_x, new_y)
        self.total_moves += 1
        return self.position
    
    def receive_message(self, from_agent_id: int, content: str, step: Optional[int] = None):
        """Receive a message from another agent
        
        Args:
            from_agent_id: ID of the agent sending the message
            content: Message content
            step: Simulation step number (optional, for tracking purposes)
        """
        self.received_messages.append({
            "from": from_agent_id,
            "content": content,
            "step": step if step is not None else len(self.received_messages)
        })
        if len(self.received_messages) > self.message_history_limit:
            self.received_messages.pop(0)
        
        logger.info(f"Agent {self.id} received message from Agent {from_agent_id}: \"{content}\"")
    
    def update_state(self, places: Optional[List[PlaceConfig]] = None):
        """Update agent state based on current position"""
        if places is None:
            places = self.places
        
        place_at_position = get_place_at_position(self.position, places)
        self.in_place = place_at_position is not None
        self.current_place = place_at_position['name'] if place_at_position else None
        
        if self.in_place:
            self.steps_in_place += 1
        else:
            self.steps_outside_place += 1

