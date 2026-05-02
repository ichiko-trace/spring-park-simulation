"""
LLM-based agent in 2D worlds with multiple places.
"""
import json
import os
import random
import re
import yaml
import logging
from typing import List, Tuple, Dict, Set, Optional
import numpy as np
from agent import Agent
from ollama_client import OllamaClient
from utils import is_position_in_place, get_place_at_position, PlaceConfig, FireConfig

logger = logging.getLogger(__name__)

# Constants
MAX_POSITION_ATTEMPTS = 1000
LOG_INTERVAL = 10


class Simulation:
    """Main simulation class for LLM-based agent in 2D worlds with multiple places."""
    
    def __init__(self, config_path: str = "config.yaml", output_dir: Optional[str] = None):
        """Initialize simulation from config file"""
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        # Output directory for logs
        self.output_dir = output_dir
        
        # Simulation parameters
        sim_config = self.config['simulation']
        self.duration = sim_config['duration']
        self.half_space_size = sim_config['half_space_size']
        self.half_place_size = sim_config.get('half_place_size', 5)
        self.script_generation_step = sim_config.get('script_generation_step', None)
        
        # Agent parameters
        agent_config = self.config['agents']
        self.num_agents = agent_config['num_agents']
        self.communication_radius = agent_config['communication_radius']
        self.memory_limit = agent_config.get('memory_limit', 20)
        self.memory_size = agent_config.get('memory_size', 5)
        self.message_history_limit = agent_config.get('message_history_limit', 10)
        self.message_context_size = agent_config.get('message_context_size', 3)
        
        # Place parameters - support multiple places
        if 'places' not in self.config:
            raise ValueError("No 'places' configuration found in config file. Please use 'places:' key.")
        
        self.places = self.config['places']
        
        # Validate places configuration
        if not isinstance(self.places, list):
            raise ValueError("'places' must be a list of place configurations.")
        
        if len(self.places) == 0:
            raise ValueError("At least one place must be configured in 'places'.")
        
        # Validate each place configuration
        required_fields = ['name', 'type', 'center_x', 'center_y', 'half_size', 'capacity']
        for i, place in enumerate(self.places):
            if not isinstance(place, dict):
                raise ValueError(f"Place at index {i} must be a dictionary.")
            
            for field in required_fields:
                if field not in place:
                    raise ValueError(f"Place at index {i} is missing required field: '{field}'")
        
        place_names = [place['name'] for place in self.places]
        place_types = [place['type'] for place in self.places]
        logger.info(f"Initialized {len(self.places)} place(s): {place_names} (types: {place_types})")
        
        # Fire parameters (multiple fires supported)
        fires_config = self.config.get('fires', [])
        self.fire_configs: List[Dict] = []
        for i, fc in enumerate(fires_config):
            config_entry = {
                'name': fc.get('name', f'fire_{i}'),
                'start_step': fc['start_step'],
                'intensity': fc['intensity'],
                'radius': fc['radius'],
            }
            if 'center_x' in fc and 'center_y' in fc:
                config_entry['center_x'] = fc['center_x']
                config_entry['center_y'] = fc['center_y']
            self.fire_configs.append(config_entry)
            pos_info = f"({fc['center_x']}, {fc['center_y']})" if 'center_x' in fc else "random"
            logger.info(
                f"Fire '{config_entry['name']}' configured: step={fc['start_step']}, "
                f"intensity={fc['intensity']}, radius={fc['radius']}, position={pos_info}"
            )
        self.fire_states: List[Dict] = []  # Active fires

        # LLM parameters
        llm_config = self.config['llm']
        self.llm_client = OllamaClient(
            base_url=llm_config['base_url'],
            model=llm_config['model'],
            temperature=llm_config.get('temperature', 0.7),
            max_tokens=llm_config.get('max_tokens', 200),
            repeat_penalty=llm_config.get('repeat_penalty', 1.1),
            repeat_last_n=llm_config.get('repeat_last_n', 128),
            min_p=llm_config.get('min_p', 0.05),
            think=llm_config.get('think', False)  # qwen3等のthinking無効化（デフォルトOff）
        )

        # 観測者（Agent 2）専用クライアント（qwen3:4b / think=False）
        obs_config = self.config.get('observer_llm', {})
        if obs_config:
            self.observer_llm_client = OllamaClient(
                base_url=llm_config['base_url'],
                model=obs_config.get('model', llm_config['model']),
                temperature=obs_config.get('temperature', 0.7),
                max_tokens=obs_config.get('max_tokens', 256),
                repeat_penalty=llm_config.get('repeat_penalty', 1.1),
                repeat_last_n=llm_config.get('repeat_last_n', 128),
                min_p=llm_config.get('min_p', 0.05),
                think=obs_config.get('think', False)
            )
            logger.info(f"観測者用LLM: {obs_config.get('model')} (think={obs_config.get('think', False)})")
        else:
            self.observer_llm_client = self.llm_client
        
        # Initialize agents
        self.agents: List[Agent] = []
        self.step = 0
        self.history: List[Dict] = []
        
        # Statistics - track per place
        self.stats = {
            'place_occupancy': [],  # Overall occupancy (all places combined)
            'agents_in_place': [],  # Total agents in any place
            'agents_outside_place': [],
            'communication_events': [],
            'places': {place['name']: {
                'occupancy': [],
                'agents_in_place': []
            } for place in self.places},
            'agents_in_fire_radius': [],  # Total agents in any fire radius
        }
        
    def _kintsugi_filter(self, text: str) -> str:
        """金継ぎフィルター: LLMのメタ発言・システム用語を詩的な言葉に昇華する。

        割れた器を金で継ぐように、AIの漏れを作品の一部として美しく直す。
        エージェントIDをキャラクター名に、AI的な言い回しを断定的な言葉に変換する。
        """
        if not text:
            return text

        # ── エージェントID → キャラクター名 ──────────────────────────────
        agent_names = {
            r'エージェント\s*0': 'いち子',
            r'エージェント\s*1': '犬',
            r'エージェント\s*2': '観測者',
            r'エージェント\s*3': '飼い主',
        }
        for pattern, name in agent_names.items():
            text = re.sub(pattern, name, text)

        # ── AI的メタ発言 → 断定的なトーンへ ─────────────────────────────
        noise_patterns = [
            (r'を示唆する',     'だ'),
            (r'と推測される',   'だ'),
            (r'の可能性が高い', 'だろう'),
            (r'と考えられる',   'だ'),
            (r'と思われる',     'だ'),
            (r'かもしれない',   ''),
        ]
        for pattern, replacement in noise_patterns:
            text = re.sub(pattern, replacement, text)

        # ── 【Phase 15】誤字修正（Kintsugi）────────────────────────────────
        # qwen2.5:3bが「桜」を「桙」と誤出力する問題を強制修正
        text = text.replace('桙', '桜')

        # ── 【Phase 16】アシスタント化防止（Kintsugi）───────────────────
        # 「親切なAI」に戻ろうとする発言を即座に削ぎ落とす
        assistant_patterns = [
            r'もちろん[、。]?[^。\n]*[。\n]?',
            r'お手伝い[^。\n]*[。\n]?',
            r'承知[しいた][^。\n]*[。\n]?',
            r'ご質問[^。\n]*[。\n]?',
            r'お役に立て[^。\n]*[。\n]?',
            r'何かお[^。\n]*[。\n]?',
            r'了解[^。\n]*[。\n]?',
            r'かしこまり[^。\n]*[。\n]?',
        ]
        for pat in assistant_patterns:
            text = re.sub(pat, '', text)

        return text

    def _is_position_in_place(self, position: Tuple[int, int]) -> bool:
        """Check if a position is inside any place"""
        return get_place_at_position(position, self.places) is not None

    def _log_message(
        self,
        from_agent_id: int,
        to_agent_id: int,
        message: str,
        reasoning: str = ""
    ) -> None:
        """Log a message to messages.jsonl file"""
        if not self.output_dir:
            return

        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)

        messages_file = os.path.join(self.output_dir, "messages.jsonl")
        record = {
            "step": self.step,
            "from": from_agent_id,
            "to": to_agent_id,
            "message": self._kintsugi_filter(message),
            "reasoning": self._kintsugi_filter(reasoning)
        }

        with open(messages_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

    def _log_memory_reasoning_batch(
        self,
        records: List[Dict]
    ) -> None:
        """Log memory and reasoning records in batch to memory_reasoning.jsonl file
        
        This is more efficient than writing one record at a time, especially
        when logging for all agents in each step.
        """
        if not self.output_dir or not records:
            return

        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)

        memory_reasoning_file = os.path.join(self.output_dir, "memory_reasoning.jsonl")
        
        # Write all records at once (buffered I/O)
        # 金継ぎフィルターをかけてからログに書く
        with open(memory_reasoning_file, 'a', encoding='utf-8') as f:
            for record in records:
                filtered_record = {
                    **record,
                    "memory":    self._kintsugi_filter(record.get("memory", "")),
                    "reasoning": self._kintsugi_filter(record.get("reasoning", "")),
                }
                f.write(json.dumps(filtered_record, ensure_ascii=False) + '\n')

    def _generate_random_position(self) -> Tuple[int, int]:
        """Generate a random position within the space (origin-centered coordinate system)"""
        return (
            random.randint(-self.half_space_size, self.half_space_size),
            random.randint(-self.half_space_size, self.half_space_size)
        )
    
    def _generate_initial_positions(self, avoid_places: bool = True) -> List[Tuple[int, int]]:
        """Generate initial positions for agents"""
        positions: List[Tuple[int, int]] = []
        used_positions: Set[Tuple[int, int]] = set()
        attempts = 0
        
        while len(positions) < self.num_agents and attempts < MAX_POSITION_ATTEMPTS:
            position = self._generate_random_position()
            
            # Skip if position is already used
            if position in used_positions:
                attempts += 1
                continue
            
            # Skip if position is in any place and we want to avoid it
            if avoid_places and self._is_position_in_place(position):
                attempts += 1
                continue
            
            positions.append(position)
            used_positions.add(position)
            attempts += 1
        
        # If we couldn't generate enough positions avoiding places, fill remaining
        if len(positions) < self.num_agents:
            logger.warning(
                f"Could only generate {len(positions)} unique positions avoiding places. "
                "Using all available space."
            )
            while len(positions) < self.num_agents:
                position = self._generate_random_position()
                if position not in used_positions:
                    positions.append(position)
                    used_positions.add(position)
        
        return positions
    
    def initialize_agents(self):
        """Initialize agents at random positions (excluding late-spawn agents)"""
        personas = self.config.get('personas', [])
        late_spawns = self.config.get('late_spawns', [])
        late_spawn_ids = {ls['agent_id'] for ls in late_spawns}

        # Only initialize agents that are NOT late-spawns
        initial_count = self.num_agents - len(late_spawn_ids)
        logger.info(f"Initializing {initial_count} agents (+ {len(late_spawn_ids)} late-spawn)...")

        positions = self._generate_initial_positions(avoid_places=True)

        pos_index = 0
        for i in range(self.num_agents):
            if i in late_spawn_ids:
                continue
            gender = random.choice(["male", "female"])
            persona = personas[i] if i < len(personas) else ""

            # いち子（id=0）はsakura_treeの中心に絶対固定。重力場は漂流しない。
            if i == 0:
                sakura = next((p for p in self.places if p['name'] == 'sakura_tree'), None)
                position = (sakura['center_x'], sakura['center_y']) if sakura else (0, 0)
                logger.info(f"★ いち子（id=0）をsakura_tree中心 {position} に固定")
            # 犬（id=1）はいち子の真南4マスに固定。至近距離からの接近で間（Ma）を生む。
            elif i == 1:
                position = (0, -4)
                logger.info(f"★ 犬（id=1）を (0, -4) に固定")
            else:
                position = positions[pos_index]
                pos_index += 1

            agent = Agent(
                agent_id=i,
                initial_position=position,
                llm_client=self.llm_client,
                communication_radius=self.communication_radius,
                half_space_size=self.half_space_size,
                places=self.places,
                num_agents=self.num_agents,
                gender=gender,
                memory_limit=self.memory_limit,
                memory_size=self.memory_size,
                message_history_limit=self.message_history_limit,
                message_context_size=self.message_context_size,
                persona=persona
            )
            agent.update_state()
            self.agents.append(agent)
            persona_name = persona.split('\n')[1].strip() if persona else "(no persona)"
            logger.info(f"Agent {i} initialized with persona: {persona_name}")
            pos_index += 1

        logger.info("Agents initialized successfully")

    def _spawn_late_agent(self, agent_id: int):
        """Spawn a late-arrival agent mid-simulation"""
        personas = self.config.get('personas', [])
        late_spawns = self.config.get('late_spawns', [])
        spawn_config = next((ls for ls in late_spawns if ls['agent_id'] == agent_id), None)
        if not spawn_config:
            return

        persona = personas[agent_id] if agent_id < len(personas) else ""
        position = (
            spawn_config.get('spawn_x', self.half_space_size),
            spawn_config.get('spawn_y', 0)
        )
        gender = spawn_config.get('gender', 'male')

        agent = Agent(
            agent_id=agent_id,
            initial_position=position,
            llm_client=self.llm_client,
            communication_radius=self.communication_radius,
            half_space_size=self.half_space_size,
            places=self.places,
            num_agents=self.num_agents,
            gender=gender,
            memory_limit=self.memory_limit,
            memory_size=self.memory_size,
            message_history_limit=self.message_history_limit,
            message_context_size=self.message_context_size,
            persona=persona
        )
        agent.update_state()
        self.agents.append(agent)
        logger.info(f"★ Agent {agent_id}（飼い主）が Step {self.step} に出現しました。位置: {position}")
    
    def _run_image_prompt_generation(self):
        """Step 29: 観測者の全memory + 犬の全reasoningから4枚の画像プロンプトを生成する。

        脚本（語ること）を捨て、4枚の視覚標本（見せること）へ転換。
        30ステップの沈黙とノイズを、英語の光と影の設計図にコンパイルする。
        """
        logger.info("★ 画像プロンプト生成フェーズ開始（Step %d）— Visual Compilation 発動", self.step)

        # ── memory_reasoning.jsonl から観測者のmemoryと犬のreasoningを収集 ──
        memory_file = os.path.join(self.output_dir, "memory_reasoning.jsonl")
        observer_memos = []
        dog_pulses = []

        if os.path.exists(memory_file):
            with open(memory_file, 'r', encoding='utf-8') as f:
                for line in f:
                    r = json.loads(line)
                    if r['id'] == 2 and r.get('memory'):
                        observer_memos.append(f"Step {r['step']}: {r['memory']}")
                    if r['id'] == 1 and r.get('reasoning'):
                        dog_pulses.append(f"Step {r['step']}: {r['reasoning']}")

        observer_text = "\n".join(observer_memos) or "（記録なし）"
        dog_text = "\n".join(dog_pulses) or "（沈黙。）"

        # ── 4枚の画像プロンプト生成プロンプト ──────────────────────
        prompt = f"""You are a cinematographer-poet who translates raw sensory fragments into cinematic image prompts.

Below are field notes from a Japanese cherry blossom park, captured over 30 steps by two presences:
- An observer sitting motionless on a bench, recording what they see (Japanese fragments)
- A dog whose sensory pulses are also recorded (Japanese fragments)

=== Observer's Field Notes ===
{observer_text}

=== Dog's Sensory Pulse ===
{dog_text}

=== Your Task ===
Create EXACTLY 4 cinematic image prompts in English.
These 4 prompts form a visual story in 4 acts (起承転結).

ACT DEFINITIONS:
[ACT 1 / 起 / The Arrival of Silence]
  Subject: A motionless woman beneath a cherry tree. Pure stillness as phenomenon.
  Draw from: the earliest observer notes. The scene before anything moves.

[ACT 2 / 承 / The Wild Approaches]
  Subject: A dog moving north by instinct alone. The first sensory encounter.
  Draw from: dog's pulses. The smell of cold. The pull toward something unnamed.

[ACT 3 / 転 / Gravity Crosses]
  Subject: Two presences in proximity without touching — OR a stranger's voice breaking the silence.
  Draw from: observer notes near the middle of the sequence. Tension without drama.

[ACT 4 / 結 / Dissolution]
  Subject: Cherry petals fall. All boundaries dissolve. Nothing remains but light.
  Draw from: the final observer notes. The ending that is also a disappearance.

=== Required Style (apply to ALL 4 prompts) ===
Cinematic 35mm photography, Yugen (profound grace), Wabi-sabi aesthetics, ethereal natural lighting, extreme silence, desaturated palette with warm highlights, shallow depth of field, fine film grain, Japanese park setting, cherry blossom season.

=== Strict Rules ===
1. Write in English only.
2. Each prompt: 2-4 sentences, dense with precise visual detail.
3. Translate the Japanese fragments into English visual imagery.
   Examples: "影、重なる。" → "two shadows overlap on the moss-covered stone path"
             "光、ずれる。" → "light shifts fractionally across the bark of the cherry tree"
             "犬、止まる。" → "a dog halts mid-step, nose raised toward an invisible threshold"
4. Use at least ONE translated Japanese fragment per prompt.
5. NO emotion words. NO narration. Only what the camera sees.
6. The woman never moves. Never speaks. She is a phenomenon, not a character.

=== Output Format (exact — no extra text) ===
[ACT 1 / 起 / Silence]
{{image prompt here}}

[ACT 2 / 承 / Approach]
{{image prompt here}}

[ACT 3 / 転 / Friction]
{{image prompt here}}

[ACT 4 / 結 / Dissolution]
{{image prompt here}}
"""

        # ── Low-temperature generation: 整合性優先、ループ防止 ─────────
        try:
            result_text = self.llm_client.generate(
                prompt,
                temperature=0.5,
                max_tokens=800,
                timeout=600
            )
            result_text = self._kintsugi_filter(result_text)

            # output/image_prompts.md に保存
            os.makedirs(self.output_dir, exist_ok=True)
            output_path = os.path.join(self.output_dir, "image_prompts.md")
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write("# TRINOIR Visual Compilation\n")
                f.write("## 4 Cinematic Image Prompts — 30 Steps Compiled\n\n")
                f.write(f"*Generated at Step {self.step} / {self.duration}*\n\n")
                f.write("---\n\n")
                f.write(result_text)

            logger.info("★ 画像プロンプトを保存しました: %s", output_path)
            logger.info("冒頭:\n%s", result_text[:400])

            self._log_message(
                from_agent_id=2,
                to_agent_id=-1,
                message=result_text,
                reasoning="30ステップの全記録を4枚の視覚標本にコンパイルした。"
            )

        except Exception as e:
            logger.error("画像プロンプト生成エラー: %s", e)

    def get_agents_in_place(self, place_name: Optional[str] = None) -> List[Agent]:
        """Get list of agents currently in a specific place or any place"""
        if place_name:
            return [agent for agent in self.agents if agent.current_place == place_name]
        return [agent for agent in self.agents if agent.in_place]
    
    def get_place_status(self, place_name: Optional[str] = None) -> Dict:
        """Get current place status for a specific place or overall status"""
        if place_name:
            # Get status for a specific place
            place_config = next((p for p in self.places if p['name'] == place_name), None)
            if not place_config:
                raise ValueError(f"Place '{place_name}' not found")
            
            agents_in_place = len(self.get_agents_in_place(place_name))
            capacity = place_config['capacity']
            occupancy_rate = agents_in_place / capacity

            return {
                "place_name": place_name,
                "agents_in_place": agents_in_place,
                "capacity": capacity,
                "occupancy_rate": occupancy_rate,
            }
        else:
            # Get overall status (all places combined)
            agents_in_place = len(self.get_agents_in_place())
            occupancy_rate = agents_in_place / self.num_agents
            
            # Get per-place status (optimized: calculate directly instead of recursive calls)
            place_statuses = {}
            for place in self.places:
                place_agents = len(self.get_agents_in_place(place['name']))
                place_capacity = place['capacity']
                place_occupancy_rate = place_agents / place_capacity

                place_statuses[place['name']] = {
                    "place_name": place['name'],
                    "agents_in_place": place_agents,
                    "capacity": place_capacity,
                    "occupancy_rate": place_occupancy_rate,
                }
            
            return {
                "agents_in_place": agents_in_place,
                "occupancy_rate": occupancy_rate,
                "places": place_statuses
            }
    
    def get_fire_info_for_agent(self, agent: Agent) -> Optional[List[Dict]]:
        """Return list of perceived fire info dicts, or None if no fires perceived.

        Implements Model B: only agents within each fire's radius get that fire's data.
        Agents outside all radii must learn about fires through messages.
        """
        if not self.fire_states:
            return None

        perceived = []
        for fire in self.fire_states:
            if not fire.get('active'):
                continue
            fire_pos = fire['position']
            distance = agent.distance_to(fire_pos)
            if distance <= fire['radius']:
                perceived.append({
                    'name': fire['name'],
                    'fire_position': fire_pos,
                    'intensity': fire['intensity'],
                    'radius': fire['radius'],
                    'agent_distance': round(distance, 2),
                })
        return perceived if perceived else None

    def step_simulation(self):
        """Execute one simulation step

        New order:
        1. All agents decide messages (without position information)
        2. Messages are sent to nearby agents (using decision-time positions)
        3. All agents decide actions (with position information and message content)
        4. Agents move to new positions
        """
        self.step += 1

        # ── 花びら時計（Petal-Driven Clock）——ステップ境界を独立したヘッダーとして刻む ──
        petal_header = f"---[ 環境: 一枚の桜の花びらが落ちた (Step {self.step} / {self.duration}) ]---"
        logger.info(petal_header)
        if self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)
            with open(os.path.join(self.output_dir, "petal_clock.log"), 'a', encoding='utf-8') as f:
                f.write(petal_header + '\n')

        # Late-spawn agent check
        late_spawns = self.config.get('late_spawns', [])
        existing_ids = {a.id for a in self.agents}
        for ls in late_spawns:
            if ls['agent_id'] not in existing_ids and self.step >= ls['spawn_step']:
                self._spawn_late_agent(ls['agent_id'])

        # ── 脚本生成フェーズ（Step 51以降）──────────────────────────
        # いち子（id=0）と犬（id=1）をフリーズ。観測者だけが動く。
        if self.script_generation_step and self.step >= self.script_generation_step:
            if self.step == self.script_generation_step:
                self._run_image_prompt_generation()
            # Step 51以降は全員フリーズ（ステップ記録だけ残す）
            agents_in_place = len(self.get_agents_in_place())
            overall_status = self.get_place_status()
            self.stats['place_occupancy'].append(overall_status['occupancy_rate'])
            self.stats['agents_in_place'].append(agents_in_place)
            self.stats['agents_outside_place'].append(self.num_agents - agents_in_place)
            for place in self.places:
                place_status = self.get_place_status(place['name'])
                self.stats['places'][place['name']]['occupancy'].append(place_status['occupancy_rate'])
                self.stats['places'][place['name']]['agents_in_place'].append(place_status['agents_in_place'])
            self.stats['agents_in_fire_radius'].append(0)
            self.history.append({
                'step': self.step,
                'place_status': overall_status,
                'agent_positions': [agent.position for agent in self.agents],
                'agents_in_place': [agent.id for agent in self.get_agents_in_place()],
                'fire_states': [],
            })
            return  # 通常のエージェント処理をスキップ

        # Fire activation check (multiple fires)
        active_names = {f['name'] for f in self.fire_states}
        for fc in self.fire_configs:
            if fc['name'] not in active_names and self.step >= fc['start_step']:
                if 'center_x' in fc and 'center_y' in fc:
                    fire_pos = (fc['center_x'], fc['center_y'])
                else:
                    fire_pos = self._generate_random_position()
                fire_state = {
                    'name': fc['name'],
                    'position': fire_pos,
                    'intensity': fc['intensity'],
                    'radius': fc['radius'],
                    'start_step': fc['start_step'],
                    'active': True,
                }
                self.fire_states.append(fire_state)
                logger.info(
                    f"FIRE '{fc['name']}' started at position {fire_pos} with intensity "
                    f"{fc['intensity']}, radius {fc['radius']}"
                )

        # Update agent states
        for agent in self.agents:
            agent.update_state(self.places)

        # Phase 1: Collect message decisions from all agents (without position information)
        message_decisions = []
        for agent in self.agents:
            nearby_agents = agent.get_nearby_agents(self.agents)
            # Get place status for the place the agent is in (or None if outside)
            agent_place_status = None
            if agent.in_place and agent.current_place:
                agent_place_status = self.get_place_status(agent.current_place)
            fire_info = self.get_fire_info_for_agent(agent)
            message_decision = agent.decide_message(agent_place_status, nearby_agents, self.step, fire_info=fire_info)
            message_decisions.append((agent, message_decision, nearby_agents))

        # Phase 2: Send messages (using decision-time nearby agents, before movement)
        for agent, message_decision, nearby_agents in message_decisions:
            message_content = message_decision.get('message', '')
            if message_content and nearby_agents:
                logger.info(
                    f"Step {self.step}: Agent {agent.id} sends message to {len(nearby_agents)} nearby agent(s): "
                    f"\"{message_content}\""
                )
                for other_agent in nearby_agents:
                    other_agent.receive_message(agent.id, message_content, step=self.step)
                    # Log message to jsonl file
                    self._log_message(
                        from_agent_id=agent.id,
                        to_agent_id=other_agent.id,
                        message=message_content,
                        reasoning=message_decision.get('reasoning', '')
                    )

        # Phase 3: Collect action decisions from all agents (with position information and message content)
        action_decisions = []
        memory_reasoning_records = []  # Batch records for efficient I/O
        for agent, message_decision, nearby_agents in message_decisions:
            # Get place status for the place the agent is in (or None if outside)
            agent_place_status = None
            if agent.in_place and agent.current_place:
                agent_place_status = self.get_place_status(agent.current_place)
            message_content = message_decision.get('message', '')
            fire_info = self.get_fire_info_for_agent(agent)
            action_decision = agent.decide_action(agent_place_status, nearby_agents, self.step, message_content, fire_info=fire_info)
            action_decisions.append((agent, action_decision))
            
            # Collect memory and reasoning records for batch writing
            memory_reasoning_records.append({
                "step": self.step,
                "id": agent.id,
                "memory": action_decision.get('memory', ''),
                "reasoning": action_decision.get('reasoning', '')
            })
        
        # Write all memory/reasoning records in batch (more efficient than individual writes)
        self._log_memory_reasoning_batch(memory_reasoning_records)

        # Phase 4: Execute movement (after messages are sent and actions are decided)
        for agent, action_decision in action_decisions:
            if action_decision['action'] == 'move' and action_decision['direction']:
                agent.move(action_decision['direction'])

        # Update states after movement
        for agent in self.agents:
            agent.update_state(self.places)
        
        # Record statistics
        agents_in_place = len(self.get_agents_in_place())
        overall_status = self.get_place_status()
        self.stats['place_occupancy'].append(overall_status['occupancy_rate'])
        self.stats['agents_in_place'].append(agents_in_place)
        self.stats['agents_outside_place'].append(self.num_agents - agents_in_place)
        
        # Record per-place statistics
        for place in self.places:
            place_status = self.get_place_status(place['name'])
            self.stats['places'][place['name']]['occupancy'].append(place_status['occupancy_rate'])
            self.stats['places'][place['name']]['agents_in_place'].append(place_status['agents_in_place'])
        
        # Record fire statistics (count agents in any active fire radius)
        if self.fire_states:
            agents_in_any_fire = set()
            for fire in self.fire_states:
                if fire.get('active'):
                    for agent in self.agents:
                        if agent.distance_to(fire['position']) <= fire['radius']:
                            agents_in_any_fire.add(agent.id)
            self.stats['agents_in_fire_radius'].append(len(agents_in_any_fire))
        else:
            self.stats['agents_in_fire_radius'].append(0)

        # Store history
        self.history.append({
            'step': self.step,
            'place_status': overall_status,
            'agent_positions': [agent.position for agent in self.agents],
            'agents_in_place': [agent.id for agent in self.get_agents_in_place()],
            'fire_states': list(self.fire_states),
        })
        
        if self.step % LOG_INTERVAL == 0:
            place_info = ", ".join([
                f"{place['name']}: {self.get_place_status(place['name'])['agents_in_place']}"
                for place in self.places
            ])
            logger.info(
                f"Step {self.step}/{self.duration}: "
                f"{agents_in_place} agents in places ({place_info}), "
                f"{overall_status['occupancy_rate']:.1%} overall occupancy"
            )
    
    def run(self):
        """Run the full simulation"""
        logger.info("Starting simulation...")
        
        # Check Ollama connection
        if not self.llm_client.check_connection():
            logger.error("Cannot connect to Ollama. Please make sure Ollama is running.")
            return
        
        # Initialize agents
        self.initialize_agents()
        
        # Run simulation
        try:
            while self.step < self.duration:
                self.step_simulation()
        except KeyboardInterrupt:
            logger.info("Simulation interrupted by user")
        except Exception as e:
            logger.error(f"Error during simulation: {e}", exc_info=True)
        
        logger.info("Simulation completed")
    
    def get_statistics(self) -> Dict:
        """Get simulation statistics"""
        if not self.stats['place_occupancy']:
            return {}
        
        place_occupancy = np.array(self.stats['place_occupancy'])
        agents_in_place = np.array(self.stats['agents_in_place'])
        
        return {
            'mean_occupancy': float(np.mean(place_occupancy)),
            'std_occupancy': float(np.std(place_occupancy)),
            'mean_agents_in_place': float(np.mean(agents_in_place)),
            'max_agents_in_place': int(np.max(agents_in_place)),
            'min_agents_in_place': int(np.min(agents_in_place)),
            'total_steps': self.step
        }

