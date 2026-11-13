"""Agent Quoridor : minimax alpha-beta, heuristique rapide et table de transposition.

Hypotheses basees sur les interfaces fournies :
- Jeu a deux joueurs.
- GameStateQuoridor expose _legal_moves(), _candidate_blocking_walls(),
  _is_wall_legal(), _opponent() et apply_action().
- Les actions sont des StatelessAction avec data['type'] et data['destination'].

La generation des murs est volontairement selective : tous les mouvements de pion
sont examines, mais seuls les murs proches du corridor adverse sont candidats.
Cela evite d'appeler _is_wall_legal() sur tous les emplacements du plateau.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections import deque
from enum import Enum
import random
from time import perf_counter
from typing import Optional
import math

from seahorse.game.action import Action
from seahorse.game.stateless_action import StatelessAction

from player_quoridor import PlayerQuoridor
from game_state_quoridor import GameStateQuoridor
from actions_quoridor import Wall, Orientation


MATE_SCORE = 1_000_000.0


class Bound(Enum):
    EXACT = 0
    LOWER = 1
    UPPER = 2


@dataclass
class TTEntry:
    signature: tuple
    depth: int
    value: float
    bound: Bound
    best_action: Optional[tuple[str, tuple[int, int]]]


class ZobristHasher:
    """Hash Zobrist 64 bits. Les cles sont generees deterministement et reutilisees."""

    def __init__(self, seed: int = 0x51A7C0DE) -> None:
        self._rng = random.Random(seed)
        self._keys: dict[tuple, int] = {}


    def _key(self, *parts) -> int:
        key = tuple(parts)
        if key not in self._keys:
            self._keys[key] = self._rng.getrandbits(64)
        return self._keys[key]

    def hash_state(self, state: GameStateQuoridor, me_id: int) -> int:
        rep = state.rep
        me = next(p for p in state.players if p.id == me_id)
        opp = state._opponent(me)
        h = self._key("dimension", rep.dimension)

        for side, player in enumerate((me, opp)):
            row, col = rep.pawn_positions[player.id]
            h ^= self._key("pawn", side, row, col)
            h ^= self._key("walls_left", side, rep.remaining_walls[player.id])
            h ^= self._key("goal", side, player.get_goal_row())

        for wall in rep.walls:
            h ^= self._key(
                "wall", wall.row, wall.col, wall.orientation.value
            )

        h ^= self._key("turn", state.active_player.id == me_id)
        return h

    @staticmethod
    def signature(state: GameStateQuoridor, me_id: int) -> tuple:
        """Signature complete pour verifier les collisions Zobrist."""
        rep = state.rep
        me = next(p for p in state.players if p.id == me_id)
        opp = state._opponent(me)
        walls = tuple(sorted(
            (w.row, w.col, w.orientation.value) for w in rep.walls
        ))
        return (
            rep.dimension,
            tuple(rep.pawn_positions[me.id]),
            tuple(rep.pawn_positions[opp.id]),
            rep.remaining_walls[me.id],
            rep.remaining_walls[opp.id],
            me.get_goal_row(),
            opp.get_goal_row(),
            state.active_player.id == me_id,
            walls,
        )


class SearchTimeout(Exception):
    pass


class MyPlayer(PlayerQuoridor):
    # Parametres de recherche ajustables.
    MAX_WALL_CANDIDATES = 20
    WALL_HORIZON = 6
    MAX_TT_ENTRIES = 150_000

    def __init__(
        self,
        piece_type: str,
        goal_row: int = 0,
        name: str = "QuoridorAI",
        *args,
        **kwargs,
    ) -> None:
        super().__init__(piece_type, goal_row, name)
        self._zobrist = ZobristHasher()
        self._tt: dict[int, TTEntry] = {}
        self._deadline = 0.0
        self._nodes = 0
        self._distance_cache = {}
        self.MAX_DISTANCE_CACHE = 512
        self._my_position_history = deque(maxlen=8)

        self._tt_probes = 0
        self._tt_hits = 0
        self._tt_usable_hits = 0
        self._tt_cutoffs = 0
        self._tt_stores = 0
        self._tt_collisions = 0

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def _estimate_remaining_turns(
        self, state: GameStateQuoridor
    ) -> int:
        me = next(p for p in state.players if p.id == self.get_id())
        opp = state._opponent(me)

        my_dist = state._shortest_path(me)
        opp_dist = state._shortest_path(opp)

        if my_dist is None or opp_dist is None:
            return 8

        # Le joueur le plus proche de son objectif détermine
        # généralement la durée restante de la partie.
        min_distance = min(my_dist, opp_dist)

        total_walls = sum(state.rep.remaining_walls.values())

        if min_distance <= 2:
            estimate = min_distance + 1
        else:
            estimate = math.ceil(
                min_distance + min(4.0, total_walls * 0.20) + 2
            )

        return max(2, min(24, estimate))

    def _fast_distance(
        self,
        state: GameStateQuoridor,
        player: PlayerQuoridor
    ) -> int:

        dimension = state.rep.dimension

        walls_key = tuple(sorted(
            (w.row, w.col, w.orientation.value)
            for w in state.rep.walls
        ))

        goal_row = player.get_goal_row()

        cache_key = (dimension, goal_row, walls_key)

        distances = self._distance_cache.get(cache_key)

        if distances is None:

            # BFS multi-source depuis toutes les cases de la ligne d'arrivée.
            distances = [-1] * (dimension * dimension)

            q = deque()

            for col in range(dimension):
                index = goal_row * dimension + col
                distances[index] = 0
                q.append((goal_row, col))

            while q:
                position = q.popleft()

                row, col = position
                current_distance = distances[row * dimension + col]

                for neighbour in state._reachable_neighbours(position):

                    r, c = neighbour
                    index = r * dimension + c

                    if distances[index] == -1:
                        distances[index] = current_distance + 1
                        q.append(neighbour)

            if len(self._distance_cache) >= self.MAX_DISTANCE_CACHE:
                self._distance_cache.pop(
                    next(iter(self._distance_cache))
                )

            self._distance_cache[cache_key] = distances

        row, col = state.rep.pawn_positions[player.id]

        return distances[row * dimension + col]

    def _evaluate(self, state: GameStateQuoridor) -> float:
        """Score du point de vue de notre agent : une valeur positive est bonne."""
        if state.is_done():
            for player in state.players:
                if state.scores.get(player.id, 0.0) == 1.0:
                    return MATE_SCORE if player.id == self.get_id() else -MATE_SCORE
            return 0.0

        me = next(p for p in state.players if p.id == self.get_id())
        opp = state._opponent(me)
        rep = state.rep

        my_walls = rep.remaining_walls[me.id]
        opp_walls = rep.remaining_walls[opp.id]
        total_walls = my_walls + opp_walls

        # Quand plus aucun mur ne reste, le BFS exact devient abordable et utile.
        my_distance = self._fast_distance(state, me)
        opp_distance = self._fast_distance(state, opp)

        # Les chemins ignorent le pion adverse. On ajoute donc explicitement
        # l'effet d'un saut direct disponible au joueur qui doit jouer maintenant.
        active = state.active_player
        active_distance = my_distance if active.id == me.id else opp_distance
        active_row, _ = rep.pawn_positions[active.id]
        active_goal = active.get_goal_row()
        current_remaining = abs(active_goal - active_row)
        for move in state._legal_moves():
            destination_row = move.data["destination"][0]
            remaining_after_move = abs(active_goal - destination_row)
            progress = current_remaining - remaining_after_move
            if progress >= 2:
                # Un saut droit fait gagner une case de progression supplementaire.
                active_distance = max(0, active_distance - 1)
                break

        if active.id == me.id:
            my_distance = active_distance
        else:
            opp_distance = active_distance

        # En fin de partie sans murs, tenir compte du joueur qui a le trait :
        # a distance egale, le joueur qui joue maintenant arrive un demi-coup avant.
        if total_walls == 0:
            my_arrival = 2 * my_distance - (1 if active.id == me.id else 0)
            opp_arrival = 2 * opp_distance - (1 if active.id == opp.id else 0)
            score = 50.0 * (opp_arrival - my_arrival)
        else:
            score = 100.0 * (opp_distance - my_distance)

        # Les murs conservent une valeur : leur poids augmente quand ils se raréfient.
        wall_weight = 5.0 + 0.5 * (20 - min(total_walls, 20))
        score += wall_weight * (my_walls - opp_walls)

        # Leger avantage de flexibilite pour une position centrale.
        center = (rep.dimension - 1) / 2.0
        my_col = rep.pawn_positions[me.id][1]
        opp_col = rep.pawn_positions[opp.id][1]
        score += 2.0 * (abs(opp_col - center) - abs(my_col - center))

        return score

    # ------------------------------------------------------------------
    # Generation selective des actions
    # ------------------------------------------------------------------

    @staticmethod
    def _action_key(action: Optional[StatelessAction]):
        if action is None:
            return None
        return (
            action.data["type"],
            tuple(action.data["destination"]),
        )

    @staticmethod
    def _action_from_key(key: Optional[tuple]) -> Optional[StatelessAction]:
        if key is None:
            return None
        action_type, destination = key
        return StatelessAction({
            "type": action_type,
            "destination": tuple(destination),
        })

    @staticmethod
    def _wall_in_bounds(wall: Wall, dimension: int) -> bool:
        if wall.orientation == Orientation.HORIZONTAL:
            return 0 <= wall.row < dimension and 0 <= wall.col < dimension - 1
        return 0 <= wall.row < dimension - 1 and 0 <= wall.col < dimension

    def _candidate_walls(self, state: GameStateQuoridor) -> list[Wall]:
        """Construit des murs candidats autour du corridor du joueur a bloquer."""
        active = state.active_player
        if state.rep.remaining_walls[active.id] <= 0:
            return []

        target = state._opponent(active)
        row, col = state.rep.pawn_positions[target.id]
        goal = target.get_goal_row()
        if row == goal:
            return []

        direction = 1 if goal > row else -1
        dimension = state.rep.dimension
        candidates: dict[Wall, int] = {}

        # Murs horizontaux touchant les prochaines transitions vers l'objectif.
        for step in range(min(self.WALL_HORIZON, abs(goal - row))):
            edge_row = row + direction * step
            next_row = edge_row + direction
            for lane_col in (col - 1, col, col + 1):
                if not (0 <= lane_col < dimension):
                    continue
                start = (edge_row, lane_col)
                end = (next_row, lane_col)
                for wall in state._candidate_blocking_walls(start, end):
                    if self._wall_in_bounds(wall, dimension):
                        priority = (
                            abs(wall.row - row) + abs(wall.col - col)
                        )
                        candidates[wall] = min(
                            priority, candidates.get(wall, 10**6)
                        )

        # Quelques murs verticaux pour limiter les changements de couloir.
        for edge_row in (row, row + direction):
            if not 0 <= edge_row < dimension:
                continue
            for left_col in range(max(0, col - 2), min(dimension - 1, col + 2)):
                start = (edge_row, left_col)
                end = (edge_row, left_col + 1)
                for wall in state._candidate_blocking_walls(start, end):
                    if self._wall_in_bounds(wall, dimension):
                        priority = (
                            abs(wall.row - row) + abs(wall.col - col) + 1
                        )
                        candidates[wall] = min(
                            priority, candidates.get(wall, 10**6)
                        )

        # Garder aussi quelques murs verticaux : ils peuvent fermer une epaule
        # et rendre beaucoup plus couteux le contournement d'un mur horizontal.
        horizontal = sorted(
            (w for w in candidates if w.orientation == Orientation.HORIZONTAL),
            key=lambda wall: candidates[wall],
        )
        vertical = sorted(
            (w for w in candidates if w.orientation == Orientation.VERTICAL),
            key=lambda wall: candidates[wall],
        )
        selected = horizontal[:8] + vertical[:4]
        return sorted(selected, key=lambda wall: candidates[wall])[: self.MAX_WALL_CANDIDATES]

    def _generate_actions(self, state: GameStateQuoridor) -> list[StatelessAction]:
        """Tous les mouvements, plus un sous-ensemble de murs legalement verifies."""
        actions = list(state._legal_moves())
        active = state.active_player

        if state.rep.remaining_walls[active.id] > 0:
            for wall in self._candidate_walls(state):
                self._check_time()
                # _is_wall_legal verifie chevauchement, croisement et chemins restants.
                if state._is_wall_legal(wall):
                    wall_type = (
                        "horizontal"
                        if wall.orientation == Orientation.HORIZONTAL
                        else "vertical"
                    )
                    actions.append(StatelessAction({
                        "type": wall_type,
                        "destination": (wall.row, wall.col),
                    }))

        return actions

    def _ordered_children(
        self,
        state: GameStateQuoridor,
        maximizing: bool,
        preferred_action: Optional[tuple],
    ) -> list[tuple[StatelessAction, GameStateQuoridor]]:
        """Ordre les successeurs par evaluation rapide (meilleurs d'abord)."""
        children = []
        for action in self._generate_actions(state):
            self._check_time()
            child = state.apply_action(action)
            action_key = self._action_key(action)
            score = self._evaluate(child)
            children.append((action, child, action_key == preferred_action, score))

        # Le meilleur coup connu de la TT est toujours explore en premier.
        children.sort(
            key=lambda item: (
                not item[2],
                -item[3] if maximizing else item[3],
            )
        )
        return [(action, child) for action, child, _, _ in children]

    # ------------------------------------------------------------------
    # Table de transposition
    # ------------------------------------------------------------------

    def _probe_tt(self, state):
        self._tt_probes += 1

        key = self._zobrist.hash_state(state, self.get_id())
        signature = self._zobrist.signature(state, self.get_id())

        entry = self._tt.get(key)

        if entry is None:
            return key, signature, None

        if entry.signature != signature:
            self._tt_collisions += 1
            return key, signature, None

        self._tt_hits += 1

        return key, signature, entry

    def _store_tt(
        self,
        key: int,
        signature: tuple,
        depth: int,
        value: float,
        alpha_original: float,
        beta_original: float,
        best_action: Optional[StatelessAction],
    ) -> None:
        if value <= alpha_original:
            bound = Bound.UPPER
        elif value >= beta_original:
            bound = Bound.LOWER
        else:
            bound = Bound.EXACT

        old = self._tt.get(key)
        if old is None or old.signature != signature or depth >= old.depth:
            self._tt[key] = TTEntry(
                signature=signature,
                depth=depth,
                value=value,
                bound=bound,
                best_action=self._action_key(best_action),
            )
            self._tt_stores += 1

        if len(self._tt) > self.MAX_TT_ENTRIES:
            oldest_key = next(iter(self._tt))
            del self._tt[oldest_key]

    # ------------------------------------------------------------------
    # Minimax alpha-beta
    # ------------------------------------------------------------------

    def _check_time(self) -> None:
        self._nodes += 1
        if (self._nodes & 0x7F) == 0 and perf_counter() >= self._deadline:
            raise SearchTimeout

    def max_value(
        self,
        state: GameStateQuoridor,
        depth: int,
        alpha: float,
        beta: float,
    ) -> tuple[float, Optional[StatelessAction]]:
        """Valeur MAX : notre agent cherche a maximiser son evaluation."""
        self._check_time()
        if state.is_done() or depth == 0:
            return self._evaluate(state), None

        alpha_original, beta_original = alpha, beta
        key, signature, entry = self._probe_tt(state)
        preferred_action = None

        if entry is not None:
            preferred_action = entry.best_action

            if entry.depth >= depth:
                self._tt_usable_hits += 1

                if entry.bound == Bound.EXACT:
                    self._tt_cutoffs += 1
                    return (
                        entry.value,
                        self._action_from_key(entry.best_action)
                    )

                if entry.bound == Bound.LOWER:
                    alpha = max(alpha, entry.value)

                elif entry.bound == Bound.UPPER:
                    beta = min(beta, entry.value)

                if alpha >= beta:
                    self._tt_cutoffs += 1
                    return (
                        entry.value,
                        self._action_from_key(entry.best_action)
                    )

        best_value = float("-inf")
        best_action = None
        children = self._ordered_children(state, True, preferred_action)

        if not children:
            return self._evaluate(state), None

        for action, child in children:
            value, _ = self.min_value(child, depth - 1, alpha, beta)
            if value > best_value:
                best_value = value
                best_action = action
            alpha = max(alpha, best_value)
            if alpha >= beta:
                break

        self._store_tt(
            key, signature, depth, best_value,
            alpha_original, beta_original, best_action,
        )
        return best_value, best_action

    def min_value(
        self,
        state: GameStateQuoridor,
        depth: int,
        alpha: float,
        beta: float,
    ) -> tuple[float, Optional[StatelessAction]]:
        """Valeur MIN : l'adversaire cherche a minimiser notre evaluation."""
        self._check_time()
        if state.is_done() or depth == 0:
            return self._evaluate(state), None

        alpha_original, beta_original = alpha, beta
        key, signature, entry = self._probe_tt(state)
        preferred_action = None

        if entry is not None:
            preferred_action = entry.best_action

            if entry.depth >= depth:
                self._tt_usable_hits += 1

                if entry.bound == Bound.EXACT:
                    self._tt_cutoffs += 1
                    return (
                        entry.value,
                        self._action_from_key(entry.best_action)
                    )

                if entry.bound == Bound.LOWER:
                    alpha = max(alpha, entry.value)

                elif entry.bound == Bound.UPPER:
                    beta = min(beta, entry.value)

                if alpha >= beta:
                    self._tt_cutoffs += 1
                    return (
                        entry.value,
                        self._action_from_key(entry.best_action)
                    )

        best_value = float("inf")
        best_action = None
        children = self._ordered_children(state, False, preferred_action)

        if not children:
            return self._evaluate(state), None

        for action, child in children:
            value, _ = self.max_value(child, depth - 1, alpha, beta)
            if value < best_value:
                best_value = value
                best_action = action
            beta = min(beta, best_value)
            if beta <= alpha:
                break

        self._store_tt(
            key, signature, depth, best_value,
            alpha_original, beta_original, best_action,
        )
        return best_value, best_action


    def _avoid_oscillation(self, state, action):

        if action is None or action.data["type"] != "move":
            return action

        if len(self._my_position_history) < 2:
            return action

        previous_position = self._my_position_history[-2]

        destination = tuple(action.data["destination"])

        # Le meilleur coup revient sur notre position d'il y a deux tours.
        if destination != previous_position:
            return action

        alternatives = [
            a for a in state._legal_moves()
            if tuple(a.data["destination"]) != previous_position
        ]

        if not alternatives:
            return action

        current_score = self._evaluate(
            state.apply_action(action)
        )

        alternative_scores = [
            (self._evaluate(state.apply_action(a)), a)
            for a in alternatives
        ]

        best_alternative_score, best_alternative = max(
            alternative_scores,
            key=lambda item: item[0]
        )

        # On accepte le détour si son évaluation reste raisonnablement proche.
        if best_alternative_score >= current_score - 50:
            return best_alternative

        return action
    # ------------------------------------------------------------------
    # Ouverture, fin de partie et choix final
    # ------------------------------------------------------------------



    def _opening_book(self, state: GameStateQuoridor) -> Optional[StatelessAction]:
        """Marche centrale simple pendant les six premiers demi-coups."""
        if state.step >= 6 or state.rep.walls:
            return None

        dimension = state.rep.dimension
        center = dimension // 2
        if any(state.rep.pawn_positions[p.id][1] != center for p in state.players):
            return None

        me = next(p for p in state.players if p.id == self.get_id())
        row, col = state.rep.pawn_positions[me.id]
        direction = 1 if me.get_goal_row() > row else -1
        forward = (row + direction, col)

        if not (0 <= forward[0] < dimension):
            return None

        for action in state._legal_moves():
            if tuple(action.data["destination"]) == forward:
                return action
        return None

    @staticmethod
    def _maximum_depth(state: GameStateQuoridor) -> int:
        total_walls = sum(state.rep.remaining_walls.values())
        if total_walls == 0:
            return 34  
        if total_walls <= 4:
            return 25
        return 20

    @staticmethod
    def _format_action(action: Optional[StatelessAction]) -> str:
        if action is None:
            return "None"
        action_type = action.data.get("type", "?")
        destination = tuple(action.data.get("destination", ()))
        if action_type == "move":
            return f"move vers {destination}"
        return f"mur {action_type} en {destination}"

    def _log_decision(
        self,
        mode: str,
        action: Optional[StatelessAction],
        elapsed: float,
        completed_depth: int = 0,
        minimax_value: Optional[float] = None,
        state: Optional[GameStateQuoridor] = None,
    ) -> None:
        heuristic_value = None
        if state is not None and action is not None:
            try:
                heuristic_value = self._evaluate(state.apply_action(action))
            except Exception:
                # Le log ne doit jamais empecher de retourner un coup valide.
                heuristic_value = None

        probes = self._tt_probes
        hit_rate = (100.0 * self._tt_hits / probes) if probes else 0.0
        minimax_text = (
            f"{minimax_value:.2f}" if minimax_value is not None else "n/a"
        )
        heuristic_text = (
            f"{heuristic_value:.2f}" if heuristic_value is not None else "n/a"
        )

        print(
            f"[SEARCH] Mode={mode} | "
            f"Action={self._format_action(action)} | "
            f"Depth={completed_depth} | Nodes={self._nodes} | "
            f"Time={elapsed:.2f}s | Minimax={minimax_text} | "
            f"H(apres coup)={heuristic_text} | "
            f"TT probes={self._tt_probes} | Hits={self._tt_hits} "
            f"({hit_rate:.1f}%) | Usable={self._tt_usable_hits} | "
            f"Cutoffs={self._tt_cutoffs} | Stores={self._tt_stores} | "
            f"Collisions={self._tt_collisions} | TT size={len(self._tt)}",
            flush=True,
        )

    def compute_action(
        self,
        current_state: GameStateQuoridor,
        remaining_time: float = 15 * 60,
        **kwargs,
    ) -> Action:
        """Recherche iterative : conserve le meilleur coup de la profondeur terminee."""
        decision_start = perf_counter()

        self._nodes = 0
        self._tt_probes = 0
        self._tt_hits = 0
        self._tt_usable_hits = 0
        self._tt_cutoffs = 0
        self._tt_stores = 0
        self._tt_collisions = 0

        current_position = tuple(
            current_state.rep.pawn_positions[self.get_id()]
        )

        if (
            not self._my_position_history
            or self._my_position_history[-1] != current_position
        ):
            self._my_position_history.append(current_position)

        if current_state.active_player.id != self.get_id():
            # Le moteur devrait appeler l'agent uniquement pendant son tour.
            actions = current_state._legal_moves()
            return actions[0]

        opening_action = self._opening_book(current_state)
        if opening_action is not None:
            return opening_action

        fallback_actions = current_state._legal_moves()
        if not fallback_actions:
            raise RuntimeError("Aucune action legale disponible.")
        best_action: Action = fallback_actions[0]
        best_value: Optional[float] = None

        # remaining_time est suppose etre exprime en secondes.
        time_left = max(0.0, float(remaining_time))

        remaining_turns = self._estimate_remaining_turns(current_state)

        # On conserve une réserve de sécurité de 8 % et 20 secondes.
        usable_time = max(0.0, 0.92 * time_left - 20.0)

        time_budget = min(
            300.0,
            max(0.05, usable_time / remaining_turns)
        )

        self._deadline = perf_counter() + time_budget
        self._nodes = 0
        max_depth = self._maximum_depth(current_state)

        search_start = perf_counter()
        completed_depth = 0
        for depth in range(1, max_depth + 1):
            try:
                value, action = self.max_value(
                    current_state,
                    depth,
                    float("-inf"),
                    float("inf"),
                )
                if action is not None:
                    best_action = action
                    best_value = value
                    completed_depth = depth
            except SearchTimeout:
                break
        
        elapsed = perf_counter() - search_start
        print(
            f"[SEARCH] "
            f"Depth={completed_depth} | "
            f"Nodes={self._nodes} | "
            f"Time={elapsed:.2f}s | "
            f"TT={len(self._tt)}"
        )

        best_action = self._avoid_oscillation(
            current_state,
            best_action
        )

        self._log_decision(
            "iterative_minimax",
            best_action,
            perf_counter() - decision_start,
            completed_depth=completed_depth,
            minimax_value=best_value,
            state=current_state,
        )


        return best_action
