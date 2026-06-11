"""Build proposition-network JSON files from extracted propositions."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Iterable, List, Set

import networkx as nx
from tqdm import tqdm


class AlignedPropositionNetwork:
    """Align entity aliases and connect propositions that share entities."""

    def __init__(self, data: Dict):
        self.sample_id = data.get("id", data.get("_id"))
        self.answer = data.get("answer")
        self.question = str(data.get("question", "")).strip()
        self.question_entities = {
            str(entity).strip().lower()
            for entity in data.get("question_entities", [])
            if str(entity).strip()
        }
        self.nodes = [self._normalize_node(node) for node in data.get("nodes", [])]
        self.embeddings = data.get("embeddings", {})
        self.entity_groups = self._align_entities()

    @staticmethod
    def _normalize_node(node: Dict) -> Dict:
        normalized = {
            "id": int(node["id"]),
            "text": str(node.get("text", "")).strip(),
            "entities": {
                str(entity).strip().lower()
                for entity in node.get("entities", [])
                if str(entity).strip()
            },
        }
        if node.get("doc_id") is not None:
            normalized["doc_id"] = node["doc_id"]
        return normalized

    @staticmethod
    def _entities_align(first: str, second: str) -> bool:
        first_words = set(first.split())
        second_words = set(second.split())
        return (
            first_words.issubset(second_words)
            or second_words.issubset(first_words)
            or len(first_words & second_words) >= 2
        )

    def _all_entities(self) -> Set[str]:
        entities = set(self.question_entities)
        for node in self.nodes:
            entities.update(node["entities"])
        return entities

    def _align_entities(self) -> Dict[str, Set[str]]:
        groups: List[Set[str]] = [{entity} for entity in sorted(self._all_entities())]
        changed = True
        while changed:
            changed = False
            for left_index in range(len(groups)):
                for right_index in range(left_index + 1, len(groups)):
                    if any(
                        self._entities_align(left, right)
                        for left in groups[left_index]
                        for right in groups[right_index]
                    ):
                        groups[left_index].update(groups.pop(right_index))
                        changed = True
                        break
                if changed:
                    break
        return {
            max(group, key=lambda value: (len(value), value)): group
            for group in groups
        }

    def _representative(self, entity: str) -> str:
        for representative, members in self.entity_groups.items():
            if entity in members:
                return representative
        return entity

    @staticmethod
    def _add_or_merge_edge(
        graph: nx.Graph,
        source: int,
        target: int,
        edge_type: str,
        shared_entities: Set[str],
    ) -> None:
        if source == target:
            return
        if graph.has_edge(source, target):
            edge = graph[source][target]
            edge["shared_entities"].update(shared_entities)
            if edge_type == "question":
                edge["type"] = "question"
            return
        graph.add_edge(
            source,
            target,
            type=edge_type,
            shared_entities=set(shared_entities),
        )

    def _build_graph(self) -> nx.Graph:
        graph = nx.Graph()
        entity_to_nodes: Dict[str, Set[int]] = {}

        for node in self.nodes:
            aligned = {self._representative(entity) for entity in node["entities"]}
            graph.add_node(node["id"], text=node["text"], entities=aligned)
            for entity in aligned:
                entity_to_nodes.setdefault(entity, set()).add(node["id"])

        for entity, node_ids in entity_to_nodes.items():
            ordered = sorted(node_ids)
            for index, source in enumerate(ordered):
                for target in ordered[index + 1 :]:
                    self._add_or_merge_edge(graph, source, target, "entity", {entity})

        aligned_question_entities = {
            self._representative(entity) for entity in self.question_entities
        }
        question_nodes = sorted(
            {
                node_id
                for entity in aligned_question_entities
                for node_id in entity_to_nodes.get(entity, set())
            }
        )
        for index, source in enumerate(question_nodes):
            for target in question_nodes[index + 1 :]:
                shared = (
                    graph.nodes[source]["entities"] | graph.nodes[target]["entities"]
                ) & aligned_question_entities
                self._add_or_merge_edge(graph, source, target, "question", shared)
        return graph

    def build(self) -> Dict:
        graph = self._build_graph()
        nodes = []
        source_nodes = {node["id"]: node for node in self.nodes}
        for node_id, attributes in graph.nodes(data=True):
            node = {
                "id": node_id,
                "text": attributes["text"],
                "entities": sorted(attributes["entities"]),
            }
            if source_nodes[node_id].get("doc_id") is not None:
                node["doc_id"] = source_nodes[node_id]["doc_id"]
            nodes.append(node)

        edges = [
            {
                "source": source,
                "target": target,
                "type": attributes["type"],
                "shared_entities": sorted(attributes["shared_entities"]),
            }
            for source, target, attributes in graph.edges(data=True)
        ]
        entity_edges = sum(edge["type"] == "entity" for edge in edges)
        question_edges = sum(edge["type"] == "question" for edge in edges)
        result = {
            "question": self.question,
            "question_entities": sorted(
                self._representative(entity) for entity in self.question_entities
            ),
            "network_stats": {
                "num_nodes": len(nodes),
                "num_edges": len(edges),
                "num_entity_edges": entity_edges,
                "num_question_edges": question_edges,
                "avg_degree": (2 * len(edges) / len(nodes)) if nodes else 0.0,
                "connected_components": nx.number_connected_components(graph),
                "entity_groups": len(self.entity_groups),
                "question_entities": len(self.question_entities),
            },
            "nodes": nodes,
            "edges": edges,
            "entity_groups": {
                representative: sorted(members)
                for representative, members in self.entity_groups.items()
            },
            "embeddings": self.embeddings,
        }
        if self.sample_id is not None:
            result["id"] = self.sample_id
        if self.answer is not None:
            result["answer"] = self.answer
        return result


def json_files(directory: Path) -> Iterable[Path]:
    return sorted(path for path in directory.rglob("*.json") if path.is_file())


def process_file(
    source: Path,
    input_dir: Path,
    output_dir: Path,
    *,
    overwrite: bool = False,
) -> Path:
    destination = output_dir / source.relative_to(input_dir)
    if destination.exists() and not overwrite:
        return destination
    data = json.loads(source.read_text(encoding="utf-8"))
    result = AlignedPropositionNetwork(data).build()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return destination


def process_directory(
    input_dir: str | Path,
    output_dir: str | Path,
    num_workers: int = 4,
    *,
    overwrite: bool = False,
) -> None:
    source_root = Path(input_dir).resolve()
    destination_root = Path(output_dir).resolve()
    destination_root.mkdir(parents=True, exist_ok=True)
    files = list(json_files(source_root))
    with ThreadPoolExecutor(max_workers=max(1, num_workers)) as pool:
        futures = [
            pool.submit(
                process_file,
                source,
                source_root,
                destination_root,
                overwrite=overwrite,
            )
            for source in files
        ]
        for future in tqdm(futures, unit="file"):
            future.result()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build proposition-network JSON files.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    process_directory(
        args.input,
        args.output,
        args.workers,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
