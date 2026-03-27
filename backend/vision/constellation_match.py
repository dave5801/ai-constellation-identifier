from __future__ import annotations

from vision.catalog_matching import evaluate_catalog_entry
from vision.clustering import cluster_star_fields
from vision.catalog_projection import catalog_star_magnitudes, project_catalog_stars
from vision.geometry import iter_triangle_indices, normalize_points
from vision.matcher_config import AMBIGUOUS_MATCH_SCORE_MARGIN, MATCHES_TO_RETURN
from vision.models import CatalogMatchEvaluation, MatchResult, Star


class ConstellationMatcher:
    def __init__(self, catalog: list[dict]) -> None:
        self.catalog = catalog
        self.prepared_catalog = {
            entry["name"]: {
                "projected_points": project_catalog_stars(entry),
                "magnitudes": catalog_star_magnitudes(entry),
                "normalized_points": normalize_points(project_catalog_stars(entry)),
                "triangles": iter_triangle_indices(len(entry["stars"])),
            }
            for entry in catalog
        }

    def evaluate(self, stars: list[Star], width: int, height: int) -> list[CatalogMatchEvaluation]:
        evaluations: list[CatalogMatchEvaluation] = []
        clusters = cluster_star_fields(stars, width, height)
        for cluster_id, cluster in enumerate(clusters):
            for catalog_entry in self.catalog:
                evaluation = evaluate_catalog_entry(
                    cluster,
                    catalog_entry,
                    cluster_id=cluster_id,
                    prepared_catalog=self.prepared_catalog[catalog_entry["name"]],
                )
                if evaluation is not None:
                    evaluations.append(evaluation)

        evaluations.sort(key=lambda item: item.confidence, reverse=True)
        return evaluations

    def match(self, stars: list[Star], width: int, height: int) -> list[MatchResult]:
        evaluations = self.evaluate(stars, width, height)
        results: list[MatchResult] = []
        seen_names: set[str] = set()
        top_by_cluster: dict[int, list[CatalogMatchEvaluation]] = {}

        for evaluation in evaluations:
            top_by_cluster.setdefault(evaluation.cluster_id, []).append(evaluation)

        ambiguous_clusters: set[int] = set()
        for cluster_id, cluster_evaluations in top_by_cluster.items():
            accepted = [item for item in cluster_evaluations if item.accepted]
            if len(accepted) < 2:
                continue
            accepted.sort(key=lambda item: item.confidence, reverse=True)
            if accepted[0].confidence - accepted[1].confidence < AMBIGUOUS_MATCH_SCORE_MARGIN:
                ambiguous_clusters.add(cluster_id)

        for evaluation in evaluations:
            if (
                not evaluation.accepted
                or evaluation.cluster_id in ambiguous_clusters
                or evaluation.name in seen_names
            ):
                continue
            seen_names.add(evaluation.name)
            results.append(
                MatchResult(
                    name=evaluation.name,
                    confidence=evaluation.confidence,
                    transformed_points=evaluation.transformed_points,
                    connections=evaluation.connections,
                    color=evaluation.color,
                )
            )

        return results[:MATCHES_TO_RETURN]
