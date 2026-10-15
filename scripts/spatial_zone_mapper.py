#!/usr/bin/env python3
"""Drishti-Edge: Spatial Zone Mapping & Zone Vulnerability Index (ZVI) Engine

Phase 0.6: Local Intelligence Dashboard, Zone Mapping & ZVI Integration

Core Responsibilities:
  1. Define canonical 3x3 spatial grid over 1280x720 video frames.
  2. Deterministically map object detections (persons, vehicles, masks) into zones.
  3. Extract localized environmental lighting deficits per zone.
  4. Compute Zone Vulnerability Index (ZVI):
       ZVI = 100 * (0.50*L + 0.35*I + 0.15*M)
     strictly enforcing component availability, bounds checking, and failure transparency.
  5. Accumulate video-level zone telemetry without persisting identifiable media.
  6. Generate scene-level spatial heatmaps and overlay visualizations.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np

# Canonical prototype resolution
CANONICAL_WIDTH = 1280
CANONICAL_HEIGHT = 720
GRID_ROWS = 3


def discover_local_videos(videos_dir: Optional[Path] = None) -> List[Path]:
    """Safely discover supported local test video files in data/videos/."""
    if videos_dir is None:
        project_root = Path(__file__).resolve().parent.parent
        videos_dir = project_root / "data" / "videos"
    if not videos_dir.is_dir():
        return []
    valid_extensions = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    video_files = [
        f for f in sorted(videos_dir.iterdir())
        if f.is_file() and f.suffix.lower() in valid_extensions and not f.name.startswith(".")
    ]
    return video_files


def get_video_metadata(video_path: Path) -> Optional[Dict[str, Any]]:
    """Extract basic non-sensitive video metadata using OpenCV."""
    if not isinstance(video_path, Path):
        video_path = Path(video_path)
    if not video_path.is_file():
        return None
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / fps if fps > 0 else 0.0
    cap.release()

    return {
        "filename": video_path.name,
        "path": str(video_path),
        "width": width,
        "height": height,
        "fps": round(fps, 2),
        "total_frames": total_frames,
        "duration_sec": round(duration_sec, 2),
        "size_mb": round(video_path.stat().st_size / (1024 * 1024), 2),
    }
GRID_COLS = 3
TOTAL_ZONES = GRID_ROWS * GRID_COLS

# Spatial 3x3 Grid Boundaries
# Width: 1280 // 3 -> [0..426], [427..853], [854..1280]
# Height: 720 // 3 -> [0..240], [241..480], [481..720]
ZONE_DEFINITIONS: Dict[int, Dict[str, Any]] = {
    1: {"name": "Zone 1 (Top-Left)", "box": (0, 0, 426, 240), "row": 0, "col": 0},
    2: {"name": "Zone 2 (Top-Center)", "box": (426, 0, 853, 240), "row": 0, "col": 1},
    3: {"name": "Zone 3 (Top-Right)", "box": (854, 0, 1280, 240), "row": 0, "col": 2},
    4: {"name": "Zone 4 (Mid-Left)", "box": (0, 240, 426, 480), "row": 1, "col": 0},
    5: {"name": "Zone 5 (Mid-Center)", "box": (426, 240, 853, 480), "row": 1, "col": 1},
    6: {"name": "Zone 6 (Mid-Right)", "box": (854, 240, 1280, 480), "row": 1, "col": 2},
    7: {"name": "Zone 7 (Bottom-Left)", "box": (0, 480, 426, 720), "row": 2, "col": 0},
    8: {"name": "Zone 8 (Bottom-Center)", "box": (426, 480, 853, 720), "row": 2, "col": 1},
    9: {"name": "Zone 9 (Bottom-Right)", "box": (854, 480, 1280, 720), "row": 2, "col": 2},
}


def map_point_to_zone(x: float, y: float) -> int:
    """Map canonical 2D pixel coordinates (x, y) to zone ID (1..9).
    
    Coordinates are clamped strictly within [0..CANONICAL_WIDTH, 0..CANONICAL_HEIGHT].
    """
    clamped_x = max(0.0, min(float(CANONICAL_WIDTH), float(x)))
    clamped_y = max(0.0, min(float(CANONICAL_HEIGHT), float(y)))

    col_w = CANONICAL_WIDTH / GRID_COLS
    row_h = CANONICAL_HEIGHT / GRID_ROWS

    col = min(GRID_COLS - 1, max(0, int(clamped_x / col_w)))
    row = min(GRID_ROWS - 1, max(0, int(clamped_y / row_h)))

    return row * GRID_COLS + col + 1


def map_box_to_zone(box: Tuple[int, int, int, int]) -> int:
    """Map a bounding box (x1, y1, x2, y2) to its primary zone using box center."""
    x1, y1, x2, y2 = box[:4]
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    return map_point_to_zone(cx, cy)


def compute_box_zone_overlap(box: Tuple[int, int, int, int], zone_id: int) -> float:
    """Compute the fraction of bounding box area intersecting the specified zone."""
    if zone_id not in ZONE_DEFINITIONS:
        return 0.0

    bx1, by1, bx2, by2 = box[:4]
    box_w = max(0, bx2 - bx1)
    box_h = max(0, by2 - by1)
    box_area = box_w * box_h
    if box_area == 0:
        return 0.0

    zx1, zy1, zx2, zy2 = ZONE_DEFINITIONS[zone_id]["box"]

    ix1 = max(bx1, zx1)
    iy1 = max(by1, zy1)
    ix2 = min(bx2, zx2)
    iy2 = min(by2, zy2)

    inter_w = max(0, ix2 - ix1)
    inter_h = max(0, iy2 - iy1)
    inter_area = inter_w * inter_h

    return float(inter_area / box_area)


def calculate_zone_lighting_deficit(
    frame: np.ndarray,
    zone_box: Tuple[int, int, int, int],
) -> Tuple[float, float]:
    """Calculate mean brightness and lighting deficit for a specific zone crop.
    
    Formula:
      V = mean(V channel of HSV) in [0..255]
      L = 1.0 - (V / 255.0) in [0.0..1.0]
    Returns: (mean_v, lighting_deficit)
    """
    zx1, zy1, zx2, zy2 = zone_box
    h, w = frame.shape[:2]
    cx1 = max(0, min(w, zx1))
    cy1 = max(0, min(h, zy1))
    cx2 = max(0, min(w, zx2))
    cy2 = max(0, min(h, zy2))

    if cx2 <= cx1 or cy2 <= cy1:
        return 0.0, 1.0

    crop = frame[cy1:cy2, cx1:cx2]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    v_channel = hsv[:, :, 2]
    mean_v = float(np.mean(v_channel))
    lighting_deficit = float(1.0 - (mean_v / 255.0))
    return mean_v, max(0.0, min(1.0, lighting_deficit))


def calculate_mobility_factor(
    person_count: int,
    two_wheeler_count: int,
    four_wheeler_count: int = 0,
    norm_capacity: float = 5.0,
) -> float:
    """Calculate normalized mobility exposure factor (M) for a zone.
    
    Represents the spatial concentration of vulnerable road users (pedestrians and two-wheelers)
    co-occurring with vehicular movement:
      raw_M = (1.0 * N_person + 0.5 * N_2w + 0.25 * N_4w) / norm_capacity
      M = min(1.0, max(0.0, raw_M)) in [0.0, 1.0]
    """
    norm_cap = max(1.0, float(norm_capacity))
    weighted_sum = (
        1.0 * max(0, person_count)
        + 0.5 * max(0, two_wheeler_count)
        + 0.25 * max(0, four_wheeler_count)
    )
    return float(max(0.0, min(1.0, weighted_sum / norm_cap)))


def compute_zvi(
    lighting_deficit: Optional[float],
    infrastructure_hazard: Optional[float],
    mobility_factor: Optional[float],
) -> Dict[str, Any]:
    """Compute the Zone Vulnerability Index (ZVI) prototype formula:
    
      ZVI = 100 * (0.50*L + 0.35*I + 0.15*M)
    
    Parameters:
      lighting_deficit (L): Environmental darkness deficit in [0.0, 1.0].
      infrastructure_hazard (I): Infrastructure defect / physical hazard factor in [0.0, 1.0].
                                 (Requires MANUAL auditor assessment; marked UNAVAILABLE if None).
      mobility_factor (M): Mobility exposure / pedestrian density in [0.0, 1.0].
    
    Guarantees:
      - Strictly validates all inputs are numeric and within [0.0, 1.0].
      - Does NOT silently substitute zero for missing components.
      - Never presents a partial calculation as a complete ZVI.
      - Discloses formula, component breakdown, and heuristic status.
    """
    formula_str = "ZVI = 100 * (0.50*L + 0.35*I + 0.15*M)"
    missing_components = []

    # Validate L
    L_val: Optional[float] = None
    if lighting_deficit is None:
        missing_components.append("L (Lighting Deficit: not computed)")
    else:
        try:
            L_val = max(0.0, min(1.0, float(lighting_deficit)))
        except (ValueError, TypeError):
            missing_components.append("L (Lighting Deficit: invalid numeric value)")

    # Validate I
    I_val: Optional[float] = None
    if infrastructure_hazard is None:
        missing_components.append(
            "I (Infrastructure Condition: requires manual auditor inspection; "
            "no automated road defect detector is loaded in this prototype)"
        )
    else:
        try:
            I_val = max(0.0, min(1.0, float(infrastructure_hazard)))
        except (ValueError, TypeError):
            missing_components.append("I (Infrastructure Condition: invalid numeric value)")

    # Validate M
    M_val: Optional[float] = None
    if mobility_factor is None:
        missing_components.append("M (Mobility Exposure: not computed)")
    else:
        try:
            M_val = max(0.0, min(1.0, float(mobility_factor)))
        except (ValueError, TypeError):
            missing_components.append("M (Mobility Exposure: invalid numeric value)")

    # If any required component is missing, return transparent INCOMPLETE status
    if missing_components:
        return {
            "zvi": None,
            "status": "INCOMPLETE",
            "is_complete": False,
            "missing_components": missing_components,
            "components": {
                "L": L_val,
                "I": I_val,
                "M": M_val,
            },
            "formula": formula_str,
            "explanation": (
                "ZVI score calculation is incomplete. "
                f"Missing or unassessed components: {'; '.join(missing_components)}. "
                "No synthetic values or zeroes have been substituted."
            ),
            "disclaimer": (
                "Heuristic index for local urban safety auditing only. "
                "Not a scientifically calibrated measure of crime or accident probability."
            ),
        }

    # All three components are valid and populated
    assert L_val is not None and I_val is not None and M_val is not None

    l_contrib = 50.0 * L_val
    i_contrib = 35.0 * I_val
    m_contrib = 15.0 * M_val
    score = round(l_contrib + i_contrib + m_contrib, 2)
    score = max(0.0, min(100.0, score))

    # Identify primary contributing driver
    contribs = [
        ("Lighting Deficit (50% weight)", l_contrib),
        ("Infrastructure Hazard (35% weight)", i_contrib),
        ("Mobility Exposure (15% weight)", m_contrib),
    ]
    contribs.sort(key=lambda item: item[1], reverse=True)
    primary_driver = contribs[0][0]

    return {
        "zvi": score,
        "status": "COMPLETE",
        "is_complete": True,
        "missing_components": [],
        "components": {
            "L": L_val,
            "I": I_val,
            "M": M_val,
        },
        "contributions": {
            "L_contribution": round(l_contrib, 2),
            "I_contribution": round(i_contrib, 2),
            "M_contribution": round(m_contrib, 2),
        },
        "formula": formula_str,
        "primary_driver": primary_driver,
        "explanation": (
            f"ZVI Score: {score:.2f} / 100. "
            f"Primary factor: {primary_driver} ({contribs[0][1]:.2f} pts). "
            f"Component values: L={L_val:.3f}, I={I_val:.3f} (manual), M={M_val:.3f}."
        ),
        "disclaimer": (
            "Heuristic index for comparative local auditing only. "
            "Not a scientifically calibrated measure of crime or accident probability."
        ),
    }


@dataclass
class ZoneTelemetry:
    """Accumulator for zone-level metrics across a video sequence."""
    zone_id: int
    name: str
    box: Tuple[int, int, int, int]
    frames_count: int = 0
    lighting_sum: float = 0.0
    persons_sum: int = 0
    vehicles_2w_sum: int = 0
    vehicles_4w_sum: int = 0
    peak_persons: int = 0
    peak_vehicles: int = 0
    masks_sum: int = 0
    obscured_fraction_sum: float = 0.0

    @property
    def mean_lighting_deficit(self) -> float:
        return self.lighting_sum / self.frames_count if self.frames_count > 0 else 0.0

    @property
    def mean_persons(self) -> float:
        return self.persons_sum / self.frames_count if self.frames_count > 0 else 0.0

    @property
    def mean_vehicles(self) -> float:
        tot_v = self.vehicles_2w_sum + self.vehicles_4w_sum
        return tot_v / self.frames_count if self.frames_count > 0 else 0.0

    @property
    def mean_mobility_factor(self) -> float:
        if self.frames_count == 0:
            return 0.0
        return calculate_mobility_factor(
            int(round(self.mean_persons)),
            int(round(self.vehicles_2w_sum / self.frames_count)),
            int(round(self.vehicles_4w_sum / self.frames_count)),
        )


class SpatialZoneMapper:
    """Manages spatial zone mapping, temporal accumulation, and scene heatmap generation."""

    def __init__(self) -> None:
        self.zones: Dict[int, ZoneTelemetry] = {
            zid: ZoneTelemetry(
                zone_id=zid,
                name=zdef["name"],
                box=zdef["box"],
            )
            for zid, zdef in ZONE_DEFINITIONS.items()
        }
        self.total_processed_frames = 0
        self.total_fail_closed_frames = 0

    def reset(self) -> None:
        """Clear all accumulated statistics across video runs."""
        self.__init__()

    def analyze_frame(
        self,
        canonical_frame: np.ndarray,
        metrics: Dict[str, Any],
        manual_infrastructure: Optional[float] = None,
        accumulate: bool = True,
    ) -> Dict[str, Any]:
        """Analyze a single canonical 1280x720 frame across all 9 spatial zones.
        
        Extracts:
          - Zone lighting deficit
          - Bounding-box zone assignments for persons, vehicles, and redaction masks
          - Mobility exposure
          - ZVI score per zone
        """
        if accumulate:
            self.total_processed_frames += 1

        # Extract entity bounding boxes from privacy engine metrics
        person_boxes = metrics.get("person_boxes", [])
        vehicle_boxes = metrics.get("vehicle_boxes", [])
        masks_detail = metrics.get("masks_detail", [])

        # Categorize vehicles into 2W/3W vs 4W
        v_2w_boxes = []
        v_4w_boxes = []
        for v in vehicle_boxes:
            v_box = v[:4]
            cls_id = v[4] if len(v) > 4 else None
            vw = v_box[2] - v_box[0]
            vh = v_box[3] - v_box[1]
            is_2w3w = (
                cls_id == 3
                or (cls_id is not None and cls_id in {1, 2, 7} and vw / max(1, vh) <= 0.85)
            )
            if is_2w3w:
                v_2w_boxes.append(v_box)
            else:
                v_4w_boxes.append(v_box)

        frame_zone_results: Dict[int, Dict[str, Any]] = {}

        for zid, zdef in ZONE_DEFINITIONS.items():
            zbox = zdef["box"]
            _, l_def = calculate_zone_lighting_deficit(canonical_frame, zbox)

            # Count persons mapped to this zone
            z_persons = [p for p in person_boxes if map_box_to_zone(p) == zid]
            p_count = len(z_persons)

            # Count 2W/3W vehicles mapped to this zone
            z_2w = [v for v in v_2w_boxes if map_box_to_zone(v) == zid]
            v2w_count = len(z_2w)

            # Count 4W vehicles mapped to this zone
            z_4w = [v for v in v_4w_boxes if map_box_to_zone(v) == zid]
            v4w_count = len(z_4w)

            # Count redaction masks mapped to this zone
            z_masks = [m for m in masks_detail if map_box_to_zone(m["box"]) == zid]
            m_count = len(z_masks)

            # Compute zone mobility factor
            m_factor = calculate_mobility_factor(p_count, v2w_count, v4w_count)

            # Compute ZVI for zone
            zvi_res = compute_zvi(l_def, manual_infrastructure, m_factor)

            # Accumulate
            if accumulate:
                zt = self.zones[zid]
                zt.frames_count += 1
                zt.lighting_sum += l_def
                zt.persons_sum += p_count
                zt.vehicles_2w_sum += v2w_count
                zt.vehicles_4w_sum += v4w_count
                zt.peak_persons = max(zt.peak_persons, p_count)
                zt.peak_vehicles = max(zt.peak_vehicles, v2w_count + v4w_count)
                zt.masks_sum += m_count

            frame_zone_results[zid] = {
                "zone_id": zid,
                "name": zdef["name"],
                "box": zbox,
                "lighting_deficit": round(l_def, 4),
                "person_count": p_count,
                "vehicle_2w_count": v2w_count,
                "vehicle_4w_count": v4w_count,
                "vehicle_total_count": v2w_count + v4w_count,
                "mask_count": m_count,
                "mobility_factor": round(m_factor, 4),
                "zvi_result": zvi_res,
            }

        return {
            "frame_zones": frame_zone_results,
            "scene_lighting_deficit": metrics.get("lighting_deficit", 0.0),
            "scene_person_count": len(person_boxes),
            "scene_vehicle_count": len(vehicle_boxes),
            "scene_mask_count": len(masks_detail),
            "scene_zvi": compute_zvi(
                metrics.get("lighting_deficit", 0.0),
                manual_infrastructure,
                calculate_mobility_factor(len(person_boxes), len(v_2w_boxes), len(v_4w_boxes)),
            ),
        }

    def get_video_summary(self, manual_infrastructure: Optional[float] = None) -> Dict[str, Any]:
        """Generate complete video-level spatial zone summary across all processed frames."""
        summary_zones = {}
        for zid, zt in self.zones.items():
            zvi_agg = compute_zvi(
                zt.mean_lighting_deficit,
                manual_infrastructure,
                zt.mean_mobility_factor,
            )
            summary_zones[zid] = {
                "zone_id": zid,
                "name": zt.name,
                "box": zt.box,
                "frames_evaluated": zt.frames_count,
                "mean_lighting_deficit": round(zt.mean_lighting_deficit, 4),
                "mean_persons": round(zt.mean_persons, 2),
                "peak_persons": zt.peak_persons,
                "total_persons_observed": zt.persons_sum,
                "mean_vehicles": round(zt.mean_vehicles, 2),
                "peak_vehicles": zt.peak_vehicles,
                "total_vehicles_observed": zt.vehicles_2w_sum + zt.vehicles_4w_sum,
                "total_masks_applied": zt.masks_sum,
                "mean_mobility_factor": round(zt.mean_mobility_factor, 4),
                "zvi_result": zvi_agg,
            }

        return {
            "total_frames_processed": self.total_processed_frames,
            "zones_summary": summary_zones,
        }

    def render_spatial_heatmap(
        self,
        base_frame: np.ndarray,
        metric: str = "lighting",
        manual_infrastructure: Optional[float] = None,
        alpha: float = 0.35,
    ) -> np.ndarray:
        """Render the 3x3 spatial grid and color-coded heatmap onto a canonical 1280x720 canvas.
        
        metric: 'lighting' (L), 'mobility' (M), or 'zvi' (ZVI).
        """
        vis = base_frame.copy()
        if vis.shape[:2] != (CANONICAL_HEIGHT, CANONICAL_WIDTH):
            vis = cv2.resize(vis, (CANONICAL_WIDTH, CANONICAL_HEIGHT), interpolation=cv2.INTER_AREA)

        overlay = vis.copy()

        # Color mapping helper: 0.0 (green) -> 0.5 (yellow/amber) -> 1.0 (red)
        def get_heat_color(val: Optional[float]) -> Tuple[int, int, int]:
            if val is None:
                return (128, 128, 128)  # Gray for incomplete/unavailable
            v_clamped = max(0.0, min(1.0, float(val)))
            if v_clamped <= 0.5:
                # Green (0, 200, 0) to Yellow (0, 220, 220) in BGR
                ratio = v_clamped / 0.5
                b = 0
                g = int(round(200 + ratio * 20))
                r = int(round(ratio * 220))
            else:
                # Yellow (0, 220, 220) to Red (0, 0, 230) in BGR
                ratio = (v_clamped - 0.5) / 0.5
                b = 0
                g = int(round(220 * (1.0 - ratio)))
                r = int(round(220 + ratio * 10))
            return (b, g, r)

        for zid, zdef in ZONE_DEFINITIONS.items():
            zx1, zy1, zx2, zy2 = zdef["box"]
            zt = self.zones[zid]

            if metric == "lighting":
                val = zt.mean_lighting_deficit
                label_val = f"L: {val:.2f}"
            elif metric == "mobility":
                val = zt.mean_mobility_factor
                label_val = f"M: {val:.2f}"
            elif metric == "zvi":
                zvi_dict = compute_zvi(zt.mean_lighting_deficit, manual_infrastructure, zt.mean_mobility_factor)
                if zvi_dict["is_complete"]:
                    val = zvi_dict["zvi"] / 100.0
                    label_val = f"ZVI: {zvi_dict['zvi']:.1f}"
                else:
                    val = None
                    label_val = "ZVI: Incomplete (I pending)"
            else:
                val = zt.mean_lighting_deficit
                label_val = f"L: {val:.2f}"

            color = get_heat_color(val)
            # Tint zone
            cv2.rectangle(overlay, (zx1, zy1), (zx2, zy2), color, -1)

        # Blend semi-transparent overlay
        cv2.addWeighted(overlay, alpha, vis, 1.0 - alpha, 0, vis)

        # Draw grid borders and text labels
        for zid, zdef in ZONE_DEFINITIONS.items():
            zx1, zy1, zx2, zy2 = zdef["box"]
            zt = self.zones[zid]
            cv2.rectangle(vis, (zx1, zy1), (zx2, zy2), (255, 255, 255), 2)

            # Zone HUD text
            zt_mean_v = zt.mean_vehicles
            zt_mean_p = zt.mean_persons
            title_text = f"Z{zid}: {zdef['name'].split()[1][1:-1]}"
            sub_text1 = f"Deficit L: {zt.mean_lighting_deficit:.2f}"
            sub_text2 = f"Avg Ped: {zt_mean_p:.1f} | Veh: {zt_mean_v:.1f}"

            if metric == "zvi":
                zvi_d = compute_zvi(zt.mean_lighting_deficit, manual_infrastructure, zt.mean_mobility_factor)
                score_str = f"ZVI: {zvi_d['zvi']:.1f}" if zvi_d["is_complete"] else "ZVI: Incomplete (I pending)"
            elif metric == "mobility":
                score_str = f"Mobility M: {zt.mean_mobility_factor:.2f}"
            else:
                score_str = f"Lighting L: {zt.mean_lighting_deficit:.2f}"

            # Badge box
            badge_x = zx1 + 10
            badge_y = zy1 + 12
            cv2.rectangle(vis, (badge_x, badge_y), (badge_x + 220, badge_y + 80), (15, 15, 15), -1)
            cv2.rectangle(vis, (badge_x, badge_y), (badge_x + 220, badge_y + 80), (200, 200, 200), 1)

            cv2.putText(vis, title_text, (badge_x + 8, badge_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(vis, sub_text1, (badge_x + 8, badge_y + 38), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 255), 1, cv2.LINE_AA)
            cv2.putText(vis, sub_text2, (badge_x + 8, badge_y + 54), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)
            cv2.putText(vis, score_str, (badge_x + 8, badge_y + 72), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 180), 1, cv2.LINE_AA)

        # Top Header Banner
        cv2.rectangle(vis, (0, 0), (CANONICAL_WIDTH, 30), (20, 20, 20), -1)
        cv2.putText(
            vis,
            f"DRISHTI-EDGE VIDEO-SCENE SPATIAL MAP (3x3 CANONICAL 1280x720 GRID) | VIEW: {metric.upper()}",
            (15, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 200),
            1,
            cv2.LINE_AA,
        )

        return vis
