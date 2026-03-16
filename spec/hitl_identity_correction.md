# HITL Identity Correction for Bodypipe Multi-Person Pipeline

> **Deliverable:** Copy this file to `~/GVHMR/spec/hitl_identity_correction.md` so it can be used as a Claude Code handoff on Powerhouse.

## Context

The bodypipe multi-person motion capture pipeline (GVHMR) uses OC-SORT for person detection and tracking, but loses or swaps identities during occlusion events — people crossing paths, walking behind furniture, or leaving and re-entering frame. The existing Identity Inspector panel provides bbox editing and a basic swap operation, but lacks the tools needed to efficiently correct the most common failure modes: **track fragmentation** (same person gets two track IDs) and **silent identity swaps** (tracker follows the wrong person after a crossing).

This plan adds five features to the Identity Inspector panel, prioritized by impact. The implementation target is Powerhouse (`shinyscale@100.125.206.12`, RTX PRO 6000 Blackwell).

## Implementation Status

All 5 phases implemented in this commit. See the modified files below.

## Files Modified

| File | Role |
|------|------|
| `identity_panel.py` | All UI components, callbacks, session state. Every feature touches this. |
| `identity_tracking.py` | Data model. New `merge_identity_tracks()` and `split_identity_track()` functions. |
| `multi_person_split.py` | Pipeline orchestration. New `merge_persons()` and `split_person()` functions. |
| `gvhmr_gui.py` | GUI integration. `populate_panel()` output list updated. |
| `identity_reid.py` | Shape re-ID. Read-only (used by review scanner). |
| `identity_confidence.py` | Confidence scoring. Read-only (used by review scanner). |

## Phase Summary

### Phase 1: Confidence-Guided Review
- `compute_review_issues()` scans for low-confidence spans, potential swaps, track gaps, shape drift
- Review navigation row: Scan button, Next/Prev Issue buttons, issue summary
- Issue markers on confidence timeline plot (red triangles for swaps, orange squares for drift)
- Auto-scans on panel init

### Phase 2: Visual ReID Gallery
- `_extract_thumbnails()` crops person images from high-confidence keyframe frames
- `_build_gallery_items()` builds gallery with selected person first
- Gallery updates on person change, merge, split, reprocess

### Phase 3: Track Merge
- `merge_identity_tracks()` in identity_tracking.py — frame-by-frame merge logic
- `merge_persons()` in multi_person_split.py — full pipeline orchestration
- Merge UI: dropdown, merge button, undo button, status
- Single-level undo via operation_log

### Phase 4: Enhanced Identity Swap
- Existing `on_swap_ids()` now re-establishes betas and logs swaps for undo
- Swap undo: re-executes swap (self-inverse)

### Phase 5: Track Split
- `split_identity_track()` in identity_tracking.py — splits at frame boundary
- `split_person()` in multi_person_split.py — full pipeline orchestration
- Split button: splits selected person at current frame

## Verification Plan

### Per-phase testing

**Phase 1 (Review):** Run a multi-person video with a known crossing. Verify "Scan" detects the low-confidence span and potential swap. Verify "Next Issue" navigates to the correct frame. Verify issue markers appear on the confidence plot.

**Phase 2 (Gallery):** Run pipeline. Verify 3 thumbnails per person extracted from high-confidence frames. Verify gallery updates when switching persons. Verify thumbnails refresh after reprocessing.

**Phase 3 (Merge):** Process a video where a person walks behind furniture (track fragments into person_0 + person_2). Verify merge combines them into one continuous track. Verify the gap gets bridged. Verify undo restores original state.

**Phase 4 (Swap):** Process a video with two people crossing. Verify swap at the crossing frame + reprocess produces correct per-person output. Verify undo reverses the swap.

**Phase 5 (Split):** Process a video where tracker merged two people. Verify split creates two valid separate tracks. Verify each gets their own isolation + GVHMR output.
