#!/bin/bash
#
# fix_data_filenames.sh
#
# Creates symlinks/copies to make data files match expected names in workload definitions.
# Principle: Add symlinks/copies instead of changing workload files.
#
# Usage:
#   ./scripts/fix_data_filenames.sh         # Dry run (show what would be done)
#   ./scripts/fix_data_filenames.sh --apply # Actually create symlinks
#   ./scripts/fix_data_filenames.sh --undo  # Remove created symlinks and restore originals
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

MODE="dry-run"
if [[ "$1" == "--apply" ]]; then
    MODE="apply"
elif [[ "$1" == "--undo" ]]; then
    MODE="undo"
fi

create_symlink() {
    local target="$1"  # Original file
    local link="$2"    # New symlink

    if [[ "$MODE" == "undo" ]]; then
        if [[ -L "$link" ]]; then
            rm "$link"
            echo "  [REMOVED] Symlink: $(basename "$link")"
        fi
        return
    fi

    if [[ ! -f "$target" ]]; then
        echo "  [SKIP] Target not found: $target"
        return
    fi

    if [[ -f "$link" && ! -L "$link" ]]; then
        echo "  [SKIP] File already exists (not a symlink): $link"
        return
    fi

    if [[ -L "$link" ]]; then
        echo "  [SKIP] Symlink already exists: $(basename "$link")"
        return
    fi

    if [[ "$MODE" == "dry-run" ]]; then
        echo "  [DRY-RUN] Would create symlink: $(basename "$link") -> $(basename "$target")"
    else
        ln -s "$(basename "$target")" "$link"
        echo "  [CREATED] Symlink: $(basename "$link") -> $(basename "$target")"
    fi
}

restore_rename() {
    local src="$1"
    local dst="$2"

    if [[ "$MODE" != "undo" ]]; then
        return
    fi

    # If dst exists and src doesn't, rename back
    if [[ -f "$dst" && ! -L "$dst" && ! -f "$src" ]]; then
        mv "$dst" "$src"
        echo "  [RESTORED] $(basename "$dst") -> $(basename "$src")"
    fi
}

echo "=============================================="
echo "Data File Symlink Script"
echo "=============================================="
echo "Project root: $PROJECT_ROOT"
echo "Mode: $MODE"
echo ""

# -----------------------------------------------------------------------------
# 1. Wildfire: Create .tsv symlinks pointing to .csv files
#    (Workload uses both .csv and .tsv references)
# -----------------------------------------------------------------------------
echo "1. Wildfire domain - create .tsv symlinks to .csv files"
WILDFIRE_DIR="$PROJECT_ROOT/data/wildfire/input"

# First, undo any previous renames
restore_rename "$WILDFIRE_DIR/nifc_suppression_costs.csv" "$WILDFIRE_DIR/nifc_suppression_costs.tsv"
restore_rename "$WILDFIRE_DIR/nifc_human_caused_acres.csv" "$WILDFIRE_DIR/nifc_human_caused_acres.tsv"

# Create symlinks
create_symlink "$WILDFIRE_DIR/nifc_suppression_costs.csv" "$WILDFIRE_DIR/nifc_suppression_costs.tsv"
create_symlink "$WILDFIRE_DIR/nifc_human_caused_acres.csv" "$WILDFIRE_DIR/nifc_human_caused_acres.tsv"

echo ""

# -----------------------------------------------------------------------------
# 2. Astronomy: Create omni2.txt symlink pointing to omni2.text
# -----------------------------------------------------------------------------
echo "2. Astronomy domain - create omni2.txt symlink to omni2.text"
ASTRONOMY_DIR="$PROJECT_ROOT/data/astronomy/input"

# First, undo any previous renames
restore_rename "$ASTRONOMY_DIR/omni2_low_res/omni2.text" "$ASTRONOMY_DIR/omni2_low_res/omni2.txt"

# Create symlink
create_symlink "$ASTRONOMY_DIR/omni2_low_res/omni2.text" "$ASTRONOMY_DIR/omni2_low_res/omni2.txt"

echo ""

# -----------------------------------------------------------------------------
# 3. Environment: Beach names - No changes needed
# -----------------------------------------------------------------------------
echo "3. Environment domain - beach name files"
echo "  [INFO] No changes needed for beach name files"
echo "  [INFO] 'Pleasure Bay Beach' in workload refers to 'pleasure_bay_and_castle_island_beach_datasheet.csv'"
echo "  [INFO] This requires fuzzy matching logic update, not file renaming"

echo ""
echo "=============================================="
case "$MODE" in
    "dry-run")
        echo "Dry run complete. Run with --apply to execute."
        ;;
    "apply")
        echo "Symlinks created successfully."
        ;;
    "undo")
        echo "Undo complete. Symlinks removed, originals restored."
        ;;
esac
echo "=============================================="
