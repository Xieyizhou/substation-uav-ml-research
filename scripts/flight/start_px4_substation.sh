#!/usr/bin/env bash
# Launch PX4 SITL with this repository's substation Gazebo world.
#
# The script copies the local SDF world into the PX4 checkout and starts
# `make px4_sitl gz_x500`. It does not modify the flight code in PX4; it only
# provides the world file required by this project's experiments.
set -euo pipefail

CHECK_ONLY=false
case "${1:-}" in
  "") ;;
  --check)
    CHECK_ONLY=true
    shift
    ;;
  -h|--help)
    echo "Usage: bash scripts/flight/start_px4_substation.sh [--check]"
    echo "  --check  Validate paths and dependencies without copying files or starting PX4."
    exit 0
    ;;
  *)
    echo "ERROR: unknown argument: $1" >&2
    echo "Use --help for usage." >&2
    exit 2
    ;;
esac
if (( $# > 0 )); then
  echo "ERROR: unexpected argument: $1" >&2
  exit 2
fi

fail() {
  echo "ERROR: $*" >&2
  exit 1
}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DETECTED_PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
if [[ -n "${PROJECT_ROOT:-}" ]] && [[ "$PROJECT_ROOT" != "$DETECTED_PROJECT_ROOT" ]]; then
  echo "WARNING: ignoring stale PROJECT_ROOT=$PROJECT_ROOT" >&2
  echo "Using the repository that contains this launcher: $DETECTED_PROJECT_ROOT" >&2
fi
PROJECT_ROOT="$DETECTED_PROJECT_ROOT"
PX4_ROOT="${PX4_ROOT:-$HOME/PX4-Autopilot}"

[[ -d "$PROJECT_ROOT" ]] || fail "project root directory not found: $PROJECT_ROOT"
PROJECT_ROOT="$(cd -- "$PROJECT_ROOT" && pwd)"
[[ -f "$PROJECT_ROOT/main.py" ]] || fail \
  "invalid PROJECT_ROOT (main.py not found): $PROJECT_ROOT"

MAP_SWITCHER="$PROJECT_ROOT/scripts/maps/switch_map.py"
TARGET_PREPARER="$PROJECT_ROOT/scripts/maps/prepare_selected_world.py"
VEHICLE_PREPARER="$PROJECT_ROOT/scripts/maps/prepare_research_vehicle.py"
MAP_PYTHON="${UAV_SANDBOX_PYTHON:-$PROJECT_ROOT/.venv/bin/python}"
if [[ ! -x "$MAP_PYTHON" ]]; then
  MAP_PYTHON=""
  while IFS= read -r candidate; do
    if [[ -x "$candidate" ]] && "$candidate" -c \
      'import sys; raise SystemExit(sys.version_info < (3, 11))' 2>/dev/null; then
      MAP_PYTHON="$candidate"
      break
    fi
  done < <(type -aP python3 2>/dev/null || true)
fi
[[ -n "$MAP_PYTHON" ]] || fail \
  "Compatible Python 3.11+ was not found. Create .venv or select a verified runtime."
GZ_EXECUTABLE="${UAV_SANDBOX_GZ_EXECUTABLE:-$(command -v gz || true)}"

if [[ -z "${WORLD_NAME+x}" ]] && [[ -f "$MAP_SWITCHER" ]]; then
  MAP_ID="$("$MAP_PYTHON" "$MAP_SWITCHER" current --field id)"
  WORLD_NAME="$("$MAP_PYTHON" "$MAP_SWITCHER" current --field world_name)"
  WORLD_RELATIVE_PATH="$("$MAP_PYTHON" "$MAP_SWITCHER" current --field world_file)"
  WORLD_SRC="$PROJECT_ROOT/$WORLD_RELATIVE_PATH"
  MODEL_SPAWN_POSE="$("$MAP_PYTHON" "$MAP_SWITCHER" current --field spawn_pose)"
else
  MAP_ID="${MAP_ID:-custom}"
  WORLD_NAME="${WORLD_NAME:-substation_simple}"
  WORLD_SRC="${WORLD_SRC:-$PROJECT_ROOT/simulation/worlds/${WORLD_NAME}.sdf}"
MODEL_SPAWN_POSE="${PX4_GZ_MODEL_POSE:--10,-10,0,0,0,0}"
fi
SIM_MODEL="${SIM_MODEL:-x500}"

[[ "$WORLD_NAME" =~ ^[A-Za-z0-9_-]+$ ]] || fail \
  "invalid world name '$WORLD_NAME'; use letters, numbers, underscores, or hyphens"
[[ "$SIM_MODEL" =~ ^[A-Za-z0-9_-]+$ ]] || fail \
  "invalid simulator model '$SIM_MODEL'"

WORLD_DST="$PX4_ROOT/Tools/simulation/gz/worlds/${WORLD_NAME}.sdf"
RUNTIME_DIR="$PROJECT_ROOT/.runtime"
PX4_PID_FILE="$RUNTIME_DIR/px4_launcher.pid"

echo "===================================="
echo "Starting PX4 + Gazebo Test Map"
echo "===================================="
echo "Project root: $PROJECT_ROOT"
echo "PX4 root:     $PX4_ROOT"
echo "Map ID:       $MAP_ID"
echo "World:        $WORLD_NAME"
echo "World file:   $WORLD_SRC"
echo "Model spawn:  $MODEL_SPAWN_POSE"
echo "Vehicle model: $SIM_MODEL"
echo

[[ -f "$WORLD_SRC" ]] || fail "world file not found: $WORLD_SRC"
[[ -d "$PX4_ROOT" ]] || fail "PX4 root directory not found: $PX4_ROOT"
[[ -f "$PX4_ROOT/Makefile" ]] || fail "PX4 Makefile not found: $PX4_ROOT/Makefile"
command -v make >/dev/null 2>&1 || fail "make command not found"
[[ -x "$GZ_EXECUTABLE" ]] || fail "Verified Gazebo 'gz' executable not found: $GZ_EXECUTABLE"
if [[ "$MAP_ID" != "custom" ]]; then
  [[ -f "$TARGET_PREPARER" ]] || fail "target world preparer not found: $TARGET_PREPARER"
  "$MAP_PYTHON" "$TARGET_PREPARER" --help >/dev/null || fail \
    "target world preparer could not be loaded"
fi
if [[ "$SIM_MODEL" == "x500_research" ]]; then
  RESEARCH_MODEL_SRC="${RESEARCH_MODEL_SRC:-$PROJECT_ROOT/simulation/models/x500_research/model.sdf}"
  [[ -f "$RESEARCH_MODEL_SRC" ]] || fail \
    "research vehicle model is missing"
  [[ -f "$VEHICLE_PREPARER" ]] || fail "research vehicle preparer is missing"
fi

if [[ "$CHECK_ONLY" == true ]]; then
  echo "Preflight check passed. No files were copied and PX4 was not started."
  exit 0
fi

mkdir -p "$RUNTIME_DIR"
if [[ -f "$PX4_PID_FILE" ]]; then
  previous_pid="$(cat "$PX4_PID_FILE")"
  if [[ "$previous_pid" =~ ^[0-9]+$ ]] && kill -0 "$previous_pid" 2>/dev/null; then
    previous_command="$(ps -p "$previous_pid" -o command= 2>/dev/null || true)"
    if [[ "$previous_command" == *"start_px4_substation.sh"* ]]; then
      echo "ERROR: this project already has a PX4 launcher running as PID $previous_pid." >&2
      echo "Stop that launcher before starting another one." >&2
      exit 1
    fi
  fi
  rm -f "$PX4_PID_FILE"
fi

# Never terminate simulator processes not owned by this launcher. A concurrent
# PX4 or Gazebo instance can share ports and transport namespaces, so fail with
# a precise conflict instead of relying on global process cleanup.
if pgrep -x px4 >/dev/null 2>&1; then
  fail "another PX4 instance is already running; stop it explicitly before launch"
fi
if pgrep -f '(^|/)gz sim' >/dev/null 2>&1; then
  fail "another Gazebo Sim instance is already running; stop it explicitly before launch"
fi

echo "Copying world file into PX4 Gazebo worlds folder..."
mkdir -p "$(dirname "$WORLD_DST")"
WORLD_COPY_SRC="$WORLD_SRC"
if [[ "$MAP_ID" != "custom" ]] && [[ -f "$TARGET_PREPARER" ]]; then
  RUNTIME_WORLD="$RUNTIME_DIR/worlds/${WORLD_NAME}.sdf"
  "$MAP_PYTHON" "$TARGET_PREPARER" \
    --map "$MAP_ID" \
    --source "$WORLD_SRC" \
    --output "$RUNTIME_WORLD"
  WORLD_COPY_SRC="$RUNTIME_WORLD"
fi
MAKE_SIM_MODEL="$SIM_MODEL"
unset PX4_GZ_MODEL_NAME
if [[ "$SIM_MODEL" == "x500_research" ]]; then
  RESEARCH_WORLD="$RUNTIME_DIR/worlds/${WORLD_NAME}_research.sdf"
  "$MAP_PYTHON" "$VEHICLE_PREPARER" \
    --source "$WORLD_COPY_SRC" \
    --output "$RESEARCH_WORLD" \
    --model x500_research \
    --name x500_research_0 \
    "--pose=$MODEL_SPAWN_POSE"
  WORLD_COPY_SRC="$RESEARCH_WORLD"
  # PX4 only creates Make targets for registered airframes. Use the standard
  # x500 airframe and attach its bridge to the custom model already in-world.
  MAKE_SIM_MODEL="x500"
  export PX4_GZ_MODEL_NAME="x500_research_0"
fi
[[ -f "$WORLD_COPY_SRC" ]] || fail "prepared world file not found: $WORLD_COPY_SRC"
cp "$WORLD_COPY_SRC" "$WORLD_DST"
if [[ "$SIM_MODEL" == "x500_research" ]]; then
  RESEARCH_MODEL_DST="$PX4_ROOT/Tools/simulation/gz/models/x500_research"
  mkdir -p "$RESEARCH_MODEL_DST"
  cp "$RESEARCH_MODEL_SRC" "$RESEARCH_MODEL_DST/model.sdf"
  cp "$PROJECT_ROOT/simulation/models/x500_research/model.config" "$RESEARCH_MODEL_DST/model.config"
fi

printf '%s\n' "$$" > "$PX4_PID_FILE"
cleanup_pid_file() {
  if [[ -f "$PX4_PID_FILE" ]] && [[ "$(cat "$PX4_PID_FILE")" == "$$" ]]; then
    rm -f "$PX4_PID_FILE"
  fi
}
trap cleanup_pid_file EXIT

cd "$PX4_ROOT"

if [[ -f ".venv/bin/activate" ]]; then
  echo "Activating PX4 virtual environment..."
  source .venv/bin/activate
else
  echo "PX4 .venv not found, continuing without activating it."
fi

if [[ -n "${UAV_SANDBOX_OPENCV_PREFIX:-}" ]]; then
  OPENCV_PREFIX="$UAV_SANDBOX_OPENCV_PREFIX"
elif command -v brew >/dev/null 2>&1; then
  if brew --prefix opencv@4 >/dev/null 2>&1; then
    OPENCV_PREFIX="$(brew --prefix opencv@4)"
  elif brew --prefix opencv >/dev/null 2>&1; then
    OPENCV_PREFIX="$(brew --prefix opencv)"
  else
    OPENCV_PREFIX=""
  fi
fi

if [[ -n "${OPENCV_PREFIX:-}" ]]; then
  export OpenCV_DIR="${OpenCV_DIR:-$OPENCV_PREFIX/lib/cmake/opencv4}"
  export CMAKE_PREFIX_PATH="$OPENCV_PREFIX${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"
  export PKG_CONFIG_PATH="$OPENCV_PREFIX/lib/pkgconfig${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"
  OPENCV_LEGACY_HEADER="$OPENCV_PREFIX/include/opencv4/opencv2/core/types_c.h"
  [[ -f "$OPENCV_LEGACY_HEADER" ]] || fail \
    "PX4 requires OpenCV 4 compatibility. Select a verified opencv@4 runtime."
  echo "OpenCV compatibility prefix: $OPENCV_PREFIX"
  echo "OpenCV_DIR=$OpenCV_DIR"
fi

if [[ -n "${UAV_SANDBOX_QT_PREFIX:-}" ]]; then
  QT5_PREFIX="$UAV_SANDBOX_QT_PREFIX"
elif command -v brew >/dev/null 2>&1; then
  if brew --prefix qt@5 >/dev/null 2>&1; then
    QT5_PREFIX="$(brew --prefix qt@5)"
  fi
fi

[[ -n "${QT5_PREFIX:-}" ]] || fail "Verified Qt 5 runtime was not found"
QT5_CONFIG="$QT5_PREFIX/lib/cmake/Qt5/Qt5Config.cmake"
[[ -f "$QT5_CONFIG" ]] || fail "Qt 5 CMake configuration was not found: $QT5_CONFIG"
export Qt5_DIR="${Qt5_DIR:-$QT5_PREFIX/lib/cmake/Qt5}"
export CMAKE_PREFIX_PATH="$QT5_PREFIX${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"
echo "Qt 5 compatibility prefix: $QT5_PREFIX"
echo "Qt5_DIR=$Qt5_DIR"

echo
echo "Launching PX4 SITL..."
echo "Do not close this terminal while flying."
echo

export PX4_GZ_MODEL_POSE="$MODEL_SPAWN_POSE"
if [[ -n "${OpenCV_DIR:-}" ]]; then
  echo "Refreshing PX4 build configuration for current Homebrew dependencies..."
  cmake \
    -S "$PX4_ROOT" \
    -B "$PX4_ROOT/build/px4_sitl_default" \
    -G Ninja \
    -U 'GSTREAMER_*' \
    -U 'PC_GSTREAMER_*' \
    -U 'GSTREAMER_APP_*' \
    -U 'PC_GSTREAMER_APP_*' \
    -DCONFIG=px4_sitl_default \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DOpenCV_DIR="$OpenCV_DIR" \
    -DQt5_DIR="$Qt5_DIR"
fi
PX4_GZ_WORLD="$WORLD_NAME" make px4_sitl "gz_$MAKE_SIM_MODEL"
