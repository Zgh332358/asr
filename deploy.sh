#!/usr/bin/env bash
# =============================================================================
# StepFun Cloud ASR API — Non-invasive Deployment Script
# =============================================================================
# Runs the cloud gateway in an isolated Python environment.
# - Creates an isolated venv (does NOT touch system Python)
# - No local model download or GPU is required
# - Manages the server via PID file
# - Supports HTTP proxy for pip installs
#
# Usage:
#   bash deploy.sh start          # Start the server
#   bash deploy.sh stop           # Stop the server
#   bash deploy.sh restart        # Restart the server
#   bash deploy.sh status         # Check server status
#   bash deploy.sh logs [N]       # Tail last N lines of logs (default 50)
#   bash deploy.sh setup          # Install dependencies (no start)
#   bash deploy.sh update         # Git pull + pip install + restart
#   bash deploy.sh auto-update    # Start background auto-update daemon
#   bash deploy.sh auto-update-stop # Stop the auto-update daemon
#   bash deploy.sh install-systemd # Install systemd service + timer units
# =============================================================================

set -euo pipefail

# --- Configuration -----------------------------------------------------------
# These can be overridden by environment variables or a .env file.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Load .env if present
if [ -f ".env" ]; then
    set -a
    source .env
    set +a
fi

# Defaults (matching .env.example)
APP_NAME="whisper_api"
VENV_DIR="${VENV_DIR:-venv}"
PID_FILE="${PID_FILE:-${APP_NAME}.pid}"
AUTO_UPDATE_PID_FILE="${AUTO_UPDATE_PID_FILE:-${APP_NAME}_auto_update.pid}"
LOG_DIR="${LOG_DIR:-logs}"
LOG_FILE="${LOG_DIR}/${APP_NAME}.log"
AUTO_UPDATE_LOG="${LOG_DIR}/auto_update.log"
AUTO_UPDATE_INTERVAL="${AUTO_UPDATE_INTERVAL:-300}"  # seconds between checks (default 5 min)

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8080}"
ASR_MODEL="${ASR_MODEL:-stepaudio-3-chat-preview}"
OPENAI_BASE_URL="${OPENAI_BASE_URL:-https://api.stepfun.com/v1}"
OPENAI_API_KEY="${OPENAI_API_KEY:-}"
DEFAULT_LANGUAGE="${DEFAULT_LANGUAGE:-zh}"
LOG_LEVEL="${LOG_LEVEL:-info}"
LOG_FORMAT="${LOG_FORMAT:-json}"

# Proxy settings
HTTP_PROXY="${HTTP_PROXY:-}"
HTTPS_PROXY="${HTTPS_PROXY:-}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# --- Helpers -----------------------------------------------------------------

log_info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*"; }
log_step()  { echo -e "${BLUE}[STEP]${NC}  $*"; }

die() {
    log_error "$*"
    exit 1
}

# --- Proxy Setup -------------------------------------------------------------

setup_proxy() {
    if [ -n "$HTTP_PROXY" ]; then
        log_info "Using configured HTTP proxy"
        export http_proxy="$HTTP_PROXY"
        export HTTP_PROXY="$HTTP_PROXY"
    fi
    if [ -n "$HTTPS_PROXY" ]; then
        log_info "Using configured HTTPS proxy"
        export https_proxy="$HTTPS_PROXY"
        export HTTPS_PROXY="$HTTPS_PROXY"
    fi
}

# --- Prerequisite Checks -----------------------------------------------------

check_prerequisites() {
    log_step "Checking prerequisites..."

    # Python 3.10+
    if ! command -v python3 &>/dev/null; then
        die "python3 not found. Please install Python 3.10+."
    fi

    PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
    log_info "Python version: $PYTHON_VERSION"

    if ! command -v ffmpeg &>/dev/null || ! command -v ffprobe &>/dev/null; then
        die "ffmpeg/ffprobe are required. Install them with your system package manager."
    fi
}

# --- Virtual Environment -----------------------------------------------------

setup_venv() {
    log_step "Setting up virtual environment..."

    if [ ! -d "$VENV_DIR" ]; then
        python3 -m venv "$VENV_DIR"
        log_info "Created venv at $VENV_DIR"
    else
        log_info "Venv already exists at $VENV_DIR"
    fi

    # Activate venv
    source "$VENV_DIR/bin/activate" 2>/dev/null || source "$VENV_DIR/Scripts/activate" 2>/dev/null

    # Upgrade pip
    log_info "Upgrading pip..."
    pip install --upgrade pip -q

    # Install dependencies
    log_info "Installing Python dependencies..."
    pip install -r requirements.txt -q

    log_info "Dependencies installed successfully"
}

# --- Server Management -------------------------------------------------------

is_running() {
    if [ -f "$PID_FILE" ]; then
        local pid
        pid=$(cat "$PID_FILE" 2>/dev/null || echo "")
        if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
            # Guard against PID recycling: verify the process is actually our
            # uvicorn server, not an unrelated process that reused the PID.
            if [ -r "/proc/$pid/cmdline" ]; then
                local cmdline
                cmdline=$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null)
                case "$cmdline" in
                    *uvicorn*|*app.main*) return 0 ;;
                    *)
                        log_warn "PID $pid in $PID_FILE is not the ASR server (recycled?) — ignoring"
                        rm -f "$PID_FILE"
                        return 1
                        ;;
                esac
            fi
            return 0
        fi
    fi
    return 1
}

start_server() {
    if is_running; then
        local pid
        pid=$(cat "$PID_FILE")
        log_warn "Server is already running (PID: $pid)"
        return 1
    fi

    log_step "Starting server..."

    # Ensure log directory exists
    mkdir -p "$LOG_DIR"

    # Activate venv
    source "$VENV_DIR/bin/activate" 2>/dev/null || source "$VENV_DIR/Scripts/activate" 2>/dev/null

    # Export env vars for the server process
    export HOST PORT ASR_MODEL OPENAI_BASE_URL OPENAI_API_KEY
    export DEFAULT_LANGUAGE LOG_LEVEL LOG_FORMAT

    # Start server in background
    nohup python3 -m uvicorn app.main:app \
        --host "$HOST" \
        --port "$PORT" \
        --log-level "$LOG_LEVEL" \
        --workers 1 \
        >> "$LOG_FILE" 2>&1 &

    local pid=$!
    echo "$pid" > "$PID_FILE"

    # Poll process liveness; readiness remains 503 until a key is configured
    log_info "Waiting for server to start..."
    local waited=0
    local max_wait=60
    while [ $waited -lt $max_wait ]; do
        if command -v curl &>/dev/null; then
            if curl -sf "http://127.0.0.1:${PORT}/health/live" >/dev/null 2>&1; then
                break
            fi
        fi
        sleep 2
        waited=$((waited + 2))
    done

    if kill -0 "$pid" 2>/dev/null; then
        log_info "Server started successfully (PID: $pid)"
        log_info "Listening on http://${HOST}:${PORT}"
        log_info "Health check: http://${HOST}:${PORT}/health/live"
        log_info "API docs:   http://${HOST}:${PORT}/docs"
        log_info "Logs:       tail -f $LOG_FILE"
    else
        log_error "Server failed to start. Check logs:"
        tail -20 "$LOG_FILE"
        rm -f "$PID_FILE"
        return 1
    fi
}

stop_server() {
    if ! is_running; then
        log_warn "Server is not running"
        rm -f "$PID_FILE"
        return 0
    fi

    local pid
    pid=$(cat "$PID_FILE")
    log_step "Stopping server (PID: $pid)..."

    # Send SIGTERM for graceful shutdown
    kill -TERM "$pid" 2>/dev/null || true

    # Wait up to 30 seconds for graceful shutdown
    local waited=0
    while kill -0 "$pid" 2>/dev/null && [ $waited -lt 30 ]; do
        sleep 1
        waited=$((waited + 1))
    done

    # Force kill if still running
    if kill -0 "$pid" 2>/dev/null; then
        log_warn "Server did not stop gracefully, force killing..."
        kill -KILL "$pid" 2>/dev/null || true
        sleep 1
    fi

    rm -f "$PID_FILE"
    log_info "Server stopped"
}

show_status() {
    if is_running; then
        local pid
        pid=$(cat "$PID_FILE")
        echo -e "${GREEN}Server is running${NC}"
        echo "  PID:       $pid"
        echo "  Host:      $HOST"
        echo "  Port:      $PORT"
        echo "  Model:     $ASR_MODEL"
        echo "  Log file:  $LOG_FILE"

        # Try health check
        if command -v curl &>/dev/null; then
            echo ""
            echo "Health check:"
            curl -s "http://${HOST}:${PORT}/health/live" 2>/dev/null || echo "  (unreachable)"
        fi
    else
        echo -e "${RED}Server is not running${NC}"
        rm -f "$PID_FILE"
    fi
}

show_logs() {
    local lines="${1:-50}"
    if [ -f "$LOG_FILE" ]; then
        tail -n "$lines" "$LOG_FILE"
    else
        log_warn "No log file found at $LOG_FILE"
    fi
}

# --- Update ------------------------------------------------------------------

GIT_REMOTE="${GIT_REMOTE:-origin}"
GIT_BRANCH="${GIT_BRANCH:-master}"

do_update() {
    log_step "Checking for updates from ${GIT_REMOTE}/${GIT_BRANCH}..."

    # Ensure we're in a git repo
    if ! git rev-parse --git-dir &>/dev/null; then
        log_warn "Not a git repository — skipping git operations"
        return 1
    fi

    local remote_url
    remote_url=$(git remote get-url "$GIT_REMOTE") || die "Configured Git remote is unavailable"
    case "$remote_url" in
        https://github.com/Zgh332358/asr|https://github.com/Zgh332358/asr.git|git@github.com:Zgh332358/asr.git) ;;
        *) die "Update is restricted to the Zgh332358/asr fork. Review your Git remote." ;;
    esac

    local stashed=false
    if ! git diff --quiet || ! git diff --cached --quiet; then
        die "Working tree has local changes; review them before updating."
    fi

    # Fetch latest
    if ! git fetch "$GIT_REMOTE" "$GIT_BRANCH" 2>/dev/null; then
        log_error "git fetch failed — check network or GIT_REMOTE/GIT_BRANCH"
        return 1
    fi

    LOCAL=$(git rev-parse HEAD 2>/dev/null || echo "")
    REMOTE=$(git rev-parse "${GIT_REMOTE}/${GIT_BRANCH}" 2>/dev/null || echo "")

    if [ -z "$LOCAL" ] || [ -z "$REMOTE" ]; then
        log_error "Could not determine local or remote HEAD"
        return 1
    fi

    if [ "$LOCAL" = "$REMOTE" ]; then
        log_info "Already up-to-date (${LOCAL:0:8})"
        if $stashed; then
            log_info "Restoring stashed local changes"
            git stash pop 2>/dev/null || true
        fi
        return 0
    fi

    # Show what's new
    log_info "Updates available — new commits:"
    git log --oneline "${LOCAL}..${REMOTE}" 2>/dev/null || true
    echo ""

    # Pull
    if ! git pull --ff-only "$GIT_REMOTE" "$GIT_BRANCH" 2>/dev/null; then
        log_error "git pull failed"
        return 1
    fi

    log_info "Code updated to ${REMOTE:0:8}"

    # Update Python dependencies
    log_step "Updating Python dependencies..."
    source "$VENV_DIR/bin/activate" 2>/dev/null || source "$VENV_DIR/Scripts/activate" 2>/dev/null
    pip install -r requirements.txt -q 2>/dev/null || log_warn "pip install had warnings"

    if $stashed; then
        log_info "Attempting to re-apply stashed local changes (may conflict)..."
        git stash pop 2>/dev/null || log_warn "Stash pop had conflicts — local changes in working tree"
    fi

    log_info "Update complete"
    return 2  # Return 2 means "updated" (caller can decide to restart)
}

# --- Auto-Update Daemon -----------------------------------------------------

auto_update_is_running() {
    if [ -f "$AUTO_UPDATE_PID_FILE" ]; then
        local pid
        pid=$(cat "$AUTO_UPDATE_PID_FILE" 2>/dev/null || echo "")
        if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
            return 0
        fi
    fi
    return 1
}

auto_update_daemon() {
    # The actual daemon loop — runs in background via nohup
    log_info "Auto-update daemon started (PID: $$, interval: ${AUTO_UPDATE_INTERVAL}s)"
    log_info "Auto-update log: $AUTO_UPDATE_LOG"

    while true; do
        sleep "$AUTO_UPDATE_INTERVAL"

        echo "[$(date -Iseconds)] Checking for updates..." >> "$AUTO_UPDATE_LOG"

        # Capture update output
        local result=0
        do_update >> "$AUTO_UPDATE_LOG" 2>&1 || result=$?

        if [ "$result" = "2" ]; then
            # Code was updated — restart the server
            echo "[$(date -Iseconds)] Update applied, restarting server..." >> "$AUTO_UPDATE_LOG"
            if is_running; then
                # Source env for restart
                setup_proxy >> "$AUTO_UPDATE_LOG" 2>&1 || true
                source "$VENV_DIR/bin/activate" 2>/dev/null || source "$VENV_DIR/Scripts/activate" 2>/dev/null
                stop_server >> "$AUTO_UPDATE_LOG" 2>&1 || true
                sleep 2
                start_server >> "$AUTO_UPDATE_LOG" 2>&1 || {
                    echo "[$(date -Iseconds)] [ERROR] Server restart failed!" >> "$AUTO_UPDATE_LOG"
                }
                echo "[$(date -Iseconds)] Server restarted" >> "$AUTO_UPDATE_LOG"
            fi
        fi

        # Rotate auto-update log if > 1MB
        if [ -f "$AUTO_UPDATE_LOG" ]; then
            local logsize
            logsize=$(stat -c%s "$AUTO_UPDATE_LOG" 2>/dev/null || stat -f%z "$AUTO_UPDATE_LOG" 2>/dev/null || echo 0)
            if [ "$logsize" -gt 1048576 ] 2>/dev/null; then
                mv "$AUTO_UPDATE_LOG" "${AUTO_UPDATE_LOG}.old" 2>/dev/null || true
            fi
        fi
    done
}

start_auto_update() {
    if auto_update_is_running; then
        local pid
        pid=$(cat "$AUTO_UPDATE_PID_FILE")
        log_warn "Auto-update daemon is already running (PID: $pid)"
        return 1
    fi

    log_step "Starting auto-update daemon (interval: ${AUTO_UPDATE_INTERVAL}s)..."

    # Activate venv if it exists (so git/python/pip are available)
    if [ -d "$VENV_DIR" ]; then
        source "$VENV_DIR/bin/activate" 2>/dev/null || source "$VENV_DIR/Scripts/activate" 2>/dev/null
    fi

    mkdir -p "$LOG_DIR"

    # Start daemon in background
    nohup bash "$0" _auto_update_daemon >> "$AUTO_UPDATE_LOG" 2>&1 &

    local pid=$!
    echo "$pid" > "$AUTO_UPDATE_PID_FILE"

    if kill -0 "$pid" 2>/dev/null; then
        log_info "Auto-update daemon started (PID: $pid)"
        log_info "  Check interval: ${AUTO_UPDATE_INTERVAL}s"
        log_info "  Log file:       $AUTO_UPDATE_LOG"
    else
        log_error "Auto-update daemon failed to start"
        rm -f "$AUTO_UPDATE_PID_FILE"
        return 1
    fi
}

stop_auto_update() {
    if ! auto_update_is_running; then
        log_warn "Auto-update daemon is not running"
        rm -f "$AUTO_UPDATE_PID_FILE"
        return 0
    fi

    local pid
    pid=$(cat "$AUTO_UPDATE_PID_FILE")
    log_step "Stopping auto-update daemon (PID: $pid)..."

    kill -TERM "$pid" 2>/dev/null || true
    sleep 1
    kill -KILL "$pid" 2>/dev/null || true

    rm -f "$AUTO_UPDATE_PID_FILE"
    log_info "Auto-update daemon stopped"
}

# --- Systemd Installation ---------------------------------------------------

install_systemd() {
    log_step "Installing systemd service units..."

    if ! command -v systemctl &>/dev/null; then
        die "systemctl not found — systemd is required"
    fi

    local SVC_DIR="${SCRIPT_DIR}/deploy/systemd"

    if [ ! -f "${SVC_DIR}/whisper-asr.service" ]; then
        die "Systemd unit files not found at ${SVC_DIR}"
    fi

    # Ensure venv exists (run 'bash deploy.sh setup' first if missing)
    if [ ! -f "${SCRIPT_DIR}/${VENV_DIR}/bin/python" ]; then
        log_warn "Virtual environment not found. Run 'bash deploy.sh setup' first."
        log_step "Running setup now..."
        setup_proxy
        check_prerequisites
        setup_venv
    fi

    # --- Create whisper system user ---
    if ! id -u whisper &>/dev/null; then
        log_info "Creating system user 'whisper'..."
        sudo useradd -r -s /usr/sbin/nologin -M whisper 2>/dev/null || {
            # Some distros use different flags; try without -M
            sudo useradd -r -s /usr/sbin/nologin whisper 2>/dev/null || \
            die "Failed to create 'whisper' user"
        }
        log_info "System user 'whisper' created"
    else
        log_info "User 'whisper' already exists"
    fi

    # Ensure whisper owns the install directory (so git/python can work)
    sudo chown -R whisper:whisper "$SCRIPT_DIR"

    # --- Determine writable paths from .env ---
    local TEMP_DIR_VAL="/tmp/whisper_api"
    local STORAGE_PATH_VAL="/data/asr_storage"
    local LOG_DIR_VAL="${SCRIPT_DIR}/logs"

    if [ -f "${SCRIPT_DIR}/.env" ]; then
        # Source .env safely — only extract simple KEY=VALUE pairs
        local env_temp_dir
        env_temp_dir=$(grep -E '^TEMP_DIR=' "${SCRIPT_DIR}/.env" 2>/dev/null | tail -1 | cut -d= -f2- | tr -d '"' | tr -d "'" || echo "")
        [ -n "$env_temp_dir" ] && TEMP_DIR_VAL="$env_temp_dir"

        local env_storage_path
        env_storage_path=$(grep -E '^STORAGE_PATH=' "${SCRIPT_DIR}/.env" 2>/dev/null | tail -1 | cut -d= -f2- | tr -d '"' | tr -d "'" || echo "")
        [ -n "$env_storage_path" ] && STORAGE_PATH_VAL="$env_storage_path"
    fi

    # Ensure writable directories exist with correct ownership
    mkdir -p "$TEMP_DIR_VAL" "$LOG_DIR_VAL"
    sudo chown -R whisper:whisper "$TEMP_DIR_VAL" "$LOG_DIR_VAL" 2>/dev/null || true

    if [ -n "$STORAGE_PATH_VAL" ]; then
        mkdir -p "$STORAGE_PATH_VAL"
        sudo chown -R whisper:whisper "$STORAGE_PATH_VAL" 2>/dev/null || true
    fi

    # Build ReadWritePaths (space-separated, systemd format)
    local RW_PATHS="${LOG_DIR_VAL} ${TEMP_DIR_VAL}"
    [ -n "$STORAGE_PATH_VAL" ] && RW_PATHS="${RW_PATHS} ${STORAGE_PATH_VAL}"

    # There are no downloaded model files in the cloud-backed service.
    local RO_PATHS="${SCRIPT_DIR}/app"

    # --- Copy unit files ---
    sudo cp "${SVC_DIR}/whisper-asr.service" /etc/systemd/system/
    sudo cp "${SVC_DIR}/whisper-asr-update.service" /etc/systemd/system/
    sudo cp "${SVC_DIR}/whisper-asr-update.timer" /etc/systemd/system/

    # Replace placeholder paths in service files with actual install path
    sudo sed -i "s|WorkingDirectory=/opt/whisper-asr|WorkingDirectory=${SCRIPT_DIR}|g" \
        /etc/systemd/system/whisper-asr.service \
        /etc/systemd/system/whisper-asr-update.service
    sudo sed -i "s|/opt/whisper-asr|${SCRIPT_DIR}|g" \
        /etc/systemd/system/whisper-asr.service \
        /etc/systemd/system/whisper-asr-update.service

    # Replace ReadWritePaths / ReadOnlyPaths placeholders
    sudo sed -i "s|ReadWritePaths=__READ_WRITE_PATHS__|ReadWritePaths=${RW_PATHS}|g" \
        /etc/systemd/system/whisper-asr.service
    sudo sed -i "s|ReadOnlyPaths=__READ_ONLY_PATHS__|ReadOnlyPaths=${RO_PATHS}|g" \
        /etc/systemd/system/whisper-asr.service

    # --- Reload and enable ---
    sudo systemctl daemon-reload

    log_info "Enabling whisper-asr.service..."
    sudo systemctl enable whisper-asr.service

    log_info "Auto-update timer remains opt-in."
    # Auto-update timer is opt-in; do not enable it during service setup.

    # --- Start services ---
    log_info "Starting whisper-asr.service..."
    sudo systemctl start whisper-asr.service || log_warn "Service start failed — check 'sudo journalctl -u whisper-asr -f'"

    log_info "Auto-update timer was not started."
    # Enable manually only after reviewing the update policy.

    log_info "Systemd units installed and started."
    echo ""
    echo "  Service control:"
    echo "    sudo systemctl start whisper-asr"
    echo "    sudo systemctl stop whisper-asr"
    echo "    sudo systemctl status whisper-asr"
    echo "    sudo journalctl -u whisper-asr -f"
    echo ""
    echo "  Auto-update timer:"
    echo "    sudo systemctl status whisper-asr-update.timer"
    echo "    sudo systemctl list-timers whisper-asr-update"
    echo ""
    echo "  To change the update check interval:"
    echo "    sudo systemctl edit whisper-asr-update.timer"
    echo "    # Add under [Timer]: OnUnitActiveSec=600"
}

# --- Main --------------------------------------------------------------------

main() {
    local cmd="${1:-}"

    case "$cmd" in
        setup)
            setup_proxy
            check_prerequisites
            setup_venv
            log_info "Setup complete. Run 'bash deploy.sh start' to start the server."
            ;;

        start)
            setup_proxy
            check_prerequisites
            setup_venv
            start_server
            ;;

        stop)
            stop_server
            ;;

        restart)
            stop_server
            sleep 2
            # Re-export proxy vars in case they were lost after stop
            setup_proxy
            start_server
            ;;

        status)
            show_status
            ;;

        logs)
            show_logs "${2:-50}"
            ;;

        update)
            setup_proxy
            local result=0
            do_update || result=$?
            if [ "$result" = "2" ]; then
                # If running as root (e.g., from systemd update service), fix file ownership
                # so the main service (which runs as whisper) can read updated files
                if [ "$(id -u)" = "0" ]; then
                    log_info "Fixing file ownership for whisper user..."
                    chown -R whisper:whisper "$SCRIPT_DIR" 2>/dev/null || true
                fi

                # Code was updated — restart if server is running
                # Detect if managed by systemd and restart via systemctl
                if command -v systemctl &>/dev/null && systemctl is-active --quiet whisper-asr 2>/dev/null; then
                    log_info "Restarting whisper-asr via systemctl..."
                    if [ "$(id -u)" = "0" ]; then
                        systemctl restart whisper-asr || log_warn "systemctl restart failed"
                    else
                        sudo systemctl restart whisper-asr || log_warn "systemctl restart failed"
                    fi
                elif is_running; then
                    log_info "Restarting server (PID-based) to apply updates..."
                    stop_server
                    sleep 2
                    start_server
                fi
            fi
            ;;

        auto-update)
            # Warn if systemd is managing the service (timer handles updates)
            if command -v systemctl &>/dev/null && systemctl is-active --quiet whisper-asr 2>/dev/null; then
                log_warn "Systemd is managing whisper-asr. Use the timer for auto-updates:"
                log_warn "  sudo systemctl status whisper-asr-update.timer"
                log_warn "Starting daemon anyway (may conflict with systemd timer)..."
            fi
            start_auto_update
            ;;

        auto-update-stop)
            stop_auto_update
            ;;

        install-systemd)
            install_systemd
            ;;

        _auto_update_daemon)
            # Internal: called by nohup to run the daemon loop
            auto_update_daemon
            ;;

        *)
            echo "StepFun Cloud ASR API — Deployment Script"
            echo ""
            echo "Usage: bash deploy.sh <command> [options]"
            echo ""
            echo "Commands:"
            echo "  setup             Install dependencies and download model (don't start)"
            echo "  start             Start the server (setup + start)"
            echo "  stop              Stop the server gracefully"
            echo "  restart           Stop then start the server"
            echo "  status            Show server status and health"
            echo "  logs [N]          Tail last N lines of logs (default: 50)"
            echo "  update            Git pull + pip install + restart (if running)"
            echo "  auto-update       Start background auto-update daemon"
            echo "  auto-update-stop  Stop the auto-update daemon"
            echo "  install-systemd   Install systemd service + timer units (requires sudo)"
            echo ""
            echo "Auto-update config (.env or export):"
            echo "  AUTO_UPDATE_INTERVAL  Seconds between checks (default: 300 = 5 min)"
            echo "  GIT_REMOTE            Git remote name (default: origin)"
            echo "  GIT_BRANCH            Git branch to track (default: master)"
            echo ""
            echo "Environment (.env or export):"
            echo "  OPENAI_API_KEY       StepFun key (server-only)"
            echo "  ASR_MODEL            StepFun model ID"
            echo "  HTTP_PROXY           Proxy for pip/downloads"
            echo "  PORT                 Server port (default: 8080)"
            echo "  OPENAI_BASE_URL      Defaults to https://api.stepfun.com/v1"
            echo ""
            echo "First time: copy .env.example to .env and configure."
            exit 0
            ;;
    esac
}

main "$@"
