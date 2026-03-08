#!/usr/bin/env bash
set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/Bard-Ha/ECESSP-Final.git}"
APP_DIR="${APP_DIR:-$HOME/ECESSP}"
API_DOMAIN="${API_DOMAIN:-api.ecessp.online}"
WEB_ORIGINS="${WEB_ORIGINS:-https://ecessp.online,https://www.ecessp.online}"

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "Missing required command: $1" >&2
    exit 1
  }
}

set_env_value() {
  local file="$1"
  local key="$2"
  local value="$3"

  if grep -q "^${key}=" "$file"; then
    sed -i "s|^${key}=.*|${key}=${value}|" "$file"
  else
    printf '%s=%s\n' "$key" "$value" >>"$file"
  fi
}

install_docker() {
  if command -v docker >/dev/null 2>&1; then
    return
  fi

  sudo apt update
  sudo apt install -y ca-certificates curl gnupg lsb-release git
  sudo install -m 0755 -d /etc/apt/keyrings
  curl -fsSL https://download.docker.com/linux/ubuntu/gpg | \
    sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
  echo \
    "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
    $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
    sudo tee /etc/apt/sources.list.d/docker.list >/dev/null
  sudo apt update
  sudo apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
  sudo usermod -aG docker "$USER" || true
}

install_caddy() {
  if command -v caddy >/dev/null 2>&1; then
    return
  fi

  sudo apt install -y debian-keyring debian-archive-keyring apt-transport-https
  curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' | \
    sudo gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
  curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' | \
    sudo tee /etc/apt/sources.list.d/caddy-stable.list >/dev/null
  sudo apt update
  sudo apt install -y caddy
}

prepare_repo() {
  if [ -d "$APP_DIR/.git" ]; then
    git -C "$APP_DIR" fetch origin
    git -C "$APP_DIR" pull --ff-only
  else
    git clone "$REPO_URL" "$APP_DIR"
  fi
}

prepare_env() {
  local env_file="$APP_DIR/ecessp-ml/.env"

  if [ ! -f "$env_file" ]; then
    cp "$APP_DIR/ecessp-ml/.env.example" "$env_file"
  fi

  set_env_value "$env_file" "PORT" "8000"
  set_env_value "$env_file" "USE_GPU" "0"
  set_env_value "$env_file" "ECESSP_ALLOWED_ORIGINS" "$WEB_ORIGINS"
  set_env_value "$env_file" "REQUIRE_API_KEY" "0"
  set_env_value "$env_file" "RATE_LIMIT_ENABLED" "1"
  set_env_value "$env_file" "RATE_LIMIT_REQUESTS" "120"
  set_env_value "$env_file" "RATE_LIMIT_WINDOW_SEC" "60"
  set_env_value "$env_file" "REQUEST_TIMEOUT_SEC" "60"
}

prepare_asset_dirs() {
  mkdir -p "$APP_DIR/ecessp-ml/data/processed"
  mkdir -p "$APP_DIR/ecessp-ml/graphs"
  mkdir -p "$APP_DIR/ecessp-ml/reports/models"
}

configure_caddy() {
  local src="$APP_DIR/deploy/Caddyfile.api.ecessp.online"
  local tmp
  tmp="$(mktemp)"
  sed "s/api\.ecessp\.online/${API_DOMAIN}/g" "$src" >"$tmp"
  sudo cp "$tmp" /etc/caddy/Caddyfile
  rm -f "$tmp"
  sudo systemctl reload caddy
}

start_backend() {
  cd "$APP_DIR"
  docker compose up -d --build ecessp-ml
}

health_check() {
  sleep 5
  curl --fail --silent http://127.0.0.1:8000/health >/dev/null
  echo "Backend health check passed on http://127.0.0.1:8000/health"
}

main() {
  require_cmd git
  require_cmd curl
  install_docker
  install_caddy
  prepare_repo
  prepare_env
  prepare_asset_dirs

  cat <<EOF
Upload these runtime assets to $APP_DIR/ecessp-ml before relying on the API:
  data/processed/material_catalog.csv
  data/processed/batteries_parsed.csv
  data/processed/batteries_parsed_curated.csv
  data/processed/batteries_ml_curated.csv
  graphs/masked_battery_graph_normalized_v2.pt
  graphs/battery_hetero_graph_v1.pt
  reports/training_summary.json
  reports/final_family_ensemble_manifest.json
  reports/three_model_ensemble_manifest.json
  reports/active_learning_queue.jsonl
  reports/models/
EOF

  configure_caddy
  start_backend
  health_check
}

main "$@"
