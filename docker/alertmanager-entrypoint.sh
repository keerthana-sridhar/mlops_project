#!/bin/sh
set -eu

CONFIG_PATH="/etc/alertmanager/alertmanager.yml"

if [ -n "${SMTP_SMARTHOST:-}" ] && [ -n "${ALERT_EMAIL_FROM:-}" ] && [ -n "${ALERT_EMAIL_TO:-}" ]; then
cat > "${CONFIG_PATH}" <<EOF
global:
  resolve_timeout: 5m
  smtp_smarthost: "${SMTP_SMARTHOST}"
  smtp_from: "${ALERT_EMAIL_FROM}"
  smtp_auth_username: "${SMTP_AUTH_USERNAME:-${ALERT_EMAIL_FROM}}"
  smtp_auth_password: "${SMTP_AUTH_PASSWORD:-}"
  smtp_require_tls: ${SMTP_REQUIRE_TLS:-true}

route:
  receiver: default
  group_wait: 0s
  group_interval: 5s
  repeat_interval: 10s

receivers:
  - name: default
    webhook_configs:
      - url: http://backend:8000/internal/alert
        send_resolved: true
    email_configs:
      - to: "${ALERT_EMAIL_TO}"
        send_resolved: true
        headers:
          subject: "[Malaria Monitor] {{ .CommonLabels.alertname }}"
EOF
else
cat > "${CONFIG_PATH}" <<'EOF'
global:
  resolve_timeout: 5m

route:
  receiver: default
  group_wait: 0s
  group_interval: 5s
  repeat_interval: 10s

receivers:
  - name: default
    webhook_configs:
      - url: http://backend:8000/internal/alert
        send_resolved: true
EOF
fi

exec /bin/alertmanager --config.file="${CONFIG_PATH}" --storage.path=/alertmanager
