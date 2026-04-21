FROM prom/prometheus

COPY monitoring/prometheus/prometheus.yml /etc/prometheus/prometheus.yml
COPY alert_rules.yml /etc/prometheus/alert_rules.yml
