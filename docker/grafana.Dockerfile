FROM grafana/grafana

COPY monitoring/grafana/provisioning /etc/grafana/provisioning
COPY grafana.json /etc/grafana/dashboards/malaria-dashboard.json
