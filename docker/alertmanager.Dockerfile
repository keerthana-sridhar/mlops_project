FROM prom/alertmanager

COPY monitoring/alertmanager/alertmanager.yml /etc/alertmanager/alertmanager.yml
