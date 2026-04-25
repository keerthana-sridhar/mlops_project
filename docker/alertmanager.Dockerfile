FROM prom/alertmanager

USER root

COPY docker/alertmanager-entrypoint.sh /usr/local/bin/alertmanager-entrypoint.sh
RUN chmod +x /usr/local/bin/alertmanager-entrypoint.sh

USER nobody

ENTRYPOINT ["/usr/local/bin/alertmanager-entrypoint.sh"]